import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from .configuration_avey import AveyConfig


class StaticLayer(nn.Module):
    def __init__(self, config: AveyConfig):
        super().__init__()
        self.norm = nn.RMSNorm(config.d_embed, eps=config.eps)
        self.enricher = nn.Linear(config.d_embed, config.d_embed * 4)

        proj_size = config.chunk_size
        self.spatial_proj = nn.Parameter(torch.empty(proj_size, proj_size))
        nn.init.xavier_normal_(self.spatial_proj)

        self.fuser = nn.Linear(int(config.d_embed * 3), config.d_embed)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    @torch.compile
    def forward(self, x):
        _, T, _ = x.shape
        res = x
        x = self.norm(x)
        x = self.enricher(x)
        x = F.gelu(x)
        x, bypass = x.chunk(2, dim=-1)

        x, gate = x.chunk(2, dim=-1)
        x = self.spatial_proj[:T, :T] @ x
        x = gate * x

        x = torch.cat([x, bypass], dim=-1)
        x = self.fuser(x)
        return x + (self.alpha * res)


class DynamicLayer(nn.Module):
    def __init__(self, config: AveyConfig):
        super().__init__()
        self.norm = nn.RMSNorm(config.d_embed, eps=config.eps)
        self.enricher = nn.Linear(config.d_embed, config.d_embed * 4)
        self.fuser = nn.Linear(config.d_embed * 3, config.d_embed)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    @torch.compile
    def forward(self, x):
        _, T, _ = x.shape
        res = x
        x = self.norm(x)
        x = self.enricher(x)
        x = F.gelu(x)
        x, bypass = x.chunk(2, dim=-1)

        x, gate = x.chunk(2, dim=-1)
        x_norm = F.normalize(x, p=2, dim=-1)
        sim_scores = x_norm @ x_norm.mT
        x = F.normalize(sim_scores, p=1, dim=-1) @ x
        x = gate * x

        x = torch.cat([x, bypass], dim=-1)
        x = self.fuser(x)
        return x + (self.alpha * res)


class Ranker(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size = config.chunk_size
        self.k = config.k + 1
        self.extended_len = self.k * config.chunk_size
        self.eps = config.eps
        self.down_proj = nn.Parameter(torch.empty(self.chunk_size, self.extended_len))
        nn.init.xavier_normal_(self.down_proj)

    def preprocess(self, x):
        B, T, E = x.shape
        cs, L = self.chunk_size, self.extended_len

        padded = False
        orig_T = T
        if T % cs != 0:
            pad_len = cs - (T % cs)
            pad = torch.zeros(B, pad_len, E, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
            T += pad_len
            padded = True

        N = T // cs
        x_chunks = x.view(B, N, cs, E)

        extended = []
        for i in range(0, N):
            cur = x_chunks[:, i]
            others = x_chunks[:, :i]
            cat = self._extend(others, cur)  # (B, ≤k⋅cs+cs, E)

            # pad or truncate to length L
            cur_len = cat.size(1)
            if cur_len < L:
                pad2 = torch.zeros(B, L - cur_len, E, device=x.device, dtype=x.dtype)
                cat = torch.cat([pad2, cat], dim=1)
            else:
                cat = cat[:, -L:]

            extended.append(cat)

        ext = torch.stack(extended, dim=1)  # (B, N, L, E)
        ext = (self.down_proj @ ext) + x_chunks
        h = ext.view(B * N, cs, E)

        state = {"B": B, "N": N, "orig_T": orig_T, "padded": padded}
        return h, state

    def contract(self, h, st):
        B, cs = st["B"], self.chunk_size
        N = st["N"]
        padded = st["padded"]
        orig_T = st["orig_T"]

        E = h.size(-1)
        final_chunks = h.view(B, N, cs, E)

        out = final_chunks.reshape(B, N * cs, E)

        if padded:
            out = out[:, :orig_T, :]

        return out

    def _extend(self, other_chunks, cur_chunk):
        B, cs, E = cur_chunk.shape
        if other_chunks is None or other_chunks.size(1) == 0:
            return cur_chunk

        i = other_chunks.size(1)
        num_sel = min(i, self.k - 1)
        if num_sel <= 0:
            return cur_chunk

        # l2 normalize
        cn = other_chunks / (other_chunks.norm(dim=-1, keepdim=True) + self.eps)
        cm = cur_chunk / (cur_chunk.norm(dim=-1, keepdim=True) + self.eps)

        # cosine sim
        cm_e = cm.unsqueeze(1)  # (B, 1, cs, E)
        ct = cn.transpose(-1, -2)  # (B, i, E, cs)
        sims = torch.matmul(cm_e, ct)  # (B, i, cs, cs)
        mx, _ = sims.max(dim=-1)  # (B, i, cs)
        scores = mx.sum(dim=-1)  # (B, i)

        # topk
        topk_vals, topk_idx = scores.topk(num_sel, dim=1)

        # normalize weights
        v_min = topk_vals.min(dim=-1, keepdim=True)[0]  # (B, 1)
        w = topk_vals / (v_min + self.eps)  # (B, num_sel)
        w = w.unsqueeze(-1).unsqueeze(-1)  # (B, num_sel, 1, 1)

        # gather
        idx_e = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, cs, E)
        sel = other_chunks.gather(1, idx_e)  # (B, num_sel, cs, E)

        # weight & flatten
        wt = (sel * w).reshape(B, num_sel * cs, E)

        return torch.cat([wt, cur_chunk], dim=1)  # (B, ≤k⋅cs+cs, E)


class AveyPreTrainedModel(PreTrainedModel):
    config_class = AveyConfig

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class AveyModel(AveyPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.chunk_size = config.chunk_size
        self.embed = nn.Embedding(config.vocab_size, config.d_embed)
        self.layers = nn.ModuleList(
            [
                DynamicLayer(config) if (i + 1) % 2 == 0 else StaticLayer(config)
                for i in range(config.n_layers)
            ]
        )
        self.ranker = Ranker(config)
        self.apply(self._init_weights)

    def _get_hidden(self, input_ids):
        x = self.embed(input_ids)
        x, state = self.ranker.preprocess(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ranker.contract(x, state)
        return x

    def forward(self, input_ids, **kwargs):
        x = self._get_hidden(input_ids)
        return BaseModelOutput(last_hidden_state=x)


class AveyForMaskedLM(AveyModel):
    def __init__(self, config):
        super().__init__(config)
        self.apply(self._init_weights)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self._get_hidden(input_ids)
        logits = F.linear(x, self.embed.weight)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )

        return MaskedLMOutput(logits=logits, loss=loss)


class AveyForSequenceClassification(AveyModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dense = nn.Sequential(
            nn.Linear(self.config.d_embed, self.config.d_embed * 2),
            nn.GELU(),
            nn.Linear(self.config.d_embed * 2, self.config.d_embed * 2),
            nn.GELU(),
            nn.Linear(self.config.d_embed * 2, self.config.d_embed),
        )
        self.classifier = nn.Linear(config.d_embed, config.num_labels)
        self.apply(self._init_weights)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self._get_hidden(input_ids)
        x = x.mean(dim=1)
        x = self.dense(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = MSELoss()(logits.squeeze(), labels.squeeze())
            elif labels.dtype in (torch.long, torch.int):
                loss = CrossEntropyLoss()(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            else:
                loss = BCEWithLogitsLoss()(logits, labels)

        return SequenceClassifierOutput(logits=logits, loss=loss)


class AveyForTokenClassification(AveyModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dense = nn.Sequential(nn.Linear(config.d_embed, config.d_embed), nn.Tanh())
        self.classifier = nn.Linear(config.d_embed, config.num_labels)
        self.apply(self._init_weights)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self._get_hidden(input_ids)
        x = self.dense(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(logits=logits, loss=loss)
