import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from .configuration_avey import AveyConfig


class Contextualizer(nn.Module):
    def __init__(self, config: AveyConfig, layer_idx):
        super().__init__()
        self.eps = config.eps
        self.layer_idx = layer_idx
        if self.layer_idx % 2 == 0:
            self.spatial_proj = nn.Parameter(
                torch.empty(config.chunk_size, config.chunk_size)
            )
            nn.init.xavier_normal_(self.spatial_proj)

    def cosim(self, embeddings: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.sum(embeddings**2, dim=-1, keepdim=True) + self.eps)
        normalized = embeddings / norm
        cosim = torch.matmul(normalized, normalized.transpose(-1, -2))
        return cosim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        x0, x1 = x.chunk(2, dim=-1)
        if self.layer_idx % 2 == 0:
            x0 = self.spatial_proj[:T, :T] @ x0
        else:
            sim_scores = self.cosim(x0)
            row_sums = sim_scores.sum(dim=-1, keepdim=True)
            sim_scores = sim_scores / (row_sums + self.eps)
            x0 = sim_scores @ x0
        output = x0 * x1
        return output


class ContextualizerLayer(nn.Module):
    def __init__(self, config: AveyConfig, layer_idx):
        super().__init__()
        expanded_dim = config.d_embed * config.expansion_factor
        self.split_factor = [
            int(expanded_dim * config.context_proportion),
            int(expanded_dim * (1 - config.context_proportion)),
        ]
        diff = expanded_dim - (self.split_factor[0] + self.split_factor[1])
        self.split_factor[1] += diff
        if self.split_factor[0] % 2 != 0:
            self.split_factor[0] += 1
            self.split_factor[1] -= 1

        self.enricher = nn.Linear(config.d_embed, expanded_dim)
        self.contextualizer = Contextualizer(config, layer_idx)
        proj_in_features = int(self.split_factor[0] / 2 + self.split_factor[1])
        self.fuser = nn.Linear(proj_in_features, config.d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = F.relu(self.enricher(x)).square()
        x0, x1 = x_proj.split(self.split_factor, dim=-1)
        x0 = self.contextualizer(x0)
        return self.fuser(torch.cat([x0, x1], dim=-1))


class AveyLayer(nn.Module):
    def __init__(self, config: AveyConfig, layer_idx):
        super().__init__()
        self.rms_norm = nn.RMSNorm(config.d_embed, eps=config.eps)
        self.ctxt = ContextualizerLayer(config, layer_idx)

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ctxt(self.rms_norm(x))


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

        state = {
            "B": B,
            "N": N,
            "orig_T": orig_T,
            "padded": padded,
        }
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

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

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
    def __init__(self, config: AveyConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_embed)
        self.layers = nn.ModuleList(
            [AveyLayer(config, i) for i in range(config.n_layers)]
        )
        self.ranker = Ranker(config)
        self.post_init()

    def forward(self, input_ids: torch.Tensor, attention_mask=None, **kwargs):
        h = self.embeddings(input_ids)
        if attention_mask is not None:
            h = h * attention_mask.unsqueeze(-1)

        B, T, E = h.shape
        padded = False
        orig_T = T
        if T % self.config.chunk_size != 0:
            pad_len = self.config.chunk_size - (T % self.config.chunk_size)
            pad_tensor = torch.zeros(B, pad_len, E, device=h.device, dtype=h.dtype)
            h = torch.cat([h, pad_tensor], dim=1)
            T = h.shape[1]
            padded = True

        h, state = self.ranker.preprocess(h)
        for layer in self.layers:
            h = layer(h)
        h = self.ranker.contract(h, state)

        if padded:
            h = h[:, :orig_T, :]

        return BaseModelOutput(last_hidden_state=h)


class AveyForMaskedLM(AveyPreTrainedModel):
    def __init__(self, config: AveyConfig):
        super().__init__(config)
        self.config = config

        self.base_avey_model = AveyModel(config)
        self.ln_f = nn.RMSNorm(config.d_embed, eps=config.eps)

        self.post_init()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        h = self.base_avey_model(input_ids, **kwargs).last_hidden_state
        logits = F.linear(self.ln_f(h), self.base_avey_model.embeddings.weight)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
            return MaskedLMOutput(logits=logits, loss=loss)

        return MaskedLMOutput(logits=logits)


class AveyForSequenceClassification(AveyPreTrainedModel):
    def __init__(self, config: AveyConfig, avey_model: AveyForMaskedLM = None):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        if avey_model is None:
            self.avey_model = AveyForMaskedLM(config)
        else:
            self.avey_model = avey_model

        self.classifier = nn.Linear(config.d_embed, config.num_labels)
        self.dense = nn.Sequential(
            nn.Linear(self.config.d_embed, self.config.d_embed * 2),
            nn.GELU(),
            nn.Linear(self.config.d_embed * 2, self.config.d_embed * 2),
            nn.GELU(),
            nn.Linear(self.config.d_embed * 2, self.config.d_embed),
        )
        self.post_init()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        h = self.avey_model.base_avey_model(input_ids, **kwargs).last_hidden_state
        h = h.mean(dim=1)
        h = self.avey_model.ln_f(h)
        h = self.dense(h)
        h = F.gelu(h)
        logits = self.classifier(h)
        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(logits=logits, loss=loss)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = AveyConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        archs = getattr(config, "architectures", [])
        is_mlm = any("MaskedLM" in a for a in archs)

        if is_mlm:
            mlm_model = AveyForMaskedLM.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
            return cls(config, avey_model=mlm_model)
        else:
            return super().from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )


class AveyForTokenClassification(AveyPreTrainedModel):
    def __init__(self, config: AveyConfig, avey_model: AveyForMaskedLM = None):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        if avey_model is None:
            self.avey_model = AveyForMaskedLM(config)
        else:
            self.avey_model = avey_model

        self.classifier = nn.Linear(config.d_embed, config.num_labels)
        self.dense = nn.Sequential(nn.Linear(config.d_embed, config.d_embed), nn.Tanh())
        self.post_init()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        outputs = self.avey_model.base_avey_model(input_ids, **kwargs)

        h = outputs.last_hidden_state
        h = self.avey_model.ln_f(h)
        h = self.dense(h)
        logits = self.classifier(h)
        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = AveyConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        archs = getattr(config, "architectures", [])
        is_mlm = any("MaskedLM" in a for a in archs)

        if is_mlm:
            mlm_model = AveyForMaskedLM.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
            return cls(config, avey_model=mlm_model)
        else:
            return super().from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
