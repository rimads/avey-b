from datasets import load_dataset
import torch
import os
from typing import List


class DataLoader_MLM:
    def __init__(self,
                 B: int,
                 T: int,
                 process_rank: int,
                 num_processes: int,
                 tokenizer: str,
                 **kwargs):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        dataset_name = "HuggingFaceFW/fineweb"
        dataset_config = "sample-350BT"
        self.dataset = load_dataset(
            dataset_name,
            name=dataset_config,
            split="train",
            num_proc=os.cpu_count()
        )
        self.dataset_length = len(self.dataset)

        # load tokenizer and add a mask token
        self.tokenizer = tokenizer

        self.reset()

    def reset(self) -> None:
        # interleave documents across processes
        self.current_idx = self.process_rank
        self.current_buffer: List[int] = []

    def _get_next_tokens(self) -> None:
        # fill buffer until we have at least B*T tokens
        required = self.B * self.T
        while len(self.current_buffer) < required:
            if self.current_idx >= self.dataset_length:
                self.current_idx = self.process_rank  # wrap around
            text = self.dataset[self.current_idx]["text"]
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if self.tokenizer.eos_token_id is not None:
                tokens.append(self.tokenizer.eos_token_id)
            self.current_buffer.extend(tokens)
            self.current_idx += self.num_processes

    def next_batch(self, mask_fraction):
        """
        Args:
            mask_fraction: fraction of total tokens to mask (replace with mask_token_id)

        Returns:
            x: (B, T) input token IDs, with some replaced by mask_token_id
            y: (B, T) labels, original token IDs at masked positions, -100 elsewhere
        """
        self._get_next_tokens()
        B, T = self.B, self.T
        total = B * T

        # 1) grab next total tokens and remove from buffer
        buf = torch.tensor(self.current_buffer[:total], dtype=torch.long)
        self.current_buffer = self.current_buffer[total:]

        # 2) clone for x and y
        buf_flat = buf.view(-1)
        x_flat = buf_flat.clone()
        y_flat = torch.full_like(buf_flat, -100)

        # 3) compute how many tokens to mask
        n_mask = int(round(mask_fraction * total))

        # 4) enforce at least one token masked
        if n_mask < 1:
            n_mask = 1

        # 5) enforce at least one token left untouched
        if n_mask > total - 1:
            n_mask = total - 1

        # 6) sample positions to mask
        device = buf_flat.device
        mask_pos = torch.randperm(total, device=device)[:n_mask]

        # 7) apply masking
        x_flat[mask_pos] = self.tokenizer.mask_token_id
        y_flat[mask_pos] = buf_flat[mask_pos]

        # 8) reshape back to (B, T)
        x = x_flat.view(B, T)
        y = y_flat.view(B, T)

        return x, y

    def state_dict(self):
        """
        Return a serializable dict capturing the current streaming state.
        current_buffer may be large — acceptable for resuming exact position.
        """
        return {
            "current_idx": self.current_idx,
            "current_buffer": list(self.current_buffer),  # ensure plain list of ints
        }

    def load_state_dict(self, state: dict):
        """
        Restore loader state from a dict previously returned by state_dict().
        """
        self.current_idx = int(state["current_idx"])
        # restore buffer as list[int]
        self.current_buffer = list(state["current_buffer"])
