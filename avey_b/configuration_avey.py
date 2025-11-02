from transformers import PretrainedConfig


class AveyConfig(PretrainedConfig):
    model_type = "avey"

    def __init__(
            self,
            vocab_size: int = 50304,
            context_len: int = 512,
            d_embed: int = 768,
            n_layers: int = 26,
            expansion_factor: int = 4,
            chunk_size: int = 128,
            k: int = 3,
            context_proportion: float = 0.5,
            eps=1e-12,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = context_len
        self.d_embed = d_embed
        self.hidden_size = d_embed
        self.n_layers = n_layers
        self.expansion_factor = expansion_factor
        self.chunk_size = chunk_size
        self.k = k
        self.context_proportion = context_proportion
        self.eps = eps
        super().__init__(**kwargs)
