from transformers import PretrainedConfig


class AveyConfig(PretrainedConfig):
    model_type = "avey-b"

    def __init__(
        self,
        vocab_size: int = 50304,
        context_len: int = 512,
        d_embed: int = 768,
        n_layers: int = 26,
        chunk_size: int = 128,
        k: int = 3,
        eps=1e-12,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.n_layers = n_layers
        self.chunk_size = chunk_size
        self.k = k
        self.eps = eps

        # for compatibility with the eval lib
        self.max_position_embeddings = context_len
        self.hidden_size = d_embed
        super().__init__(**kwargs)
