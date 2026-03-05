from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryModelConfig:
    vocab_size: int
    d_model: int = 64
    num_layers: int = 4
    num_heads: int = 4
    mlp_dim: int = 128
    memory_dim: int = 16
    num_memory_heads: int = 1
    alpha_init: float = 0.2
    beta_init: float = 0.1
    memory_write_init: float = 0.05
    memory_keep_init: float = 0.02
    memory_read_init: float = 0.2
    memory_write_interval: int = 2
    memory_write_layers: int = 1
    keep_write_scale: float = 0.0
    keep_read_scale: float = 0.25
    keep_token_mix_scale: float = 0.1
    rms_norm_eps: float = 1e-6
    max_seq_len: int = 256
    causal: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.memory_dim > self.d_model:
            raise ValueError("memory_dim must not exceed d_model.")
        if self.num_memory_heads > self.num_heads:
            raise ValueError("num_memory_heads must not exceed num_heads.")
        if self.memory_dim % self.num_memory_heads != 0:
            raise ValueError("memory_dim must be divisible by num_memory_heads.")
        if self.memory_write_interval < 1:
            raise ValueError("memory_write_interval must be positive.")
        if self.memory_write_layers < 1:
            raise ValueError("memory_write_layers must be positive.")


@dataclass(frozen=True)
class ProbeConfig:
    probe_layers: tuple[int, ...]
    k_eigs: int = 8
    power_iters: int = 20
    subspace_iters: int = 12
    subspace_oversample: int = 4
    seed: int = 0
    operator: str = "covariance"
    state_view: str = "memory"

    def __post_init__(self) -> None:
        if not self.probe_layers:
            raise ValueError("probe_layers must not be empty.")
        if self.k_eigs < 1:
            raise ValueError("k_eigs must be positive.")
        if self.power_iters < 1:
            raise ValueError("power_iters must be positive.")
        if self.subspace_iters < 1:
            raise ValueError("subspace_iters must be positive.")
        if self.subspace_oversample < 1:
            raise ValueError("subspace_oversample must be positive.")
        if self.state_view not in {"full", "token", "memory"}:
            raise ValueError("state_view must be one of: full, token, memory.")
