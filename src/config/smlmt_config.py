# src/config/smlmt.py

from dataclasses import dataclass, field


@dataclass
class ClassifierHeadConfig:
    # hidden size of the MLP head
    hidden_dim: int = 256
    # dropout between layers
    dropout: float = 0.1
    # how many layers (not counting final projection)
    num_layers: int = 2
    # initialization for all head weights
    init_method: str = "xavier"


@dataclass
class SMLMTConfig:
    """
    Default Semi-Supervised Meta-Learning (SMLMT) configuration.

    Fields:
      enabled: Whether to enable SMLMT loss during training.
      hybrid_ratio: Probability of choosing an SMLMT batch vs. AR batch.
      min_token_freq: Minimum frequency for a token to be considered maskable.
      max_token_freq: Maximum frequency for a token to be considered maskable.
    """

    enabled: bool = True
    hybrid_ratio: float = 0.5
    min_token_freq: int = 1
    max_token_freq: int = 1000000
    inner_steps: int = 10
    inner_lr: float = 0.001
    k_shot: int = 4
    n_way: int = 32
    classifier_head: ClassifierHeadConfig = field(default_factory=ClassifierHeadConfig)
