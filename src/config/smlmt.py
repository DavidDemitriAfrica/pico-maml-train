# src/config/smlmt.py

from dataclasses import dataclass


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

    enabled: bool = False
    hybrid_ratio: float = 0.6
    min_token_freq: int = 30
    max_token_freq: int = 100
