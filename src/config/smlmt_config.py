# smlmt_config.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class SMLMTConfig:
    enabled: bool = False
    probability: float = 0.2
    num_classes: int = 3
    support_per_class: int = 2
    query_per_class: int = 2
    sentences: List[str] = field(default_factory=list)
    vocabulary: List[str] = field(default_factory=list)
    inner_lr: float = 0.001
    inner_steps: int = 10
    max_length: int = 1024
    hidden_dims: List[int] = field(default_factory=lambda: [96, 96])
    dropout: float = 0
    weight_decay: float = 0.1
