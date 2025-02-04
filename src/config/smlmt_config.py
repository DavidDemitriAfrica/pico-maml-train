# smlmt_config.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class SMLMTConfig:
    enabled: bool = False
    probability: float = 0.3
    num_classes: int = 3
    support_per_class: int = 2
    query_per_class: int = 2
    sentences: List[str] = field(default_factory=list)
    vocabulary: List[str] = field(default_factory=list)
