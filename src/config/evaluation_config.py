"""
Evaluation Config

Specifies the hyperparameters for the evaluation process, i.e. what metrics to compute, etc.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from src.config._constants import MAX_SEQ_LEN


@dataclass
class PalomaEvaluationConfig:
    dataset_name: str = "pico-lm/pretokenized-paloma-tinsy"
    dataset_split: str = "val"
    max_length: int = MAX_SEQ_LEN
    batch_size: int = 16


@dataclass
class UniversalNEREvaluationConfig:
    dataset_name: str = "universalner/universal_ner"
    dataset_split: str = "test"  # or "val", depending on your use case
    batch_size: int = 16


@dataclass
class EvaluationConfig:
    # Evaluation metrics to compute: by default, we compute the perplexity of the model
    metrics: Optional[List[str]] = field(default_factory=lambda: ["paloma"])

    # NOTE: Add other evaluation configs here
    # Each evaluation metric should have its own config
    paloma: PalomaEvaluationConfig = field(default_factory=PalomaEvaluationConfig)
    universal_ner: UniversalNEREvaluationConfig = field(
        default_factory=UniversalNEREvaluationConfig
    )
