"""
Paloma is a comprehensive evaluation benchmark for large language models (LLMs) that focuses
on measuring perplexity across diverse text domains.

To evaluate on Paloma, we use the huggingface evaluation framework.

For more details, see: https://huggingface.co/datasets/allenai/paloma
"""

import evaluate
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
import torch

# typing imports
from src.config.evaluation_config import PalomaEvaluationConfig


def run_paloma_evaluation(model_path: str, paloma_config: PalomaEvaluationConfig):
    disable_progress_bar()

    # 1) load the metric
    perplexity = evaluate.load("pico-lm/perplexity")

    # 2) load your data
    dataset = load_dataset(
        paloma_config.dataset_name, split=paloma_config.dataset_split
    )["text"]

    # 3) monkey-patch torch.cuda.is_available → False
    orig_cuda_avail = torch.cuda.is_available
    torch.cuda.is_available = lambda: False

    try:
        # 4) this compute will now “see” no GPU and load everything on CPU
        result = perplexity.compute(
            model_id=model_path,
            predictions=dataset,
            add_start_token=False,
            max_length=paloma_config.max_length,
            batch_size=paloma_config.batch_size,
            trust_remote_code=True,
        )
    finally:
        # 5) restore the original function
        torch.cuda.is_available = orig_cuda_avail

    enable_progress_bar()
    return result["mean_perplexity"]
