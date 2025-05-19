#!/usr/bin/env python3
import logging
import random
import re

import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from huggingface_hub import list_repo_files

from src.model.pico_decoder import PicoDecoderHF, PicoDecoderHFConfig, RoPE

# ─── Seeds & Determinism ───────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Model variants ────────────────────────────────────────────────────────────
SIZES = ["tiny", "small", "medium", "large"]
BASE_REPO = "davidafrica/pico-maml-decoder-{}"


# ─── ER / PER computation ──────────────────────────────────────────────────────
def compute_er_per(tensor: torch.Tensor):
    # flatten to 2D
    mat = tensor.view(tensor.size(0), -1) if tensor.ndim > 2 else tensor
    S = torch.linalg.svd(mat, full_matrices=False).S
    p = S / S.sum()
    er = float(torch.exp(-torch.sum(p * torch.log(p))))
    per = er / p.numel()
    return er, per


def get_checkpoint_steps(repo_id):
    # list all files in the repo, extract unique step dirs
    files = list_repo_files(repo_id)
    steps = {m.group(1) for f in files if (m := re.match(r"checkpoints/(step_\d+)", f))}
    return sorted(steps, key=lambda s: int(s.split("_")[1]))


# ─── Main logging routine ─────────────────────────────────────────────────────
def process_variant(size):
    repo_id = BASE_REPO.format(size)
    steps = get_checkpoint_steps(repo_id)
    model_slug = repo_id.split("/")[-1]  # e.g. pico-maml-decoder-small

    # start a new W&B run
    run = wandb.init(
        entity="pico-lm",
        project="pico-maml-analysis",
        name=f"{model_slug}_learning_dynamics",
        tags=["learning_dynamics", "variant:maml", f"size:{size}"],
        reinit=True,
    )

    records = []
    for step_dir in steps:
        step = int(step_dir.split("_")[1])
        subfolder = f"checkpoints/{step_dir}"

        # load config + model
        cfg = PicoDecoderHFConfig.from_pretrained(
            repo_id, trust_remote_code=True, subfolder=subfolder
        )
        RoPE._freqs_cis_tensor = None  # reset rotary frequencies
        model = PicoDecoderHF.from_pretrained(
            repo_id, config=cfg, trust_remote_code=True, subfolder=subfolder
        )
        model.eval()

        ers, pers = [], []
        for _, param in model.named_parameters():
            if param.ndim >= 2:
                er, per = compute_er_per(param.detach().cpu())
                ers.append(er)
                pers.append(per)

        avg_er = sum(ers) / len(ers)
        avg_per = sum(pers) / len(pers)
        records.append({"step": step, "ER": avg_er, "PER": avg_per})

        # log to W&B
        run.log({"ER": avg_er, "PER": avg_per}, step=step)
        logger.info(f"[{size}] step={step} → ER={avg_er:.4f}, PER={avg_per:.4f}")

    run.finish()

    # local CSV + plot
    df = pd.DataFrame(records).sort_values("step")
    csv_path = f"er_per_{size}.csv"
    plot_path = f"er_per_{size}.png"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {csv_path}")

    plt.figure()
    plt.plot(df["step"], df["ER"], label="Effective Rank")
    plt.plot(df["step"], df["PER"], label="Proportional Effective Rank")
    plt.xlabel("Training Step")
    plt.ylabel("Rank Metric")
    plt.title(f"ER & PER over Training ({size})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved {plot_path}")


if __name__ == "__main__":
    # ensure you’ve run `wandb login` first
    for sz in SIZES:
        logger.info(f"==== Processing size={sz} ====")
        process_variant(sz)
