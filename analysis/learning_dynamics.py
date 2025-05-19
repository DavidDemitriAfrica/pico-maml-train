#!/usr/bin/env python3
import os

# ─── Fix for deterministic CuBLAS operations ─────────────────────────────────
# Must set BEFORE any CUDA/cuBLAS calls or torch.use_deterministic_algorithms()
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import logging
import random
import re

import pandas as pd
import torch
import wandb
from datasets import load_dataset
from huggingface_hub import list_repo_files
from torch.utils.data import DataLoader

# ─── Monkey-patch RoPE to be identity ────────────────────────────────────────
import src.model.pico_decoder as _pdmod
from src.model.pico_decoder import PicoDecoderHF, PicoDecoderHFConfig, RoPE

_pdmod.RoPE.forward = lambda self, queries, keys, start_pos: (queries, keys)

# ─── Reproducibility (no deterministic algorithms enforced) ─────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
SIZES = [  # "tiny",
    "small",
    "medium",
    "large",
]
BASE_REPO = "davidafrica/pico-maml-decoder-{}"
ENTITY = "pico-lm"
PROJECT = "pico-maml-analysis"

DATASET_NAME = "pico-lm/pretokenized-dolma"
BATCH_SIZE = 8


# ─── Helpers ─────────────────────────────────────────────────────────────────
def compute_er_per(tensor: torch.Tensor):
    """Return (ER, PER) for a 2D matrix or  >2D tensor (flattened to 2D)."""
    mat = tensor.view(tensor.size(0), -1) if tensor.ndim > 2 else tensor
    S = torch.linalg.svd(mat, full_matrices=False).S
    p = S / S.sum()
    er = float(torch.exp(-torch.sum(p * torch.log(p))))
    per = er / p.numel()
    return er, per


def get_checkpoint_steps(repo_id):
    """Scan `checkpoints/step_*/` folders in HF repo."""
    files = list_repo_files(repo_id)
    steps = {m.group(1) for f in files if (m := re.match(r"checkpoints/(step_\d+)", f))}
    return sorted(steps, key=lambda s: int(s.split("_")[1]))


# ─── Main processing function ────────────────────────────────────────────────
def process_variant(size: str):
    repo_id = BASE_REPO.format(size)
    steps = get_checkpoint_steps(repo_id)
    model_tag = repo_id.split("/")[-1]  # e.g. pico-maml-decoder-small

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=f"{model_tag}_learning_dynamics",
        tags=["learning_dynamics", "variant:maml", f"size:{size}"],
        reinit=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = []

    for step_dir in steps:
        step = int(step_dir.split("_")[1])
        subfolder = f"checkpoints/{step_dir}"

        # ─── Load model ────────────────────────────────────────────────────
        cfg = PicoDecoderHFConfig.from_pretrained(
            repo_id, trust_remote_code=True, subfolder=subfolder
        )
        RoPE._freqs_cis_tensor = None  # reset cache
        model = PicoDecoderHF.from_pretrained(
            repo_id, config=cfg, trust_remote_code=True, subfolder=subfolder
        ).to(device)
        model.eval()

        # 1) Collect all weight tensors
        name2weights = {
            n: p.detach().cpu() for n, p in model.named_parameters() if p.ndim >= 2
        }

        # 2) Pull one real batch → forward + backward for gradients
        ds_loader = DataLoader(
            load_dataset(DATASET_NAME, split="train", streaming=True),
            batch_size=BATCH_SIZE,
            collate_fn=lambda batch: {"input_ids": [ex["input_ids"] for ex in batch]},
        )
        batch = next(iter(ds_loader))
        input_ids = torch.tensor(batch["input_ids"], device=device)

        model.zero_grad()
        out = model(input_ids, return_dict=True)
        loss = out.logits.mean()
        loss.backward()

        name2grads = {
            n: p.grad.detach().cpu()
            for n, p in model.named_parameters()
            if (p.ndim >= 2 and p.grad is not None)
        }

        # 3) Break down by component
        components = {
            "mlp": lambda n: "mlp" in n.lower(),
            "attn": lambda n: ("attn" in n.lower()) or ("attention" in n.lower()),
        }

        rec = {"step": step}
        for data_type, name2tensor in [
            ("weights", name2weights),
            ("gradients", name2grads),
        ]:
            for comp, filt in components.items():
                ers, pers = [], []
                for n, t in name2tensor.items():
                    if filt(n):
                        er, per = compute_er_per(t)
                        ers.append(er)
                        pers.append(per)
                if not ers:
                    continue
                avg_er = sum(ers) / len(ers)
                avg_per = sum(pers) / len(pers)

                # log to W&B
                run.log(
                    {
                        f"{data_type}/{comp}/ER": avg_er,
                        f"{data_type}/{comp}/PER": avg_per,
                    },
                    step=step,
                )
                logger.info(
                    f"[{size}] step={step} "
                    f"{data_type}/{comp} → ER={avg_er:.4f}, PER={avg_per:.4f}"
                )

                rec[f"{data_type}_{comp}_ER"] = avg_er
                rec[f"{data_type}_{comp}_PER"] = avg_per

        records.append(rec)

    run.finish()

    # Save CSV
    df = pd.DataFrame(records).sort_values("step")
    out_csv = f"er_per_breakdown_{size}.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved {out_csv}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Ensure you've done `wandb login` in your environment
    for sz in SIZES:
        logger.info(f"=== Processing size={sz} ===")
        process_variant(sz)
        logger.info(f"=== Finished size={sz} ===")
