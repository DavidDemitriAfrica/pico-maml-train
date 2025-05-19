#!/usr/bin/env python3
import logging
import random
import re

import pandas as pd
import torch
import wandb
from datasets import load_dataset
from huggingface_hub import list_repo_files
from torch.utils.data import DataLoader

# import your Pico model
from src.model.pico_decoder import PicoDecoderHF, PicoDecoderHFConfig, RoPE

# ─── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.benchmark = False

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────────────
SIZES = ["tiny", "small", "medium", "large"]
BASE_REPO = "davidafrica/pico-maml-decoder-{}"
ENTITY = "pico-lm"
PROJECT = "pico-maml-analysis"

DATASET_NAME = "pico-lm/pretokenized-dolma"
BATCH_SIZE = 8


# ─── Helpers ───────────────────────────────────────────────────────────────────
def compute_er_per(tensor: torch.Tensor):
    """Return (effective_rank, proportional_effective_rank)."""
    mat = tensor.view(tensor.size(0), -1) if tensor.ndim > 2 else tensor
    S = torch.linalg.svd(mat, full_matrices=False).S
    p = S / S.sum()
    er = float(torch.exp(-torch.sum(p * torch.log(p))))
    per = er / p.numel()
    return er, per


def get_checkpoint_steps(repo_id):
    """List all unique checkpoint step folders under checkpoints/."""
    files = list_repo_files(repo_id)
    steps = {m.group(1) for f in files if (m := re.match(r"checkpoints/(step_\d+)", f))}
    return sorted(steps, key=lambda s: int(s.split("_")[1]))


# ─── Main ─────────────────────────────────────────────────────────────────────
def process_variant(size):
    repo_id = BASE_REPO.format(size)
    steps = get_checkpoint_steps(repo_id)
    model_slug = repo_id.split("/")[-1]  # e.g. "pico-maml-decoder-small"

    # start W&B run
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=f"{model_slug}_learning_dynamics",
        tags=["learning_dynamics", "variant:maml", f"size:{size}"],
        reinit=True,
    )

    records = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for step_dir in steps:
        step = int(step_dir.split("_")[1])
        subfolder = f"checkpoints/{step_dir}"

        # load config + model
        cfg = PicoDecoderHFConfig.from_pretrained(
            repo_id, trust_remote_code=True, subfolder=subfolder
        )
        RoPE._freqs_cis_tensor = None
        model = PicoDecoderHF.from_pretrained(
            repo_id, config=cfg, trust_remote_code=True, subfolder=subfolder
        ).to(device)
        model.eval()

        # 1) collect all weight tensors
        name2weights = {
            n: p.detach().cpu() for n, p in model.named_parameters() if p.ndim >= 2
        }

        # 2) pull one real batch, forward + backward to get grads
        ds = load_dataset(DATASET_NAME, split="train", streaming=True)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=lambda b: b)
        batch = next(iter(loader))
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], device=device)
        model.zero_grad()
        out = model(input_ids, return_dict=True)
        loss = out.logits.mean()
        loss.backward()

        name2grads = {
            n: p.grad.detach().cpu()
            for n, p in model.named_parameters()
            if (p.ndim >= 2 and p.grad is not None)
        }

        # 3) break down by component
        components = {
            "mlp": lambda n: "mlp" in n.lower(),
            "attn": lambda n: ("attn" in n.lower()) or ("attention" in n.lower()),
        }

        rec = {"step": step}
        for data_type, name2tensor in [
            ("weights", name2weights),
            ("gradients", name2grads),
        ]:
            for comp, name_filter in components.items():
                ers, pers = [], []
                for n, t in name2tensor.items():
                    if name_filter(n):
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
                # record for CSV
                rec[f"{data_type}_{comp}_ER"] = avg_er
                rec[f"{data_type}_{comp}_PER"] = avg_per

        records.append(rec)

    run.finish()

    # save CSV locally
    df = pd.DataFrame(records).sort_values("step")
    out_csv = f"er_per_breakdown_{size}.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved {out_csv}")


if __name__ == "__main__":
    # ensure `wandb login` has been run
    for sz in SIZES:
        logger.info(f"=== Processing size={sz} ===")
        process_variant(sz)
