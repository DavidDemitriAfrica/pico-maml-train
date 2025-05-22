#!/usr/bin/env python3
import gc
import logging
import random
import re

import pandas as pd
import torch
import wandb
from datasets import load_dataset
from huggingface_hub import list_repo_files
from torch.utils.data import DataLoader

import src.model.pico_decoder as _pdmod
from src.model.pico_decoder import PicoDecoderHF, PicoDecoderHFConfig, RoPE

_pdmod.RoPE.forward = lambda self, queries, keys, start_pos: (queries, keys)

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
SIZES = ["tiny", "small", "medium", "large"]
DATASET_NAME = "pico-lm/pretokenized-dolma"
BATCH_SIZE = 8
ENTITY = "pico-lm"
PROJECT = "pico-maml-analysis"

COMMIT_MAP = {
    "tiny": {
        0: "a6cdb56366f88792f5e4e15daf707e6c7b308bac",
        1000: "c3fee915709901f9b419c1832f765c00bf4a3f9a",
        2000: "3d74f9791df735882b8bf022b906ca0dbf107abc",
        3000: "45ad29a4dae0a4bad7a73cb43bd96799e6e45586",
        4000: "ed16792adee15d031d7845d4ae7c221a553ec0f1",
        5000: "7c36fee2a4deb3c6f2c57bce476c60ca8a678b59",
        6000: "8f42ded9c1c37cb188d68d46ba09aafa045a2a1d",
    },
    "small": {
        0: "530d5adedbc18916e1e2aeaa435c8ddd72342cd1",
        1000: "c31076b94449248738ca68f50ac3bd5dd3866794",
        2000: "551049ac959202f24266a4d6a7757e908f003928",
        3000: "e3bc6632c4f3a672f5f04209de2b85f587363b55",
        4000: "3f6139d5410da4c3a48e0def49ec45b74a6c5d82",
        5000: "efe54c3ee29fc420c999e2af83b815a35600d33b",
        6000: "1efec115d29eb94670bc3e62686c8b2b14acf2e0",
    },
    "medium": {
        0: "383771963911ed0fe9969baf863ef5a492f0c9c6",
        1000: "31219822df031d7d35bd7a7408c9e6dc43d72d7e",
        2000: "d99c5a544948ce84c99033ceae50a9ddf35117cd",
        3000: "b35ce01cdd3f83f411c8522263be6f68c6db40a0",
        4000: "217e0bc737a489ba58dd883701c21fc0b4e4e274",
        5000: "78ff1d279304cbb68cdc40aff80ec802e5a33287",
        6000: "46f9b7e6fbb7a075600fe12de0b351b6363620cf",
    },
    "large": {
        0: "09b4a5452d0ce34d1030fa20c61a4701ac37a77a",
        1000: "523d939cba8f8f37907f88f3fbd2a27338ef6200",
        2000: "cae3d876304a6e7a980acb15f8f95583934cabac",
        3000: "25d9636f6e2e34afde07fc5333cbd11ad349fb60",
        4000: "08a3d0620f5efafaf9e636dd0719ce86575c25ed",
        5000: "508d443aafa8020c7fc164733e02b2ccc255cde9",
        6000: "ce5fa8fe69acb265cf38773bd7f9c92325b863f3",
    },
}
REPO_MAP = {
    "tiny": "pico-decoder-tiny",
    "small": "pico-decoder-small",
    "medium": "pico-decoder-medium",
    "large": "pico-decoder-large",
}


def compute_er_per(tensor: torch.Tensor):
    mat = tensor.view(tensor.size(0), -1) if tensor.ndim > 2 else tensor
    S = torch.linalg.svd(mat, full_matrices=False).S
    p = S / S.sum()
    er = float(torch.exp(-torch.sum(p * torch.log(p))))
    per = er / p.numel()
    return er, per


def get_checkpoint_steps(repo_id):
    files = list_repo_files(repo_id)
    steps = {m.group(1) for f in files if (m := re.match(r"checkpoints/(step_\d+)", f))}
    return sorted(steps, key=lambda s: int(s.split("_")[1]))


def process_variant(size: str, variant: str):
    is_maml = variant == "maml"
    repo_id = (
        f"davidafrica/pico-maml-decoder-{size}"
        if is_maml
        else f"pico-lm/{REPO_MAP[size]}"
    )
    step_list = (
        [int(s.split("_")[1]) for s in get_checkpoint_steps(repo_id)]
        if is_maml
        else list(COMMIT_MAP[size].keys())
    )

    model_tag = f"{variant}-{size}"
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=f"{model_tag}_learning_dynamics",
        tags=["learning_dynamics", f"variant:{variant}", f"size:{size}"],
        reinit=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = []

    for step in sorted(step_list):
        logger.info(f"[{variant}:{size}] Processing step {step}")
        try:
            if is_maml:
                subfolder = f"checkpoints/step_{step}"
                cfg = PicoDecoderHFConfig.from_pretrained(
                    repo_id, trust_remote_code=True, subfolder=subfolder
                )
                RoPE._freqs_cis_tensor = None
                model = PicoDecoderHF.from_pretrained(
                    repo_id, config=cfg, trust_remote_code=True, subfolder=subfolder
                ).to(device)
            else:
                revision = COMMIT_MAP[size][step]
                cfg = PicoDecoderHFConfig.from_pretrained(
                    repo_id, trust_remote_code=True, revision=revision
                )
                RoPE._freqs_cis_tensor = None
                model = PicoDecoderHF.from_pretrained(
                    repo_id, config=cfg, trust_remote_code=True, revision=revision
                ).to(device)

            model.eval()
            name2weights = {
                n: p.detach().cpu() for n, p in model.named_parameters() if p.ndim >= 2
            }

            ds_loader = DataLoader(
                load_dataset(DATASET_NAME, split="train", streaming=True),
                batch_size=BATCH_SIZE,
                collate_fn=lambda batch: {
                    "input_ids": [ex["input_ids"] for ex in batch]
                },
            )
            batch = next(iter(ds_loader))
            input_ids = torch.tensor(batch["input_ids"], device=device)
            model.zero_grad()
            out = model(input_ids, return_dict=True)
            out.logits.mean().backward()

            name2grads = {
                n: p.grad.detach().cpu()
                for n, p in model.named_parameters()
                if (p.ndim >= 2 and p.grad is not None)
            }

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
                    # collect (ER, PER) tuples for this component
                    vals = [
                        compute_er_per(t) for n, t in name2tensor.items() if filt(n)
                    ]
                    if not vals:
                        # nothing to log for this comp
                        continue
                    ers, pers = zip(*vals)
                    avg_er, avg_per = sum(ers) / len(ers), sum(pers) / len(pers)
                    run.log(
                        {
                            f"{data_type}/{comp}/ER": avg_er,
                            f"{data_type}/{comp}/PER": avg_per,
                        },
                        step=step,
                    )
                    rec[f"{data_type}_{comp}_ER"] = avg_er
                    rec[f"{data_type}_{comp}_PER"] = avg_per

            records.append(rec)

        finally:
            model.cpu()
            del model, name2weights, name2grads, ds_loader, batch, input_ids
            torch.cuda.empty_cache()
            gc.collect()

    run.finish()
    pd.DataFrame(records).sort_values("step").to_csv(
        f"er_per_breakdown_{model_tag}.csv", index=False
    )


if __name__ == "__main__":
    for variant in ["maml", "vanilla"]:
        for sz in SIZES:
            logger.info(f"=== Processing variant={variant} size={sz} ===")
            process_variant(sz, variant)
