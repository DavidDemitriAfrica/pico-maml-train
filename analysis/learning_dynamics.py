#!/usr/bin/env python3
import logging
import random
import re

import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from huggingface_hub import list_repo_files
from lightning.fabric import Fabric

from src.config.base import BaseComponentConfig

# your dataset initialization utilities
from src.config.data_config import DataConfig
from src.config.training_config import TrainingConfig
from src.data_utils import (
    initialize_dataloader,
    initialize_dataset,
    # initialize_tokenizer,
)
from src.metrics._registry import get_metric
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


# ─── Main processing function ─────────────────────────────────────────────────
def process_variant(size: str):
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
    per_metric = get_metric("per")()
    components = {
        "mlp": lambda n: "mlp" in n.lower(),
        "attn": lambda n: ("attn" in n.lower()) or ("attention" in n.lower()),
    }

    # setup Fabric and data pipeline once per variant
    fabric = Fabric(accelerator="cpu", devices=1)
    data_conf = DataConfig()
    dataset, _ = initialize_dataset(data_conf, fabric, return_fast_forward_steps=True)
    # tokenizer = initialize_tokenizer(data_conf)
    train_conf = TrainingConfig()
    dataloader = initialize_dataloader(data_conf, train_conf, fabric, dataset)
    batch_iter = iter(dataloader)

    for step_dir in steps:
        step = int(step_dir.split("_")[1])
        subfolder = f"checkpoints/{step_dir}"

        # load model
        cfg = PicoDecoderHFConfig.from_pretrained(
            repo_id, trust_remote_code=True, subfolder=subfolder
        )
        RoPE._freqs_cis_tensor = None
        model = PicoDecoderHF.from_pretrained(
            repo_id, config=cfg, trust_remote_code=True, subfolder=subfolder
        )
        model.eval().to(fabric.device)

        # get one batch to populate gradients
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(dataloader)
            batch = next(batch_iter)

        input_ids = torch.tensor(batch["input_ids"]).to(model.device)
        model.zero_grad()
        outputs = model(input_ids, return_dict=True)
        # dummy scalar loss
        loss = outputs.logits.mean()
        loss.backward()

        # compute broken-down ER/PER
        for data_type in ["weights", "gradients"]:
            # assemble name→tensor map
            name2tensor = {}
            for name, param in model.named_parameters():
                if param.ndim < 2:
                    continue
                if data_type == "weights":
                    name2tensor[name] = param.detach().cpu()
                else:
                    if param.grad is not None:
                        name2tensor[name] = param.grad.detach().cpu()

            # for each component
            for comp_name, name_filter in components.items():
                ers, pers = [], []
                # validate component
                cfg_base = BaseComponentConfig(
                    component_name=comp_name, data_type=data_type
                )
                per_metric.validate_component(cfg_base)

                for name, tensor in name2tensor.items():
                    if name_filter(name):
                        per = per_metric.compute_metric(tensor)
                        er = per * tensor.numel()
                        pers.append(per)
                        ers.append(er)

                if not ers:
                    continue

                avg_er = sum(ers) / len(ers)
                avg_per = sum(pers) / len(pers)
                # log to W&B
                run.log(
                    {
                        f"{data_type}/{comp_name}/ER": avg_er,
                        f"{data_type}/{comp_name}/PER": avg_per,
                    },
                    step=step,
                )
                logger.info(
                    f"[{size}] step={step} "
                    f"{data_type}/{comp_name} → ER={avg_er:.4f}, PER={avg_per:.4f}"
                )

        records.append(
            {
                "step": step,
                **{
                    f"{dt}_{cmp}_{mtr}": (avg_er if mtr == "ER" else avg_per)
                    for dt in ["weights", "gradients"]
                    for cmp in components
                    for mtr, (avg_er, avg_per) in [
                        (
                            sum([x for x in ([avg_er] if dt == "weights" else [])]),
                            sum([x for x in ([avg_per] if dt == "weights" else [])]),
                        )
                    ]
                },
            }
        )

    run.finish()

    # save local CSV + plot
    df = pd.DataFrame(records).sort_values("step")
    csv_path = f"er_per_{size}.csv"
    plot_path = f"er_per_{size}.png"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {csv_path}")

    plt.figure()
    for dt in ["weights", "gradients"]:
        for cmp in components:
            plt.plot(df["step"], df[f"{dt}_{cmp}_PER"], label=f"{dt}/{cmp}/PER")
    plt.xlabel("Step")
    plt.ylabel("PER")
    plt.title(f"PER over Training ({size})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved {plot_path}")


# ─── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # make sure: `wandb login` before running
    for sz in SIZES:
        logger.info(f"==== Processing size={sz} ====")
        process_variant(sz)
