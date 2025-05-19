import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# ─── Configuration ────────────────────────────────────────────────────────────
ENTITY = "pico-lm"
PROJECT = "pico-maml"

IN_LANGUAGE = [
    "da_ddt",
    "en_ewt",
    "hr_set",
    "pt_bosque",
    "sk_snk",
    "sr_set",
    "sv_talbanken",
    "zh_gsd",
    "zh_gsdsimp",
]
PUD = ["de_pud", "en_pud", "pt_pud", "ru_pud", "sv_pud", "zh_pud"]
OTHER = ["ceb_gja", "tl_trg", "tl_ugnayan"]

EVAL_CONFIGS = IN_LANGUAGE + PUD + OTHER
FINETUNE_CONFIGS = IN_LANGUAGE + ["all"]
TUNE_MODES = ["head", "full"]

MODEL_SLUGS = [
    "vanilla_tiny",
    "vanilla_small",
    "vanilla_medium",
    "vanilla_large",
    "maml_tiny",
    "maml_small",
    "maml_medium",
    "maml_large",
]

# ─── Prepare output directory ─────────────────────────────────────────────────
output_dir = "heatmaps"
os.makedirs(output_dir, exist_ok=True)

# ─── Fetch runs from W&B ─────────────────────────────────────────────────────
api = wandb.Api()
all_runs = api.runs(f"{ENTITY}/{PROJECT}")
runs_map = {run.name: run.summary for run in all_runs}

# ─── Plot and save heatmaps ───────────────────────────────────────────────────
for slug in MODEL_SLUGS:
    for mode in TUNE_MODES:
        # assemble matrix
        matrix = []
        for ft in FINETUNE_CONFIGS:
            run_name = f"ner_{slug}_{mode}_finetune_{ft}"
            summary = runs_map.get(run_name, {})
            row = [summary.get(f"{cfg}/f1", np.nan) for cfg in EVAL_CONFIGS]
            matrix.append(row)
        df = pd.DataFrame(matrix, index=FINETUNE_CONFIGS, columns=EVAL_CONFIGS)

        # create panels
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        cmap = "inferno"
        vmin, vmax = 0.0, 1.0

        # panel 0: In-Language
        ax = axes[0]
        im0 = ax.imshow(
            df[IN_LANGUAGE].values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax.set_title("In-Language Results")
        ax.set_xticks(range(len(IN_LANGUAGE)))
        ax.set_xticklabels(IN_LANGUAGE, rotation=90)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        # annotate numbers and highlight diagonal
        for i in range(df.shape[0]):
            for j in range(len(IN_LANGUAGE)):
                val = df.iloc[i, j]
                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

        # panel 1: PUD
        ax = axes[1]
        im1 = ax.imshow(df[PUD].values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title("PUD Results")
        ax.set_xticks(range(len(PUD)))
        ax.set_xticklabels(PUD, rotation=90)
        ax.set_yticks([])  # no y-ticks
        for i in range(df.shape[0]):
            for j in range(len(PUD)):
                val = df.iloc[i, len(IN_LANGUAGE) + j]
                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

        # panel 2: Other
        ax = axes[2]
        im2 = ax.imshow(
            df[OTHER].values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax.set_title("Other Results")
        ax.set_xticks(range(len(OTHER)))
        ax.set_xticklabels(OTHER, rotation=90)
        ax.set_yticks([])
        for i in range(df.shape[0]):
            for j in range(len(OTHER)):
                val = df.iloc[i, len(IN_LANGUAGE) + len(PUD) + j]
                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

        # single colorbar
        cbar = fig.colorbar(im2, ax=axes, fraction=0.046, pad=0.04)
        cbar.set_label("Micro-F1")

        title = f"{slug}_{mode}_heatmap.png"
        fig.suptitle(f"{slug} ({mode} fine-tune) Micro-F1 Heatmap", fontsize=16)
        # Save figure
        fig.savefig(os.path.join(output_dir, title), dpi=300)
        plt.close(fig)

print(f"All heatmaps saved to ./{output_dir}/")
