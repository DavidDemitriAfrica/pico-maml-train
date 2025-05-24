import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# ─── Configuration ────────────────────────────────────────────────────────────
ENTITY = "pico-lm"
PROJECT = "pico-maml-ner"

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
FINETUNE_CONFIGS = IN_LANGUAGE
MODEL_SIZES = ["tiny", "small", "medium", "large"]
MODES = ["head", "full"]

# Interleave by size: vanilla tiny, maml tiny, vanilla small, maml small, ...
MODEL_SLUGS_ORDERED = []
index_labels = []
for size in MODEL_SIZES:
    for prefix in ["vanilla", "maml"]:
        slug = f"{prefix}_{size}"
        MODEL_SLUGS_ORDERED.append(slug)
        index_labels.append(slug)

# ─── Fetch real W&B summaries ────────────────────────────────────────────────
api = wandb.Api()
all_runs = api.runs(f"{ENTITY}/{PROJECT}")
runs_map = {run.name: run.summary for run in all_runs}

# ─── Collect DataFrames for head and full ─────────────────────────────────────
dfs = {"head": [], "full": []}

for slug in MODEL_SLUGS_ORDERED:
    for mode in MODES:
        row = []
        for ft in FINETUNE_CONFIGS:
            run_name = f"ner_{slug}_{mode}_finetune_{ft}"
            summary = runs_map.get(run_name, {})
            val = summary.get(f"{ft}/f1", np.nan)
            row.append(val)
        dfs[mode].append(row)

# Create DataFrames
df_head = pd.DataFrame(dfs["head"], index=index_labels, columns=FINETUNE_CONFIGS)
df_full = pd.DataFrame(dfs["full"], index=index_labels, columns=FINETUNE_CONFIGS)

# ─── Combine and format for LaTeX ─────────────────────────────────────────────
df_combined = []

for i, slug in enumerate(MODEL_SLUGS_ORDERED):
    row = [slug]
    row.extend(df_head.iloc[i].tolist())
    row.extend(df_full.iloc[i].tolist())
    df_combined.append(row)

columns = ["Model"]
columns.extend([f"{ft}_head" for ft in FINETUNE_CONFIGS])
columns.extend([f"{ft}_full" for ft in FINETUNE_CONFIGS])

df_latex = pd.DataFrame(df_combined, columns=columns)

# Convert to LaTeX
latex_output = df_latex.to_latex(index=False, float_format="%.3f")

# Save to file
output_dir = "heatmaps"
os.makedirs(output_dir, exist_ok=True)
latex_path = os.path.join(output_dir, "finetune_eval_results_table.tex")
with open(latex_path, "w") as f:
    f.write(latex_output)

latex_output

# Plot side-by-side heatmaps
fig, axes = plt.subplots(
    1,
    2,
    figsize=(len(FINETUNE_CONFIGS) * 1.2, len(index_labels) * 0.5),
    sharey=True,
    constrained_layout=True,
)
cmap = "inferno"
vmin, vmax = 0.0, 1.0

# Head
ax = axes[0]
im0 = ax.imshow(df_head.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
ax.set_title("Head Finetune-Eval")
ax.set_xticks(range(len(FINETUNE_CONFIGS)))
ax.set_xticklabels(FINETUNE_CONFIGS, rotation=90)
ax.set_yticks(range(len(index_labels)))
ax.set_yticklabels(index_labels)
for i in range(df_head.shape[0]):
    for j in range(df_head.shape[1]):
        val = df_head.iloc[i, j]
        if not np.isnan(val):
            ax.text(
                j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=7
            )

# Full
ax = axes[1]
im1 = ax.imshow(df_full.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
ax.set_title("Full Finetune-Eval")
ax.set_xticks(range(len(FINETUNE_CONFIGS)))
ax.set_xticklabels(FINETUNE_CONFIGS, rotation=90)
ax.set_yticks(range(len(index_labels)))
ax.set_yticklabels(index_labels)
for i in range(df_full.shape[0]):
    for j in range(df_full.shape[1]):
        val = df_full.iloc[i, j]
        if not np.isnan(val):
            ax.text(
                j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=7
            )

# Add colorbar
cbar = fig.colorbar(im1, ax=axes, fraction=0.046, pad=0.04)
cbar.set_label("Micro-F1")

# Save figure
output_dir = "heatmaps"
os.makedirs(output_dir, exist_ok=True)
title = "finetune_eval_pair_heatmaps_combined.png"
fig.suptitle("Same Finetune-Eval Results by Model and Tuning Regime", fontsize=16)
fig.savefig(os.path.join(output_dir, title), dpi=300)
