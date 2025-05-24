import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# ─── Configuration ────────────────────────────────────────────────────────────
ENTITY = "pico-lm"
PROJECT = "pico-maml-ner"

ALL_LANGS = [
    "da_ddt",
    "en_ewt",
    "hr_set",
    "pt_bosque",
    "sk_snk",
    "sr_set",
    "sv_talbanken",
    "zh_gsd",
    "zh_gsdsimp",
    "de_pud",
    "en_pud",
    "pt_pud",
    "ru_pud",
    "sv_pud",
    "zh_pud",
    "ceb_gja",
    "tl_trg",
    "tl_ugnayan",
]

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

FT_FROM_EN = ["en_ewt"]
FT_TO_EN = [lang for lang in ALL_LANGS if lang != "en_ewt"]
EN_ONLY = ["en_ewt"]

# ─── Output directory ─────────────────────────────────────────────────────────
output_dir = "heatmaps_en_directional"
os.makedirs(output_dir, exist_ok=True)

# ─── Fetch W&B runs ───────────────────────────────────────────────────────────
api = wandb.Api()
all_runs = api.runs(f"{ENTITY}/{PROJECT}")
runs_map = {run.name: run.summary for run in all_runs}


# ─── Function: Create heatmaps for a given adaptation direction ───────────────
def create_heatmap_and_summary(finetune_configs, eval_configs, tag):
    summary_rows = []

    for slug in MODEL_SLUGS:
        row = [slug]
        for mode in TUNE_MODES:
            matrix = []
            for ft in finetune_configs:
                run_name = f"ner_{slug}_{mode}_finetune_{ft}"
                summary = runs_map.get(run_name, {})
                row_vals = [summary.get(f"{cfg}/f1", np.nan) for cfg in eval_configs]
                matrix.append(row_vals)
            df = pd.DataFrame(matrix, index=finetune_configs, columns=eval_configs)

            # Save heatmap
            fig, ax = plt.subplots(figsize=(len(eval_configs), len(finetune_configs)))
            im = ax.imshow(df.values, aspect="auto", cmap="inferno", vmin=0.0, vmax=1.0)
            ax.set_xticks(range(len(eval_configs)))
            ax.set_xticklabels(eval_configs, rotation=90)
            ax.set_yticks(range(len(df.index)))
            ax.set_yticklabels(df.index)
            for i in range(df.shape[0]):
                for j in range(len(eval_configs)):
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
            fig.colorbar(im, ax=ax).set_label("Micro-F1")
            fig.suptitle(f"{slug} ({mode} fine-tune) Micro-F1 Heatmap ({tag})")
            fig.tight_layout()
            fig.savefig(
                os.path.join(output_dir, f"{slug}_{mode}_heatmap_{tag}.png"), dpi=300
            )
            plt.close(fig)

            # Compute group-wise means
            mean_score = df.mean(axis=1).mean()
            row.append(mean_score)
        summary_rows.append(row)

    # Build summary table
    df_summary = pd.DataFrame(
        summary_rows, columns=["Model", f"{tag} (Head)", f"{tag} (Full)"]
    )
    df_summary.to_csv(f"summary_micro_f1_{tag}.csv", index=False)
    print(df_summary.to_latex(index=False, float_format="%.3f"))
    return df_summary


# ─── Execute both directions ──────────────────────────────────────────────────
df_from_en = create_heatmap_and_summary(FT_FROM_EN, FT_TO_EN, tag="from_en")
df_to_en = create_heatmap_and_summary(FT_TO_EN, EN_ONLY, tag="to_en")
