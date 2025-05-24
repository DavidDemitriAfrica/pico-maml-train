import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# ─── Configure scienceplots for LaTeX-style figures ──────────────────────────
plt.style.use(["science", "no-latex"])

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
FINETUNE_CONFIGS = IN_LANGUAGE
MODEL_SIZES = ["tiny", "small", "medium", "large"]
PREFIXES = ["vanilla", "maml"]
MODES = ["head", "full"]
METRICS = ["PER_f1", "LOC_f1", "ORG_f1"]

# Build ordered model slugs and labels
MODEL_SLUGS_ORDERED = [
    f"{prefix}_{size}" for size in MODEL_SIZES for prefix in PREFIXES
]
index_labels = MODEL_SLUGS_ORDERED

# Ensure output directory exists
os.makedirs("heatmaps", exist_ok=True)

# ─── Fetch W&B summaries ───────────────────────────────────────────────────────
api = wandb.Api()
all_runs = api.runs(f"{ENTITY}/{PROJECT}")
runs_map = {run.name: run.summary for run in all_runs}

# ─── 1) Collect raw breakdown DataFrames ───────────────────────────────────────
dfs = {}
for mode in MODES:
    cols = [f"{ft}_{metric}" for ft in FINETUNE_CONFIGS for metric in METRICS]
    df = pd.DataFrame(index=index_labels, columns=cols, dtype=float)

    for slug in MODEL_SLUGS_ORDERED:
        for ft in FINETUNE_CONFIGS:
            for metric in METRICS:
                run_name = f"ner_{slug}_{mode}_finetune_{ft}"
                summary = runs_map.get(run_name, {})
                df.at[slug, f"{ft}_{metric}"] = summary.get(f"{ft}/{metric}", np.nan)

    dfs[mode] = df
    df.to_latex(f"heatmaps/finetune_{mode}_breakdown.tex", float_format="%.3f")

# ─── 2) Average across languages, keeping tag breakdown ────────────────────────
head_avg = pd.DataFrame(index=index_labels, columns=METRICS, dtype=float)
full_avg = pd.DataFrame(index=index_labels, columns=METRICS, dtype=float)

for metric in METRICS:
    cols = [f"{lang}_{metric}" for lang in FINETUNE_CONFIGS]
    head_avg[metric] = dfs["head"][cols].mean(axis=1)
    full_avg[metric] = dfs["full"][cols].mean(axis=1)

head_avg.to_latex("heatmaps/head_avg_breakdown.tex", float_format="%.3f")
full_avg.to_latex("heatmaps/full_avg_breakdown.tex", float_format="%.3f")

# ─── 3) Compute vanilla vs. maml difference by size ───────────────────────────
diff_head = pd.DataFrame(index=MODEL_SIZES, columns=METRICS, dtype=float)
diff_full = pd.DataFrame(index=MODEL_SIZES, columns=METRICS, dtype=float)

for size in MODEL_SIZES:
    diff_head.loc[size] = head_avg.loc[f"maml_{size}"] - head_avg.loc[f"vanilla_{size}"]
    diff_full.loc[size] = full_avg.loc[f"maml_{size}"] - full_avg.loc[f"vanilla_{size}"]

# diff_head.to_latex("heatmaps/head_diff.tex", float_format="%.3f")
# diff_full.to_latex("heatmaps/full_diff.tex", float_format="%.3f")

# ─── Compute F1 difference (MAML minus Vanilla) by size ────────────────────────
diff_head = pd.DataFrame(index=MODEL_SIZES, columns=METRICS, dtype=float)
diff_full = pd.DataFrame(index=MODEL_SIZES, columns=METRICS, dtype=float)
for size in MODEL_SIZES:
    diff_head.loc[size] = head_avg.loc[f"maml_{size}"] - head_avg.loc[f"vanilla_{size}"]
    diff_full.loc[size] = full_avg.loc[f"maml_{size}"] - full_avg.loc[f"vanilla_{size}"]

# ─── Plot both regimes in one figure ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, constrained_layout=True)

for ax, (diff_df, mode) in zip(axes, [(diff_head, "Head"), (diff_full, "Full")]):
    diff_df.plot.bar(ax=ax)
    ax.set_xlabel("Model Size")
    ax.set_ylabel("F1 difference (MAML minus Vanilla)")
    ax.set_title(f"{mode} regime")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(title="Tag", frameon=False, loc="upper left")
    # Annotate bars with three-decimal precision
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(
            f"{h:.3f}",
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=12,
        )

fig.suptitle("MAML vs. Vanilla: F1 Improvement by Tag and Regime")
plt.savefig("heatmaps/f1_diff_both_regimes.png", dpi=300)
plt.show()

OTHER = ["ceb_gja", "tl_trg", "tl_ugnayan"]

# collect overall + per‐tag metrics for OTHER langs
dfs_other = {"head": [], "full": []}
for slug in MODEL_SLUGS_ORDERED:
    for mode in MODES:
        run_name = f"ner_{slug}_{mode}_finetune_all"  # finetuned on 'all'
        summary = runs_map.get(run_name, {})
        vals = [summary.get(f"{lang}/f1", np.nan) for lang in OTHER]
        overall = np.nanmean(vals)
        dfs_other[mode].append([overall] + vals)

# build pandas DataFrames
df_other_head = pd.DataFrame(
    dfs_other["head"], index=index_labels, columns=["overall"] + OTHER
)
df_other_full = pd.DataFrame(
    dfs_other["full"], index=index_labels, columns=["overall"] + OTHER
)

# combine into a single table with a 'mode' column
df_other = (
    pd.concat([df_other_head.assign(mode="head"), df_other_full.assign(mode="full")])
    .reset_index()
    .rename(columns={"index": "model"})[
        # reorder columns
        ["model", "mode", "overall"] + OTHER
    ]
)

# export to LaTeX
latex_other = df_other.to_latex(
    index=False,
    float_format="%.3f",
    caption="Zero‐shot transfer F1 on OTHER languages",
    label="tab:transfer_other",
)
with open(
    os.path.join("heatmaps", "other_transfer_results_table.tex"), "w", encoding="utf-8"
) as f:
    f.write(latex_other)
