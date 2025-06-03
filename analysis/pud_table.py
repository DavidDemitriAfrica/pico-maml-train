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
PUD_CONFIGS = [
    "de_pud",
    "en_pud",
    "pt_pud",
    "ru_pud",
    "sv_pud",
    "zh_pud",
]
MODEL_SIZES = ["tiny", "small", "medium", "large"]
PREFIXES = ["vanilla", "maml"]
MODES = ["head", "full"]
METRICS = ["PER_f1", "LOC_f1", "ORG_f1"]

# Build ordered model slugs
MODEL_SLUGS = [f"{prefix}_{size}" for size in MODEL_SIZES for prefix in PREFIXES]

# Ensure output directory exists
os.makedirs("heatmaps", exist_ok=True)

# ─── Fetch W&B summaries ───────────────────────────────────────────────────────
api = wandb.Api()
all_runs = api.runs(f"{ENTITY}/{PROJECT}")
runs_map = {run.name: run.summary for run in all_runs}

# ─── 1) Collect raw breakdown DataFrames on PUD evals ─────────────────────────
dfs = {mode: {} for mode in MODES}
for mode in MODES:
    for ft in IN_LANGUAGE:
        cols = [f"{pud}_{metric}" for pud in PUD_CONFIGS for metric in METRICS]
        df = pd.DataFrame(index=MODEL_SLUGS, columns=cols, dtype=float)
        for slug in MODEL_SLUGS:
            run_name = f"ner_{slug}_{mode}_finetune_{ft}"
            summary = runs_map.get(run_name, {})
            for pud in PUD_CONFIGS:
                for metric in METRICS:
                    df.at[slug, f"{pud}_{metric}"] = summary.get(
                        f"{pud}/{metric}", np.nan
                    )
        dfs[mode][ft] = df
        df.to_latex(
            f"heatmaps/{mode}_finetune_{ft}_pud_breakdown.tex", float_format="%.3f"
        )

# ─── 2) Average across PUD sets (per ft) ──────────────────────────────────────
avg_pud = {mode: {} for mode in MODES}
for mode in MODES:
    for ft, df in dfs[mode].items():
        avg = pd.DataFrame(index=MODEL_SLUGS, columns=METRICS, dtype=float)
        for metric in METRICS:
            cols = [f"{pud}_{metric}" for pud in PUD_CONFIGS]
            avg[metric] = df[cols].mean(axis=1)
        avg_pud[mode][ft] = avg
        avg.to_latex(f"heatmaps/{mode}_pud_avg_{ft}.tex", float_format="%.3f")

# ─── 3) Compute MAML vs. Vanilla difference by size on avg PUD ───────────────
diff = {mode: {} for mode in MODES}
for mode in MODES:
    for ft, avg in avg_pud[mode].items():
        ddf = pd.DataFrame(index=MODEL_SIZES, columns=METRICS, dtype=float)
        for size in MODEL_SIZES:
            ddf.loc[size] = avg.loc[f"maml_{size}"] - avg.loc[f"vanilla_{size}"]
        diff[mode][ft] = ddf
        ddf.to_latex(f"heatmaps/{mode}_pud_diff_{ft}.tex", float_format="%.3f")

# ─── 4) Compute & Save diffs averaged across finetune langs ───────────────────
overall_diff = {}
for mode in MODES:
    # Collect all diff DataFrames for this mode
    dfs_mode = list(diff[mode].values())
    tags = dfs_mode[0].columns
    mean_diff = pd.DataFrame(index=MODEL_SIZES, columns=tags, dtype=float)
    for tag in tags:
        mat = np.stack([df[tag].values for df in dfs_mode], axis=1)
        mean_diff[tag] = mat.mean(axis=1)
    overall_diff[mode] = mean_diff
    mean_diff.to_latex(f"heatmaps/{mode}_pud_diff_avg.tex", float_format="%.3f")

# ─── 5) Plot averaged diffs ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, constrained_layout=True)
for ax, (mode, df) in zip(axes, overall_diff.items()):
    df.plot.bar(ax=ax)
    ax.set_xlabel("Model Size")
    ax.set_ylabel("F1 diff (MAML - Vanilla)")
    ax.set_title(f"{mode.capitalize()} regime (PUD eval avg)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(title="Tag", frameon=False, loc="upper left")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(
            f"{h:.3f}",
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=12,
        )
fig.suptitle("MAML vs. Vanilla on PUD Evaluations (Avg over finetune langs)")
plt.savefig("heatmaps/pud_f1_overall_diff.png", dpi=300)
plt.show()

# ─── 6) Write all generated .tex files to a text file ─────────────────────────
output_file = os.path.join("heatmaps", "pud_tex_files.txt")
with open(output_file, "w", encoding="utf-8") as out:
    for fname in sorted(os.listdir("heatmaps")):
        if fname.endswith(".tex"):
            out.write(f"--- {fname} ---\n")
            with open(os.path.join("heatmaps", fname), encoding="utf-8") as f:
                out.write(f.read())
                out.write("\n")
print(f"Wrote all .tex contents to {output_file}")
