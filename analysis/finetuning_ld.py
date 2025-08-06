    import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from scipy.integrate import trapezoid
from sklearn.linear_model import LinearRegression

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

# Build ordered slugs
MODEL_SLUGS = [f"{p}_{s}" for s in MODEL_SIZES for p in PREFIXES]

# Ensure output directory exists
os.makedirs("heatmaps", exist_ok=True)

# ─── Fetch all runs from WandB ─────────────────────────────────────────────────
api = wandb.Api()
all_runs = api.runs(f"{ENTITY}/{PROJECT}")
runs_map = {run.name: run for run in all_runs}

# ─── 1) Collect fine-tuning time-series ────────────────────────────────────────
TS_KEYS = ["train/loss", "train/grad_norm"]
history = {mode: {} for mode in MODES}

for mode in MODES:
    for slug in MODEL_SLUGS:
        for lang in FINETUNE_CONFIGS:
            run_name = f"ner_{slug}_{mode}_finetune_{lang}"
            run = runs_map.get(run_name)
            if not run:
                print(f"[WARN] Run not found: {run_name}")
                continue
            df = run.history(keys=TS_KEYS, pandas=True)
            # rename step column if needed
            if "_step" in df.columns:
                df = df.rename(columns={"_step": "step"})
            df = df[["step"] + TS_KEYS].dropna()
            history[mode][run_name] = df
            df.to_csv(f"heatmaps/{run_name}_history.csv", index=False)


# ─── 2) Compute speed metrics per run ──────────────────────────────────────────
def compute_speed_metrics(df, frac=0.9, early_frac=0.1):
    steps = df["step"].values
    loss = df["train/loss"].values
    L0, L_inf = loss[0], loss[-1]
    thresh = L0 - frac * (L0 - L_inf)
    idx90 = np.where(loss <= thresh)[0]
    t90 = steps[idx90[0]] if len(idx90) else np.nan
    auc = trapezoid(loss, steps) / (steps[-1] - steps[0])
    cutoff = steps[0] + early_frac * (steps[-1] - steps[0])
    mask = steps <= cutoff
    X = steps[mask].reshape(-1, 1)
    y = loss[mask]
    slope = LinearRegression().fit(X, y).coef_[0]
    return t90, auc, slope


records = []
pattern = re.compile(
    r"ner_(vanilla|maml)_(tiny|small|medium|large)_(head|full)_finetune_(.+)"
)
for mode, runs in history.items():
    for run_name, df in runs.items():
        m = pattern.match(run_name)
        if not m:
            continue
        prefix, size, _, lang = m.groups()
        t90, auc, slope = compute_speed_metrics(df)
        records.append(
            {
                "run": run_name,
                "prefix": prefix,
                "size": size,
                "mode": mode,
                "lang": lang,
                "t90": t90,
                "auc": auc,
                "slope": slope,
            }
        )

metrics_df = pd.DataFrame(records)
metrics_df.to_csv("heatmaps/fine_tune_speed_metrics.csv", index=False)

# ─── 3) Visualizations ─────────────────────────────────────────────────────────

# 3a) Example loss curves (medium, full, one language)
example_lang = FINETUNE_CONFIGS[0]
plt.figure(figsize=(6, 3))
for prefix in PREFIXES:
    run_name = f"ner_{prefix}_medium_full_finetune_{example_lang}"
    df = history["full"].get(run_name)
    if df is not None:
        plt.plot(df["step"], df["train/loss"], label=prefix)
plt.xlabel("Fine-tuning Step")
plt.ylabel("Train Loss")
plt.title(f"Loss Curve on {example_lang} (medium/full)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("heatmaps/example_loss_curve.png", dpi=300)
plt.close()

# 3b) Scatter t₉₀: Vanilla vs. MAML
pivot = metrics_df.pivot_table(
    index=["size", "mode", "lang"], columns="prefix", values="t90"
).dropna()

for mode in MODES:
    for size in MODEL_SIZES:
        sel = pivot.loc[(size, mode)]
        if sel.empty:
            continue
        plt.figure(figsize=(4, 4))
        plt.scatter(sel["vanilla"], sel["maml"], alpha=0.7)
        lims = [
            np.nanmin([sel["vanilla"], sel["maml"]]),
            np.nanmax([sel["vanilla"], sel["maml"]]),
        ]
        plt.plot(lims, lims, "--", color="gray")
        plt.xlabel("t₉₀ (Vanilla)")
        plt.ylabel("t₉₀ (MAML)")
        plt.title(f"t₉₀: {size}, {mode}")
        plt.tight_layout()
        plt.savefig(f"heatmaps/scatter_t90_{size}_{mode}.png", dpi=300)
        plt.close()

# 3c) Boxplots of paired differences (MAML – Vanilla), with no fliers
diff = metrics_df.pivot_table(
    index=["size", "mode", "lang"], columns="prefix", values=["t90", "auc", "slope"]
).dropna()

# compute MAML − Vanilla
diff["dt90"] = diff[("t90", "maml")] - diff[("t90", "vanilla")]
diff["dauc"] = diff[("auc", "maml")] - diff[("auc", "vanilla")]
diff["dslope"] = diff[("slope", "maml")] - diff[("slope", "vanilla")]

for metric, nice in [("dt90", "t₉₀"), ("dauc", "AUC"), ("dslope", "slope")]:
    plt.figure(figsize=(5, 3))
    data = [
        diff[metric][diff.index.get_level_values("size") == sz] for sz in MODEL_SIZES
    ]
    bp = plt.boxplot(
        data, labels=MODEL_SIZES, notch=True, showfliers=False, patch_artist=True
    )
    # style boxes
    for box in bp["boxes"]:
        box.set_facecolor("white")
        box.set_edgecolor("black")
        box.set_linewidth(1)
    # jittered points
    for i, vals in enumerate(data):
        x = np.random.normal(i + 1, 0.05, size=len(vals))
        plt.scatter(x, vals, alpha=0.6, s=20, color="C0", edgecolors="none")
    # zero line
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    # labels
    plt.ylabel(f"MAML − Vanilla ({nice})")
    plt.xlabel("Model Size")
    plt.title(f"Paired Differences by Size (MAML − Vanilla, {nice})")
    plt.tight_layout()
    plt.savefig(f"heatmaps/box_diff_{metric}.png", dpi=300)
    plt.close()

print("✅ All histories, metrics, and plots saved under ./heatmaps/")

# ─── 4) Summarize into a LaTeX table ───────────────────────────────────────────

# compute mean and std by size,mode,prefix
grp = metrics_df.groupby(["size", "mode", "prefix"])[["t90", "auc", "slope"]]
mean_df = grp.mean().unstack("prefix")
std_df = grp.std().unstack("prefix")

# flatten columns
mean_df.columns = [f"{m}_{p}" for m, p in mean_df.columns]
std_df.columns = [f"{m}_{p}_std" for m, p in std_df.columns]

summary = mean_df.join(std_df)

# compute deltas: MAML − Vanilla
summary["Δt90"] = summary["t90_maml"] - summary["t90_vanilla"]
summary["ΔAUC"] = summary["auc_maml"] - summary["auc_vanilla"]
summary["Δslope"] = summary["slope_maml"] - summary["slope_vanilla"]

# reorder and rename for LaTeX
cols = [
    "t90_vanilla",
    "t90_maml",
    "Δt90",
    "auc_vanilla",
    "auc_maml",
    "ΔAUC",
    "slope_vanilla",
    "slope_maml",
    "Δslope",
]
col_labels = {
    "t90_vanilla": "Vanilla $t_{90}$",
    "t90_maml": "MAML $t_{90}$",
    "Δt90": "$\\Delta t_{90}$",
    "auc_vanilla": "Vanilla AUC",
    "auc_maml": "MAML AUC",
    "ΔAUC": "$\\Delta\\mathrm{AUC}$",
    "slope_vanilla": "Vanilla slope",
    "slope_maml": "MAML slope",
    "Δslope": "$\\Delta\\mathrm{slope}$",
}

summary = summary[cols].rename(columns=col_labels)

# round for readability
summary = summary.round(
    {
        "Vanilla $t_{90}$": 1,
        "MAML $t_{90}$": 1,
        "$\\Delta t_{90}$": 1,
        "Vanilla AUC": 3,
        "MAML AUC": 3,
        "$\\Delta\\mathrm{AUC}$": 3,
        "Vanilla slope": 5,
        "MAML slope": 5,
        "$\\Delta\\mathrm{slope}$": 5,
    }
)

# export to LaTeX
latex = summary.to_latex(
    index=[
        summary.index.get_level_values("size"),
        summary.index.get_level_values("mode"),
    ],
    index_names=["Model Size", "Regime"],
    caption="Fine-tuning convergence speed metrics (mean over languages).",
    label="tab:finetune_speed",
    escape=False,
)

with open("heatmaps/fine_tune_speed_summary.tex", "w") as f:
    f.write(latex)

print("✅ Summary LaTeX table saved to heatmaps/fine_tune_speed_summary.tex")
