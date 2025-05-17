import matplotlib.pyplot as plt
import pandas as pd
import wandb

# 1. Configuration
ENTITY = "pico-lm"
PROJECT = "pico-maml"

DATASET_CONFIGS = [
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
TEST_ONLY_CONFIGS = [
    "ceb_gja",
    "tl_trg",
    "tl_ugnayan",
    "de_pud",
    "en_pud",
    "pt_pud",
    "ru_pud",
    "sv_pud",
    "zh_pud",
]
EVAL_CONFIGS = DATASET_CONFIGS + TEST_ONLY_CONFIGS

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

# 2. Fetch all runs and index by their run.name
api = wandb.Api()
all_runs = api.runs(f"{ENTITY}/{PROJECT}")
runs_map = {run.name: run for run in all_runs}

# 3. For each model, assemble a DataFrame of shape (finetune_cfg × eval_cfg)
for slug in MODEL_SLUGS:
    # Build the list of finetune configurations for this model
    finetune_cfgs = DATASET_CONFIGS + ["all"]

    # Prepare a matrix to hold micro-F1 scores
    data = []
    for ft in finetune_cfgs:
        run_name = f"ner_{slug}_full_finetune_{ft}"
        run = runs_map.get(run_name)
        if run is None:
            # If missing, fill with NaNs
            row = [float("nan")] * len(EVAL_CONFIGS)
        else:
            summary = run.summary
            row = [summary.get(f"{cfg}/f1", float("nan")) for cfg in EVAL_CONFIGS]
        data.append(row)

    df = pd.DataFrame(data, index=finetune_cfgs, columns=EVAL_CONFIGS)

    # 4. Plot heatmap for this model
    plt.figure()
    plt.imshow(df.values, aspect="auto")
    plt.xticks(range(len(df.columns)), df.columns, rotation=90)
    plt.yticks(range(len(df.index)), df.index)
    plt.title(f"Micro-F1 heatmap: {slug} (full fine-tune)")

    # Annotate each cell with the F1 numeric value
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.iat[i, j]
            if not pd.isna(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center")

    plt.tight_layout()
    plt.show()
