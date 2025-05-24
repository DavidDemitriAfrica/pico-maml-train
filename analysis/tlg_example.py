#!/usr/bin/env python3
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.model.pico_decoder import PicoDecoderHF, PicoDecoderHFConfig, RoPE

# ─── Settings ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

VANILLA_MODEL = "pico-lm/pico-decoder-large"
MAML_MODEL = "davidafrica/pico-maml-decoder-large"
SUBFOLDER = "checkpoints/step_6000"
DATASET = "universalner/universal_ner"
FT_SPLIT = "en_ewt"
EVAL_SPLIT = "tl_trg"
NUM_EPOCHS = 10
BATCH_SIZE = 16
MAX_EX = 50

# ─── Initialize W&B ──────────────────────────────────────────────────────────
wandb.init(
    project="pico-maml-ner",
    job_type="head_only_enewt",
    name="head_only_enewt_tagalog_diff",
    reinit=True,
)

# ─── Load label list from EN EWT train ────────────────────────────────────────
ds_train_full = load_dataset(DATASET, FT_SPLIT, trust_remote_code=True)
label_list = ds_train_full["train"].features["ner_tags"].feature.names
num_labels = len(label_list)

# ─── Tokenization + alignment fn ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(VANILLA_MODEL, trust_remote_code=True)


def tokenize_and_align_labels(examples):
    tok = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=128,
        is_split_into_words=True,
    )
    label_ids = []
    for i, labs in enumerate(examples["ner_tags"]):
        wids = tok.word_ids(batch_index=i)
        prev = None
        ids = []
        for wid in wids:
            if wid is None or wid == prev:
                ids.append(-100)
            else:
                ids.append(labs[wid])
            prev = wid
        label_ids.append(ids)
    tok["labels"] = label_ids
    return tok


# ─── Prepare head-only model loader ───────────────────────────────────────────
def load_head_only(model_id, subfolder=None, revision=None):
    load_kw = {"trust_remote_code": True}
    if subfolder:
        load_kw["subfolder"] = subfolder
    if revision:
        load_kw["revision"] = revision
    config = PicoDecoderHFConfig.from_pretrained(model_id, **load_kw)
    config.num_labels = num_labels
    RoPE._freqs_cis_tensor = None
    base = PicoDecoderHF.from_pretrained(
        model_id, config=config, **load_kw
    ).pico_decoder
    # freeze backbone
    for p in base.parameters():
        p.requires_grad = False
    # head
    classifier = torch.nn.Linear(config.d_model, num_labels)
    model = torch.nn.Sequential(base, classifier).to(device)
    return model


# ─── Load tokenized EN_EWT ───────────────────────────────────────────────────
train_ds = ds_train_full["train"].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=ds_train_full["train"].column_names,
)
train_val = ds_train_full["validation"].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=ds_train_full["validation"].column_names,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

# ─── Fine-tune head-only for both variants ────────────────────────────────────
histories = {}
for variant, model_id, subf in [
    ("vanilla", VANILLA_MODEL, None),
    ("maml", MAML_MODEL, SUBFOLDER),
]:
    model = load_head_only(model_id, subfolder=subf)
    args = TrainingArguments(
        output_dir=f"temp_{variant}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=3e-5,
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        seed=SEED,
        fp16=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=train_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    histories[variant] = model

# ─── Gather logprob differences on TL_TRG ─────────────────────────────────────
records = []
ds_eval = load_dataset(DATASET, EVAL_SPLIT, trust_remote_code=True)["test"]
examples = [ex for ex in ds_eval if any(tag != 0 for tag in ex["ner_tags"])][:MAX_EX]

for idx, ex in enumerate(examples):
    toks = ex["tokens"]
    tags = ex["ner_tags"]
    enc = tokenizer(
        toks,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(device)
    wids, prev = enc.word_ids(batch_index=0), None
    aligned = []
    for wid in wids:
        if wid is None or wid == prev:
            aligned.append(-100)
        else:
            aligned.append(tags[wid])
        prev = wid

    with torch.no_grad():
        logps_v = F.log_softmax(histories["vanilla"](**enc), dim=-1)
        logps_m = F.log_softmax(histories["maml"](**enc), dim=-1)

    lp_v = [logps_v[0, i, lab].item() for i, lab in enumerate(aligned) if lab != -100]
    lp_m = [logps_m[0, i, lab].item() for i, lab in enumerate(aligned) if lab != -100]
    avg_v = sum(lp_v) / len(lp_v)
    avg_m = sum(lp_m) / len(lp_m)

    records.append(
        {
            "example_idx": idx,
            "sentence": " ".join(toks),
            "vanilla_lp": avg_v,
            "maml_lp": avg_m,
            "diff_lp": avg_m - avg_v,
        }
    )

df = pd.DataFrame(records)

# ─── Report & visualize ──────────────────────────────────────────────────────
top_gain = df.nlargest(5, "diff_lp")
top_loss = df.nsmallest(5, "diff_lp")

print("\n--- Top 5 MAML Gains (head-only on en_ewt; eval tl_trg) ---")
print(top_gain[["example_idx", "diff_lp", "sentence"]].to_string(index=False))

print("\n--- Top 5 MAML Losses ---")
print(top_loss[["example_idx", "diff_lp", "sentence"]].to_string(index=False))

plt.style.use(["science", "no-latex"])
plt.figure(figsize=(6, 4))
df["diff_lp"].hist(bins=20)
plt.title("Δ avg log-prob (MAML – Vanilla)")
plt.xlabel("Log-prob difference")
plt.ylabel("Count")
plt.tight_layout()
wandb.log({"head_only_enewt_diff_hist": wandb.Image(plt)})

wandb.finish()
