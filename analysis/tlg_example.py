#!/usr/bin/env python3
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from ace_tools import display_dataframe_to_user
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import TokenClassifierOutput

from src.model.pico_decoder import PicoDecoderHF, PicoDecoderHFConfig, RoPE

# ─── Seeds & Determinism ──────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

# ─── Config ──────────────────────────────────────────────────────────────────
VANILLA_MODEL = "pico-lm/pico-decoder-large"
MAML_MODEL = "davidafrica/pico-maml-decoder-large"
STEP = 6000
SUBFOLDER = f"checkpoints/step_{STEP}"
FT_SPLIT = "en_ewt"
EVAL_SPLIT = "tl_trg"
DATASET = "universalner/universal_ner"
NUM_EPOCHS = 10
BATCH_SIZE = 16
MAX_EXAMPLES = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── W&B init ────────────────────────────────────────────────────────────────
wandb.init(
    project="pico-maml-ner",
    entity="pico-lm",
    job_type="head_only_enewt",
    name="head_only_enewt_tagalog_diff",
    reinit=True,
)

# ─── Load label list from EN_EWT ──────────────────────────────────────────────
ds_ft = load_dataset(DATASET, FT_SPLIT, trust_remote_code=True)
label_list = ds_ft["train"].features["ner_tags"].feature.names
num_labels = len(label_list)

# ─── Tokenization + alignment ─────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(VANILLA_MODEL, trust_remote_code=True)


def tokenize_and_align_labels(examples):
    tok = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=128,
        is_split_into_words=True,
    )
    lab_ids = []
    for i, labs in enumerate(examples["ner_tags"]):
        wids, prev = tok.word_ids(batch_index=i), None
        ids = []
        for wid in wids:
            if wid is None or wid == prev:
                ids.append(-100)
            else:
                ids.append(labs[wid])
            prev = wid
        lab_ids.append(ids)
    tok["labels"] = lab_ids
    return tok


# ─── Head-only model loader ───────────────────────────────────────────────────
commit_map = {
    "tiny": "8f42ded9c1c37cb188d68d46ba09aafa045a2a1d",
    "small": "1efec115d29eb94670bc3e62686c8b2b14acf2e0",
    "medium": "46f9b7e6fbb7a075600fe12de0b351b6363620cf",
    "large": "ce5fa8fe69acb265cf38773bd7f9c92325b863f3",
}


def load_head_only(model_id, revision=None, subfolder=None):
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
    # Freeze backbone
    for p in base.parameters():
        p.requires_grad = False

    class HeadOnlyModel(PreTrainedModel):
        config_class = config.__class__
        base_model_prefix = "pico_decoder"

        def __init__(self, config):
            super().__init__(config)
            self.pico_decoder = base
            self.classifier = torch.nn.Linear(config.d_model, config.num_labels)

        def forward(self, input_ids, attention_mask=None, labels=None):
            hidden, _ = self.pico_decoder(
                input_ids, use_cache=False, return_hidden=True
            )
            logits = self.classifier(hidden)
            loss = None
            if labels is not None:
                loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
                    logits.view(-1, config.num_labels),
                    labels.view(-1),
                )
            return TokenClassifierOutput(loss=loss, logits=logits)

    model = HeadOnlyModel(config).to(DEVICE)
    model.eval()
    return model


# ─── Prepare EN_EWT datasets ──────────────────────────────────────────────────
train_ds = ds_ft["train"].map(
    tokenize_and_align_labels, batched=True, remove_columns=ds_ft["train"].column_names
)
val_ds = ds_ft["validation"].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=ds_ft["validation"].column_names,
)
data_collator = DataCollatorForTokenClassification(tokenizer)

# ─── Fine-tune head-only on EN_EWT ─────────────────────────────────────────────
models = {}
for variant, mid, rev, sub in [
    ("vanilla", VANILLA_MODEL, commit_map["large"], None),
    ("maml", MAML_MODEL, None, SUBFOLDER),
]:
    model = load_head_only(mid, revision=rev, subfolder=sub)
    args = TrainingArguments(
        output_dir=f"tmp_{variant}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=3e-5,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        remove_unused_columns=False,
        seed=SEED,
        fp16=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    logger.info(f"Training head-only {variant}")
    trainer.train()
    models[variant] = model

# ─── Tagalog evaluation + log-prob inspection ─────────────────────────────────
records = []
ds_eval = load_dataset(DATASET, EVAL_SPLIT, trust_remote_code=True)["test"]
examples = [ex for ex in ds_eval if any(tag != 0 for tag in ex["ner_tags"])]
examples = examples[:MAX_EXAMPLES]

for idx, ex in enumerate(examples):
    toks = ex["tokens"]
    tags = ex["ner_tags"]
    enc = tokenizer(
        toks,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(DEVICE)
    wids, prev = enc.word_ids(batch_index=0), None
    aligned = []
    for wid in wids:
        if wid is None or wid == prev:
            aligned.append(-100)
        else:
            aligned.append(tags[wid])
        prev = wid

    with torch.no_grad():
        logps_v = F.log_softmax(models["vanilla"](**enc).logits, dim=-1)
        logps_m = F.log_softmax(models["maml"](**enc).logits, dim=-1)

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

# ─── Display top gains & losses ───────────────────────────────────────────────
top_gain = df.nlargest(5, "diff_lp")
top_loss = df.nsmallest(5, "diff_lp")
display_dataframe_to_user("Top 5 MAML Gains (head-only en_ewt → tl_trg)", top_gain)
display_dataframe_to_user("Top 5 MAML Losses", top_loss)

# ─── Histogram of Δ log-probs ─────────────────────────────────────────────────
plt.style.use(["science", "no-latex"])
plt.figure(figsize=(6, 4))
df["diff_lp"].hist(bins=20)
plt.title("Δ avg log-prob (MAML – Vanilla)")
plt.xlabel("Log-prob difference")
plt.ylabel("Count")
plt.tight_layout()
wandb.log({"head_only_enewt_diff_hist": wandb.Image(plt)})

wandb.finish()
