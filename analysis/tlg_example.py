#!/usr/bin/env python3
import logging
import random

import matplotlib.colors as mcolors
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
pd.np.random.seed(SEED)
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

# ─── Entity classes ────────────────────────────────────────────────────────────
en_classes = ["PER", "LOC", "ORG"]

# ─── W&B init ────────────────────────────────────────────────────────────────
wandb.init(
    project="pico-maml-ner",
    job_type="head_only_enewt",
    name="head_only_enewt_colored",
    reinit=True,
)

# ─── Load label list and build class→indices map ───────────────────────────────
ds_ft = load_dataset(DATASET, FT_SPLIT, trust_remote_code=True)
label_list = ds_ft["train"].features["ner_tags"].feature.names
# map each entity class to its B- and I- label indices
class_idx_map = {
    c: [i for i, lab in enumerate(label_list) if lab.endswith(f"-{c}")]
    for c in en_classes
}
num_labels = len(label_list)

# ─── Tokenizer + alignment ────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(VANILLA_MODEL, trust_remote_code=True)


def tokenize_and_align_labels(examples):
    tok = tokenizer(
        examples["tokens"], truncation=True, max_length=128, is_split_into_words=True
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


# ─── Load head-only model ──────────────────────────────────────────────────────
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
                    logits.view(-1, config.num_labels), labels.view(-1)
                )
            return TokenClassifierOutput(loss=loss, logits=logits)

    model = HeadOnlyModel(config).to(DEVICE)
    model.eval()
    return model


# ─── Prepare datasets ─────────────────────────────────────────────────────────
train_ds = ds_ft["train"].map(
    tokenize_and_align_labels, batched=True, remove_columns=ds_ft["train"].column_names
)
val_ds = ds_ft["validation"].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=ds_ft["validation"].column_names,
)
data_collator = DataCollatorForTokenClassification(tokenizer)

# ─── Fine-tune heads ──────────────────────────────────────────────────────────
models = {}
for variant, mid, rev, sub in [
    ("vanilla", VANILLA_MODEL, commit_map["large"], None),
    ("maml", MAML_MODEL, None, SUBFOLDER),
]:
    m = load_head_only(mid, revision=rev, subfolder=sub)
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
        report_to=["wandb"],
    )
    trainer = Trainer(
        model=m,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    logger.info(f"Training head-only {variant}")
    trainer.train()
    models[variant] = m

# ─── Load evaluation set and pick examples ───────────────────────────────────
ds_eval = load_dataset(DATASET, EVAL_SPLIT, trust_remote_code=True)["test"]
examples = [ex for ex in ds_eval if any(tag != 0 for tag in ex["ner_tags"])][
    :MAX_EXAMPLES
]

# ─── Compute avg log-prob diffs ───────────────────────────────────────────────
records = []
for idx, ex in enumerate(examples):
    toks, tags = ex["tokens"], ex["ner_tags"]
    enc = tokenizer(
        toks,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(DEVICE)
    wids, prev = enc.word_ids(batch_index=0), None
    aligned = [tags[wid] if wid is not None and wid != prev else -100 for wid in wids]
    with torch.no_grad():
        lv = F.log_softmax(
            models["vanilla"](
                enc["input_ids"], attention_mask=enc["attention_mask"]
            ).logits,
            dim=-1,
        )[0]
        lm = F.log_softmax(
            models["maml"](
                enc["input_ids"], attention_mask=enc["attention_mask"]
            ).logits,
            dim=-1,
        )[0]
    lp_v = [lv[i, lab].item() for i, lab in enumerate(aligned) if lab != -100]
    lp_m = [lm[i, lab].item() for i, lab in enumerate(aligned) if lab != -100]
    records.append(
        {
            "idx": idx,
            "sentence": " ".join(toks),
            "van_lp": np.mean(lp_v),
            "mam_lp": np.mean(lp_m),
            "diff": np.mean(lp_m) - np.mean(lp_v),
        }
    )

df = pd.DataFrame(records)

top_gain = df.nlargest(5, "diff")
top_loss = df.nsmallest(5, "diff")


# ─── Function: plot per-class colored sentence ─────────────────────────────────
def plot_class_colored(words, logps, example_idx):
    # compute per-token log-prob for each class via logsumexp
    class_vals = {}
    for c, idxs in class_idx_map.items():
        vals = torch.logsumexp(logps[:, idxs], dim=1).cpu().numpy()
        class_vals[c] = vals

    # normalize across all values for color mapping
    all_vals = np.concatenate(list(class_vals.values()))
    norm = mcolors.Normalize(vmin=all_vals.min(), vmax=all_vals.max())
    cmap = plt.cm.RdYlGn

    fig, axes = plt.subplots(
        len(en_classes),
        1,
        figsize=(len(words) * 0.6, 2 * len(en_classes)),
        constrained_layout=True,
    )
    for ax, c in zip(axes, en_classes):
        ax.axis("off")
        x = 0.01
        for w, v in zip(words, class_vals[c]):
            ax.text(
                x,
                0.5,
                w,
                fontsize=12,
                bbox=dict(
                    facecolor=cmap(norm(v)), edgecolor="none", boxstyle="round,pad=0.2"
                ),
                va="center",
            )
            x += (len(w) + 1) * 0.04
        ax.set_title(f"{c} log-prob", pad=10)

    fname = f"class_colored_{example_idx}.png"
    plt.savefig(fname, dpi=300)
    wandb.log({f"class_colored_{example_idx}": wandb.Image(plt)})
    plt.close()


# ─── Generate plots for top gain/loss ─────────────────────────────────────────
for idx in list(top_gain["idx"]) + list(top_loss["idx"]):
    ex = examples[idx]
    toks = ex["tokens"]
    enc = tokenizer(
        toks,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(DEVICE)
    with torch.no_grad():
        logps = F.log_softmax(
            models["maml"](
                enc["input_ids"], attention_mask=enc["attention_mask"]
            ).logits,
            dim=-1,
        )[0]
    wids, prev = enc.word_ids(batch_index=0), None
    words = [toks[wid] for i, wid in enumerate(wids) if wid is not None and wid != prev]
    plot_class_colored(words, logps, idx)

# ─── Histogram of Δ log-probs ─────────────────────────────────────────────────
plt.figure(figsize=(6, 4))
df["diff"].hist(bins=20)
plt.title("Avg log-prob difference (MAML – Vanilla)")
plt.xlabel("Log-prob difference")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("diff_hist.png", dpi=300)
wandb.log({"diff_hist": wandb.Image(plt)})

wandb.finish()
