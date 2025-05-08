#!/usr/bin/env python3
"""
Sweep over Hugging Face checkpoints to run:
 1) devinterp LLC (via Timaeus) time‑series
 2) NER fine‑tune + evaluation
Log results per checkpoint to W&B.
"""

import logging
import random

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import TokenClassifierOutput

# Devinterp imports
from analysis.dev_interp import run_devinterp_llc

# ─── 0. Boilerplate ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("devinterp_ner.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 1. Configuration ─────────────────────────────────────────────────────────
REPO_ID = "davidafrica/pico-maml-decoder-tiny"
CHECKPOINTS = [
    "checkpoints/step_0",
    "checkpoints/step_1000",
    "checkpoints/step_2000",
    "checkpoints/step_3000",
    "checkpoints/step_4000",
    "checkpoints/step_5000",
    "checkpoints/step_6000",
    "checkpoints/step_7000",
]

# ─── 2. Prepare NER dataset once ────────────────────────────────────────────────
raw_ds = load_dataset("universalner/universal_ner", "en_ewt", trust_remote_code=True)
base = raw_ds["train"] if "train" in raw_ds else next(iter(raw_ds.values()))
label_list = base.features["ner_tags"].feature.names

# Tokenization & label alignment
max_len = 128


def tokenize_and_align_labels(examples):
    tok = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_len,
        is_split_into_words=True,
    )
    all_labels = []
    for i, labs in enumerate(examples["ner_tags"]):
        word_ids = tok.word_ids(batch_index=i)
        prev = None
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev:
                label_ids.append(labs[wid])
            else:
                label_ids.append(-100)
            prev = wid
        all_labels.append(label_ids)
    tok["labels"] = all_labels
    return tok


# We'll fill tokenizer later after first checkpoint load
ner_splits = None

# ─── 3. Metrics for NER ───────────────────────────────────────────────────────
metric = evaluate.load("seqeval")


def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=2)
    true_labels, true_preds = [], []
    for lab_seq, prd_seq in zip(labels, preds):
        tl = [
            label_list[label]
            for (label, pred) in zip(lab_seq, prd_seq)
            if label != -100
        ]
        tp = [
            label_list[pred] for (label, pred) in zip(lab_seq, prd_seq) if label != -100
        ]
        true_labels.append(tl)
        true_preds.append(tp)
    res = metric.compute(predictions=true_preds, references=true_labels)
    metrics = {
        "precision": res["overall_precision"],
        "recall": res["overall_recall"],
        "f1": res["overall_f1"],
        "accuracy": res["overall_accuracy"],
    }
    for key, stats in res.items():
        if isinstance(stats, dict):
            metrics[f"{key}_precision"] = stats.get("precision")
            metrics[f"{key}_recall"] = stats.get("recall")
            metrics[f"{key}_f1"] = stats.get("f1")
            metrics[f"{key}_support"] = stats.get("number")
    return metrics


# ─── 4. Helper: NER eval function ──────────────────────────────────────────────
def run_ner_eval(
    base_lm,
    tokenizer,
    tokenized_ds,
    label_list,
    seed=SEED,
    batch_size=16,
    device=device,
):
    cfg = base_lm.config
    cfg.num_labels = len(label_list)

    class PicoForTokenClassification(PreTrainedModel):
        config_class = cfg.__class__
        base_model_prefix = "pico_decoder"

        def __init__(self, config):
            super().__init__(config)
            self.pico_decoder = base_lm.pico_decoder
            self.classifier = torch.nn.Linear(config.d_model, config.num_labels)
            self.init_weights()

        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            hidden, _ = self.pico_decoder(
                input_ids, past_key_values=None, use_cache=False, return_hidden=True
            )
            logits = self.classifier(hidden)
            loss = None
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, cfg.num_labels), labels.view(-1))
            return TokenClassifierOutput(loss=loss, logits=logits)

    model = PicoForTokenClassification(cfg).to(device)
    for p in model.pico_decoder.parameters():
        p.requires_grad = False

    data_collator = DataCollatorForTokenClassification(tokenizer)
    args = TrainingArguments(
        output_dir="tmp_ner",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        do_train=True,
        do_eval=True,
        evaluation_strategy="no",
        seed=seed,
        logging_dir="tmp_ner/logs",
        logging_steps=100,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    raw = trainer.evaluate(tokenized_ds["test"])
    return {
        k[len("eval_") :]: v if k.startswith("eval_") else v for k, v in raw.items()
    }


# ─── 5. Main sweep ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Tokenize NER once we have tokenizer
    # we need a temporary tokenizer; use a fast HF one
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
    tokenized = raw_ds.map(
        tokenize_and_align_labels, batched=True, remove_columns=base.column_names
    )
    ner_splits = {
        split: tokenized[split]
        for split in ["train", "validation", "test"]
        if split in tokenized
    }

    # Use the validation split as the devinterp dataset
    dev_batch = ner_splits["validation"]

    import wandb

    for ckpt in CHECKPOINTS:
        logger.info(f"Processing {ckpt}")
        local_dir = snapshot_download(repo_id=REPO_ID, subfolder=ckpt)
        try:
            from src.model.pico_decoder import PicoDecoderHF, PicoDecoderHFConfig, RoPE

            cfg = PicoDecoderHFConfig.from_pretrained(local_dir, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
            RoPE._freqs_cis_tensor = None
            lm = PicoDecoderHF.from_pretrained(
                local_dir, config=cfg, trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Wrapper load failed: {e}, falling back to AutoModel")
            from transformers import AutoModel

            cfg = PicoDecoderHFConfig.from_pretrained(
                local_dir, trust_remote_code=False
            )
            tokenizer = AutoTokenizer.from_pretrained(local_dir)
            lm = AutoModel.from_pretrained(local_dir, config=cfg)

        # Run devinterp LLC
        dev_results = run_devinterp_llc(model=lm, dataset=dev_batch)
        llc_mean = dev_results.get("llc/mean")

        # Run NER eval
        ner_results = run_ner_eval(
            base_lm=lm,
            tokenizer=tokenizer,
            tokenized_ds=ner_splits,
            label_list=label_list,
            seed=SEED,
            batch_size=16,
            device=device,
        )

        # Log both
        step = int(ckpt.split("_")[-1]) if "step_" in ckpt else 0
        wandb.init(
            project="pico-maml-devinterp", name=ckpt.replace("/", "_"), reinit=True
        )
        log_dict = {"llc/mean": llc_mean}
        log_dict.update({f"ner/{k}": v for k, v in ner_results.items()})
        wandb.log(log_dict, step=step)
        wandb.finish()
        logger.info(f"Logged {ckpt}: {{**dev_results, **ner_results}}")
