#!/usr/bin/env python3
import json
import logging
import os
import random

import evaluate
import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EvalPrediction,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import WandbCallback
from transformers.modeling_outputs import TokenClassifierOutput

# import your local HF wrapper & config
from src.model.pico_decoder import PicoDecoderHF, PicoDecoderHFConfig, RoPE

# ─── -1. Setting seeds for reproducibility ────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# make CUDA ops deterministic (may be slower)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─── 0. Logging setup ─────────────────────────────────────────────────────────
LOG_FILE = "evaluation.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PrefixedWandbCallback(WandbCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Intercept Trainer.log calls and re-log to W&B
        with the appropriate 'train/' or 'valid/' prefix.
        """
        if logs is None:
            return
        step = state.global_step
        wandb_logs = {}
        for key, val in logs.items():
            # validation metrics come prefixed by "eval_"
            if key.startswith("eval_"):
                clean = key[len("eval_") :]  # e.g. "loss", "f1"
                wandb_logs[f"valid/{clean}"] = val
            else:
                # everything else is training: loss, learning_rate, etc.
                wandb_logs[f"train/{key}"] = val
        wandb.log(wandb_logs, step=step)
        # no need to call super()


# ─── 1. CONFIGURATION ─────────────────────────────────────────────────────────
MODEL_NAMES = [
    "pico-lm/pico-decoder-tiny",
    "pico-lm/pico-decoder-small",
    "pico-lm/pico-decoder-medium",
    "pico-lm/pico-decoder-large",
]
DATASET_NAME = "universalner/universal_ner"
DATASET_CONFIGS = ["en_ewt"]
SPLIT = "test"
BATCH_SIZE = 1024

# ─── 2. Metric ────────────────────────────────────────────────────────────────
metric = evaluate.load("seqeval")


def compute_metrics(eval_pred: EvalPrediction):
    # unpack
    logits = eval_pred.predictions  # (batch, seq_len, num_labels)
    labels = eval_pred.label_ids  # (batch, seq_len)

    preds = np.argmax(logits, axis=2)
    # convert label ids → actual label strings, filter out -100
    true_labels = []
    true_preds = []
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

    # get the full seqeval breakdown
    res = metric.compute(predictions=true_preds, references=true_labels)

    # start flattening
    metrics = {
        "precision": res["overall_precision"],
        "recall": res["overall_recall"],
        "f1": res["overall_f1"],
        "accuracy": res["overall_accuracy"],
    }

    # per-entity metrics
    for key, stats in res.items():
        if isinstance(stats, dict):
            # a tag like "PER", "LOC", etc.
            # stats has keys: precision, recall, f1, number
            metrics[f"{key}_precision"] = stats.get("precision", None)
            metrics[f"{key}_recall"] = stats.get("recall", None)
            metrics[f"{key}_f1"] = stats.get("f1", None)
            metrics[f"{key}_support"] = stats.get("number", None)

    return metrics


# ─── 3. Main evaluation loop ─────────────────────────────────────────────────
for cfg in DATASET_CONFIGS:
    logger.info(f"Loading dataset {DATASET_NAME} config={cfg}")
    ds = load_dataset(DATASET_NAME, cfg, trust_remote_code=True)
    if "train" in ds:
        base = ds["train"]
    else:
        base = next(iter(ds.values()))
    label_list = base.features["ner_tags"].feature.names
    logger.info(f"Found {len(label_list)} labels: {label_list}")

    for model_name in MODEL_NAMES:
        logger.info(f"→ Evaluating model '{model_name}' on config='{cfg}'")

        # ─── Initialize a new W&B run ──────────────────────────────────────────
        run_id = f"ner_eval_head_only_{model_name.split('/')[-1]}_{cfg}"
        wandb.init(
            project="pico-maml",  # your W&B project
            entity="pico-lm",  # your W&B entity/org
            name=run_id,  # run name
            tags=[run_id, "ner_eval", "head_only"],  # tags list
            reinit=True,  # allow multiple in one script
        )
        logger.info(f"Started W&B run: {run_id}")

        # 3a. Load *local* HF config & wrapper (which uses your src/model/pico_decoder.py)
        config = PicoDecoderHFConfig.from_pretrained(
            model_name,
            trust_remote_code=True,  # still needed if you have remote custom code
        )
        config.num_labels = len(label_list)

        # Reset RoPE buffer
        RoPE._freqs_cis_tensor = None

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=True
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)
        logger.info("Data collator ready")

        # only keep up to 128 tokens per example
        max_len = min(128, config.max_seq_len)
        logger.info(f"Enforcing max_seq_len={max_len} for tokenization")

        # 3c. Load the PicoDecoderHF (local) wrapper, not the remote AutoModel
        base_lm = PicoDecoderHF.from_pretrained(
            model_name, config=config, trust_remote_code=True
        )

        class PicoForTokenClassification(PreTrainedModel):
            config_class = config.__class__
            base_model_prefix = "pico_decoder"

            def __init__(self, config):
                super().__init__(config)
                self.pico_decoder = base_lm.pico_decoder
                self.classifier = nn.Linear(config.d_model, config.num_labels)
                self.init_weights()

            def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
                hidden_states, _ = self.pico_decoder(
                    input_ids,
                    past_key_values=None,
                    use_cache=False,
                    return_hidden=True,
                )
                logits = self.classifier(hidden_states)
                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(
                        logits.view(-1, self.config.num_labels),
                        labels.view(-1),
                    )
                return TokenClassifierOutput(loss=loss, logits=logits)

        model = PicoForTokenClassification(config)
        logger.info("Wrapped LM into token‐classification model")

        # ─── Freeze the PicoDecoder backbone ─────────────────────────────────
        # We only want to train the classifier head; the pretrained decoder stays fixed.
        for name, param in model.pico_decoder.named_parameters():
            param.requires_grad = False
        logger.info(
            "Froze pico_decoder backbone; only the classifier head will be trained"
        )

        # 3d. Tokenize & align labels
        def tokenize_and_align_labels(examples):
            tok = tokenizer(
                examples["tokens"],
                truncation=True,
                max_length=max_len,
                is_split_into_words=True,
            )
            all_labels = []
            for i, labs in enumerate(examples["ner_tags"]):
                word_ids, prev = tok.word_ids(batch_index=i), None
                label_ids = []
                for wid in word_ids:
                    if wid is None:
                        label_ids.append(-100)
                    elif wid != prev:
                        label_ids.append(labs[wid])
                    else:
                        label_ids.append(-100)
                    prev = wid
                # if sentence shorter than max_len, this is already padded to length max_len
                all_labels.append(label_ids)
            tok["labels"] = all_labels
            return tok

        logger.info("Applying tokenization & label alignment...")
        tokenized = ds.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=ds["train"].column_names,
        )
        logger.info(f"Tokenized dataset columns: {tokenized.column_names}")

        # 3e. Trainer setup
        output_dir = os.path.join("results", model_name.replace("/", "_"), cfg)
        log_dir = os.path.join("logs", model_name.replace("/", "_"), cfg)
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=3,  # train for 3 epochs (adjust as you like)
            do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",  # run validation at end of each epoch
            fp16=True,
            dataloader_num_workers=4,
            logging_dir=log_dir,
            logging_steps=100,
            save_strategy="no",
            seed=SEED,
            report_to=["wandb"],
            run_name=run_id,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],  # fine‐tune on train split
            eval_dataset=tokenized["validation"],  # validate on validation split
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[PrefixedWandbCallback],
        )
        logger.info("Trainer initialized for fine‐tuning")

        # 3f. Fine‐tune the model
        logger.info("Starting training")
        trainer.train()

        # 3g. Evaluate on the held‐out test set
        logger.info("Training complete — evaluating on test split")
        test_metrics = trainer.evaluate(tokenized["test"])
        wandb.log({f"ner/test/{k}": v for k, v in test_metrics.items()})
        logger.info(f"Test results for {model_name} on {cfg}: {test_metrics}")
        print(f"\n→ {model_name} / {cfg} / test: {test_metrics}")

        # ─── Save metrics locally ─────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Saved test metrics to {metrics_path}")

        # ─── Log metrics to W&B and finish run ───────────────────────────────
        wandb.log(test_metrics)
        wandb.finish()
