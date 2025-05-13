#!/usr/bin/env python3
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

# local HF wrapper & config
from src.model.pico_decoder import PicoDecoderHF, PicoDecoderHFConfig, RoPE

# ─── -1. Seeds & Determinism ─────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_seed(SEED)
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

# ─── 0.5. MAML subfolder ──────────────────────────────────────────────────────
STEP = 5000
SUBFOLDER = f"checkpoints/step_{STEP}"
logger.info(f"MAML subfolder (for davidafrica/*): {SUBFOLDER}")

# ─── 1. CONFIG ────────────────────────────────────────────────────────────────
MODEL_NAMES = [
    # vanilla HF
    # "pico-lm/pico-decoder-tiny",
    # "pico-lm/pico-decoder-small",
    # "pico-lm/pico-decoder-medium",
    # "pico-lm/pico-decoder-large",
    # MAML‐variants
    # "davidafrica/pico-maml-decoder-tiny",
    "davidafrica/pico-maml-decoder-small",
    "davidafrica/pico-maml-decoder-medium",
    "davidafrica/pico-maml-decoder-large",
]
DATASET_NAME = "universalner/universal_ner"
DATASET_CONFIGS = ["en_ewt"]
BATCH_SIZE = 16
NUM_EPOCHS = 10  # align with paper


# ─── 2. W&B callback ──────────────────────────────────────────────────────────
class PrefixedWandbCallback(WandbCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        wandb_logs = {}
        for k, v in logs.items():
            if k.startswith("eval_"):
                wandb_logs[f"valid/{k[len('eval_'):]}"] = v
            else:
                wandb_logs[f"train/{k}"] = v
        wandb.log(wandb_logs, step=state.global_step)


# ─── 3. Metric w/ per‐tag breakdown ────────────────────────────────────────────
metric = evaluate.load("seqeval")


def compute_metrics(eval_pred: EvalPrediction):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
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
    out = {
        "precision": res["overall_precision"],
        "recall": res["overall_recall"],
        "f1": res["overall_f1"],
        "accuracy": res["overall_accuracy"],
    }
    for ent, stats in res.items():
        if isinstance(stats, dict):
            out[f"{ent}_precision"] = stats.get("precision")
            out[f"{ent}_recall"] = stats.get("recall")
            out[f"{ent}_f1"] = stats.get("f1")
            out[f"{ent}_support"] = stats.get("number")
    return out


# ─── 4. Main loop ────────────────────────────────────────────────────────────
for cfg in DATASET_CONFIGS:
    logger.info(f"Loading {DATASET_NAME}, config={cfg}")
    ds = load_dataset(DATASET_NAME, cfg, trust_remote_code=True)
    base = ds["train"] if "train" in ds else next(iter(ds.values()))
    label_list = base.features["ner_tags"].feature.names
    logger.info(f"Labels: {label_list}")

    for model_name in MODEL_NAMES:
        is_maml = model_name.startswith("davidafrica/pico-maml")
        run_id = f"ner_finetune_{model_name.split('/')[-1]}_{cfg}"
        logger.info(f"→ W&B run: {run_id}")

        wandb.init(
            project="pico-maml",
            entity="pico-lm",
            name=run_id,
            tags=[run_id, "ner_finetune", "head_only"],
            reinit=True,
        )

        # ─── a) Load config/tokenizer/model
        load_kw = {"trust_remote_code": True}
        if is_maml:
            load_kw["subfolder"] = SUBFOLDER

        config = PicoDecoderHFConfig.from_pretrained(model_name, **load_kw)
        config.num_labels = len(label_list)
        RoPE._freqs_cis_tensor = None

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **load_kw)
        data_collator = DataCollatorForTokenClassification(tokenizer)

        base_lm = PicoDecoderHF.from_pretrained(model_name, config=config, **load_kw)

        # ─── b) Wrap & freeze
        class PicoForTokenClassification(PreTrainedModel):
            config_class = config.__class__
            base_model_prefix = "pico_decoder"

            def __init__(self, config):
                super().__init__(config)
                self.pico_decoder = base_lm.pico_decoder
                self.classifier = nn.Linear(config.d_model, config.num_labels)
                self.init_weights()

            def forward(self, input_ids, attention_mask=None, labels=None, **kw):
                hidden, _ = self.pico_decoder(
                    input_ids, past_key_values=None, use_cache=False, return_hidden=True
                )
                logits = self.classifier(hidden)
                loss = None
                if labels is not None:
                    loss = nn.CrossEntropyLoss(ignore_index=-100)(
                        logits.view(-1, config.num_labels),
                        labels.view(-1),
                    )
                return TokenClassifierOutput(loss=loss, logits=logits)

        model = PicoForTokenClassification(config)
        for _, p in model.pico_decoder.named_parameters():
            p.requires_grad = False

        # ─── c) Tokenize + align
        max_len = min(128, config.max_seq_len)

        def tokenize_and_align_labels(examples):
            tok = tokenizer(
                examples["tokens"],
                truncation=True,
                max_length=max_len,
                is_split_into_words=True,
            )
            lab_ids = []
            for i, labs in enumerate(examples["ner_tags"]):
                wids, prev = tok.word_ids(batch_index=i), None
                ids = []
                for wid in wids:
                    if wid is None:
                        ids.append(-100)
                    elif wid != prev:
                        ids.append(labs[wid])
                    else:
                        ids.append(-100)
                    prev = wid
                lab_ids.append(ids)
            tok["labels"] = lab_ids
            return tok

        tokenized = ds.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=ds["train"].column_names,
        )
        logger.info(f"Tokenized; columns now: {tokenized.column_names}")

        # ─── d) Trainer args (paper-style) ───────────────────────────────────────
        output_dir = os.path.join("results", model_name.replace("/", "_"), cfg)
        log_dir = os.path.join("logs", model_name.replace("/", "_"), cfg)
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=2e-5,
            weight_decay=0.01,
            num_train_epochs=NUM_EPOCHS,
            evaluation_strategy="steps",
            eval_steps=500,
            logging_strategy="steps",
            logging_steps=500,
            save_strategy="no",
            save_steps=None,
            save_total_limit=1,
            load_best_model_at_end=False,
            label_names=["labels"],
            run_name=run_id,
            seed=SEED,
            fp16=True,
            dataloader_num_workers=4,
            report_to=["wandb"],
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[PrefixedWandbCallback],
        )
        logger.info("Trainer ready")

        # ─── e) Train only the head
        logger.info("Training classifier head")
        trainer.train()

        # ─── f) Final test eval
        logger.info("Evaluating on test split")
        raw_test = trainer.evaluate(tokenized["test"])
        test_logs = {}
        for k, v in raw_test.items():
            if not k.startswith("eval_"):
                test_logs[f"test/{k}"] = v
            else:
                nm = k[len("eval_") :]
                if nm in ("loss", "precision", "recall", "f1", "accuracy"):
                    test_logs[f"test/{nm}"] = v
                else:
                    test_logs[f"in_depth/{nm}"] = v

        wandb.log(test_logs)
        logger.info(f"Test results: {test_logs}")
        print(f"\n→ {run_id}: {test_logs}")

        # ─── g) Save + finish
        # os.makedirs(output_dir, exist_ok=True)
        # with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        #    json.dump(test_logs, f, indent=2)
        wandb.finish()
        logger.info(f"Finished run {run_id}")
