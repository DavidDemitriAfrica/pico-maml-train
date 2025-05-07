#!/usr/bin/env python3
import logging
import os
import random

import evaluate
import numpy as np
import torch
import torch.nn as nn
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

# ─── 1. CONFIGURATION ─────────────────────────────────────────────────────────
MODEL_NAMES = [
    "pico-lm/pico-decoder-tiny",
    "pico-lm/pico-decoder-small",
    "pico-lm/pico-decoder-medium",
    "pico-lm/pico-decoder-large",
]
DATASET_NAME = "universalner/universal_ner"
DATASET_CONFIGS = ["en_ewt", "en_pud"]
SPLIT = "test"
BATCH_SIZE = 16

# ─── 2. Metric ────────────────────────────────────────────────────────────────
metric = evaluate.load("seqeval")


def compute_metrics(eval_pred: EvalPrediction):
    # eval_pred.predictions is a [batch, seq_len, num_labels] array
    # eval_pred.label_ids   is a [batch, seq_len] array
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=2)
    true_labels = [
        [
            label_list[label]
            for (label, pred) in zip(label_seq, pred_seq)
            if label != -100
        ]
        for label_seq, pred_seq in zip(labels, preds)
    ]
    true_preds = [
        [
            label_list[pred]
            for (label, pred) in zip(label_seq, pred_seq)
            if label != -100
        ]
        for label_seq, pred_seq in zip(labels, preds)
    ]
    res = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": res["overall_precision"],
        "recall": res["overall_recall"],
        "f1": res["overall_f1"],
        "accuracy": res["overall_accuracy"],
    }


# ─── 3. Main evaluation loop ─────────────────────────────────────────────────
for cfg in DATASET_CONFIGS:
    logger.info(f"Loading dataset {DATASET_NAME} config={cfg}")
    ds = load_dataset(DATASET_NAME, cfg, trust_remote_code=True)
    label_list = ds["train"].features["ner_tags"].feature.names
    logger.info(f"Found {len(label_list)} labels: {label_list}")

    for model_name in MODEL_NAMES:
        logger.info(f"→ Evaluating model '{model_name}' on config='{cfg}'")

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

        max_len = config.max_seq_len
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

        # 3d. Tokenize & align labels
        def tokenize_and_align_labels(examples):
            tok = tokenizer(
                examples["tokens"],
                truncation=True,
                max_length=max_len,
                padding="max_length",
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
            per_device_eval_batch_size=BATCH_SIZE,
            do_train=False,
            do_eval=True,
            logging_dir=log_dir,
            report_to=[],
        )
        logger.info(f"TrainerArguments: {args}")

        trainer = Trainer(
            model=model,
            args=args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            eval_dataset=tokenized[SPLIT],
        )
        logger.info("Trainer initialized with data collator and eval_dataset")

        # 3f. Run evaluation
        logger.info(f"Starting evaluation on split='{SPLIT}'")
        metrics = trainer.evaluate()
        logger.info(f"Results for {model_name} on {cfg}: {metrics}")
        print(f"\n→ {model_name} / {cfg} / {SPLIT}: {metrics}")
