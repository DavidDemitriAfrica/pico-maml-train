#!/usr/bin/env python3
import logging
import os

import evaluate
import numpy as np
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import TokenClassifierOutput

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


def compute_metrics(pred):
    logits, labels = pred
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
    ds = load_dataset(DATASET_NAME, cfg)
    label_list = ds["train"].features["ner_tags"].feature.names
    logger.info(f"Found {len(label_list)} labels: {label_list}")

    for model_name in MODEL_NAMES:
        logger.info(f"→ Evaluating model '{model_name}' on config='{cfg}'")

        # 3a. Load model config dynamically with remote code
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.num_labels = len(label_list)
        logger.debug(f"Loaded config: {config}")

        # 3b. Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=True
        )
        logger.info(f"Tokenizer loaded (vocab size={tokenizer.vocab_size})")

        # 3c. Data collator for dynamic padding
        data_collator = DataCollatorForTokenClassification(tokenizer)
        logger.info("Data collator ready")

        # 3d. Load base causal LM with updated config
        base_lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            config=config,
        )
        logger.info("Base LM loaded with dynamic config")

        # 3e. Define token-classification wrapper
        class PicoForTokenClassification(PreTrainedModel):
            config_class = type(config)
            base_model_prefix = base_lm.base_model_prefix

            def __init__(self, config):
                super().__init__(config)
                self.base_lm = base_lm
                self.classifier = nn.Linear(config.d_model, config.num_labels)
                self.init_weights()

            def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
                outputs = self.base_lm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                hidden = outputs.hidden_states[-1]
                logits = self.classifier(hidden)
                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(
                        logits.view(-1, self.config.num_labels),
                        labels.view(-1),
                    )
                return TokenClassifierOutput(loss=loss, logits=logits)

        model = PicoForTokenClassification(config)
        logger.info("Wrapped LM in PicoForTokenClassification")

        # 3f. Tokenize & align
        def tokenize_and_align_labels(examples):
            tok = tokenizer(
                examples["tokens"], truncation=True, is_split_into_words=True
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
                all_labels.append(label_ids)
            tok["labels"] = all_labels
            return tok

        logger.info("Tokenizing dataset")
        tokenized = ds.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=ds["train"].column_names,
        )
        logger.info(f"Tokenized columns: {tokenized.column_names}")

        # 3g. Trainer setup
        output_dir = os.path.join("results", model_name.replace("/", "_"), cfg)
        log_dir = os.path.join("logs", model_name.replace("/", "_"), cfg)
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=BATCH_SIZE,
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
        logger.info("Trainer initialized")

        # 3h. Evaluate
        logger.info(f"Evaluating split='{SPLIT}'")
        metrics = trainer.evaluate()
        logger.info(f"Metrics {model_name} / {cfg}: {metrics}")
        print(f"\n→ {model_name} / {cfg} / {SPLIT}:\n  {metrics}")
