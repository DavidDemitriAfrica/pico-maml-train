#!/usr/bin/env python3
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# 1. CONFIGURATION
MODEL_NAMES = [
    "pico-lm/pico-decoder-tiny",
    "pico-lm/pico-decoder-small",
    "pico-lm/pico-decoder-medium",
    "pico-lm/pico-decoder-large",
]
DATASET_NAME = "universalner/universal_ner"
DATASET_CONFIGS = ["en_ewt", "en_pud"]
SPLIT = "test"  # or "train"/"validation"


# 3. TOKENIZER + ALIGN LABELS
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_id = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_id:
                label_ids.append(labels[word_idx])
            else:
                # set to -100 so we ignore sub‐word pieces in loss/metrics
                label_ids.append(-100)
            previous_id = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


# 4. METRIC
metric = evaluate.load("seqeval")


def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=2)

    # Remove ignored index (-100) and convert to label names
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

    results = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


for DATASET_CONFIG in DATASET_CONFIGS:
    print(f"\nLoading {DATASET_NAME} — config={DATASET_CONFIG}")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    label_list = dataset["train"].features["ner_tags"].feature.names
    for model_name in MODEL_NAMES:
        print(f"\n→ Evaluating {model_name} on {DATASET_NAME}/{DATASET_CONFIG} {SPLIT}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label_list),
        )

        # tokenize & align
        tokenized = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        # set up Trainer
        args = TrainingArguments(
            output_dir=f"./results/{model_name.replace('/', '_')}",
            per_device_eval_batch_size=16,
            do_train=False,
            do_eval=True,
            logging_dir=f"./logs/{model_name.replace('/', '_')}",
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=args,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # run evaluation
        metrics = trainer.evaluate(tokenized[SPLIT])
        print(metrics)
