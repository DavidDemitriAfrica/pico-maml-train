import os

import evaluate
import numpy as np
import torch.nn as nn
import wandb
from transformers import (
    DataCollatorForTokenClassification,
    EvalPrediction,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    WandbCallback,
)
from transformers.modeling_outputs import TokenClassifierOutput


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


def compute_metrics(
    eval_pred: EvalPrediction, label_list: list, metric=evaluate.load("seqeval")
):
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


def run_ner_eval(
    base_lm,
    tokenizer,
    tokenized_ds: dict,
    label_list: list,
    seed: int = 42,
    batch_size: int = 16,
    device: str = "cpu",
):
    """
    Fine-tunes a linear NER head on base_lm and evaluates on tokenized_ds.
    Returns a dict like {'precision':..,'recall':..,'f1':..,'PER_f1':.., ...}.
    """
    # ── 1. Prepare config & wrapper ─────────────────────────────
    cfg = base_lm.config
    cfg.num_labels = len(label_list)

    class PicoForTokenClassification(PreTrainedModel):
        config_class = cfg.__class__
        base_model_prefix = "pico_decoder"

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.pico_decoder = base_lm.pico_decoder
            self.classifier = nn.Linear(config.d_model, config.num_labels)
            self.init_weights()

        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            # get hidden states from all positions
            hidden, _ = self.pico_decoder(
                input_ids, past_key_values=None, use_cache=False, return_hidden=True
            )
            logits = self.classifier(hidden)  # (batch, seq_len, num_labels)
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    logits.view(-1, self.config.num_labels),
                    labels.view(-1),
                )
            return TokenClassifierOutput(loss=loss, logits=logits)

    model = PicoForTokenClassification(cfg).to(device)

    # ── 2. Freeze backbone ────────────────────────────────────────
    for p in model.pico_decoder.parameters():
        p.requires_grad = False

    # ── 3. Data collator ─────────────────────────────────────────
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # ── 4. Training args ─────────────────────────────────────────
    output_dir = os.path.join("tmp_ner", f"seed{seed}")
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        do_train=True,
        do_eval=True,
        evaluation_strategy="no",
        seed=seed,
        logging_dir=output_dir + "/logs",
        logging_steps=100,
        save_strategy="no",
        report_to=[],  # disable built-in loggers
    )

    # ── 5. Trainer setup ──────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # reuse your existing function
    )

    # ── 6. Train & eval ───────────────────────────────────────────
    trainer.train()
    raw = trainer.evaluate(tokenized_ds["test"])

    # ── 7. Flatten metrics ────────────────────────────────────────
    results = {}
    for k, v in raw.items():
        if k.startswith("eval_"):
            name = k[len("eval_") :]
            results[name] = v
        else:
            results[k] = v
    return results
