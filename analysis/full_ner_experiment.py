#!/usr/bin/env python3
import logging
import os
import random

import evaluate
import numpy as np
import torch
import wandb
from datasets import concatenate_datasets, load_dataset
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
    "pico-lm/pico-decoder-tiny",
    "pico-lm/pico-decoder-small",
    "pico-lm/pico-decoder-medium",
    "pico-lm/pico-decoder-large",
    # MAML‐variants
    "davidafrica/pico-maml-decoder-tiny",
    "davidafrica/pico-maml-decoder-small",
    "davidafrica/pico-maml-decoder-medium",
    "davidafrica/pico-maml-decoder-large",
]

# the ten UNER configs from the paper
DATASET_CONFIGS = [
    "da_ddt",
    "en_ewt",
    "hr_set",
    "pt_bosque",
    # "qaf_arabizi", # not actually available
    "sk_snk",
    "sr_set",
    "sv_talbanken",
    "zh_gsd",
    "zh_gsdsimp",
]

# these have no train/dev splits – we only eval on their test sets
TEST_ONLY_CONFIGS = ["ceb_gja", "tl_trg", "tl_ugnayan"]

# FINETUNE on each single dataset, then FINETUNE on ALL
FINETUNE_CONFIGS = DATASET_CONFIGS + ["all"]

# training hyper-parameters from the paper
BATCH_SIZE = 16
NUM_EPOCHS = 10
DATASET_NAME = "universalner/universal_ner"

# ─── 1.5. Pre-load all individual datasets ────────────────────────────────────
logger.info("Loading all UNER configs into memory…")
ds_dict = {
    cfg: load_dataset(DATASET_NAME, cfg, trust_remote_code=True)
    for cfg in DATASET_CONFIGS
}


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
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=2)
    true_labels, true_preds = [], []
    for lab_seq, prd_seq in zip(labels, preds):
        tl, tp = [], []
        for label, pred in zip(lab_seq, prd_seq):
            if label == -100:
                continue
            tl.append(label_list[label])
            tp.append(label_list[pred])
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


# ─── tokenization + label alignment fn (unchanged) ────────────────────────────
def tokenize_and_align_labels(examples):
    tok = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=min(128, config.max_seq_len),
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


# ─── 4. Main loop ────────────────────────────────────────────────────────────
for finetune_cfg in FINETUNE_CONFIGS:
    # build train/validation
    if finetune_cfg != "all":
        ds = ds_dict[finetune_cfg]
        train_split = ds["train"]
        val_split = ds["validation"]
    else:
        # concatenate *all* training & dev
        train_split = concatenate_datasets(
            [ds_dict[c]["train"] for c in DATASET_CONFIGS]
        )
        val_split = concatenate_datasets(
            [ds_dict[c]["validation"] for c in DATASET_CONFIGS]
        )

    # shared label list
    label_list = train_split.features["ner_tags"].feature.names

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAMES[0], use_fast=True
    )  # dummy to get tokenizer config
    tokenized_train = train_split.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=train_split.column_names,
    )
    tokenized_val = val_split.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=val_split.column_names,
    )

    for model_name in MODEL_NAMES:
        is_maml = model_name.startswith("davidafrica/pico-maml")

        # derive size & variant
        model_slug = model_name.split("/")[-1]  # e.g. "pico-maml-decoder-small"
        size = model_slug.split("-")[-1]  # "small"
        variant = "maml" if "maml" in model_slug else "vanilla"

        run_id = f"ner_fulltune_{model_name.split('/')[-1]}_{finetune_cfg}"
        logger.info(f"→ W&B run: {run_id}")

        tags = [
            f"size:{size}",  # e.g. "size:small"
            f"variant:{variant}",  # "variant:maml" or "variant:vanilla"
            f"lang:{finetune_cfg}",
            "ner_finetune",
        ]
        wandb.init(
            project="pico-maml",
            entity="pico-lm",
            name=run_id,
            tags=tags,
            reinit=True,
        )

        # a) load config/tokenizer/model
        load_kw = {"trust_remote_code": True}
        if is_maml:
            load_kw["subfolder"] = SUBFOLDER

        config = PicoDecoderHFConfig.from_pretrained(model_name, **load_kw)
        config.num_labels = len(label_list)
        RoPE._freqs_cis_tensor = None

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **load_kw)
        data_collator = DataCollatorForTokenClassification(tokenizer)

        base_lm = PicoDecoderHF.from_pretrained(model_name, config=config, **load_kw)

        # define token‐classification wrapper (unchanged)
        class PicoForTokenClassification(PreTrainedModel):
            config_class = config.__class__
            base_model_prefix = "pico_decoder"

            def __init__(self, config):
                super().__init__(config)
                self.pico_decoder = base_lm.pico_decoder
                self.classifier = torch.nn.Linear(config.d_model, config.num_labels)
                self.init_weights()

            def forward(self, input_ids, attention_mask=None, labels=None, **kw):
                hidden, _ = self.pico_decoder(
                    input_ids, past_key_values=None, use_cache=False, return_hidden=True
                )
                logits = self.classifier(hidden)
                loss = None
                if labels is not None:
                    loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
                        logits.view(-1, config.num_labels),
                        labels.view(-1),
                    )
                return TokenClassifierOutput(loss=loss, logits=logits)

        model = PicoForTokenClassification(config)

        # d) Trainer args
        args = TrainingArguments(
            output_dir=os.path.join(
                "results", model_name.replace("/", "_"), finetune_cfg
            ),
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=3e-5,
            num_train_epochs=NUM_EPOCHS,
            evaluation_strategy="steps",
            eval_steps=500,
            logging_strategy="steps",
            logging_steps=500,
            save_strategy="no",
            seed=SEED,
            fp16=True,
            dataloader_num_workers=4,
            report_to=["wandb"],
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[PrefixedWandbCallback],
        )

        # e) Train
        logger.info("Training classifier head")
        trainer.train()

        # f) Evaluate *on every* dataset’s test split
        EVAL_CONFIGS = DATASET_CONFIGS + TEST_ONLY_CONFIGS
        for eval_cfg in EVAL_CONFIGS:
            if eval_cfg in ds_dict:
                eval_ds = ds_dict[eval_cfg]["test"]
            else:
                # load on-the-fly for test-only configs
                eval_ds = load_dataset(DATASET_NAME, eval_cfg, trust_remote_code=True)[
                    "test"
                ]
            tokenized_eval = eval_ds.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=eval_ds.column_names,
            )
            raw = trainer.evaluate(tokenized_eval)
            # flatten and tag logs so you can distinguish finetune_cfg→eval_cfg
            test_logs = {}
            for k, v in raw.items():
                key = k[len("eval_") :] if k.startswith("eval_") else k
                test_logs[f"{eval_cfg}/{key}"] = v
            wandb.log(test_logs)

        wandb.finish()
        logger.info(f"Finished run {run_id}")
