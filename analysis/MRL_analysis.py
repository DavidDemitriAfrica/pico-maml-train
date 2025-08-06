#!/usr/bin/env python3
# grid_runner.py --------------------------------------------------------------
#
# Sweep head-only fine-tuning for Pico-MAML-medium checkpoints.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import itertools
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from huggingface_hub import HfApi
from seqeval.metrics import classification_report, f1_score
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import TokenClassifierOutput

from src.model.pico_decoder import PicoDecoderHF, PicoDecoderHFConfig, RoPE

import torch.nn.functional as F
from pathlib import Path

# ─── where raw per-sentence records go ───────────────────────────────────────
LOGPROB_DIR = Path("logits_by_sent")
LOGPROB_DIR.mkdir(exist_ok=True)

# ─────────────────────────── user knobs ──────────────────────────────────────
SEED = 42
PROJECT = "pico-maml-grid"
HF_REPO_MAML = "davidafrica/pico-maml-decoder-medium"
DATASET = "universalner/universal_ner"
FT_SPLIT = "hr_set"
EVAL_SPLITS = ["tl_trg", "ceb_gja"]
NUM_EPOCHS = 10
BATCH_SIZE = 16
RESULT_CSV = "maml_results.csv"
MAX_STEP = 6000
STRIDE = 100
WANDB_ENTITY = None  # set if needed
# ─────────────────────────────────────────────────────────────────────────────

# ── CLI sharding -------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--n_shards", type=int, default=1)
ARGS = parser.parse_args()
assert 0 <= ARGS.shard < ARGS.n_shards

# ── logging & seed -----------------------------------------------------------
set_seed(SEED)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger("grid")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── discover checkpoints -----------------------------------------------------
api = HfApi()
file_list = api.list_repo_files(HF_REPO_MAML, repo_type="model")
ALL_STEPS = sorted(
    {
        int(re.search(r"checkpoints/step_(\d+)", p).group(1))
        for p in file_list
        if p.startswith("checkpoints/step_")
    }
)
CHECKPOINT_STEPS = [
    s for s in ALL_STEPS if 0 <= s <= MAX_STEP and s % STRIDE == 0
]
MY_STEPS = [s for i, s in enumerate(CHECKPOINT_STEPS)
            if i % ARGS.n_shards == ARGS.shard]
LOG.info(
    "Shard %d/%d will run %d checkpoints",
    ARGS.shard,
    ARGS.n_shards,
    len(MY_STEPS),
)

# ── dataset & tokenizer ------------------------------------------------------
ds_ft = load_dataset(DATASET, FT_SPLIT, trust_remote_code=True)
LABEL_LIST: List[str] = ds_ft["train"].features["ner_tags"].feature.names
NUM_LABELS = len(LABEL_LIST)

# --- NEW: grab the first checkpoint that exists and load the tokenizer from it
FIRST_STEP   = min(CHECKPOINT_STEPS)          # e.g. 0
TOKEN_FOLDER = f"checkpoints/step_{FIRST_STEP}"

tokenizer = AutoTokenizer.from_pretrained(HF_REPO_MAML, 
                                        subfolder=TOKEN_FOLDER,                  # ← the crucial line
                                        trust_remote_code=True)

PARTICLES = {"si", "ni", "ang", "ng", "sa"}


def tok_align(batch):
    enc = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=128,
    )
    aligned = []
    for i, labs in enumerate(batch["ner_tags"]):
        wids = enc.word_ids(batch_index=i)
        prev = None
        ids = []
        for wid in wids:
            ids.append(-100 if wid is None or wid == prev else labs[wid])
            prev = wid
        aligned.append(ids)
    enc["labels"] = aligned
    return enc


train_ds = ds_ft["train"].map(
    tok_align, batched=True, remove_columns=ds_ft["train"].column_names
)
val_ds = ds_ft["validation"].map(
    tok_align, batched=True, remove_columns=ds_ft["validation"].column_names
)
collator = DataCollatorForTokenClassification(tokenizer)

# ── head-only model wrapper ---------------------------------------------------


def load_head_only(step: int) -> torch.nn.Module:
    subfolder = f"checkpoints/step_{step}"
    cfg = PicoDecoderHFConfig.from_pretrained(
        HF_REPO_MAML, subfolder=subfolder, trust_remote_code=True
    )
    cfg.num_labels = NUM_LABELS
    RoPE._freqs_cis_tensor = None

    backbone = PicoDecoderHF.from_pretrained(
        HF_REPO_MAML, subfolder=subfolder, trust_remote_code=True
    ).pico_decoder
    for param in backbone.parameters():
        param.requires_grad = False

    class HeadOnly(torch.nn.Module):
        def __init__(self, cfg_):
            super().__init__()
            self.cfg = cfg_
            self.backbone = backbone
            self.classifier = torch.nn.Linear(cfg_.d_model, cfg_.num_labels)

        def forward(self, input_ids, attention_mask=None, labels=None):
            hidden, _ = self.backbone(
                input_ids, use_cache=False, return_hidden=True
            )
            logits = self.classifier(hidden)
            loss = None
            if labels is not None:
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(
                    logits.view(-1, self.cfg.num_labels), labels.view(-1)
                )
            return TokenClassifierOutput(loss=loss, logits=logits)

    return HeadOnly(cfg).to(DEVICE)


# ── seqeval micro-F1 ---------------------------------------------------------
def micro_f1(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_tags, pred_tags = [], []
    for p_row, l_row in zip(preds, labels):
        t_seq, p_seq = [], []
        for p_tok, l_tok in zip(p_row, l_row):
            if l_tok != -100:
                t_seq.append(LABEL_LIST[l_tok])
                p_seq.append(LABEL_LIST[p_tok])
        true_tags.append(t_seq)
        pred_tags.append(p_seq)
    return {"f1": f1_score(true_tags, pred_tags)}


# ─── particle-aware recall helper ────────────────────────────────────────────
def particle_hits(tokens: list, spans: list[tuple[int, str]]) -> int:
    """
    Count gold entities that are *immediately* preceded by a Filipino
    particle (“si / ni / ang / ng / sa”).
    """
    hits = 0
    for idx, _ in spans:
        if idx == 0 or idx > len(tokens) - 1:       # out-of-range safety
            continue
        tok = tokens[idx - 1]
        if isinstance(tok, bytes):                  # rare byte token
            tok = tok.decode("utf-8", errors="ignore")
        if isinstance(tok, list):                   # sub-token list → join
            tok = "".join(map(str, tok))
        if str(tok).lower() in PARTICLES:
            hits += 1
    return hits



def eval_full(trainer: Trainer,
              split: str,
              step: int,
              store_logits: bool = True,
              keep_full_logits: bool = False) -> Dict:    
    raw = load_dataset(DATASET, split, trust_remote_code=True)["test"]
    tok_ds = raw.map(tok_align, batched=True, remove_columns=raw.column_names)
    output = trainer.predict(tok_ds)
    logits = output.predictions                       # (N, T, C)
    pred_ids = np.argmax(logits, axis=-1)
    labels = output.label_ids
    spans_gold, spans_pred = [], []
    for p_row, l_row in zip(pred_ids, labels):
        g_spans, p_spans = [], []
        for idx, lab in enumerate(l_row):
            if lab != -100:
                g_spans.append((idx, LABEL_LIST[lab]))
                p_spans.append((idx, LABEL_LIST[p_row[idx]]))
        spans_gold.append(g_spans)
        spans_pred.append(p_spans)

    # --- class-wise F1 (macro) ---------------------------------------------------
    # seqeval expects: List[List[str]]  → one inner list per sentence
    gold_labels = [[lab for _, lab in g] for g in spans_gold]
    pred_labels = [[lab for _, lab in p] for p in spans_pred]

    macro = classification_report(
        gold_labels,
        pred_labels,
        output_dict=True,
        zero_division=0,
    )["macro avg"]
    # ---------------------------------------------------------------------------


    # taxonomy counts
    miss = span_err = typ = spur = 0
    for g_sp, p_sp in zip(spans_gold, spans_pred):
        g_set = {(i, c) for i, c in g_sp}
        p_set = {(i, c) for i, c in p_sp}

        spur += len({(i, c) for i, c in p_set if i not in {gi for gi, _ in g_set}})
        miss += len({(i, c) for i, c in g_set if i not in {pi for pi, _ in p_set}})

        overlap = {(i, c) for i, c in p_set if any(i == gi for gi, _ in g_set)}
        for i, c in overlap:
            g_cls = next(gc for gi, gc in g_set if gi == i)
            if c != g_cls:
                typ += 1

        span_err += sum(
            1
            for (i, c) in p_set
            for (j, d) in g_set
            if i != j and c == d
        )

    # tokenisation stats
    part_hits = oov_cnt = tot_tok = 0
    for tokens, g_sp in zip(raw["tokens"], spans_gold):
        part_hits += particle_hits(tokens, g_sp)
        oov_cnt += sum(tok not in tokenizer.vocab for tok in tokens)
        tot_tok += len(tokens)

    # qualitative snippets
    deltas = []
    for idx, (g_sp, p_sp) in enumerate(zip(spans_gold, spans_pred)):
        correct = sum(1 for s in p_sp if s in g_sp)
        err = len(p_sp) - correct + len(g_sp) - correct
        deltas.append((correct - err, idx))
    deltas.sort()
    worst = [raw[i]["tokens"] for _, i in deltas[:2]]
    best = [raw[i]["tokens"] for _, i in deltas[-2:]]

    # ─── NEW: persist per-sentence log-probs ───────────────────────────────
    if store_logits:
        records = []
        logp = F.log_softmax(torch.from_numpy(logits), dim=-1).cpu().numpy()  # (N,T,C)

        for sent_ix, (tok_row, lab_row, lp_row) in enumerate(
            zip(raw["tokens"], labels, logp)
        ):
            # 1) re-encode once to get sub-token → word mapping
            enc  = tokenizer(tok_row, is_split_into_words=True,
                            truncation=True, max_length=128)
            wids = enc.word_ids()           # len == nr. sub-tokens

            for sub_ix, (lab, lp_vec) in enumerate(zip(lab_row, lp_row)):
                if lab == -100:             # padding / extra sub-piece
                    continue
                wid = wids[sub_ix]          # word index (0…len(tok_row)-1)
                tok_txt = tok_row[wid] if wid is not None else \
                        tokenizer.convert_ids_to_tokens(
                            int(enc["input_ids"][sub_ix])
                        )

                entry = {
                    "sent_id":  sent_ix,
                    "tok_id":   sub_ix,      # sub-token position
                    "word_id":  wid,         # original word position
                    "token":    tok_txt,
                    "gold_lab": LABEL_LIST[lab],
                    "gold_lp":  lp_vec[lab].astype(np.float32),
                }
                if keep_full_logits:
                    entry["logits"] = lp_vec.astype(np.float16)
                records.append(entry)
        # ------------------------------------------------------------------

        df_lp = pd.DataFrame.from_records(records)
        out_path = LOGPROB_DIR / f"{split}_step{step:04d}.parquet"
        df_lp.to_parquet(out_path, compression="zstd")

        # attach lightweight preview to W&B, heavy file as Artifact
        wandb.log({f"{split}_logprob_preview": wandb.Table(dataframe=df_lp.head(30))})
        art = wandb.Artifact(f"{split}_logprobs_step{step:04d}", type="logprobs")
        art.add_file(str(out_path))
        wandb.log_artifact(art)

    return {
        "f1": macro["f1-score"],
        "per_f1": macro.get("PER", 0),
        "loc_f1": macro.get("LOC", 0),
        "org_f1": macro.get("ORG", 0),
        "miss": miss,
        "span": span_err,
        "type": typ,
        "spur": spur,
        "n_gold": sum(len(g) for g in spans_gold),
        "particle_recall": part_hits / max(1, sum(len(g) for g in spans_gold)),
        "oov_rate": oov_cnt / tot_tok,
        "snippets_best": best,
        "snippets_worst": worst,
    }


# ── run one checkpoint -------------------------------------------------------
def run_step(step: int) -> Dict:
    model = load_head_only(step)
    run = wandb.init(
        project=PROJECT,
        entity=WANDB_ENTITY,
        group="maml-head",
        name=f"maml_s{step:04d}",
        config=dict(step=step, finetune_lang=FT_SPLIT, regime="head_only", seed=SEED),
        reinit=True,
    )

    args = TrainingArguments(
        output_dir=f"tmp/maml_s{step}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=3e-5,
        weight_decay=0.01,          # mirrors the reference script
        num_train_epochs=NUM_EPOCHS,

        # ─── match the working setup ─────────────────────────────────────────
        evaluation_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="no",            # ← nothing is written to disk
        load_best_model_at_end=False,  # ← so we don’t try to reload it
        # ─────────────────────────────────────────────────────────────────────

        seed=SEED,
        fp16=True,
        dataloader_num_workers=4,
        report_to=["wandb"],
        disable_tqdm=True,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=micro_f1,
    )
    trainer.train()

    hist = pd.DataFrame(trainer.state.log_history)
    best_dev = hist["eval_f1"].max()
    t90 = hist.loc[hist["eval_f1"] >= 0.9 * best_dev, "epoch"].min()

    test_scores: Dict[str, float] = {}
    for split in EVAL_SPLITS:
        scores = eval_full(trainer, split, step)

        wandb.log(
            {f"{split}_metrics": {k: v for k, v in scores.items() if not k.startswith("snippets")}}
        )

        # snippets
        table = wandb.Table(columns=["kind", "sentence"])
        for sent in scores["snippets_best"]:
            table.add_data("best", " ".join(sent))
        for sent in scores["snippets_worst"]:
            table.add_data("worst", " ".join(sent))
        wandb.log({f"{split}_snippets": table})

        test_scores.update(
            {f"{split}_{k}": v for k, v in scores.items() if not k.startswith("snippets")}
        )

    run.finish()
    return dict(step=step, dev_f1=best_dev, t90=t90, **test_scores)


# ── sweep loop ---------------------------------------------------------------
RESULTS: List[Dict] = []
# ─── resume support ──────────────────────────────────────────────────────────
DONE_STEPS = set()
if Path(RESULT_CSV).exists():
    try:
        DONE_STEPS = set(pd.read_csv(RESULT_CSV)["step"].tolist())
    except Exception:
        pass  # corrupt / empty file → start fresh

CHECKPOINT_STEPS = [s for s in CHECKPOINT_STEPS if s not in DONE_STEPS]
MY_STEPS = [s for i, s in enumerate(CHECKPOINT_STEPS)
            if i % ARGS.n_shards == ARGS.shard]

start = time.time()
for step in MY_STEPS:
    try:
        RESULTS.append(run_step(step))
        LOG.info("✓ step %d done", step)
    except Exception as exc:
        LOG.exception("⚠️  step %d failed: %s", step, exc)

pd.DataFrame(RESULTS).sort_values("step").to_csv(RESULT_CSV, index=False)
LOG.info(
    "Finished %d runs in %.1f h → %s",
    len(RESULTS),
    (time.time() - start) / 3600,
    RESULT_CSV,
)
