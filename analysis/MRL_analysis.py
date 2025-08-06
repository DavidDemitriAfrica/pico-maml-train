#!/usr/bin/env python3
# grid_runner.py --------------------------------------------------------------
#
# Sweep head-only fine-tuning for Pico-MAML-medium checkpoints.
# Runs every checkpoints/step_{0..6000} at stride 100, sharded if desired.
#
# ---------------------------------------------------------------------------

import argparse, logging, os, re, time, itertools, sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
import wandb
from seqeval.metrics import classification_report
from collections import Counter

from transformers import (
    AutoTokenizer, DataCollatorForTokenClassification,
    Trainer, TrainingArguments, set_seed
)
from transformers.modeling_outputs import TokenClassifierOutput
from src.model.pico_decoder import PicoDecoderHF, PicoDecoderHFConfig, RoPE  # noqa

# ─────────────────────────── user knobs ──────────────────────────────────────
SEED           = 42
PROJECT        = "pico-maml-grid"
HF_REPO_MAML   = "davidafrica/pico-maml-decoder-medium"
DATASET        = "universalner/universal_ner"
FT_SPLIT       = "hr_set"
EVAL_SPLITS    = ["tl_trg", "ceb_gja"]
NUM_EPOCHS     = 10
BATCH_SIZE     = 16
RESULT_CSV     = "maml_results.csv"
MAX_STEP       = 6000
STRIDE         = 100
WANDB_ENTITY   = None        # add if needed
# ─────────────────────────────────────────────────────────────────────────────

# ------------- parse shard args ---------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--shard",    type=int, default=0)
ap.add_argument("--n_shards", type=int, default=1)
args = ap.parse_args()
assert 0 <= args.shard < args.n_shards

# ------------- logging & seed -----------------------------------------------
set_seed(SEED)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("grid")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PARTICLES = {"si", "ni", "ang", "ng", "sa"}
def particle_stats(tokens, spans):
    hits = 0
    for (idx, _cls) in spans:
        if idx > 0 and tokens[idx-1].lower() in PARTICLES:
            hits += 1
    return hits
def oov_rate(tokens):
    return sum(tok not in tokenizer.vocab for tok in tokens) / len(tokens)


# ------------- discover checkpoints -----------------------------------------
api   = HfApi()
files = api.list_repo_files(HF_REPO_MAML, repo_type="model")
all_steps = sorted({
    int(re.search(r"checkpoints/step_(\d+)", p).group(1))
    for p in files if p.startswith("checkpoints/step_")
})
STEPS = [s for s in all_steps if 0 <= s <= MAX_STEP and s % STRIDE == 0]
MY_STEPS = [s for s in STEPS if s % args.n_shards == args.shard]
log.info("Shard %d/%d runs %d checkpoints", args.shard, args.n_shards, len(MY_STEPS))

# ------------- datasets & tokenizer -----------------------------------------
ds_ft = load_dataset(DATASET, FT_SPLIT, trust_remote_code=True)
label_list: List[str] = ds_ft["train"].features["ner_tags"].feature.names
num_labels = len(label_list)
tok = AutoTokenizer.from_pretrained(HF_REPO_MAML, trust_remote_code=True)

def tok_align(batch):
    t = tok(batch["tokens"], is_split_into_words=True,
            truncation=True, max_length=128)
    aligned = []
    for i, labs in enumerate(batch["ner_tags"]):
        wids, prev = t.word_ids(batch_index=i), None
        lab_ids = []
        for wid in wids:
            lab_ids.append(-100 if wid is None or wid == prev else labs[wid])
            prev = wid
        aligned.append(lab_ids)
    t["labels"] = aligned
    return t

train_ds = ds_ft["train"].map(tok_align, batched=True,
                              remove_columns=ds_ft["train"].column_names)
val_ds   = ds_ft["validation"].map(tok_align, batched=True,
                                   remove_columns=ds_ft["validation"].column_names)
collator = DataCollatorForTokenClassification(tok)

# ------------- head-only wrapper --------------------------------------------
def load_head_only(step: int):
    subfolder = f"checkpoints/step_{step}"
    cfg = PicoDecoderHFConfig.from_pretrained(
        HF_REPO_MAML, subfolder=subfolder, trust_remote_code=True)
    cfg.num_labels = num_labels
    RoPE._freqs_cis_tensor = None

    backbone = PicoDecoderHF.from_pretrained(
        HF_REPO_MAML, subfolder=subfolder, trust_remote_code=True
    ).pico_decoder
    for p in backbone.parameters():
        p.requires_grad = False

    class HeadOnly(torch.nn.Module):
        def __init__(self, cfg_):
            super().__init__()
            self.cfg = cfg_
            self.backbone = backbone
            self.cls = torch.nn.Linear(cfg_.d_model, cfg_.num_labels)

        def forward(self, input_ids, attention_mask=None, labels=None):
            h, _ = self.backbone(input_ids, use_cache=False, return_hidden=True)
            logits = self.cls(h)
            if labels is not None:
                loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
                    logits.view(-1, cfg_.num_labels), labels.view(-1))
            else:
                loss = None
            return TokenClassifierOutput(loss=loss, logits=logits)

    return HeadOnly(cfg).to(DEVICE)

def seqeval_f1(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    true_, pred_ = [], []
    for p, l in zip(preds, labels):
        t_seq, p_seq = [], []
        for pi, li in zip(p, l):
            if li != -100:
                t_seq.append(label_list[li])
                p_seq.append(label_list[pi])
        true_.append(t_seq); pred_.append(p_seq)
    return {"f1": f1_score(true_, pred_)}

def eval_full(trainer, split: str):
    raw = load_dataset(DATASET, split, trust_remote_code=True)["test"]
    tok_ds = raw.map(tok_align, batched=True,
                     remove_columns=raw.column_names)
    preds, labels, _ = trainer.predict(tok_ds)
    pred_ids = np.argmax(preds, axis=-1)

    # token-level → span-level conversion
    spans_gold, spans_pred = [], []
    for p_row, l_row in zip(pred_ids, labels):
        g_spans = []; p_spans = []
        for idx, lab in enumerate(l_row):
            if lab != -100:
                g_spans.append((idx, label_list[lab]))
                p_spans.append((idx, label_list[p_row[idx]]))
        spans_gold.append(g_spans)
        spans_pred.append(p_spans)

    # -------- F1 per class --------
    class_f1 = dict(classification_report(
        [list(zip(*spans_gold)[1])], [list(zip(*spans_pred)[1])],
        output_dict=True, zero_division=0)["macro avg"]
    )

    # -------- error taxonomy --------
    miss = span = typ = spur = 0
    for gold, pred in zip(spans_gold, spans_pred):
        g_set = {(i,c) for i,c in gold}
        p_set = {(i,c) for i,c in pred}
        spur += len(p_set - {x for x in p_set if any(i==gi for gi,_ in g_set)})
        miss += len(g_set - {x for x in g_set if any(i==pi for pi,_ in p_set)})

        for i,c in p_set & {x for x in p_set if any(i==gi for gi,_ in g_set)}:
            g_cls = next(gc for gi,gc in g_set if gi==i)
            if c != g_cls: typ += 1
        span += len([1 for (i,_c) in p_set for (j,_d) in g_set if i!=j and _c==_d])

    part_hits = oov_cnt = tot_tok = 0
    for row, gold in zip(raw["tokens"], spans_gold):
        part_hits += particle_stats(row, gold)
        oov_cnt   += sum(tok not in tokenizer.vocab for tok in row)
        tot_tok   += len(row)

    return {
        "f1": class_f1["f1-score"],
        "per_f1": class_f1.get("PER", 0),
        "loc_f1": class_f1.get("LOC", 0),
        "org_f1": class_f1.get("ORG", 0),
        "miss": miss, "span": span, "type": typ, "spur": spur,
        "n_gold": sum(len(g) for g in spans_gold),
        "particle_recall": part_hits / max(1, sum(len(g) for g in spans_gold)),
        "oov_rate": oov_cnt / tot_tok
    }
# ------------- run one checkpoint -------------------------------------------
def run_step(step: int) -> Dict:
    model = load_head_only(step)
    run = wandb.init(
        project = PROJECT,
        entity  = WANDB_ENTITY,
        group   = "maml-head",
        name    = f"maml_s{step:04d}",
        config  = dict(step=step, finetune_lang=FT_SPLIT,
                       regime="head_only", seed=SEED),
        reinit  = True,
    )

    args = TrainingArguments(
        output_dir              = f"tmp/maml_s{step}",
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate           = 3e-5,
        num_train_epochs        = NUM_EPOCHS,
        evaluation_strategy     = "epoch",
        save_strategy           = "no",
        report_to               = ["wandb"],
        seed                    = SEED,
        disable_tqdm            = True,
        load_best_model_at_end  = True,
        metric_for_best_model   = "eval_f1",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, tokenizer=tok,
        compute_metrics=seqeval_f1,
    )
    trainer.train()

    hist = pd.DataFrame(trainer.state.log_history)
    best_dev = hist["eval_f1"].max()
    t90 = hist.loc[hist["eval_f1"] >= 0.9 * best_dev, "epoch"].min()

    test_scores = {}
    for split in EVAL_SPLITS:
        scores = eval_full(trainer, split)
        wandb.log({f"{split}_metrics": scores})
        test_scores.update({f"{split}_{k}": v for k,v in scores.items()})
        # pick 2 best-improved and 2 worst-regressed sentences
        delta = preds - labels  # crude: positive = correction, negative = new error
        top = np.argsort(delta)[-2:]; worst = np.argsort(delta)[:2]
        snips = [raw[i]["tokens"] for i in np.concatenate([top, worst])]
        wandb.log({f"snippets_step{step}": wandb.Table(
            columns=["tokens"], data=[[ " ".join(x) ] for x in snips])})


    run.finish()
    return dict(step=step, dev_f1=best_dev, t90=t90, **test_scores)

# ------------- sweep loop ----------------------------------------------------
results = []
start = time.time()
for step in MY_STEPS:
    try:
        results.append(run_step(step))
        log.info("✓ step %d done", step)
    except Exception as e:
        log.exception("⚠️  step %d failed: %s", step, e)

pd.DataFrame(results).sort_values("step").to_csv(RESULT_CSV, index=False)
log.info("Finished %d runs in %.1f h → %s",
         len(results), (time.time()-start)/3600, RESULT_CSV)
