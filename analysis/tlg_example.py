#!/usr/bin/env python3
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, PicoDecoderHF, PicoDecoderHFConfig

# ─── Settings ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
MODEL_NAME = "davidafrica/pico-maml-decoder-large"
SUBFOLDER = "checkpoints/step_6000"
DATASET = "universalner/universal_ner"
TAGALOG_SPLITS = ["tl_trg"]
MAX_EX = 50  # how many examples per split to inspect

# ─── Load model & tokenizer ──────────────────────────────────────────────────
config = PicoDecoderHFConfig.from_pretrained(
    MODEL_NAME, trust_remote_code=True, subfolder=SUBFOLDER
)
# we only need the number of labels
ds0 = load_dataset(DATASET, TAGALOG_SPLITS[0], trust_remote_code=True)
config.num_labels = len(ds0["train"].features["ner_tags"].feature.names)
model = PicoDecoderHF.from_pretrained(
    MODEL_NAME, config=config, trust_remote_code=True, subfolder=SUBFOLDER
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, subfolder=SUBFOLDER
)

records = []
for split in TAGALOG_SPLITS:
    ds = load_dataset(DATASET, split, trust_remote_code=True)["test"]
    # collect a few that actually contain an entity
    examples = [ex for ex in ds if any(tag != 0 for tag in ex["ner_tags"])]
    examples = examples[:MAX_EX]

    for idx, ex in enumerate(examples):
        toks = ex["tokens"]
        tags = ex["ner_tags"]
        # tokenize + align
        tok = tokenizer(
            toks,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        wids = tok.word_ids(batch_index=0)
        aligned = []
        prev = None
        for wid in wids:
            if wid is None or wid == prev:
                aligned.append(-100)
            else:
                aligned.append(tags[wid])
            prev = wid

        # forward
        model.eval()
        with torch.no_grad():
            hidden, _ = model.pico_decoder(
                tok["input_ids"], use_cache=False, return_hidden=True
            )
            logits = model.classifier(hidden)
            logps = F.log_softmax(logits, dim=-1)

        # compute avg log‐prob on true labels
        lp_vals = [
            logps[0, i, lab].item() for i, lab in enumerate(aligned) if lab != -100
        ]
        avg_lp = sum(lp_vals) / len(lp_vals) if lp_vals else float("nan")

        records.append(
            {
                "split": split,
                "example_idx": idx,
                "sentence": " ".join(toks),
                "avg_logprob": avg_lp,
            }
        )

df = pd.DataFrame(records)

# show hardest & easiest
hardest = df.nsmallest(5, "avg_logprob")
easiest = df.nlargest(5, "avg_logprob")

print("\n--- Hardest Tagalog Examples (lowest avg log-prob) ---")
print(
    hardest[["split", "example_idx", "avg_logprob", "sentence"]].to_string(index=False)
)

print("\n--- Easiest Tagalog Examples (highest avg log-prob) ---")
print(
    easiest[["split", "example_idx", "avg_logprob", "sentence"]].to_string(index=False)
)

# plot distribution
plt.figure(figsize=(6, 4))
df["avg_logprob"].hist(bins=20)
plt.title("Tagalog True‐token avg log‐prob distribution")
plt.xlabel("Average token log‐prob")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
