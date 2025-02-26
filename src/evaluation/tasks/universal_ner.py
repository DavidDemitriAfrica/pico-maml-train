import evaluate
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from src.model.pico import PicoForTokenClassification
from src.config.evaluation_config import UniversalNEREvaluationConfig


def run_universal_ner_evaluation(
    model_path: str, ner_config: UniversalNEREvaluationConfig
) -> dict:
    """
    Evaluate a custom token classification (NER) model without using pipeline("ner").
    Works with a PicoForTokenClassification or any other HF-compatible token classification model
    that outputs [batch_size, seq_len, num_labels] logits.
    """

    # 1. Load the dataset
    dataset = load_dataset(
        ner_config.dataset_name,
        ner_config.dataset_config,
        split=ner_config.dataset_split,
    )
    if ner_config.limit_eval_examples is not None:
        dataset = dataset.select(
            range(min(len(dataset), ner_config.limit_eval_examples))
        )

    # 2. Label List & References
    #    We'll need the label names for conversion from ID -> string label.
    #    E.g. dataset.features["ner_tags"].feature.names is often something like
    #    ["O", "B-PER", "I-PER", ...].
    label_list = dataset.features["ner_tags"].feature.names
    num_labels = len(label_list)

    # 3. Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.model_max_length = ner_config.max_length  # or some fixed length

    model = PicoForTokenClassification.from_pretrained(
        model_path, num_labels=num_labels
    )
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Evaluate loop
    predictions = []
    references = []

    # SeqEval needs lists of label-names per token
    # e.g. predictions = [["O","B-LOC","I-LOC", "O"], ...]
    # and references =  [["O","B-LOC","I-LOC", "O"], ...]

    for example in dataset:
        tokens = example["tokens"]  # a list of tokens
        gold_tags = example["ner_tags"]  # list of label IDs (integers)

        # Tokenize (handle subwords)
        # Note: for token classification, use `is_split_into_words=True`
        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,  # so we can merge subwords later
            truncation=True,
            max_length=ner_config.max_length,
            return_tensors="pt",
        )
        # offset_mapping = inputs.pop("offset_mapping")
        # Move everything to GPU/CPU
        for k, v in inputs.items():
            inputs[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Expect shape = [batch_size=1, seq_len_subwords, num_labels]
            logits = outputs.logits  # in PicoForTokenClassification forward
            pred_ids = torch.argmax(logits, dim=-1)  # shape [1, seq_len_subwords]

        pred_ids = pred_ids[0].cpu().numpy()  # now shape [seq_len_subwords]

        # 5. Map subword predictions back to original "token" predictions
        # We'll store a predicted label *per original token*.

        # subword_to_token_index = []
        # current_token_idx = 0
        # prev_end = 0

        # offset_mapping[i][0] == prev_end (same token continuing).

        word_ids = inputs.word_ids(
            batch_index=0
        )  # directly provided by HF fast tokenizers
        # word_ids is a list of token-index or None for special tokens

        # We'll accumulate predicted IDs for each original token
        token_pred_ids = [None] * len(tokens)
        for subword_idx, token_idx in enumerate(word_ids):
            if token_idx is None:
                continue  # skip special tokens like [CLS], [SEP] etc.
            # If that slot is still None, set its label:
            if token_pred_ids[token_idx] is None:
                token_pred_ids[token_idx] = pred_ids[subword_idx]
            # Otherwise, you might decide to override or do majority voting.
            # But let's keep it simple: use the first subword's predicted label.

        # Fill any missing predictions with "O" (or 0) if needed
        for i in range(len(token_pred_ids)):
            if token_pred_ids[i] is None:
                token_pred_ids[i] = 0  # 'O' label ID

        # 6. Convert label IDs → label strings
        pred_tags = [label_list[label_id] for label_id in token_pred_ids]
        # references are the gold label IDs → strings
        ref_tags = [label_list[label_id] for label_id in gold_tags]

        predictions.append(pred_tags)
        references.append(ref_tags)

    # 7. Evaluate with seqeval
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=predictions, references=references)
    f1 = results.get("overall_f1", 0.0)

    return {"f1": f1, "detailed": results}
