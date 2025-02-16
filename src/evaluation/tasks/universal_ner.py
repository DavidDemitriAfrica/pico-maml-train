import evaluate
from datasets import load_dataset
from src.config.evaluation_config import UniversalNEREvaluationConfig
from src.model.pico import PicoForTokenClassification
from transformers import AutoTokenizer, pipeline


def run_universal_ner_evaluation(
    model_path: str, ner_config: UniversalNEREvaluationConfig
) -> dict:
    # Load the dataset and optionally limit its size.
    dataset = load_dataset(
        ner_config.dataset_name,
        ner_config.dataset_config,
        split=ner_config.dataset_split,
    )
    if ner_config.limit_eval_examples is not None:
        dataset = dataset.select(
            range(min(len(dataset), ner_config.limit_eval_examples))
        )

    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    num_labels = 6  # Update as needed for your task.
    model = PicoForTokenClassification.from_pretrained(
        model_path, num_labels=num_labels
    )

    # Create the NER pipeline. We pass the desired batch_size.
    ner_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        batch_size=ner_config.batch_size,
    )

    predictions = []
    references = []

    # Get label mapping.
    label_list = dataset.features["ner_tags"].feature.names

    for example in dataset:
        tokens = example["tokens"]
        gold_tags = example["ner_tags"]

        # Reconstruct the sentence from tokens and ensure truncation.
        sentence = " ".join(tokens)
        ner_results = ner_pipe(
            sentence, truncation=True, max_length=ner_config.max_length
        )

        # Build predictions.
        pred_tags = ["O"] * len(tokens)
        for entity in ner_results:
            entity_label = entity["entity_group"]
            entity_tokens = entity["word"].split()
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if tokens[i : i + len(entity_tokens)] == entity_tokens:
                    pred_tags[i] = f"B-{entity_label}"
                    for j in range(1, len(entity_tokens)):
                        pred_tags[i + j] = f"I-{entity_label}"
                    break

        predictions.append(pred_tags)
        references.append(gold_tags)

    # Convert integer references to string labels.
    converted_references = [[label_list[tag] for tag in ref] for ref in references]

    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=predictions, references=converted_references)
    f1 = results.get("overall_f1", 0.0)
    return {"f1": f1, "detailed": results}
