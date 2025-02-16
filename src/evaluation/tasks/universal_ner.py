import evaluate
from datasets import load_dataset
from src.config.evaluation_config import UniversalNEREvaluationConfig
from src.model.pico import PicoForTokenClassification
from transformers import AutoTokenizer, pipeline


def run_universal_ner_evaluation(
    model_path: str, ner_config: UniversalNEREvaluationConfig
) -> dict:
    # Load the Universal NER dataset using the provided config
    dataset = load_dataset(
        ner_config.dataset_name,
        ner_config.dataset_config,  # e.g. "en_pud"
        split=ner_config.dataset_split,
    )
    # Load the tokenizer from the checkpoint/model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    num_labels = 6  # Update as appropriate for your task
    # Load your custom token classification model from checkpoint
    model = PicoForTokenClassification.from_pretrained(
        model_path, num_labels=num_labels
    )

    # Create a token classification pipeline with an aggregation strategy
    ner_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        batch_size=ner_config.batch_size,  # ensure your config includes a reasonable batch_size
    )

    predictions = []
    references = []

    # Get the mapping from label IDs to label strings from the dataset features.
    label_list = dataset.features["ner_tags"].feature.names

    for example in dataset:
        tokens = example["tokens"]
        gold_tags = example["ner_tags"]  # these are integers

        # Reconstruct the sentence from tokens
        sentence = " ".join(tokens)
        ner_results = ner_pipe(sentence)

        # Initialize prediction tags with "O"
        pred_tags = ["O"] * len(tokens)
        for entity in ner_results:
            entity_label = entity["entity_group"]
            # Split the predicted entity text into tokens
            entity_tokens = entity["word"].split()
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if tokens[i : i + len(entity_tokens)] == entity_tokens:
                    pred_tags[i] = f"B-{entity_label}"
                    for j in range(1, len(entity_tokens)):
                        pred_tags[i + j] = f"I-{entity_label}"
                    break

        predictions.append(pred_tags)
        references.append(gold_tags)  # still integers

    # Convert integer references to their string labels using the mapping.
    converted_references = [[label_list[tag] for tag in ref] for ref in references]

    # Compute NER metrics using the seqeval metric
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=predictions, references=converted_references)
    f1 = results.get("overall_f1", 0.0)
    return {"f1": f1, "detailed": results}
