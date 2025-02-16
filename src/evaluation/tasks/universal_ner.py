"""
Universal NER Evaluation

This evaluation script loads the Universal NER dataset, runs token classification using
a Hugging Face model loaded from the provided checkpoint, and computes NER metrics.
"""

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
        ner_config.dataset_config,  # <-- pass the dataset config
        split=ner_config.dataset_split,
    )
    # Load the model and tokenizer for token classification
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Instantiate your custom token classification model.
    # Decide on the number of labels (e.g., the number of entity types + 1 for 'O')
    num_labels = 6  # <-- update this as appropriate for your task
    model = PicoForTokenClassification.from_pretrained(model_path, num_labels)

    # Create a token classification pipeline with an aggregation strategy
    ner_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )

    predictions = []
    references = []

    for example in dataset:
        # The dataset provides pre-tokenized data.
        tokens = example["tokens"]
        gold_tags = example["ner_tags"]

        # Reconstruct the sentence (assuming whitespace separation matches the tokenization)
        sentence = " ".join(tokens)

        # Run the NER pipeline on the sentence
        ner_results = ner_pipe(sentence)

        # Initialize prediction tags with "O"
        pred_tags = ["O"] * len(tokens)

        # For each predicted entity, align it with the tokens.
        for entity in ner_results:
            entity_label = entity["entity_group"]
            # Split the predicted entity text into tokens
            entity_tokens = entity["word"].split()
            # Try to find the entity_tokens in the original tokens list
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if tokens[i : i + len(entity_tokens)] == entity_tokens:
                    pred_tags[i] = f"B-{entity_label}"
                    for j in range(1, len(entity_tokens)):
                        pred_tags[i + j] = f"I-{entity_label}"
                    break

        predictions.append(pred_tags)
        references.append(gold_tags)

    # Compute NER metrics using the seqeval metric
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=predictions, references=references)
    f1 = results.get("overall_f1", 0.0)
    return {"f1": f1, "detailed": results}
