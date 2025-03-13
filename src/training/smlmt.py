import re
import random
from torch.utils.data import Dataset
import torch.nn as nn


class ClassifierMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.1):
        """
        Args:
            input_dim (int): Size of the input features.
            hidden_dims (List[int]): A list where each element specifies the number of units
                                     in that hidden layer.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate applied after each hidden layer.
        """
        super(ClassifierMLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SMLMTTask:
    """
    Class for generating a Self-supervised Meta-learning Task (SMLMT).

    Each task is built from a subset of vocabulary words. For each selected word,
    it samples support and query sentences from those sentences that contain the word,
    then masks the target word with a mask token.
    """

    def __init__(
        self,
        sentences,
        vocabulary,
        num_classes,
        support_per_class,
        query_per_class,
        mask_token="[MASK]",
    ):
        """
        Args:
            sentences (List[str]): A list of unsupervised sentences.
            vocabulary (List[str]): A list of vocabulary words to choose from.
            num_classes (int): Number of unique words (classes) to select (K).
            support_per_class (int): Number of support examples per class (S).
            query_per_class (int): Number of query examples per class (Q).
            mask_token (str): The token to use to mask the target word.
        """
        self.sentences = sentences
        self.vocabulary = vocabulary
        self.num_classes = num_classes
        self.support_per_class = support_per_class
        self.query_per_class = query_per_class
        self.mask_token = mask_token

        # Precompute the minimum required sentence count per word.
        self.min_required = self.support_per_class + self.query_per_class

        # Pre-filter the vocabulary to only include words that appear enough times.
        self.valid_vocabulary = [
            word
            for word in self.vocabulary
            if self._count_occurrences(word) >= self.min_required
        ]
        if len(self.valid_vocabulary) < self.num_classes:
            print(
                f"Warning: Only {len(self.valid_vocabulary)} words have at least {self.min_required} matching sentences. "
                "Tasks will be generated using these words."
            )
            # If not enough words pass the filter, fall back to full vocabulary.
            self.valid_vocabulary = self.vocabulary

    def _count_occurrences(self, target_word):
        """Counts how many sentences contain the target word (as a full token)."""
        return sum(1 for s in self.sentences if self._contains_word(s, target_word))

    def generate_task(self):
        """
        Generate a single SMLMT task.

        Returns:
            support_set (List[Tuple[str, int]]): List of (masked_sentence, label) for the support set.
            query_set (List[Tuple[str, int]]): List of (masked_sentence, label) for the query set.
        """
        # Resample until we have the required number of words with sufficient support.
        selected_words = random.sample(self.valid_vocabulary, self.num_classes)

        support_set = []
        query_set = []

        for label, word in enumerate(selected_words):
            # Find all sentences that contain the target word.
            matching_sentences = [
                s for s in self.sentences if self._contains_word(s, word)
            ]
            if len(matching_sentences) < self.min_required:
                # If this happens (should be rare thanks to filtering), print a warning and skip this word.
                print(
                    f"Warning: Even after filtering, not enough sentences for word '{word}'. Skipping this word."
                )
                continue

            # Randomly sample S + Q sentences.
            sampled = random.sample(matching_sentences, self.min_required)
            support_sentences = sampled[: self.support_per_class]
            query_sentences = sampled[self.support_per_class :]

            # Mask the target word in all sampled sentences.
            support_masked = [self._mask_word(s, word) for s in support_sentences]
            query_masked = [self._mask_word(s, word) for s in query_sentences]

            # Extend the task sets with (masked sentence, label) pairs.
            support_set.extend([(sent, label) for sent in support_masked])
            query_set.extend([(sent, label) for sent in query_masked])

        return support_set, query_set

    def _mask_word(self, sentence, target_word):
        """
        Replace all full-word occurrences of target_word in sentence with the mask token.
        If the sentence is None, returns an empty string.
        """
        if sentence is None:
            return ""
        replacement = self.mask_token if self.mask_token is not None else "[MASK]"
        pattern = r"\b" + re.escape(target_word) + r"\b"
        return re.sub(pattern, replacement, sentence)

    def _contains_word(self, sentence, target_word):
        """
        Returns True if the target_word occurs as a full word in the sentence.
        If the sentence is None, returns False.
        """
        if sentence is None:
            return False
        pattern = r"\b" + re.escape(target_word) + r"\b"
        return re.search(pattern, sentence) is not None


class SMLMTDataset(Dataset):
    """
    PyTorch Dataset for SMLMT tasks.

    Each item from this dataset is a meta-learning episode containing a support set and query set.
    """

    def __init__(
        self,
        sentences,
        vocabulary,
        num_classes=5,
        support_per_class=5,
        query_per_class=5,
        mask_token="[MASK]",
        num_tasks=10000,
    ):
        """
        Args:
            sentences (List[str]): A list of unsupervised sentences.
            vocabulary (List[str]): A list of vocabulary words.
            num_classes (int): Number of classes (K).
            support_per_class (int): Number of support examples per class (S).
            query_per_class (int): Number of query examples per class (Q).
            mask_token (str): The token to use for masking.
            num_tasks (int): How many tasks (episodes) the dataset can generate.
        """
        self.sentences = sentences
        self.vocabulary = vocabulary
        self.num_classes = num_classes
        self.support_per_class = support_per_class
        self.query_per_class = query_per_class
        self.mask_token = mask_token
        self.num_tasks = num_tasks

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, idx):
        task_generator = SMLMTTask(
            self.sentences,
            self.vocabulary,
            self.num_classes,
            self.support_per_class,
            self.query_per_class,
            self.mask_token,
        )
        support_set, query_set = task_generator.generate_task()
        return {"support": support_set, "query": query_set}
