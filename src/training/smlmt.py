import re
import random
from torch.utils.data import Dataset
import torch.nn as nn
from collections import defaultdict


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
    it samples support and query sentences (with the target word masked) from the sentences
    that contain that word.
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

        # Minimum required sentences for each word.
        self.min_required = self.support_per_class + self.query_per_class

        # Build an index mapping each vocabulary word to the list of sentences that contain it.
        # We assume that the sentences are already tokenized or use whitespace to separate tokens.
        # This is much faster than scanning all sentences repeatedly.
        self.word_to_sentences = defaultdict(list)
        vocab_set = set(self.vocabulary)
        for s in self.sentences:
            if s is None:
                continue
            # Use split() for speed. Adjust if you need a different tokenization.
            tokens = set(s.split())
            # For each token in the sentence that is in our vocabulary, record the sentence.
            for token in tokens:
                if token in vocab_set:
                    self.word_to_sentences[token].append(s)

        # Filter the vocabulary to those words that appear in at least min_required sentences.
        self.valid_vocabulary = [
            word
            for word in self.vocabulary
            if len(self.word_to_sentences[word]) >= self.min_required
        ]
        if len(self.valid_vocabulary) < self.num_classes:
            print(
                f"Warning: Only {len(self.valid_vocabulary)} words have at least {self.min_required} occurrences. "
                "Falling back to full vocabulary."
            )
            self.valid_vocabulary = self.vocabulary

    def generate_task(self):
        """
        Generate a single SMLMT task.

        Returns:
            support_set (List[Tuple[str, int]]): List of (masked_sentence, label) for the support set.
            query_set (List[Tuple[str, int]]): List of (masked_sentence, label) for the query set.
        """
        # Sample target words from the filtered vocabulary.
        selected_words = random.sample(self.valid_vocabulary, self.num_classes)
        support_set = []
        query_set = []

        for label, word in enumerate(selected_words):
            matching_sentences = self.word_to_sentences[word]
            if len(matching_sentences) < self.min_required:
                # This should rarely happen, but if it does, warn and skip this word.
                print(
                    f"Warning: Even after filtering, not enough sentences for word '{word}'. Skipping this word."
                )
                continue

            # Randomly sample exactly the number of required sentences.
            sampled = random.sample(matching_sentences, self.min_required)
            support_sentences = sampled[: self.support_per_class]
            query_sentences = sampled[self.support_per_class :]

            # Mask the target word in each sampled sentence.
            support_masked = [self._mask_word(s, word) for s in support_sentences]
            query_masked = [self._mask_word(s, word) for s in query_sentences]

            support_set.extend([(sent, label) for sent in support_masked])
            query_set.extend([(sent, label) for sent in query_masked])

        return support_set, query_set

    def _mask_word(self, sentence, target_word):
        """
        Replace all full-word occurrences of target_word in sentence with the mask token.
        """
        if sentence is None:
            return ""
        replacement = self.mask_token if self.mask_token is not None else "[MASK]"
        pattern = r"\b" + re.escape(target_word) + r"\b"
        return re.sub(pattern, replacement, sentence)

    def _contains_word(self, sentence, target_word):
        """
        Returns True if the target_word occurs as a full word in the sentence.
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
