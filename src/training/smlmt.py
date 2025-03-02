import re
import random
from torch.utils.data import Dataset


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

    def generate_task(self):
        """
        Generate a single SMLMT task.

        Returns:
            support_set (List[Tuple[str, int]]): List of (masked_sentence, label) for the support set.
            query_set (List[Tuple[str, int]]): List of (masked_sentence, label) for the query set.
        """
        # (1) Randomly select num_classes unique words from the vocabulary.
        selected_words = random.sample(self.vocabulary, self.num_classes)

        support_set = []
        query_set = []

        # For each selected word, sample sentences and mask the word.
        for label, word in enumerate(selected_words):
            # (2) Find all sentences that contain the target word.
            matching_sentences = [
                s for s in self.sentences if self._contains_word(s, word)
            ]

            # In case there are not enough sentences, you might fall back to all sentences.
            if len(matching_sentences) < self.support_per_class + self.query_per_class:
                matching_sentences = [s for s in self.sentences if s is not None]

            # Randomly sample S + Q sentences
            sampled = random.sample(
                matching_sentences, self.support_per_class + self.query_per_class
            )
            support_sentences = sampled[: self.support_per_class]
            query_sentences = sampled[self.support_per_class :]

            # (3) Replace the target word with the mask token.
            support_masked = [self._mask_word(s, word) for s in support_sentences]
            query_masked = [self._mask_word(s, word) for s in query_sentences]

            # (4) Assign the same label to all examples for this word.
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
        # Use a default mask token if self.mask_token is None.
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
