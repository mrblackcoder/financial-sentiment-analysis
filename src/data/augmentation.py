"""Data augmentation for text classification"""
import random
import numpy as np


# Ensure reproducibility
def set_seed(seed=42):
    random.seed(seed)


class TextAugmenter:
    """Text data augmentation using various techniques"""

    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Synonym mappings for financial terms
        self.synonyms = {
            'increase': ['rise', 'growth', 'gain', 'surge'],
            'decrease': ['decline', 'drop', 'fall', 'loss'],
            'company': ['firm', 'corporation', 'business', 'enterprise'],
            'profit': ['earnings', 'revenue', 'income', 'gains'],
            'stock': ['share', 'equity', 'security'],
            'market': ['trading', 'exchange', 'marketplace'],
            'investor': ['trader', 'shareholder', 'stakeholder'],
            'positive': ['bullish', 'optimistic', 'favorable', 'strong'],
            'negative': ['bearish', 'pessimistic', 'weak', 'poor'],
            'good': ['excellent', 'strong', 'solid', 'favorable'],
            'bad': ['poor', 'weak', 'disappointing', 'unfavorable']
        }

    def synonym_replacement(self, text, n=2):
        """Replace n words with synonyms"""
        words = text.split()
        new_words = words.copy()

        # Find words that have synonyms
        replaceable = [i for i, word in enumerate(words)
                      if word.lower() in self.synonyms]

        if not replaceable:
            return text

        # Randomly select n words to replace
        n_replacements = min(n, len(replaceable))
        indices = random.sample(replaceable, n_replacements)

        for idx in indices:
            word = words[idx].lower()
            synonyms = self.synonyms[word]
            new_word = random.choice(synonyms)

            # Preserve capitalization
            if words[idx][0].isupper():
                new_word = new_word.capitalize()

            new_words[idx] = new_word

        return ' '.join(new_words)

    def random_swap(self, text, n=2):
        """Randomly swap n pairs of words"""
        words = text.split()

        if len(words) < 2:
            return text

        new_words = words.copy()

        for _ in range(n):
            if len(new_words) < 2:
                break

            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

        return ' '.join(new_words)

    def random_deletion(self, text, p=0.1):
        """Randomly delete words with probability p"""
        words = text.split()

        if len(words) == 1:
            return text

        new_words = [word for word in words if random.random() > p]

        # If all words deleted, return original
        if len(new_words) == 0:
            return text

        return ' '.join(new_words)

    def random_insertion(self, text, n=1):
        """Randomly insert synonyms n times"""
        words = text.split()
        new_words = words.copy()

        for _ in range(n):
            # Find words with synonyms
            candidates = [i for i, word in enumerate(new_words)
                         if word.lower() in self.synonyms]

            if not candidates:
                continue

            idx = random.choice(candidates)
            word = new_words[idx].lower()
            synonym = random.choice(self.synonyms[word])

            # Insert synonym nearby
            insert_pos = min(idx + random.randint(0, 2), len(new_words))
            new_words.insert(insert_pos, synonym)

        return ' '.join(new_words)

    def augment(self, text, num_augmented=3, techniques='all'):
        """
        Generate augmented versions of text

        Args:
            text: Input text
            num_augmented: Number of augmented samples to generate
            techniques: Which techniques to use ('all' or list)

        Returns:
            List of augmented texts
        """
        augmented_texts = []

        available_techniques = [
            self.synonym_replacement,
            self.random_swap,
            self.random_deletion,
            self.random_insertion
        ]

        for _ in range(num_augmented):
            # Randomly select a technique
            technique = random.choice(available_techniques)
            augmented = technique(text)

            # Avoid duplicates
            if augmented != text and augmented not in augmented_texts:
                augmented_texts.append(augmented)

        return augmented_texts


def augment_dataset(texts, labels, num_augmented_per_sample=2, random_seed=42):
    """
    Augment entire dataset

    Args:
        texts: List of text samples
        labels: List of labels
        num_augmented_per_sample: Number of augmented samples per original
        random_seed: Random seed

    Returns:
        augmented_texts, augmented_labels
    """
    set_seed(random_seed)  # Set seed for reproducibility

    augmenter = TextAugmenter(random_seed=random_seed)

    augmented_texts = []
    augmented_labels = []

    for text, label in zip(texts, labels):
        # Keep original
        augmented_texts.append(text)
        augmented_labels.append(label)

        # Generate augmented versions
        aug_samples = augmenter.augment(text, num_augmented=num_augmented_per_sample)

        for aug_text in aug_samples:
            augmented_texts.append(aug_text)
            augmented_labels.append(label)

    return augmented_texts, augmented_labels


# Example usage
if __name__ == "__main__":
    # Test augmentation
    text = "The company reported strong profit increase in the market"
    augmenter = TextAugmenter()

    print("Original:", text)
    print("\nAugmented versions:")

    for i, aug_text in enumerate(augmenter.augment(text, num_augmented=5), 1):
        print(f"{i}. {aug_text}")
