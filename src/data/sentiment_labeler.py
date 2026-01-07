"""Sentiment Labeling Module for Financial Text

This module provides rule-based sentiment labeling for financial news.
Uses financial domain-specific keywords and phrases.

Sentiment Classes:
- Positive (2): Bullish, growth, profit, gains
- Negative (0): Bearish, loss, decline, crash
- Neutral (1): Mixed, unchanged, stable

This is a common approach in financial NLP when manual labeling
is not feasible for large datasets.
"""

import re
from typing import List, Tuple, Dict
import json


class FinancialSentimentLabeler:
    """
    Rule-based sentiment labeling for financial text.

    Uses comprehensive keyword dictionaries specific to financial domain.
    """

    def __init__(self):
        # Positive sentiment keywords (Bullish)
        self.positive_keywords = {
            # Strong positive
            'strong': ['surge', 'surged', 'surging', 'soar', 'soared', 'soaring',
                      'skyrocket', 'rally', 'rallied', 'rallying', 'boom', 'booming',
                      'record high', 'all-time high', 'breakout', 'breakthrough'],

            # Growth indicators
            'growth': ['growth', 'grew', 'growing', 'gain', 'gained', 'gains',
                      'rise', 'risen', 'rising', 'increase', 'increased', 'increasing',
                      'up', 'higher', 'climbed', 'climbing', 'advance', 'advancing'],

            # Performance
            'performance': ['beat', 'beats', 'beating', 'exceeded', 'exceeds',
                           'outperform', 'outperformed', 'strong earnings',
                           'profit', 'profitable', 'profits', 'revenue growth'],

            # Positive outlook
            'outlook': ['bullish', 'optimistic', 'positive', 'upgrade', 'upgraded',
                       'buy rating', 'strong buy', 'outperform rating', 'overweight',
                       'opportunity', 'promising', 'confident', 'confidence'],

            # Business positive
            'business': ['expansion', 'expanding', 'dividend', 'buyback',
                        'acquisition', 'partnership', 'deal', 'contract',
                        'innovation', 'launch', 'success', 'successful']
        }

        # Negative sentiment keywords (Bearish)
        self.negative_keywords = {
            # Strong negative
            'strong': ['crash', 'crashed', 'crashing', 'plunge', 'plunged', 'plunging',
                      'collapse', 'collapsed', 'tumble', 'tumbled', 'plummet',
                      'freefall', 'tank', 'tanked', 'disaster'],

            # Decline indicators
            'decline': ['decline', 'declined', 'declining', 'drop', 'dropped', 'dropping',
                       'fall', 'fell', 'falling', 'decrease', 'decreased',
                       'down', 'lower', 'slump', 'slumped', 'slide', 'sliding'],

            # Performance
            'performance': ['miss', 'missed', 'misses', 'disappoint', 'disappointed',
                           'disappointing', 'underperform', 'weak', 'weakness',
                           'loss', 'losses', 'losing', 'deficit', 'shortfall'],

            # Negative outlook
            'outlook': ['bearish', 'pessimistic', 'negative', 'downgrade', 'downgraded',
                       'sell rating', 'underweight', 'underperform rating',
                       'concern', 'concerns', 'worried', 'fear', 'fears', 'warning'],

            # Business negative
            'business': ['layoff', 'layoffs', 'cut', 'cuts', 'cutting',
                        'bankruptcy', 'default', 'lawsuit', 'investigation',
                        'scandal', 'fraud', 'fine', 'fined', 'penalty',
                        'recall', 'regulatory', 'challenges', 'struggling']
        }

        # Neutral indicators
        self.neutral_keywords = [
            'unchanged', 'steady', 'stable', 'flat', 'mixed',
            'hold', 'holds', 'holding', 'maintain', 'maintained',
            'neutral', 'sideways', 'consolidate', 'consolidating',
            'wait', 'await', 'awaiting', 'expect', 'expected',
            'announce', 'announced', 'report', 'reported', 'says',
            'according', 'analysis', 'review', 'update'
        ]

        # Compile patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for keyword matching"""
        # Flatten positive keywords
        pos_words = []
        for category in self.positive_keywords.values():
            pos_words.extend(category)
        self.pos_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(w) for w in pos_words) + r')\b',
            re.IGNORECASE
        )

        # Flatten negative keywords
        neg_words = []
        for category in self.negative_keywords.values():
            neg_words.extend(category)
        self.neg_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(w) for w in neg_words) + r')\b',
            re.IGNORECASE
        )

        # Neutral pattern
        self.neu_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(w) for w in self.neutral_keywords) + r')\b',
            re.IGNORECASE
        )

    def _count_matches(self, text: str) -> Tuple[int, int, int]:
        """Count keyword matches for each sentiment"""
        pos_matches = len(self.pos_pattern.findall(text))
        neg_matches = len(self.neg_pattern.findall(text))
        neu_matches = len(self.neu_pattern.findall(text))
        return pos_matches, neg_matches, neu_matches

    def label(self, text: str) -> Tuple[int, str, float]:
        """
        Label single text with sentiment

        Args:
            text: Financial text to label

        Returns:
            Tuple of (label_id, label_name, confidence)
            - label_id: 0=Negative, 1=Neutral, 2=Positive
            - label_name: String name
            - confidence: Score 0-1
        """
        pos, neg, neu = self._count_matches(text)
        total = pos + neg + neu

        # Calculate scores
        if total == 0:
            # No keywords found - default to neutral
            return 1, 'Neutral', 0.5

        pos_score = pos / total
        neg_score = neg / total
        neu_score = neu / total

        # Determine sentiment based on dominant score
        if pos_score > neg_score and pos_score > 0.3:
            confidence = pos_score
            return 2, 'Positive', confidence
        elif neg_score > pos_score and neg_score > 0.3:
            confidence = neg_score
            return 0, 'Negative', confidence
        else:
            confidence = max(neu_score, 0.5)
            return 1, 'Neutral', confidence

    def label_batch(self, texts: List[str]) -> List[Dict]:
        """
        Label batch of texts

        Args:
            texts: List of texts to label

        Returns:
            List of dicts with text, label, label_name, confidence
        """
        results = []
        for text in texts:
            label_id, label_name, confidence = self.label(text)
            results.append({
                'text': text,
                'label': label_id,
                'sentiment': label_name,
                'confidence': round(confidence, 3)
            })
        return results


def label_scraped_data(input_file: str, output_file: str) -> Dict:
    """
    Label scraped financial news data

    Args:
        input_file: Path to scraped JSON data
        output_file: Path to save labeled data

    Returns:
        Statistics dict
    """
    # Load scraped data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from {input_file}")

    # Label
    labeler = FinancialSentimentLabeler()

    labeled_data = []
    label_counts = {0: 0, 1: 0, 2: 0}

    for item in data:
        text = item['text']
        label_id, label_name, confidence = labeler.label(text)

        labeled_data.append({
            'text': text,
            'sentiment': label_name,
            'label': label_id,
            'confidence': confidence,
            'source': item.get('source', ''),
            'url': item.get('url', ''),
        })
        label_counts[label_id] += 1

    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(labeled_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(labeled_data)} labeled items to {output_file}")

    # Statistics
    label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    stats = {
        'total': len(labeled_data),
        'distribution': {label_names[k]: v for k, v in label_counts.items()}
    }

    print("\nLabel distribution:")
    for label_id, count in label_counts.items():
        pct = count / len(labeled_data) * 100
        print(f"  {label_names[label_id]}: {count} ({pct:.1f}%)")

    return stats


if __name__ == "__main__":
    # Test labeling
    labeler = FinancialSentimentLabeler()

    test_texts = [
        "Stock prices surged after strong earnings report",
        "Company faces bankruptcy amid declining sales",
        "Market remained steady following quarterly results",
        "Apple beats estimates with record revenue growth",
        "Tesla stock plunged on disappointing delivery numbers",
        "Analysts maintain neutral outlook on the sector",
    ]

    print("Testing sentiment labeler:\n")
    for text in test_texts:
        label_id, label_name, conf = labeler.label(text)
        print(f"[{label_name:8}] ({conf:.2f}) {text}")

    print("\n" + "="*60)

    # Label scraped data
    label_scraped_data(
        'data/raw/real_scraped_data.json',
        'data/raw/labeled_scraped_data.json'
    )
