"""Data collection module for web scraping

This module provides web scraping functionality for collecting financial text data.

Supported sources:
- Financial news websites (via RSS feeds)
- Social media (Twitter/Reddit via APIs)
- Financial forums

Requirements:
    pip install beautifulsoup4 requests feedparser

For social media:
    pip install tweepy praw  # Twitter and Reddit APIs
"""

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

import time
import random
from typing import List, Dict
import csv
from pathlib import Path


class DataCollector:
    """Base class for data collection"""

    def __init__(self, output_dir='data/raw', delay_range=(1, 3)):
        """
        Initialize data collector

        Args:
            output_dir: Directory to save collected data
            delay_range: Tuple of (min, max) seconds to wait between requests
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay_range = delay_range
        self.collected_data = []

    def _random_delay(self):
        """Random delay to avoid rate limiting"""
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)

    def save_data(self, filename='collected_data.csv'):
        """Save collected data to CSV"""
        filepath = self.output_dir / filename

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if self.collected_data:
                writer = csv.DictWriter(f, fieldnames=self.collected_data[0].keys())
                writer.writeheader()
                writer.writerows(self.collected_data)

        print(f" Saved {len(self.collected_data)} samples to {filepath}")
        return filepath


class RSSFeedCollector(DataCollector):
    """Collect data from RSS feeds"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Sample RSS feeds for financial news
        self.feeds = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'https://www.ft.com/?format=rss',
        ]

    def collect_from_feed(self, feed_url, max_items=100):
        """
        Collect articles from RSS feed

        Args:
            feed_url: URL of RSS feed
            max_items: Maximum number of items to collect
        """
        try:
            import feedparser
        except ImportError:
            print("Warning: feedparser not installed. Install with: pip install feedparser")
            return self._generate_sample_data(max_items)

        print(f"Collecting from {feed_url}...")

        try:
            feed = feedparser.parse(feed_url)

            for i, entry in enumerate(feed.entries[:max_items]):
                # Extract text and metadata
                text = entry.get('title', '') + ' ' + entry.get('summary', '')
                text = text.strip()

                if text:
                    self.collected_data.append({
                        'text': text,
                        'source': feed_url,
                        'title': entry.get('title', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', '')
                    })

                self._random_delay()

            print(f" Collected {len(feed.entries[:max_items])} items")

        except Exception as e:
            print(f" Error collecting from {feed_url}: {e}")
            return self._generate_sample_data(max_items)

    def collect_all(self, items_per_feed=100):
        """Collect from all configured feeds"""
        for feed_url in self.feeds:
            self.collect_from_feed(feed_url, max_items=items_per_feed)

        return self.collected_data

    def _generate_sample_data(self, n=100):
        """Generate sample financial news data"""
        templates = [
            "{company} reports {metric} {direction} in Q{quarter}",
            "{company} shares {movement} after {event}",
            "Analysts {sentiment} on {company} following {event}",
            "{company} announces {event}, stock {movement}",
            "Market {movement} as {company} {event}",
        ]

        companies = ["Apple", "Tesla", "Amazon", "Microsoft", "Google", "Meta"]
        metrics = ["revenue", "profit", "earnings", "sales"]
        directions = ["increase", "decrease", "surge", "drop", "growth", "decline"]
        quarters = ["1", "2", "3", "4"]
        movements = ["surged", "dropped", "climbed", "fell", "rallied"]
        events = ["strong earnings", "merger announcement", "product launch",
                 "disappointing results", "expansion plans"]
        sentiments = ["bullish", "bearish", "optimistic", "cautious"]

        print(f" Generating {n} sample financial texts...")

        for i in range(n):
            template = random.choice(templates)
            text = template.format(
                company=random.choice(companies),
                metric=random.choice(metrics),
                direction=random.choice(directions),
                quarter=random.choice(quarters),
                movement=random.choice(movements),
                event=random.choice(events),
                sentiment=random.choice(sentiments)
            )

            self.collected_data.append({
                'text': text,
                'source': 'generated',
                'title': text,
                'link': '',
                'published': ''
            })

        return self.collected_data


class SocialMediaCollector(DataCollector):
    """Collect data from social media (Twitter/Reddit)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collect_from_twitter(self, query='#stocks', max_tweets=100):
        """
        Collect tweets (requires Twitter API credentials)

        For demo purposes, generates sample data
        """
        print("��  Twitter API not configured. Generating sample data...")
        return self._generate_sample_tweets(max_tweets)

    def collect_from_reddit(self, subreddit='wallstreetbets', max_posts=100):
        """
        Collect Reddit posts (requires Reddit API credentials)

        For demo purposes, generates sample data
        """
        print("��  Reddit API not configured. Generating sample data...")
        return self._generate_sample_reddit(max_posts)

    def _generate_sample_tweets(self, n=100):
        """Generate sample tweets"""
        templates = [
            "$TICKER to the moon!  {sentiment}",
            "Just bought more $TICKER, feeling {sentiment}",
            "$TICKER looking {sentiment} today",
            "My $TICKER position is {sentiment}",
            "{sentiment} on $TICKER after today's news"
        ]

        tickers = ["AAPL", "TSLA", "AMZN", "MSFT", "GOOGL"]
        sentiments = ["bullish", "bearish", "strong", "weak", "great", "terrible"]

        for i in range(n):
            template = random.choice(templates)
            text = template.format(
                TICKER=random.choice(tickers),
                sentiment=random.choice(sentiments)
            )

            self.collected_data.append({
                'text': text,
                'source': 'twitter',
                'platform': 'Twitter'
            })

        return self.collected_data

    def _generate_sample_reddit(self, n=100):
        """Generate sample Reddit posts"""
        templates = [
            "DD on $TICKER: {sentiment} outlook",
            "YOLO'd into $TICKER, {sentiment}",
            "$TICKER tendies incoming! {sentiment}",
            "Holding $TICKER with diamond hands ",
            "$TICKER discussion: {sentiment}"
        ]

        tickers = ["GME", "AMC", "TSLA", "AAPL", "NVDA"]
        sentiments = ["bullish", "bearish", "looking good", "not great", "very promising"]

        for i in range(n):
            template = random.choice(templates)
            text = template.format(
                TICKER=random.choice(tickers),
                sentiment=random.choice(sentiments)
            )

            self.collected_data.append({
                'text': text,
                'source': 'reddit',
                'platform': 'Reddit'
            })

        return self.collected_data


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Data Collection Demo")
    print("="*60)

    # RSS Feed Collection
    print("\n1. Collecting from RSS feeds...")
    rss_collector = RSSFeedCollector(output_dir='data/raw')
    rss_collector.collect_all(items_per_feed=50)
    rss_collector.save_data('rss_data.csv')

    # Social Media Collection
    print("\n2. Collecting from social media...")
    social_collector = SocialMediaCollector(output_dir='data/raw')
    social_collector.collect_from_twitter(max_tweets=100)
    social_collector.collect_from_reddit(max_posts=100)
    social_collector.save_data('social_media_data.csv')

    print("\n" + "="*60)
    print(" Data collection complete!")
    print(f"Total samples collected: {len(rss_collector.collected_data) + len(social_collector.collected_data)}")
    print("="*60)
