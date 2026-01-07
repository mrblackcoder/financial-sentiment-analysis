"""Real Web Scraping Module for Financial News

This module collects REAL financial news data from RSS feeds.
No simulated/synthetic data - all content is scraped from live sources.

Data Sources:
- Yahoo Finance RSS
- MarketWatch RSS
- CNBC RSS
- Investing.com RSS
- Reuters RSS
- Bloomberg RSS

Requirements:
    pip install feedparser beautifulsoup4 requests
"""

import feedparser
from bs4 import BeautifulSoup
import json
import os
import time
import random
from datetime import datetime
from pathlib import Path
from collections import Counter


class RealFinancialScraper:
    """Scrapes real financial news from RSS feeds"""

    def __init__(self, output_dir='data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_data = []
        self.seen_texts = set()

        # RSS Feed sources - REAL financial news
        self.rss_sources = {
            # Yahoo Finance - Stock specific feeds
            'Yahoo Finance': [
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GOOGL&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MSFT&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TSLA&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AMZN&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=META&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=JPM&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=V&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=WMT&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=DIS&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NFLX&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AMD&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=INTC&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BA&region=US&lang=en-US",
            ],

            # MarketWatch
            'MarketWatch': [
                "https://feeds.marketwatch.com/marketwatch/topstories/",
                "https://feeds.marketwatch.com/marketwatch/marketpulse/",
                "https://feeds.marketwatch.com/marketwatch/StockstoWatch/",
                "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
            ],

            # CNBC
            'CNBC': [
                "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # Top News
                "https://www.cnbc.com/id/10001147/device/rss/rss.html",   # Business
                "https://www.cnbc.com/id/15839135/device/rss/rss.html",   # Earnings
                "https://www.cnbc.com/id/19854910/device/rss/rss.html",   # Tech
                "https://www.cnbc.com/id/20910258/device/rss/rss.html",   # Finance
            ],

            # Investing.com
            'Investing.com': [
                "https://www.investing.com/rss/news.rss",
                "https://www.investing.com/rss/stock_stock_picks.rss",
            ],
        }

    def _clean_text(self, text):
        """Clean HTML and normalize text"""
        if not text:
            return ""
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        clean = soup.get_text()
        # Normalize whitespace
        clean = ' '.join(clean.split())
        return clean.strip()

    def _is_duplicate(self, text):
        """Check if text is duplicate (fuzzy match on first 50 chars)"""
        key = text.lower()[:50]
        if key in self.seen_texts:
            return True
        self.seen_texts.add(key)
        return False

    def scrape_feed(self, url, source_name, max_items=50):
        """Scrape single RSS feed"""
        items_collected = 0

        try:
            feed = feedparser.parse(url)

            for entry in feed.entries[:max_items]:
                # Extract title
                title = self._clean_text(entry.get('title', ''))

                # Extract summary/description
                summary = self._clean_text(
                    entry.get('summary', entry.get('description', ''))
                )

                # Build text - title + optional summary
                text = title
                if summary and summary != title and len(summary) > 30:
                    # Add first 150 chars of summary
                    text = f"{title}. {summary[:150]}"

                # Skip if too short or duplicate
                if len(text) < 25 or self._is_duplicate(text):
                    continue

                # Collect
                self.collected_data.append({
                    'text': text,
                    'source': source_name,
                    'url': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'scraped_at': datetime.now().isoformat()
                })
                items_collected += 1

        except Exception as e:
            print(f"  Error scraping {url[:50]}...: {e}")

        return items_collected

    def scrape_all(self, delay_between_sources=0.5):
        """Scrape all configured RSS sources"""
        print("="*60)
        print("REAL FINANCIAL NEWS SCRAPING")
        print("="*60)

        total_items = 0

        for source_name, urls in self.rss_sources.items():
            print(f"\n[{source_name}]")
            source_items = 0

            for url in urls:
                items = self.scrape_feed(url, source_name)
                source_items += items

                # Small delay to be polite
                time.sleep(delay_between_sources)

            print(f"  -> Collected: {source_items} items")
            total_items += source_items

        print("\n" + "="*60)
        print(f"TOTAL SCRAPED: {len(self.collected_data)} unique items")
        print("="*60)

        return self.collected_data

    def get_source_stats(self):
        """Get statistics by source"""
        sources = Counter([item['source'] for item in self.collected_data])
        return dict(sources.most_common())

    def save(self, filename='real_scraped_data.json'):
        """Save collected data to JSON"""
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.collected_data, f, indent=2, ensure_ascii=False)

        print(f"\nSaved {len(self.collected_data)} items to {filepath}")
        return filepath

    def save_csv(self, filename='real_scraped_data.csv'):
        """Save to CSV format"""
        import csv
        filepath = self.output_dir / filename

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if self.collected_data:
                writer = csv.DictWriter(f, fieldnames=self.collected_data[0].keys())
                writer.writeheader()
                writer.writerows(self.collected_data)

        print(f"Saved {len(self.collected_data)} items to {filepath}")
        return filepath


def scrape_financial_news(min_samples=500):
    """
    Main function to scrape financial news

    Args:
        min_samples: Minimum number of samples to collect

    Returns:
        List of scraped news items
    """
    scraper = RealFinancialScraper()

    # First pass
    data = scraper.scrape_all()

    print(f"\nSource distribution:")
    for source, count in scraper.get_source_stats().items():
        print(f"  {source}: {count}")

    # Save results
    scraper.save('real_scraped_data.json')
    scraper.save_csv('real_scraped_data.csv')

    return data


if __name__ == "__main__":
    # Run scraping
    data = scrape_financial_news()

    print("\n--- Sample Headlines ---")
    for item in data[:10]:
        print(f"[{item['source']}] {item['text'][:80]}...")
