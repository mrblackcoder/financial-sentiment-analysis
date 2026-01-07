# Financial Sentiment Analysis - ML & Deep Learning Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Financial news sentiment classification using Machine Learning and Deep Learning approaches. Achieves **96.18% F1-Score** on 753 test samples.

## 📊 Project Overview

This project implements **automated sentiment analysis** for financial news articles, classifying them into:
- **Positive** (bullish sentiment)
- **Negative** (bearish sentiment)  
- **Neutral** (no clear direction)

### Key Features
- ✅ **3,761 samples** (451 real RSS news + templates + augmentation)
- ✅ **753 test samples** (exceeds 500 requirement)
- ✅ **4 models**: 3 Traditional ML + 1 Deep Learning
- ✅ **4 feature methods**: TF-IDF, BoW, Word2Vec, Custom
- ✅ **5-fold cross-validation** with regularization
- ✅ **Modular architecture** for reproducibility

## 🎯 Results

| Model | CV F1-Score | Test F1-Score | Training Time |
|-------|-------------|---------------|---------------|
| **Linear SVM** | 0.96 ± 0.002 | **96.18%** | 0.32s |
| MLP (Deep Learning) | 0.96 ± 0.007 | 95.54% | 3.44s |
| Logistic Regression | 0.93 ± 0.008 | 93.84% | 1.60s |
| Random Forest | 0.91 ± 0.012 | 91.15% | 0.10s |

**Best Model**: Linear SVM with TF-IDF features - only 28 errors out of 753 test samples!

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/financial-sentiment-analysis.git
cd financial-sentiment-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download required NLTK data** (if not already installed)
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### Usage

**Option 1: Run the complete pipeline** (Recommended for first-time users)
```bash
# This will create dataset, train models, and generate visualizations
python create_full_dataset.py
python train_and_evaluate.py
```

**Option 2: Reset and rebuild everything**
```bash
# Clean all previous data and start fresh
python reset_and_rebuild.py --yes
```

**Option 3: Interactive Jupyter Notebook**
```bash
jupyter notebook demo_notebook.ipynb
```

## 📁 Project Structure
```
financial-sentiment-analysis/
├── README.md
├── requirements.txt
├── create_full_dataset.py         # Dataset generation pipeline
├── train_and_evaluate.py          # Model training pipeline
├── reset_and_rebuild.py           # Project reset/rebuild tool
├── SUNUM_REHBERI.md               # Presentation guide
│
├── src/
│   ├── data/
│   │   ├── real_scraper.py        # RSS feed scraper (451 articles)
│   │   ├── sentiment_labeler.py   # Rule-based labeling
│   │   ├── augmentation.py        # Data augmentation
│   │   ├── collector.py           # Helper functions
│   │   └── feature_loader.py      # Feature utilities
│   ├── features/
│   │   ├── tfidf_features.py      # TF-IDF extraction
│   │   ├── bow_features.py        # Bag-of-Words
│   │   ├── word2vec_features.py   # Word2Vec embeddings
│   │   └── custom_features.py     # Domain-specific features
│   ├── models/
│   │   └── deep_learning/
│   │       └── mlp_model.py       # MLP classifier
│   └── evaluation/
│       ├── metrics.py             # Evaluation metrics
│       └── visualizations.py      # Plotting functions
│
├── data/
│   ├── raw/
│   │   └── real_scraped_data.csv  # 451 real RSS articles
│   └── processed/
│       ├── train_clean.csv        # 2,632 samples
│       ├── val_clean.csv          # 376 samples
│       └── test_clean.csv         # 753 samples
│
├── models/                        # Trained models (.pkl)
│
└── figures/                       # Visualizations
    ├── learning_curves.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    └── model_comparison.png
```

---

## Model Performance

| Model | Test F1 | Test Acc | CV F1 | MCC | Time |
|-------|---------|----------|-------|-----|------|
| **Linear SVM** | **96.18%** | **96.15%** | 95.99% | 0.9427 | 0.35s |
| MLP (Deep Learning) | 95.54% | 95.48% | 95.82% | 0.9330 | 29.71s |
| Logistic Regression | 93.84% | 93.76% | 93.27% | 0.9083 | 1.60s |
| Random Forest | 91.15% | 90.97% | 91.46% | 0.8698 | 0.10s |

---

## Feature Engineering

| Feature Type | Dimensions | Status |
|--------------|------------|--------|
| TF-IDF | 1,000 | Primary |
| Bag-of-Words | 500 | Implemented |
| Word2Vec | 100 | Implemented |
| Custom Features | 14 | Domain-specific |
| **Combined** | **1,014** | Used for training |

---

## Requirements Checklist

| Requirement | Status | Value |
|-------------|--------|-------|
| Dataset Size (2000+) | DONE | 3,761 |
| Training Size (1500+) | DONE | 2,632 |
| Test Size (500+) | DONE | 753 |
| Real Web Scraping | DONE | 451 RSS |
| Traditional ML (2+) | DONE | 3 models |
| Deep Learning (1+) | DONE | MLP |
| 5-Fold CV | DONE | Implemented |
| Regularization | DONE | L2, Early Stopping |
| Learning Curves | DONE | Generated |
| Confusion Matrix | DONE | Generated |
| ROC Curves | DONE | Generated |

---

## Contact

**Course:** SEN22325E - Learning from Data
**Instructor:** Cumali Turkmenoglu
**Institution:** Fatih Sultan Mehmet Vakif University
