# Financial Sentiment Analysis

ML & Deep Learning ile finansal haber duygu analizi projesi.

## 🚀 Hızlı Başlangıç

### Kurulum
```bash
# Projeyi indir
git clone https://github.com/KULLANICI_ADI/financial-sentiment-analysis.git
cd financial-sentiment-analysis

# Sanal ortam oluştur
python3 -m venv venv
source venv/bin/activate

# Kütüphaneleri yükle
pip install -r requirements.txt

# Projeyi oluştur (2-3 dakika)
python3 reset_and_rebuild.py --yes

# Görselleri aç
open figures/
```

## 📊 Proje Özeti

- **Dataset:** 3,761 finansal haber (451 gerçek RSS + template + augmentation)
- **Test:** 753 sample (%20)
- **En İyi Model:** Linear SVM - %96.18 F1-Score
- **Modeller:** Logistic Regression, Linear SVM, Random Forest, MLP

## 📁 Proje Yapısı
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
