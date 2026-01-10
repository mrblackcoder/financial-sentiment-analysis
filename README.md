# Financial Sentiment Analysis

Finansal haberlerin duygu analizini makine ogrenmesi ve derin ogrenme yontemleriyle gerceklestiren bir proje.

## Proje Ozeti

- **Dataset:** 3,761 finansal haber (451 gercek RSS + template + augmentation)
- **Test Seti:** 753 ornek (%20)
- **En Iyi Model:** Linear SVM - %96.18 F1-Score

## Kurulum

```bash
# Sanal ortam olustur
python3 -m venv venv
source venv/bin/activate

# Kutuphaneleri yukle
pip install -r requirements.txt
```

## Kullanim

```bash
# Veri setini olustur
python3 create_full_dataset.py

# Modelleri egit
python3 train_and_evaluate.py

# Demo notebook'u calistir
jupyter notebook demo_notebook.ipynb
```

## Proje Yapisi

```
financial-sentiment-analysis/
├── README.md
├── requirements.txt
├── create_full_dataset.py      # Veri seti olusturma
├── train_and_evaluate.py       # Model egitimi
├── demo_notebook.ipynb         # Sunum notebook'u
│
├── src/
│   ├── data/
│   │   ├── real_scraper.py     # RSS scraper (451 haber)
│   │   ├── sentiment_labeler.py
│   │   └── augmentation.py
│   ├── features/
│   │   ├── tfidf_features.py
│   │   └── custom_features.py
│   └── models/
│       └── deep_learning/
│           └── mlp_model.py
│
├── data/
│   ├── raw/                    # Ham veri
│   └── processed/              # Islenmis veri
│
├── models/                     # Egitilmis modeller (.pkl)
└── figures/                    # Gorseller
```

## Model Performansi

| Model | Test F1 | Test Acc | CV F1 | Training Time |
|-------|---------|----------|-------|---------------|
| Linear SVM | 96.18% | 96.15% | 95.99% | 0.06s |
| MLP | 96.06% | 95.48% | 95.65% | 4.21s |
| Logistic Regression | 93.84% | 93.76% | 93.27% | 1.59s |
| Random Forest | 91.15% | 90.97% | 91.46% | 0.11s |

## Feature Engineering

| Ozellik Tipi | Boyut |
|--------------|-------|
| TF-IDF | 1,000 |
| Custom Features | 14 |
| Toplam | 1,014 |

## Gereksinimler

| Gereksinim | Deger |
|------------|-------|
| Dataset Boyutu (2000+) | 3,761 |
| Test Boyutu (500+) | 753 |
| Web Scraping | 451 RSS |
| Traditional ML (2+) | 3 model |
| Deep Learning (1+) | MLP |
| 5-Fold CV | Uygulandı |

## Ekip

- Mehmet Taha Boynikoglu (2121251034)
- Merve Kedersiz (2221251045)
- Elif Hande Arslan (2121251021)

## Ders Bilgileri

- **Ders:** SEN22325E - Learning from Data
- **Hoca:** Cumali Turkmenoglu
- **Universite:** Fatih Sultan Mehmet Vakif Universitesi
