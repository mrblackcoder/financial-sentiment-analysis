# SUNUM SLAYT ICERIGI (9 Slide) - GUNCELLENDI

---

## SLIDE 1: KAPAK

**Financial Sentiment Analysis**
**with Machine Learning & Deep Learning**

**Team:**
- Mehmet Taha Boynikoglu (2121251034)
- Merve Kedersiz (2221251045)
- Elif Hande Arslan (2121251021)

**Course:** SEN22325E - Learning from Data
**Instructor:** Cumali Turkmenoglu
**Date:** December 2025

---

## SLIDE 2: PROBLEM & OBJECTIVE

### The Challenge
- Financial markets generate massive text data (Twitter, RSS, Reports)
- Manual analysis is impossible at scale
- Speed is critical for algorithmic trading

### Our Goal
Build a production-ready classifier to detect sentiment:
- **Positive** (bullish, growth)
- **Negative** (bearish, loss)
- **Neutral** (stable, unchanged)

### Key Result
**96.81% F1-Score** with MLP (Deep Learning)

---

## SLIDE 3: DATA COLLECTION

### Real Data Sources
- **451 Real RSS Articles** scraped from:
  - Yahoo Finance
  - CNBC
  - MarketWatch
  - Investing.com

### Data Augmentation Techniques
1. Synonym Replacement: "profit" -> "earnings"
2. Random Swap
3. Random Deletion
4. Random Insertion

### Final Dataset
| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 2,632 | 70% |
| Validation | 376 | 10% |
| Test | 753 | 20% |
| **Total** | **3,761** | 100% |

---

## SLIDE 4: FEATURE ENGINEERING (TF-IDF)

### Why TF-IDF?
- Financial text is keyword-based
- Words like "profit", "loss", "surge" directly indicate sentiment
- Sparse matrix = Fast training

### Configuration
```python
TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 3),
    stop_words='english'
)
```

### Result
- **1,000 features** extracted
- Captures unigrams, bigrams, trigrams
- Example: "strong buy" as single feature

---

## SLIDE 5: MODELS & TRAINING

### Models Implemented
| Category | Model |
|----------|-------|
| Traditional ML | Logistic Regression |
| Traditional ML | Linear SVM |
| Traditional ML | Random Forest |
| Deep Learning | MLP (Neural Network) |

### Training Strategy
- 5-Fold Cross Validation
- L2 Regularization (LogReg, SVM, MLP)
- Early Stopping (MLP)
- Train/Val/Test Split: 70/10/20

---

## SLIDE 6: RESULTS

### Model Comparison

| Model | F1-Score | Accuracy | Training Time |
|-------|----------|----------|---------------|
| **MLP (Deep Learning)** | **96.81%** | **96.81%** | 1.45s |
| Linear SVM | 96.55% | 96.55% | 0.01s |
| Logistic Regression | 94.73% | 94.69% | 0.65s |
| Random Forest | 89.87% | 89.64% | 0.08s |

### Key Findings
1. Deep Learning (MLP) achieved best performance
2. All models exceeded 90% F1-Score
3. Proper labeling improved results significantly

---

## SLIDE 7: CONFUSION MATRIX & ERROR ANALYSIS

### Test Set: 753 samples, ~22 errors (~2.9%)

### Per-Class Performance
| Class | F1-Score | Notes |
|-------|----------|-------|
| Positive | ~97% | Clear keywords |
| Negative | ~97% | Distinct vocabulary |
| Neutral | ~96% | Improved with proper labeling |

### Error Patterns
1. **Mixed Sentiment** (45%): Complex sentences with both pos/neg
2. **Domain Jargon** (35%): Financial slang
3. **Ambiguous Context** (20%): Context-dependent words

### Key Insight
Proper rule-based labeling significantly reduced errors

---

## SLIDE 8: REGULARIZATION & OVERFITTING

### Techniques Used

| Model | Technique | Parameter |
|-------|-----------|-----------|
| Logistic Regression | L2 (Ridge) | C=1.0 |
| Linear SVM | L2 | C=1.0 |
| MLP | L2 + Early Stopping | alpha=0.0001 |
| Random Forest | Bagging | n_estimators=100 |

### CV vs Test Gap (Overfitting Check)
| Model | CV Score | Test Score | Gap |
|-------|----------|------------|-----|
| MLP | 95.33% | 96.81% | -1.48% (test better!) |
| Linear SVM | 95.23% | 96.55% | -1.32% |

**Result:** Test > CV = Good generalization!

---

## SLIDE 9: CONCLUSION

### Achievements
- **3,761 samples** (requirement: 2,000+)
- **753 test samples** (requirement: 500+)
- **451 real RSS articles** (real web scraping!)
- **4 models** (3 Traditional ML + 1 Deep Learning)
- **96.81% F1-Score** (MLP Deep Learning)

### Key Takeaway
> Real data + Proper labeling + Deep Learning = Best results
> MLP outperformed Traditional ML on this dataset

### Future Work
1. FinBERT for context-aware classification
2. 10,000+ samples via Twitter/Reddit API
3. Real-time sentiment dashboard

---

## THANK YOU!

### Questions?

**Live Demo Available:** `demo_notebook.ipynb`

---

# NOTLAR

## Her Slide Icin Timing
- Slide 1: 20 saniye (Mehmet Taha)
- Slide 2: 45 saniye (Mehmet Taha)
- Slide 3: 1.5 dakika (Mehmet Taha)
- Slide 4: 1.5 dakika (Merve)
- Slide 5: 1 dakika (Merve)
- Slide 6: 1 dakika (Merve)
- Slide 7: 1.5 dakika (Elif)
- Slide 8: 1 dakika (Elif)
- Slide 9: 30 saniye (Elif/Tum Ekip)

**Toplam: ~10 dakika**

## Demo Icin
- Cell 4: Test data istatistikleri
- Cell 16: Model karsilastirma tablosu
- Cell 18: Confusion matrix (gorsel)
- Cell 28: LIVE PREDICTION
