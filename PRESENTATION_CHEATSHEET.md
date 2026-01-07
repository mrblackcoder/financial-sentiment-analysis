# 📋 PRESENTATION DAY CHEAT SHEET

## ✅ PRE-PRESENTATION CHECKLIST (5 min before)

```bash
# 1. Navigate to project
cd /Users/metaboy/Desktop/2121251034_MEHMET_TAHA_BOYNIKOGLU

# 2. Verify all files exist
ls data/processed/  # Should see train/val/test_clean.csv
ls models/          # Should see 4 .pkl files
ls figures/         # Should see confusion_matrices.png, roc_curves.png, etc.

# 3. Run verification (MUST PASS!)
python3 verify_presentation_alignment.py
# Expected: "✅ ALL CHECKS PASSED"

# 4. Test imports
python3 -c "import numpy, pandas, sklearn, matplotlib; print('✅ All libraries OK')"

# 5. Open demo notebook
jupyter notebook demo_notebook.ipynb
```

---

## 🎯 KEY NUMBERS TO MEMORIZE

| Metric | Value | Where Used |
|--------|-------|------------|
| **Best F1-Score** | **96.18%** | Throughout presentation |
| **Best Model** | **Linear SVM** | Main result |
| **Total Samples** | 3,761 | Dataset stats |
| **Real RSS News** | 451 | Data collection |
| **Test Samples** | 753 | Requirements check |
| **Training Time (SVM)** | 0.32s | Speed comparison |
| **MCC Score** | 0.9427 | Model quality |
| **Errors** | 29/753 (3.85%) | Error analysis |
| **TF-IDF Features** | 1,000 | Feature engineering |
| **Custom Features** | 14 | Domain knowledge |

---

## 📊 MODEL COMPARISON TABLE (FOR SLIDES)

```
Model                  CV F1           Test F1    MCC      Time
----------------------------------------------------------------
Linear SVM             0.96 ± 0.002    96.18%    0.9427   0.32s  ⭐ BEST
MLP (Deep Learning)    0.96 ± 0.007    95.54%    0.9330   3.44s
Logistic Regression    0.93 ± 0.008    93.84%    0.9083   1.60s
Random Forest          0.91 ± 0.012    91.15%    0.8698   0.10s
```

---

## 💬 MUST-KNOW EXPLANATIONS

### "What is Sentiment Analysis?"
> "Bir metnin duygusal tonunu - pozitif, negatif veya nötr - otomatik olarak belirleme işlemi."

### "Why RSS Scraping?"
> "Üç nedeni var: 1) Yasal - herkes için açık, 2) Temiz veri - başlık/tarih düzgün, 3) Güncel - her gün yeni haberler."

### "What is TF-IDF?"
> "Term Frequency × Inverse Document Frequency. Nadir ama önemli kelimelere yüksek skor veriyor. 'the' düşük, 'surged' yüksek."

### "Why SVM beats MLP?"
> "Finansal sentiment lineer ayrılabilir. 'profit' = pozitif, 'loss' = negatif. Karmaşık deep learning gereksiz. Ayrıca 10 kat daha hızlı."

### "What is (2632, 1000)?"
> "2632 cümle, her biri 1000 sayı ile temsil ediliyor. Her satır bir cümle, her sütun bir özellik."

### "What is F1-Score?"
> "Precision ve Recall'ın harmonik ortalaması. Dengesiz verilerde accuracy'den daha güvenilir."

### "What is MCC?"
> "Matthews Correlation Coefficient. -1 ile +1 arası. 0 = rastgele, 1 = mükemmel. 0.94 = model şans değil, gerçekten öğrenmiş."

### "What is Cross Validation?"
> "Veriyi 5 parçaya böl, 5 kez farklı kombinasyonla test et. Tek testten daha güvenilir."

### "What is Overfitting?"
> "Model ezberleme yapıyor, yeni veriyi tahmin edemiyor. Learning curves'de train yüksek, CV düşükse overfitting var. Bizde ikisi yakın - sorun yok."

---

## 🚨 COMMON QUESTIONS & ANSWERS

### Q: "Neden template kullandınız?"
**A:** "451 gerçek haber yetersiz ve dengesiz (Negative %16). Template'lerle her sınıfı 550'ye tamamladık - dengeli veri seti elde ettik."

### Q: "Augmentation nasıl çalışıyor?"
**A:** "Synonym replacement: 'profit' → 'earnings', Random swap: kelime yerini değiştir, Random deletion: rastgele kelime sil. Anlam aynı, kelimeler farklı."

### Q: "src/ dosyaları kullanılıyor mu?"
**A:** "Evet. `sentiment_labeler.py` ve `augmentation.py` create_full_dataset.py'de import ediliyor. `real_scraper.py` ile 451 haber toplandı."

### Q: "%96 çok yüksek değil mi?"
**A:** "Template ve augmentation kullandık. Test-train benzer pattern'ler içeriyor. Gerçek dünyada biraz düşük olabilir - limitation olarak raporladık."

### Q: "Neden 1000 feature? Neden 100?"
**A:** "1000 (TF-IDF): En önemli kelimeler, fazlası gürültü ekler. 100 (Word2Vec): Anlam vektörü, standart boyut. 14 (Custom): Domain bilgisi."

### Q: "Overfitting var mı?"
**A:** "Learning curves'e bakınca train ve CV yakın - overfitting yok. L2 regularization ve early stopping kullandık."

---

## 🎤 SPEAKING TIPS

1. **Slow down** - 15 dakika var, acele etmeyin
2. **Make eye contact** - Hocaya bakın, ekrana değil
3. **Use pauses** - Her slayt sonrası 2-3 saniye duraklayın
4. **Point to visuals** - "Bakın burada..." diyerek görselleri gösterin
5. **Confidence** - "Biz yaptık, başardık" tonunda konuşun

---

## 🔧 EMERGENCY COMMANDS

```bash
# If demo notebook crashes
jupyter notebook --no-browser --port=8888

# If verification fails
python3 reset_and_rebuild.py --yes
python3 create_full_dataset.py
python3 train_and_evaluate.py

# Quick check all files exist
find . -name "*.pkl" -o -name "*_clean.csv" -o -name "*.png"

# Re-run verification
python3 verify_presentation_alignment.py
```

---

## 📸 FIGURE LOCATIONS

- `figures/confusion_matrices.png` - Confusion matrices for all models
- `figures/roc_curves.png` - ROC curves with AUC scores
- `figures/learning_curves.png` - Train vs CV scores
- `figures/model_comparison.png` - Bar chart of F1-scores

---

## ✅ FINAL PRE-PRESENTATION CHECK

- [ ] All 4 `.pkl` models exist in `models/`
- [ ] All 3 CSV files in `data/processed/`
- [ ] All figures in `figures/`
- [ ] `verify_presentation_alignment.py` passes ✅
- [ ] Demo notebook opens without errors
- [ ] Memorized key numbers (96.18%, 3761, 753, 0.32s)
- [ ] Rehearsed explanations for TF-IDF, SVM, F1-Score
- [ ] Prepared for Q&A

---

**Good luck! 🎉 You got this! 💪**
