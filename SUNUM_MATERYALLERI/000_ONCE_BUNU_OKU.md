# SUNUM HAZIRLIGI - GUNCEL REHBER (UPDATED)

## GUNCEL RAKAMLAR (EZBERLE!)

```
Dataset: 3,761 sample (train:2632 + val:376 + test:753)
Test: 753 sample (GUNCELLENDI - artik 500+ gereksinimi karsilaniyor!)
Features: 1,000 TF-IDF

EN IYI MODEL: Linear SVM - GUNCELLENDI!
  - F1-Score: 96.18%
  - Accuracy: 96.15%
  - CV: 95.99% ± 0.19%

DIGER MODELLER:
  - MLP (Deep Learning): 95.54% F1
  - Logistic Regression: 93.84% F1
  - Random Forest: 91.15% F1

GERCEK VERI:
  - 451 gercek RSS haberi (Yahoo Finance, CNBC, MarketWatch)
  - %12 gercek veri + %88 template + augmentation

HATA ANALIZI:
  - Toplam hata: 24/753 (3.19%)
  - En zor sinif: Neutral
```

## ONEMLI DEGISIKLIKLER

1. **Artik GERCEK veri kullaniliyor!**
   - `real_scraped_data.csv` -> 451 gercek haber
   - Rule-based sentiment labeling (random degil!)

2. **Test size artik 753 (eski: 390)**
   - Proje gereksinimi: 500+
   - Artik karsilaniyor!

3. **En iyi model LINEAR SVM!**
   - Linear SVM: 96.18% F1
   - MLP: 95.54% F1
   - Neden SVM kazandı? TF-IDF + SVM klasik kombinasyon, hızlı (0.33s)

4. **Dataset buyudu!**
   - Eski: 2,607 sample
   - Yeni: 3,761 sample

---

## SUNUM DOSYALARI

Bu klasorde 3 ana dosya var:

1. **000_ONCE_BUNU_OKU.md** (bu dosya) - Hizli bakis
2. **KONUSMA_METNI_BASIT.md** - Kisa konusma metni
3. **SORU_CEVAP_VE_STRATEJI.md** - Detayli soru-cevaplar

---

## DEMO ICIN

### Jupyter Notebook (Tavsiye Edilen)
```bash
cd 2121251034_MEHMET_TAHA_BOYNIKOGLU
jupyter notebook demo_notebook.ipynb
```

### Onemli Cell'ler:
- Cell 4: Test data istatistikleri
- Cell 8: Model yukleme ve skorlar
- Cell 16: Model karsilastirma tablosu
- Cell 18: Confusion Matrix (gorsel)
- Cell 28: LIVE PREDICTION DEMO

---

## TIMING (10 Dakika)

```
MEHMET TAHA (3.5 dk):
  - Giris, Problem, Data Collection
  - Slide 1-3

MERVE (3.5 dk):
  - TF-IDF, Models, Results
  - Slide 4-6

ELIF (3 dk):
  - Confusion Matrix, Error Analysis, Conclusion
  - Slide 7-9

SORU-CEVAP: 5 dakika
```

---

## KRITIK NOKTALAR

1. **En iyi model = Linear SVM** (96.18% F1-Score)
2. **MLP ikinci** (95.54% F1) - yine de çok başarılı!
3. **3,761 sample** (2,607 değil!)
4. **753 test sample** (500+ gereksinimi karşılandı!)
5. **451 GERCEK RSS haberi** (artik gercek scraping kullaniliyor!)
4. **753 test sample** (390 degil!)
5. **451 GERCEK RSS haberi** (artik gercek scraping kullaniliyor!)

---

## PROJE GEREKSINIMLERI (HEPSI KARSILANDI)

- [x] Data Collection (2000+): 3,761 sample
- [x] Real Web Scraping: 451 RSS haberi
- [x] Training Size (1500+): 2,632 sample
- [x] Test Size (500+): 753 sample
- [x] Traditional ML (2+): LogReg, SVM, RF
- [x] Deep Learning (1+): MLP
- [x] 5-Fold CV: Var
- [x] Regularization: L2, Early Stopping
- [x] TF-IDF: 1,000 features

**BASARILAR!**
