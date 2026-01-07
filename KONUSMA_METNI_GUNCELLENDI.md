# SUNUM KONUŞMA METNİ - GÜNCELLENMIŞ VERSİYON

## ⚠️ KRİTİK NOTLAR (SUNUMDAN ÖNCE OKU!)

**GERÇEK DEĞERLER (Koddan alındı):**
```
Dataset: 3,761 sample
  - Train: 2,632 (70%)
  - Val: 376 (10%)
  - Test: 753 (20%)

EN İYİ MODEL: Linear SVM ⭐
  - Test F1: 96.18%
  - Test Acc: 96.15%
  - CV F1: 95.99% ± 0.19%
  - Training Time: 0.32s
  - MCC: 0.9427

DİĞER MODELLER:
  - MLP (Deep Learning): 95.54% F1, 3.44s
  - Logistic Regression: 93.84% F1, 1.57s
  - Random Forest: 91.15% F1, 0.08s

FEATURES:
  - TF-IDF: 1000 (n-gram 1-3)
  - BoW: 500 (n-gram 1-2)
  - Word2Vec: 100
  - Custom: 14
  - Combined: 1014

HATALAR:
  - Total: 29/753 (3.85%)
  - Negative: 9/242 (3.7%)
  - Neutral: 6/248 (2.4%)
  - Positive: 14/263 (5.3%)
```

---

## ZAMANLAMA (~15 dakika)

| Kişi | Bölüm | Süre |
|------|-------|------|
| **MEHMET TAHA** | Giriş + Problem + Veri + Features | **7 dk** |
| **MERVE** | Modeller + Sonuçlar | **5 dk** |
| **ELIF** | Görseller + Sonuç | **3 dk** |

---

# BÖLÜM 1: MEHMET TAHA (7 dakika)

---

## SLAYT 1: KAPAK (30 sn)

**[KONUŞMA]**
> "İyi günler hocam. Ben Mehmet Taha Boynikoğlu, Financial Sentiment Analysis projemizi sunacağız."
>
> "Ekip arkadaşlarım Merve Kedersiz ve Elif Hande Arslan."
>
> "Ben veri toplama ve feature engineering'i anlatacağım. Merve modelleri, Elif sonuçları özetleyecek."

---

## SLAYT 2: PROBLEM TANIMI (1.5 dk)

**[KONUŞMA]**
> "Problemimiz şu: Finansal piyasalarda her gün binlerce haber yayınlanıyor. Bloomberg, Reuters, Yahoo Finance... Bir yatırımcı bunları manuel okuyamaz."
>
> "Çözümümüz **Sentiment Analysis** - duygu analizi. Bir haberin olumlu mu, olumsuz mu, yoksa nötr mü olduğunu otomatik tespit ediyoruz."
>
> "**Örnek:** 'Stock prices surged' pozitif, 'Company faces losses' negatif, 'Market remained unchanged' nötr."
>
> "Projemizde **Linear SVM modeli ile %96.18 F1-Score** elde ettik. Yani 100 haberden 96'sını doğru sınıflandırıyoruz."

---

## SLAYT 3: VERİ TOPLAMA (2 dk)

**[TERMİNAL DEMO GÖSTERİRKEN]**

> "Veri toplama sürecini göstereyim."

```bash
python3 create_full_dataset.py
```

**[ÇIKTIYI GÖSTERİRKEN]**
```
[STEP 1] Loading REAL scraped financial news...
  Loaded 451 real news articles from RSS feeds
  Real data distribution:
    Negative: 72 (16.0%)
    Neutral: 220 (48.8%)
    Positive: 159 (35.3%)

[STEP 2] Adding template-based samples for balance...
  Added 1199 template samples

[STEP 3] Combining real + template data...
  Combined dataset: 1650 samples
  Distribution before augmentation:
    Negative: 550 (33.3%)
    Neutral: 550 (33.3%)
    Positive: 550 (33.3%)

[STEP 4] Applying data augmentation...
  Augmented dataset: 3761 samples

[STEP 5] Creating train/validation/test splits...
  Training set:   2632 samples (70.0%)
  Validation set: 376 samples (10.0%)
  Test set:       753 samples (20.0%)
```

**[KONUŞMA]**
> "**Adım 1:** 451 gerçek RSS haberi topladık (Yahoo Finance, CNBC, MarketWatch)."
>
> "Ama problemi görüyorsunuz - **class imbalance**: Neutral %48, Positive %35, Negative sadece %16."
>
> "**Neden sorun?** Model çok gördüğü sınıfı iyi öğrenir, az gördüğünü ihmal eder."
>
> "**Adım 2:** Template örnekler ekledik - her sınıfı 550'ye tamamladık. Artık dengelendi."
>
> "**Adım 3:** Data augmentation - synonym replacement, random swap. 1650'den 3761'e çıkardık."
>
> "**Proje gereksinimi 2000+ idi, biz 3761 topladık - %88 fazla!**"
>
> "**Adım 4:** Train/Val/Test: 70/10/20 böldük. **Test 753 sample - gereksinim 500+ idi, %50 fazla verdik!**"

---

## SLAYT 4: FEATURE ENGINEERING (2.5 dk)

**[KONUŞMA]**
> "Şimdi en önemli kısım: **Feature Engineering**."
>
> "**Problem:** Bilgisayar 'Apple stock surged' cümlesini anlayamaz. CPU sadece sayı işler."
>
> "**Çözüm:** Cümleyi sayılara çeviriyoruz. Buna **vektör** diyoruz."
>
> "**4 farklı yöntem kullandık** - proje gereksinimi:"

> "**1. TF-IDF (2632, 1000):**"
> "Her cümle 1000 sayıya dönüştü."
>
> "**TF-IDF nedir?** Term Frequency × Inverse Document Frequency."
> "- TF: Kelime cümlede kaç kez geçiyor"
> "- IDF: Kelime tüm cümlelerde ne kadar nadir"
> "- Nadir ama önemli kelimeler yüksek skor alır"
>
> "Örnek: 'the' her yerde var → düşük skor. 'surged' nadir ve önemli → yüksek skor."
>
> "**Neden 1000?** Binlerce kelime var ama en önemli 1000'ini seçtik. Fazlası gürültü."

> "**2. Bag-of-Words (2632, 500):**"
> "Basitçe kelime sayımı. 'profit profit loss' → profit:2, loss:1"
>
> "**Bigram'lar da var** - 'strong growth' gibi 2 kelimelik ifadeler."

> "**3. Word2Vec (2632, 100):**"
> "Kelimelerin anlamını yakalayan vektör."
> "'profit' ve 'earnings' yakın vektörler çünkü anlamları benzer."

> "**4. Custom Features (2632, 14):**"
> "Finansal domain'e özel özellikler:"
> "- positive_count: Kaç pozitif kelime var (surge, profit)"
> "- negative_count: Kaç negatif kelime var (crash, loss)"
> "- sentiment_score: positive - negative"
> "- ticker_count: $AAPL, $TSLA gibi hisse sembolleri"
>
> "**14 tane finansal özellik tanımladık** - domain bilgisi ekliyoruz."

> "**Sonuç: (2632, 1014)** - TF-IDF (1000) + Custom (14) birleştirdik."
> "Her cümle 1014 sayı ile temsil ediliyor."

---

# BÖLÜM 2: MERVE (5 dakika)

---

## SLAYT 5: MODELLER (2 dk)

**[KONUŞMA]**
> "Ben Merve, modelleri anlatacağım."
>
> "**4 model eğittik** - proje 2 traditional ML + 1 deep learning istiyor, biz 3+1 yaptık:"

> "**1. Logistic Regression:** En basit lineer model. L2 regularization ile overfitting önlüyoruz."
>
> "**2. Linear SVM:** Support Vector Machine. En iyi ayırıcı çizgiyi buluyor - iki sınıf arasındaki boşluğu maksimize ediyor."
>
> "**3. Random Forest:** 100 karar ağacı oluşturup oylama yapıyor. Ensemble learning."
>
> "**4. MLP (Deep Learning):** Multi-Layer Perceptron. **3 hidden layer: 256 → 128 → 64 nöron.** ReLU activation, early stopping ile overfitting önlüyoruz."

> "**Overfitting önleme için 3 teknik:**"
> "- **L2 Regularization:** Büyük ağırlıkları cezalandırıyor (LogReg, SVM, MLP)"
> "- **Early Stopping:** Validation kötüleştiğinde durduruyor (MLP)"
> "- **5-Fold CV:** 5 farklı bölümle test - daha güvenilir skor"

---

## SLAYT 6: SONUÇLAR (3 dk)

**[TERMİNAL ÇIKTISINI GÖSTERİRKEN]**

```
Model                     CV F1                Test F1    Test Acc   MCC        Time      
------------------------------------------------------------------------------------------
Linear SVM                0.9599 ± 0.0019   0.9618     0.9615     0.9427     0.32s  ⭐
MLP (Deep Learning)       0.9582 ± 0.0070   0.9554     0.9548     0.9330     3.44s
Logistic Regression       0.9327 ± 0.0077   0.9384     0.9376     0.9083     1.57s
Random Forest             0.9146 ± 0.0116   0.9115     0.9097     0.8698     0.08s
```

**[KONUŞMA]**
> "Sonuçlar tablosu burada."
>
> "**Linear SVM %96.18 F1-Score ile en iyi!** Hem en yüksek skor hem de çok hızlı - 0.32 saniye."
>
> "**Neden SVM kazandı?**"
> "1. **TF-IDF + SVM klasik güçlü kombinasyon** - sparse features için ideal"
> "2. **Hız:** 0.32s vs 3.44s (MLP 10x daha yavaş)"
> "3. **Finansal sentiment keyword-based** - 'profit' görürsen pozitif, 'loss' görürsen negatif. Lineer ayırma yeterli."
>
> "**MLP ikinci sırada %95.54** - yine de çok başarılı! Ama deep learning için dataset biraz küçük (ideal 10K+)."

> "**F1-Score nedir?** Precision ve Recall'ın harmonik ortalaması."
> "- Precision: Pozitif dediklerimin kaçı gerçekten pozitif?"
> "- Recall: Gerçek pozitiflerin kaçını yakaladım?"
> "- Dengesiz verilerde accuracy'den daha güvenilir."
>
> "**MCC 0.9427** - Matthews Correlation Coefficient. -1 ile +1 arası, 1 = mükemmel. **0.94 modelin gerçekten öğrendiğini gösteriyor, şans değil.**"

> "**CV vs Test Scores:**"
> "- CV: 95.99% ± 0.19%"
> "- Test: 96.18%"
> "- **Test > CV = İyi generalization!** Overfitting olsa test düşük olurdu."

---

# BÖLÜM 3: ELIF (3 dakika)

---

## SLAYT 7: CONFUSION MATRIX (1 dk)

**[FİGURES/CONFUSION_MATRICES.PNG GÖSTERİRKEN]**

**[KONUŞMA]**
> "Ben Elif, görselleri özetleyeceğim."
>
> "**Confusion Matrix:** Gerçek sınıf vs tahmin edilen sınıf tablosu."

```
          Pred Neg  Pred Neu  Pred Pos
True Neg       233         8         1      (242 total)
True Neu         3       242         3      (248 total)
True Pos         0        14       249      (263 total)
```

> "**Diagonal = Doğru tahminler:** 233+242+249 = 724 doğru"
> "**Off-diagonal = Hatalar:** Toplam 29 hata"
>
> "**En çok hata nerede?** Positive → Neutral (14 hata). Neden? Mixed sentiment - hem pozitif hem nötr kelimeler var."

---

## SLAYT 8: LEARNING CURVES & ROC (1 dk)

**[FIGURES/LEARNING_CURVES.PNG GÖSTERİRKEN]**

**[KONUŞMA]**
> "**Learning Curves:** Mavi training, turuncu CV."
> "**İkisi yakın → overfitting yok!** Ezberleme olsa mavi çok yüksek, turuncu düşük olurdu."

**[FIGURES/ROC_CURVES.PNG GÖSTERİRKEN]**

> "**ROC Curves:** True Positive Rate vs False Positive Rate."
> "**AUC (egri altındaki alan) hepsi 0.99+** - mükemmel sınıf ayrımı!"

---

## SLAYT 9: PROJE GEREKSİNİMLERİ (1 dk)

**[KONUŞMA]**
> "**Tüm proje gereksinimlerini karşıladık:**"
>
> "✅ **3761 sample** (gereksinim 2000+) - %88 fazla"
> "✅ **753 test** (gereksinim 500+) - %50 fazla"
> "✅ **451 gerçek RSS haberi** - gerçek web scraping"
> "✅ **3 Traditional ML + 1 Deep Learning** - gereksinim karşılandı"
> "✅ **4 feature tipi** - BoW, TF-IDF, Word2Vec, Custom"
> "✅ **5-Fold CV** - güvenilir skor"
> "✅ **L2 + Early Stopping** - overfitting önleme"

> "**Sonuç: Linear SVM - 96.18% F1-Score**"
> "753 testten sadece 29 hata - %3.85 hata oranı!"
>
> "Teşekkürler! Sorularınızı alabiliriz."

---

# SORU-CEVAP HAZIRLIGI

## S: "Neden MLP en iyi değil? Deep learning daha iyi olmalı?"

**C:**
> "Haklısınız hocam, teoride deep learning daha iyi olmalı. Ancak 3 sebepten SVM kazandı:"
> 
> "1. **Dataset size:** 3,761 sample MLP için küçük. Deep learning 10K+ data ile performans gösterir."
> "2. **Feature type:** TF-IDF sparse features linear modeller için optimal. Financial text keyword-based - complex patterns yok."
> "3. **Speed:** Production'da 0.32s kritik. MLP 3.44s sürüyor."
> 
> "MLP yine de 95.54% aldı - çok başarılı! Future work'te FinBERT deneyebiliriz."

## S: "Word2Vec neden kullanmadınız?"

**C:**
> "Hocam, implement ettik (`word2vec_features.py`) ancak TF-IDF daha iyi sonuç verdi:"
> "- Financial sentiment keyword-based - 'surge' kelimesi her context'te pozitif"
> "- Word2Vec 3,761 sample ile제대로 train olamadı"
> "- Pre-trained Google News vectors financial domain'e spesifik değil"
> 
> "Future work: FinBERT veya domain-specific Word2Vec!"

## S: "451 haber yeterli mi?"

**C:**
> "451 gerçek RSS haberi + template + augmentation ile 3,761 sample oluşturduk."
> "Proje gereksinimi 2,000 idi - %88 astık!"
> "Rule-based labeling ile quality sağladık."
> "Future: Twitter API ile 10K+ sample toplayacağız."

## S: "Test > CV nasıl olur? Overfitting yok mu?"

**C:**
> "Harika soru hocam! Test 96.18%, CV 95.99% - gap sadece +0.19%."
> "Bu **iyi generalization** gösteriyor:"
> "- Overfitting olsa test < CV olurdu"
> "- Küçük gap = model ezberlemiyor, genelliyor"
> "- L2 + Early Stopping + 5-Fold CV ile önledik"

## S: "Confusion matrix'te en çok hata hangi sınıfta?"

**C:**
> "Positive → Neutral: 14 hata (5.3%)"
> "Sebep: Mixed sentiment - 'Stock rose but concerns remain' gibi cümleler hem pozitif hem nötr kelime içeriyor."
> "Future: Context-aware models (BERT) bu sorunu çözebilir."

---

# KOD-SUNUM EŞLEŞTIRMESI

## TERMİNAL ÇIKTI → SUNUM

| Sunum İddiası | Kod Çıktısı | Dosya | Satır |
|---------------|-------------|-------|-------|
| "3761 sample" | `Total: 3761 samples` | train_and_evaluate.py çıktı | - |
| "2632 train, 376 val, 753 test" | `Train: 2632, Val: 376, Test: 753` | train_and_evaluate.py çıktı | - |
| "TF-IDF 1000 feature" | `Shape: (2632, 1000)` | train_and_evaluate.py çıktı | - |
| "BoW 500 feature" | `Shape: (2632, 500)` | train_and_evaluate.py çıktı | - |
| "Word2Vec 100 feature" | `Shape: (2632, 100)` | train_and_evaluate.py çıktı | - |
| "Custom 14 feature" | `Shape: (2632, 14)` | train_and_evaluate.py çıktı | - |
| "Linear SVM 96.18%" | `Test F1: 0.9618` | train_and_evaluate.py çıktı | - |
| "MLP 95.54%" | `Test F1: 0.9554` | train_and_evaluate.py çıktı | - |
| "29 hata" | `Total errors: 29/753 (3.85%)` | train_and_evaluate.py çıktı | - |
| "MCC 0.9427" | `MCC: 0.9427` | train_and_evaluate.py çıktı | - |

## JUPYTER NOTEBOOK → SUNUM

| Cell | İçerik | Sunumda Nerede Kullanılır |
|------|--------|--------------------------|
| Cell 4 | Dataset stats | MEHMET TAHA - Veri toplama |
| Cell 5 | Distribution chart | MEHMET TAHA - Class balance |
| Cell 6 | Sample texts | MEHMET TAHA - Gerçek örnekler |
| Cell 11 | Model comparison table | MERVE - Sonuçlar |
| Cell 14 | Confusion matrix | ELIF - Hata analizi |
| Cell 16 | ROC curves | ELIF - Model ayrımı |
| Cell 18 | Learning curves | ELIF - Overfitting kontrolü |
| Cell 20 | Live prediction | DEMO - Canlı tahmin |

---

## FİNAL CHECKLIST (Sunumdan önce)

- [ ] `python3 train_and_evaluate.py` çalıştır - değerleri doğrula
- [ ] `demo_notebook.ipynb` tüm cell'leri çalıştır
- [ ] `figures/` klasöründeki PNG'ler var mı kontrol et
- [ ] Konuşma metnindeki rakamlar kod çıktısıyla eşleşiyor mu?
- [ ] "Linear SVM en iyi" diyorsun, "MLP değil" diyorsun?

**HAZIR! BAŞARILAR! 🚀**
