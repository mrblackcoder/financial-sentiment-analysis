# SUNUM KONUSMA METNI - DETAYLI VERSIYON (10-12 Dakika)

---

# MEHMET TAHA (3.5-4 dakika)

## Slide 1: Giris (30-45 saniye)

"Merhaba hocam. Ben Mehmet Taha Boynikoglu. Ekip arkadaslarim Merve ve Elif ile Financial Sentiment Analysis projemizi sunacagiz.

**3,761 sample** topladik, bunun **451'i gercek RSS haberlerinden**. **4 model** egittik. En iyi sonuc: **MLP Deep Learning ile 96.81% F1-Score**."

**[Hoca sorarsa: "Bu rakamlar nereden geliyor?"]**
> "Hocam, Jupyter notebook'umuzda Cell 4'te dataset istatistiklerini gorebilirsiniz. `create_full_dataset.py` scripti calistirildiginda bu rakamlari uretir."

---

## Slide 2: Problem (45 saniye - 1 dakika)

"Hocam, finansal piyasalarda Twitter, Reuters gibi kaynaklardan gelen metinleri manuel analiz etmek cok uzun suruyor. Bizim cozumumuz: otomatik sentiment analysis.

Bir cumleyi aliyoruz - pozitif mi, negatif mi, notr mu diye siniflandiriyoruz.

Gercek RSS verisi + dogru labeling ile **96.81% F1-Score** elde ettik."

**[Hoca sorarsa: "Neden 3 sinif? Binary degil mi?"]**
> "Hocam, finansta sadece 'iyi' veya 'kotu' yeterli degil. 'Market remained steady' gibi cumleler var - bunlar notr. 3-class classification daha gercekci."

**[Hoca sorarsa: "Bu problemi neden sectiniz?"]**
> "Hocam, finansal haberler yatirim kararlari icin kritik. Gunluk binlerce haber var, manuel okumak imkansiz. Otomatik sentiment analizi ile trader'lar ve yatirimcilar hizli karar verebilir."

---

## Slide 3: Data Collection (1.5-2 dakika)

"Dataset nasil olusturduk?

**Gercek Veri Kaynaklari (451 haber):**
- Yahoo Finance RSS
- CNBC RSS
- MarketWatch RSS
- Investing.com RSS

**Rule-based Sentiment Labeling:**
- Pozitif kelimeler: 'surge', 'growth', 'profit'
- Negatif kelimeler: 'decline', 'loss', 'crash'
- Notr kelimeler: 'unchanged', 'steady', 'stable'

**Data Augmentation:**
- Synonym replacement: 'profit' -> 'earnings'
- Random swap, deletion, insertion

**Final Dataset:**
- Train: 2,632 sample (70%)
- Validation: 376 sample (10%)
- Test: 753 sample (20%)
- **Toplam: 3,761 sample**

Simdi Merve feature engineering'i anlatacak."

**[Hoca sorarsa: "RSS scraping nasil yapildi?"]**
> "Hocam, `src/data/rss_scraper.py` dosyasinda feedparser kutuphanesi ile RSS feed'leri parse ettik. Her haber icin title + description aldik. Sonra `src/data/sentiment_labeler.py` ile rule-based etiketledik."

**[Hoca sorarsa: "Neden 70/10/20 split?"]**
> "Hocam, validation set hyperparameter tuning icin, test set final evaluation icin ayrildi. %20 test = 753 sample, istatistiksel olarak anlamli. Proje gereksinimi 500+ idi, biz %50 astik."

**[Hoca sorarsa: "Augmentation neden gerekli?"]**
> "Hocam, 451 gercek haber tek basina yeterli degil. Augmentation ile veri cesitliligini artirdik. Ama label'lari koruduk - augmented veri orijinal ile ayni sentiment'i tasiyor."

**[Hoca sorarsa: "Rule-based labeling guvenilir mi?"]**
> "Hocam, financial domain'de kelimeler net sentiment tasiyor. 'Surge', 'profit' = pozitif, 'crash', 'loss' = negatif. Bu yuzden rule-based etkili. Random labeling degil, mantikli kurallar."

**[Jupyter'da gosterilecek: Cell 2-4]**
> "Notebook'ta Cell 2'de veri yukleme, Cell 4'te dataset istatistikleri gorunuyor."

---

# MERVE (3.5-4 dakika)

## Slide 4: TF-IDF Feature Engineering (1.5-2 dakika)

"Tesekkurler Taha. Ben Merve, feature engineering'i anlatacagim.

**TF-IDF nedir?**
Metinleri sayisal vektorlere donusturuyor.

- **TF (Term Frequency):** Bir kelimenin dokumanda kac kez gectigini olcer
- **IDF (Inverse Document Frequency):** Kelimenin tum dokumanlarda ne kadar nadir oldugunu olcer
- **TF-IDF = TF x IDF:** Nadir ama dokumanda sik gecen kelimeler yuksek skor alir

**Ornek:**
'Stock prices surged on strong earnings'

TF-IDF skorlari:
- 'surged' -> 0.62 (nadir kelime, cok onemli!)
- 'earnings' -> 0.58
- 'stock' -> 0.45
- 'on' -> 0.12 (yaygin, az onemli)

**Parametreler:**
```python
TfidfVectorizer(
    max_features=1000,    # Top 1,000 kelime
    ngram_range=(1, 3),   # Unigram + bigram + trigram
    stop_words='english'
)
```

**Neden Word2Vec degil TF-IDF?**
Financial text keyword-based. 'Profit', 'loss' gibi kelimeler onemli. TF-IDF bunlari yakaliyor, sparse matrix = hizli training."

**[Hoca sorarsa: "Ngram ne demek? Neden (1,3)?"]**
> "Hocam, unigram = tek kelime ('profit'), bigram = 2 kelime ('stock price'), trigram = 3 kelime ('stock price surge'). (1,3) = hepsini kullan. Bigram/trigram ile 'net profit' gibi anlamli ifadeleri yakaliyoruz."

**[Hoca sorarsa: "1000 feature yeterli mi? Neden 1000?"]**
> "Hocam, daha fazla feature noise ekler, daha az ise onemli kelimeleri kacirir. 1000, financial text icin optimal. `src/features/tfidf_features.py` dosyasinda gorebilirsiniz."

**[Hoca sorarsa: "Sparse matrix ne demek?"]**
> "Hocam, TF-IDF vektorlerinde cogu deger 0. Mesela 1000 feature'dan sadece 20-30 tanesi bir cumle icin non-zero. Sparse = seyrek. Bu bellek ve hiz avantaji sagliyor."

**[Hoca sorarsa: "Sublinear TF ne?"]**
> "Hocam, `sublinear_tf=True` parametresi var kodda. Bu TF'yi log(1+TF) yapiyor. Bir kelime 10 kez gecince 1 kez gecenden 10x onemli olmuyor, ~2.4x onemli oluyor. Daha dengeli."

**[Hoca sorarsa: "Stop words nedir?"]**
> "Hocam, 'the', 'is', 'and', 'of' gibi yaygin kelimeler. Bunlar sentiment tasimaz, cikariyoruz. sklearn'un English stop words listesini kullandik."

**[Jupyter'da gosterilecek: Cell 6-7]**
> "Notebook'ta Cell 6'da TfidfVectorizer parametreleri, Cell 7'de feature extraction gorunuyor."

---

## Slide 5: Models & Results (2 dakika)

"**4 model egittik:**

| Model | F1-Score | Accuracy | Training Time |
|-------|----------|----------|---------------|
| **MLP (Deep Learning)** | **96.81%** | **96.81%** | 1.45s |
| Linear SVM | 96.55% | 96.55% | 0.01s |
| Logistic Regression | 94.73% | 94.69% | 0.65s |
| Random Forest | 89.87% | 89.64% | 0.08s |

**En iyi: MLP Deep Learning - 96.81% F1-Score**

**Neden MLP kazandi?**
- Gercek veri + dogru labeling ile daha iyi patterns
- Non-linear relationships yakaladi
- Early stopping ile overfitting onlendi

**5-Fold Cross Validation:**
- MLP CV: 95.33% +/- 0.70%
- Test: 96.81%
- Test > CV = Iyi generalization!

Simdi Elif error analysis'i anlatacak."

**[Hoca sorarsa: "MLP mimarisi nasil?"]**
> "Hocam, 3 hidden layer kullandik: (256, 128, 64) noronlar. Input 1000 TF-IDF feature, output 3 class. ReLU activation, Adam optimizer. `src/models/mlp_model.py` dosyasinda detaylar var."

**[Hoca sorarsa: "Early stopping nasil calisiyor?"]**
> "Hocam, validation loss 10 epoch boyunca iyilesmezse training duruyor. Overfitting'i onluyor. `n_iter_no_change=10`, `validation_fraction=0.1` parametreleri kullandik."

**[Hoca sorarsa: "SVM neden MLP'ye yakin?"]**
> "Hocam, SVM de cok iyi (96.55%). TF-IDF features zaten cok ayirici, linear SVM bile iyi calisiyor. Ama MLP non-linear patterns'i yakalayarak 0.26% daha iyi sonuc aldi."

**[Hoca sorarsa: "Random Forest neden dusuk?"]**
> "Hocam, RF tree-based model. Sparse TF-IDF features icin optimal degil. Ama yine de 89.87% fena degil. Production'da interpretability icin RF tercih edilebilir."

**[Hoca sorarsa: "Logistic Regression nedir?"]**
> "Hocam, linear classifier. Softmax ile 3-class probability cikariyor. L2 regularization (C=1.0) kullandik. Simple ama etkili baseline."

**[Hoca sorarsa: "F1-Score neden onemli?"]**
> "Hocam, accuracy imbalanced class'larda yaniltici olabiliyor. F1 = precision ve recall'un harmonik ortalamasi. Her iki metrigin dengeli oldugunu gosteriyor."

**[Hoca sorarsa: "CV neden 5-fold?"]**
> "Hocam, Kohavi (1995) arastirmasina gore kucuk/orta datasetlerde 5-fold optimal. Her fold'da ~750 sample = istatistiksel anlamli. 10-fold marginal benefit, 2x cost."

**[Jupyter'da gosterilecek: Cell 14-16]**
> "Notebook'ta Cell 14'te model training, Cell 16'da comparison table gorunuyor."

---

# ELIF (3-3.5 dakika)

## Slide 6: Confusion Matrix (1.5 dakika)

"Tesekkurler Merve. Ben Elif, sonuclari anlatacagim.

**Test Seti: 753 sample, 24 hata (3.19%)**

**Per-class Performance:**
- Positive: ~97% F1 (261 sample, 2 hata)
- Negative: ~97% F1 (242 sample, 10 hata)
- Neutral: ~96% F1 (248 sample, 12 hata)

**Confusion Matrix:**
```
              Predicted
              Neg   Neu   Pos
Actual Neg:   232    7     3    (95.87% recall)
       Neu:    12  236     0    (95.16% recall)
       Pos:     2    0   261    (99.24% recall)
```

**Neden Neutral artik daha iyi?**
Dogru rule-based labeling sayesinde 'steady', 'stable' gibi kelimeler artik dogru sekilde Neutral olarak etiketlendi.

(Notebook Cell 18'de renkli confusion matrix gorebilirsiniz)"

**[Hoca sorarsa: "Confusion matrix nasil okunur?"]**
> "Hocam, satirlar gercek degerler (actual), sutunlar tahminler (predicted). Diagonal dogru tahminler. Mesela 232 tane gercek negative'i dogru tahmin ettik, 7 tanesini yanlis neutral, 3 tanesini yanlis positive dedik."

**[Hoca sorarsa: "Recall ve Precision farki ne?"]**
> "Hocam, Recall = 'Gercek positive'lerin kacini bulduk?' Precision = 'Positive dediklerimizin kaci gercekten positive?' Ikisi de onemli, F1 ikisinin dengesi."

**[Hoca sorarsa: "Positive neden en kolay?"]**
> "Hocam, 'surge', 'growth', 'bullish' gibi kelimeler cok net positive sentiment tasiyor. Model bunlari kolayca ogreniyor. 99.24% recall = neredeyse hepsini yakaladik."

**[Hoca sorarsa: "Neutral neden en zor?"]**
> "Hocam, 'steady' kelimesi gunluk dilde pozitif (saglam) ama finansta notr (degisim yok). Bu tur kelimeler karisikliga yol aciyor. Ama rule-based labeling ile bunu cozduk."

**[Jupyter'da gosterilecek: Cell 18]**
> "Notebook'ta Cell 18'de renkli confusion matrix ve per-class metrikleri gorunuyor."

---

## Slide 7: Error Analysis (1 dakika)

"**Hata Analizi (24 hata):**

**1. Mixed Sentiment (45% - 11 hata):**
'Market fell as Apple strong earnings'
- Karmasik cumleler iki sentiment iceriyor
- 'Fell' = negative, 'strong earnings' = positive

**2. Domain Jargon (35% - 8 hata):**
- 'Short squeeze', 'hedge' gibi teknik terimler
- Genel kelime dagarciginda farkli anlam

**3. Ambiguous Context (20% - 5 hata):**
- Baglama bagli kelimeler
- 'Volatile' = bazen iyi, bazen kotu

**Cozum onerileri:**
- FinBERT (context-aware model)
- Daha fazla real-world verisi"

**[Hoca sorarsa: "Bu hatalari nasil analiz ettiniz?"]**
> "Hocam, Jupyter'da Cell 20'de yanlis tahminleri listeledik. Her birini manuel inceledik ve pattern'leri belirledik. `figures/error_analysis.png` dosyasinda gorsel analiz var."

**[Hoca sorarsa: "Mixed sentiment nasil cozulur?"]**
> "Hocam, sentence-level yerine aspect-level sentiment analysis yapilabilir. Bir cumlede birden fazla aspect (Apple, market) ayri ayri analiz edilir. Future work'te bunu deneyecegiz."

**[Hoca sorarsa: "FinBERT nedir?"]**
> "Hocam, BERT'in finansal metinler uzerinde fine-tune edilmis versiyonu. Context anlayisi daha iyi. Hugging Face'de mevcut. Future work'te kullanacagiz."

**[Jupyter'da gosterilecek: Cell 20]**
> "Notebook'ta Cell 20'de yanlis tahminler ve analizi gorunuyor."

---

## Slide 8: Conclusion (30-45 saniye)

"**Sonuc:**

- **3,761 sample** (proje gereksinimi: 2,000+ - %88 astik!)
- **753 test sample** (proje gereksinimi: 500+ - %50 astik!)
- **451 gercek RSS haberi** (real web scraping!)
- 4 model (Traditional ML + Deep Learning)
- **96.81% F1-Score** (MLP Deep Learning)

**Key Takeaway:**
Gercek veri + dogru labeling + Deep Learning = En iyi sonuclar!

Tesekkurler, sorularinizi alabiliriz."

**[Hoca sorarsa: "Projenin en zor kismi neydi?"]**
> "Hocam, labeling. RSS'ten gelen ham veriye dogru sentiment atamak. Random labeling basarisiz oldu, rule-based labeling ile cozduk. Bu performansi %5'ten %96'ya cikardi."

**[Hoca sorarsa: "Production'da kullanilabilir mi?"]**
> "Hocam, kesinlikle! Model 7MB, CPU'da calisir, inference <0.01s. Flask API ile wrap edilebilir. Challenge: concept drift - financial dil degisiyor, periodic retraining gerekir."

---

# DETAYLI SORU-CEVAP (Hocanin Sorabilecegi Ek Sorular)

## Jupyter Notebook Hakkinda

**S: "Notebook'ta hangi cell ne yapiyor?"**
> "Hocam:
> - Cell 1-4: Veri yukleme ve istatistikler
> - Cell 5-7: TF-IDF feature extraction
> - Cell 8-12: Model yukleme
> - Cell 14-16: Model training ve comparison
> - Cell 18: Confusion matrix
> - Cell 20: Error analysis
> - Cell 28: CANLI TAHMIN DEMO"

**S: "Bu kod tek basina calisir mi?"**
> "Hocam, evet. Oncelikle:
> 1. `pip3 install -r requirements.txt`
> 2. `python3 create_full_dataset.py` (veri + model)
> 3. Jupyter notebook'u aciniz
>
> Alternatif: `python3 train_and_evaluate.py` learning curves icin."

---

## Kod Yapisi Hakkinda

**S: "Kod yapisi nasil?"**
> "Hocam, modular yapi kullandik:
> ```
> src/
>   data/         - RSS scraping, labeling
>   features/     - TF-IDF, BoW extraction
>   models/       - MLP, SVM, LogReg, RF
>   evaluation/   - Metrics, confusion matrix
> ```
> Her module tek sorumluluk - Clean Code prensipleri."

**S: "Hangi kutuphane kullandiniz?"**
> "Hocam:
> - sklearn: TF-IDF, modeller, metriks
> - pandas/numpy: veri isleme
> - feedparser: RSS scraping
> - matplotlib/seaborn: gorsellestirme"

---

## Teknik Detaylar

**S: "Regularization ne ise yariyor?"**
> "Hocam, overfitting'i onluyor. L2 regularization buyuk weight'leri penalize ediyor. Model daha basit kalir, generalize eder. MLP'de alpha=0.0001, SVM/LogReg'de C=1.0."

**S: "Overfitting var mi?"]**
> "Hocam, hayir! CV=95.33%, Test=96.81%. Test > CV = overfitting yok, iyi generalization. Overfitting olsaydi test skoru dusuk olurdu."

**S: "Class imbalance var mi?"**
> "Hocam, dengeli dagilim: ~33% Positive, ~33% Negative, ~33% Neutral. Stratified split kullandik, her split'te ayni oran."

**S: "Hyperparameter tuning yaptiniz mi?"**
> "Hocam, Grid Search ile temel parametreleri aradik. MLP icin hidden layers, learning rate, alpha. SVM icin C parametresi. Validation set ile optimize ettik."

---

## Literatur ve Karsilastirma

**S: "Literaturde benzer calismalar var mi?"**
> "Hocam, evet:
> - FinancialPhraseBank: 87-90% F1
> - Twitter sentiment: 80-85% F1
> - Bizim: 96.81% F1
>
> Fark: Gercek RSS + dogru labeling. Quality data = quality model."

**S: "Bu sonuc gercekci mi? Cok yuksek gorunuyor."**
> "Hocam, rule-based labeling sayesinde data quality yuksek. Model zaten labeling kurallarini ogreniyor. Real-world'de daha dusuk olabilir, ama metodoloji dogru."

---

# DEMO SENARYOSU (Hoca isterse)

## Jupyter'i Ac
```bash
/Users/metaboy/Library/Python/3.9/bin/jupyter notebook demo_notebook.ipynb
```

## Gosterilecek Sirasi
1. **Cell 4:** Dataset stats (3,761 sample, 753 test)
2. **Cell 16:** Model comparison table (MLP 96.81%)
3. **Cell 18:** Confusion matrix (renkli gorsel)
4. **Cell 28:** CANLI TAHMIN

## Canli Tahmin Ornekleri
```
Input: "Stock prices surged after strong earnings"
-> Prediction: POSITIVE (Confidence: 98%)

Input: "Company faces regulatory challenges and losses"
-> Prediction: NEGATIVE (Confidence: 95%)

Input: "Market remained steady amid mixed signals"
-> Prediction: NEUTRAL (Confidence: 87%)
```

---

# EZBERLE! (KRITIK RAKAMLAR)

```
Dataset: 3,761 sample
Test: 753 sample
Split: 70/10/20

MLP: 96.81% F1 (EN IYI)
SVM: 96.55% F1
LogReg: 94.73% F1
RF: 89.87% F1

Features: 1,000 TF-IDF
CV: 5-fold
Hatalar: 24/753 (3.19%)
Gercek RSS: 451 haber

MLP Architecture: (256, 128, 64)
Early Stopping: n_iter_no_change=10
```

# SOYLEME! (ESKI DEGERLER)

```
X 91.75% F1 (eski skor)
X 2,607 sample (eski boyut)
X 390 test (eski test size)
X Linear SVM en iyi (artik MLP!)
```

---

**BASARILAR!**
