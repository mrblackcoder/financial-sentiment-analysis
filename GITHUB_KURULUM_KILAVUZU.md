# GITHUB'DAN PROJE KURULUM KILAVUZU

Bu kılavuz, başka birinin projeyi GitHub'dan indirip çalıştırması için gerekli tüm adımları içerir.

---

## ÖN GEREKSINIMLER

### 1. Python Kontrolü
```bash
# Python 3.8+ gerekli
python3 --version

# Eğer yoksa, macOS için:
brew install python3

# veya https://www.python.org/downloads/ adresinden indirin
```

### 2. Git Kontrolü
```bash
# Git kurulu mu?
git --version

# Eğer yoksa, macOS için:
brew install git

# veya https://git-scm.com/downloads adresinden indirin
```

---

## KURULUM ADIMLARI

### ADIM 1: Projeyi İndir

```bash
# Terminal'i aç ve istediğin klasöre git
cd ~/Desktop

# Projeyi klonla (GitHub repo URL'ini değiştir)
git clone https://github.com/KULLANICI_ADI/financial-sentiment-analysis.git

# Proje klasörüne gir
cd financial-sentiment-analysis
```

---

### ADIM 2: Sanal Ortam Oluştur (Önerilen)

```bash
# Sanal ortam oluştur
python3 -m venv venv

# Sanal ortamı aktive et
source venv/bin/activate

# Artık prompt (venv) ile başlayacak
```

---

### ADIM 3: Gerekli Kütüphaneleri Yükle

```bash
# requirements.txt'den kütüphaneleri yükle
pip install -r requirements.txt

# Yükleme tamamlandı mı kontrol et
pip list | grep -E "scikit-learn|pandas|numpy|matplotlib"
```

**Beklenen çıktı:**
