#!/bin/bash

echo "=== FINANCIAL SENTIMENT ANALYSIS - QUICK START ==="

# 1. Kurulum kontrolü
echo "[1/5] Checking Python..."
python3 --version || { echo "Python3 not found!"; exit 1; }

# 2. Bağımlılıkları yükle
echo "[2/5] Installing dependencies..."
python3 -m pip install -q -r requirements.txt

# 3. NLTK verilerini indir
echo "[3/5] Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# 4. Durumu kontrol et
echo "[4/5] Checking project status..."
python3 reset_and_rebuild.py --status

# 5. Tamamlandı
echo "[5/5] Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run: python3 create_full_dataset.py"
echo "  2. Run: python3 train_and_evaluate.py"
echo "  3. Open: jupyter notebook demo_notebook.ipynb"
