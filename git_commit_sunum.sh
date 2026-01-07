#!/bin/bash

# GitHub'a sunum dosyalarını commit et

cd /Users/metaboy/Desktop/2121251034_MEHMET_TAHA_BOYNIKOGLU

echo "=============================================================================="
echo "GIT COMMIT - SUNUM DOSYALARI"
echo "=============================================================================="

# Git durumunu kontrol et
echo ""
echo "[1] Git durumu kontrol ediliyor..."
git status

# Sunum dosyalarını ekle
echo ""
echo "[2] Sunum dosyaları ekleniyor..."

# Markdown sunum rehberi
git add "# SUNUM REHBERI - FINANCIAL SENTIMENT ANALYSIS.md" 2>/dev/null || echo "Markdown dosyası bulunamadı"

# Plaintext sunum rehberi
git add "# SUNUM REHBERI - FINANCIAL SENTIMENT ANALYSIS.txt" 2>/dev/null || echo "Text dosyası bulunamadı"

# Sunum hazırlık scripti
git add sunum_hazirlik.sh 2>/dev/null || echo "Hazırlık scripti bulunamadı"

# Proje gereksinimi dosyası
git add "Learning from Data - Final Project.md" 2>/dev/null || echo "Proje dosyası bulunamadı"

echo ""
echo "[3] Eklenen dosyalar:"
git status --short

# Commit yap
echo ""
echo "[4] Commit yapılıyor..."
git commit -m "feat: Add presentation guide and project requirements

- Add comprehensive Turkish presentation guide (Markdown + Text)
- Add presentation preparation script (sunum_hazirlik.sh)
- Add project requirements document
- Include detailed explanations for technical terms
- Include Q&A preparation section
- Include code examples for demonstration"

# Push yap
echo ""
echo "[5] GitHub'a gönderiliyor..."
read -p "GitHub'a push yapmak istiyor musunuz? (E/H): " confirm

if [[ $confirm == "E" || $confirm == "e" ]]; then
    git push origin main
    echo ""
    echo "✓ GitHub'a başarıyla gönderildi!"
else
    echo ""
    echo "Push iptal edildi. İsterseniz manuel olarak yapabilirsiniz:"
    echo "  git push origin main"
fi

echo ""
echo "=============================================================================="
echo "TAMAMLANDI!"
echo "=============================================================================="
