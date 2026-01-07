#!/bin/bash

# GITHUB'DAN GEREKSIZ DOSYALARI SIL

cd /Users/metaboy/Desktop/2121251034_MEHMET_TAHA_BOYNIKOGLU

echo "Git'ten gereksiz dosyalar siliniyor..."

# Dosyaları git'ten kaldır
git rm -f "# SUNUM REHBERI - FINANCIAL SENTIMENT ANALYSIS.txt" 2>/dev/null
git rm -f "sunum_hazirlik.sh" 2>/dev/null

# Commit ve push
git add .
git commit -m "cleanup: removed unnecessary files, keeping only essential documentation"
git push origin main

echo "GitHub temizlendi!"
