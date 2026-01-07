#!/bin/bash

# GEREKSIZ DOSYALARI TEMIZLE

echo "=============================================================================="
echo "GEREKSIZ DOSYALAR SILINIYOR"
echo "=============================================================================="

cd /Users/metaboy/Desktop/2121251034_MEHMET_TAHA_BOYNIKOGLU

# Silinecek dosyalar
echo ""
echo "Siliniyor: sunum_hazirlik.sh"
rm -f sunum_hazirlik.sh

echo "Siliniyor: # SUNUM REHBERI - FINANCIAL SENTIMENT ANALYSIS.txt (eski text versiyonu)"
rm -f "# SUNUM REHBERI - FINANCIAL SENTIMENT ANALYSIS.txt"

echo ""
echo "=============================================================================="
echo "KALAN DOSYALAR:"
echo "=============================================================================="
ls -lh *.md *.pdf 2>/dev/null

echo ""
echo "Temizlik tamamlandı!"
