#!/bin/bash

# SUNUM HAZIRLIGI - FINANCIAL SENTIMENT ANALYSIS
# Bu script sunumdan önce tüm hazırlıkları yapar

echo "=============================================================================="
echo "SUNUM HAZIRLIGI BAŞLIYOR"
echo "Tarih: $(date)"
echo "=============================================================================="

# Proje klasörüne git
cd /Users/metaboy/Desktop/2121251034_MEHMET_TAHA_BOYNIKOGLU

echo ""
echo "[1] Python versiyonu kontrol ediliyor..."
python3 --version

echo ""
echo "[2] Gerekli kütüphaneler kontrol ediliyor..."
pip3 list | grep -E "scikit-learn|pandas|numpy|matplotlib"

echo ""
echo "[3] Mevcut dosyalar kontrol ediliyor..."
echo "Dataset dosyaları:"
ls -lh data/processed/*.csv 2>/dev/null || echo "  UYARI: CSV dosyaları bulunamadı!"

echo ""
echo "Model dosyaları:"
ls -lh models/*.pkl 2>/dev/null || echo "  UYARI: Model dosyaları bulunamadı!"

echo ""
echo "Görseller:"
ls -lh figures/*.png 2>/dev/null || echo "  UYARI: Görsel dosyaları bulunamadı!"

echo ""
echo "[4] Proje durumu kontrol ediliyor..."
python3 reset_and_rebuild.py --status

echo ""
echo "=============================================================================="
echo "HAZIRLAMA SEÇENEKLERİ:"
echo "=============================================================================="
echo ""
echo "A) Sadece görselleri aç (hızlı)"
echo "B) Tüm projeyi yeniden oluştur (yavaş, 2-3 dakika)"
echo "C) Sadece durum kontrolü (hiçbir şey değişmez)"
echo ""
read -p "Seçiminiz (A/B/C): " choice

case $choice in
    A|a)
        echo ""
        echo "[A] Görseller açılıyor..."
        open figures/ 2>/dev/null || echo "figures/ klasörü bulunamadı!"
        echo "Hazır! Sunum için terminal açık kalacak."
        ;;
    B|b)
        echo ""
        echo "[B] Tüm proje yeniden oluşturuluyor..."
        echo "UYARI: Bu 2-3 dakika sürebilir!"
        read -p "Devam etmek istiyor musunuz? (E/H): " confirm
        if [[ $confirm == "E" || $confirm == "e" ]]; then
            python3 reset_and_rebuild.py --yes
            echo ""
            echo "Görseller açılıyor..."
            open figures/ 2>/dev/null
            echo "Tamamlandı! Sunum hazır."
        else
            echo "İşlem iptal edildi."
        fi
        ;;
    C|c)
        echo ""
        echo "[C] Sadece durum kontrolü yapıldı."
        echo "Değişiklik yapılmadı."
        ;;
    *)
        echo ""
        echo "Geçersiz seçim! Hiçbir işlem yapılmadı."
        ;;
esac

echo ""
echo "=============================================================================="
echo "SUNUM İÇİN HIZLI KOMUTLAR:"
echo "=============================================================================="
echo ""
echo "1. Dataset oluştur (canlı demo):"
echo "   python3 create_full_dataset.py"
echo ""
echo "2. Görselleri aç:"
echo "   open figures/"
echo ""
echo "3. Proje durumu:"
echo "   python3 reset_and_rebuild.py --status"
echo ""
echo "4. Tüm projeyi sıfırla ve yeniden oluştur:"
echo "   python3 reset_and_rebuild.py --yes"
echo ""
echo "=============================================================================="
echo "SUNUM İÇİN HAZIRSINIZ!"
echo "=============================================================================="
