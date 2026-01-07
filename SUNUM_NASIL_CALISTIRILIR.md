# SUNUM HAZIRLIK REHBERİ

## Sunumdan 5 dakika önce:

### 1. Terminal'i aç ve proje klasörüne git:
```bash
cd /Users/metaboy/Desktop/2121251034_MEHMET_TAHA_BOYNIKOGLU
```

### 2. Hazırlık scriptini çalıştır:
```bash
bash sunum_hazirlik.sh
```

veya

```bash
chmod +x sunum_hazirlik.sh
./sunum_hazirlik.sh
```

### 3. Seçenekleri gör:
- **A**: Sadece görselleri aç (HIZLI - ÖNERİLEN)
- **B**: Tüm projeyi yeniden oluştur (2-3 dk sürer)
- **C**: Sadece durum kontrolü

### 4. Sunum sırasında canlı demo için:

Terminal'de şunu çalıştır:
```bash
python3 create_full_dataset.py
```

Bu komut:
- RSS'den veri toplamayı gösterir
- Class imbalance problemini gösterir
- Augmentation sürecini gösterir
- Train/val/test split'i gösterir

## HIZLI KOMUTLAR (Sunum sırasında)

```bash
# Durum kontrolü
python3 reset_and_rebuild.py --status

# Görselleri aç
open figures/

# Dataset oluştur (canlı demo)
python3 create_full_dataset.py

# Tüm projeyi sıfırla (acil durum)
python3 reset_and_rebuild.py --yes
```

## ÖNEMLİ NOTLAR:

1. **Canlı demo öncesi**: Mutlaka `bash sunum_hazirlik.sh` çalıştırın
2. **Görseller**: `figures/` klasöründe olmalı
3. **Terminal**: Sunum sırasında açık tutun
4. **Yedek plan**: Eğer canlı demo çalışmazsa, görselleri gösterin

## SORUN GİDERME:

### "Permission denied" hatası:
```bash
chmod +x sunum_hazirlik.sh
```

### Script çalışmıyor:
```bash
bash sunum_hazirlik.sh  # "bash" ile çalıştır
```

### Görseller açılmıyor:
```bash
cd figures
ls  # dosyaları listele
```

## BAŞARILAR! 🎉
