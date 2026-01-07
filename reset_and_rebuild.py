#!/usr/bin/env python3
"""
Proje Sifirlama ve Yeniden Olusturma Script'i
=============================================

Bu script projedeki tum uretilen verileri siler ve isteğe bağlı olarak
yeniden olusturur.

Kullanim:
    python3 reset_and_rebuild.py          # Sadece sil
    python3 reset_and_rebuild.py --rebuild # Sil ve yeniden olustur
    python3 reset_and_rebuild.py --help    # Yardim
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def get_project_root():
    """Proje kok dizinini bul"""
    return Path(__file__).parent


def print_header(text):
    """Baslik yazdir"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_status(text, status="INFO"):
    """Durum mesaji yazdir"""
    symbols = {
        "INFO": "[*]",
        "OK": "[+]",
        "DEL": "[-]",
        "WARN": "[!]",
        "RUN": "[>]"
    }
    print(f"{symbols.get(status, '[*]')} {text}")


def delete_directory_contents(dir_path, extensions=None):
    """
    Dizin icerigini sil

    Args:
        dir_path: Silinecek dizin
        extensions: Sadece bu uzantilari sil (None = hepsini sil)
    """
    if not dir_path.exists():
        print_status(f"Dizin mevcut degil: {dir_path}", "WARN")
        return 0

    deleted_count = 0

    for item in dir_path.iterdir():
        if item.is_file():
            if extensions is None or item.suffix in extensions:
                try:
                    item.unlink()
                    print_status(f"Silindi: {item.name}", "DEL")
                    deleted_count += 1
                except Exception as e:
                    print_status(f"Silinemedi: {item.name} - {e}", "WARN")
        elif item.is_dir() and item.name != '.gitkeep':
            # Alt dizinleri de sil (pycache vb.)
            try:
                shutil.rmtree(item)
                print_status(f"Dizin silindi: {item.name}", "DEL")
                deleted_count += 1
            except Exception as e:
                print_status(f"Dizin silinemedi: {item.name} - {e}", "WARN")

    return deleted_count


def clean_project(root_path):
    """
    Projeyi temizle - uretilen tum dosyalari sil
    """
    print_header("PROJE TEMIZLEME BASLIYOR")

    total_deleted = 0

    # 1. Islenmi veriler (data/processed/)
    print("\n--- Islenmi Veriler (data/processed/) ---")
    processed_dir = root_path / "data" / "processed"
    total_deleted += delete_directory_contents(processed_dir, ['.csv', '.pkl'])

    # 2. Feature dosyalari (data/features/)
    print("\n--- Feature Dosyalari (data/features/) ---")
    features_dir = root_path / "data" / "features"
    total_deleted += delete_directory_contents(features_dir, ['.pkl', '.npy'])

    # 3. Egitilmi modeller (models/)
    print("\n--- Egitilmis Modeller (models/) ---")
    models_dir = root_path / "models"
    total_deleted += delete_directory_contents(models_dir, ['.pkl', '.joblib', '.h5'])

    # 4. Gorseller (figures/)
    print("\n--- Gorseller (figures/) ---")
    figures_dir = root_path / "figures"
    total_deleted += delete_directory_contents(figures_dir, ['.png', '.jpg', '.pdf', '.svg'])

    # 5. Gecici dosyalar
    print("\n--- Gecici Dosyalar ---")

    # __pycache__ dizinleri
    for pycache in root_path.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            print_status(f"Silindi: {pycache.relative_to(root_path)}", "DEL")
            total_deleted += 1
        except:
            pass

    # .pyc dosyalari
    for pyc in root_path.rglob("*.pyc"):
        try:
            pyc.unlink()
            print_status(f"Silindi: {pyc.name}", "DEL")
            total_deleted += 1
        except:
            pass

    # .DS_Store dosyalari
    for ds_store in root_path.rglob(".DS_Store"):
        try:
            ds_store.unlink()
            print_status(f"Silindi: .DS_Store", "DEL")
            total_deleted += 1
        except:
            pass

    print_header(f"TEMIZLIK TAMAMLANDI: {total_deleted} dosya/dizin silindi")

    return total_deleted


def rebuild_project(root_path):
    """
    Projeyi yeniden olustur
    """
    print_header("PROJE YENIDEN OLUSTURULUYOR")

    # Gerekli dizinlerin mevcut oldugundan emin ol
    dirs_to_create = [
        "data/processed",
        "data/features",
        "data/raw",
        "models",
        "figures"
    ]

    for dir_name in dirs_to_create:
        dir_path = root_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print_status(f"Dizin hazir: {dir_name}", "OK")

    # 1. Dataset olustur
    print("\n--- Adim 1: Dataset Olusturma ---")
    print_status("create_full_dataset.py calistiriliyor...", "RUN")

    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, "create_full_dataset.py"],
            cwd=root_path,
            capture_output=True,
            text=True,
            timeout=300  # 5 dakika timeout
        )

        if result.returncode == 0:
            print_status("Dataset basariyla olusturuldu!", "OK")
            # Son satiri goster
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    print(f"    {line}")
        else:
            print_status("Dataset olusturma hatasi!", "WARN")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print_status("Dataset olusturma zaman asimina ugradi!", "WARN")
        return False
    except Exception as e:
        print_status(f"Hata: {e}", "WARN")
        return False

    # 2. Model egitimi
    print("\n--- Adim 2: Model Egitimi ---")
    print_status("train_and_evaluate.py calistiriliyor...", "RUN")
    print_status("Bu islem birkaç dakika surebilir...", "INFO")

    try:
        result = subprocess.run(
            [sys.executable, "train_and_evaluate.py"],
            cwd=root_path,
            capture_output=True,
            text=True,
            timeout=600  # 10 dakika timeout
        )

        if result.returncode == 0:
            print_status("Model egitimi basariyla tamamlandi!", "OK")
            # Sonuclari goster
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                # Son 20 satiri goster (sonuclar)
                print("\n--- Egitim Sonuclari ---")
                for line in lines[-20:]:
                    print(f"    {line}")
        else:
            print_status("Model egitimi hatasi!", "WARN")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print_status("Model egitimi zaman asimina ugradi!", "WARN")
        return False
    except Exception as e:
        print_status(f"Hata: {e}", "WARN")
        return False

    print_header("YENIDEN OLUSTURMA TAMAMLANDI!")
    return True


def show_project_status(root_path):
    """
    Proje durumunu goster
    """
    print_header("PROJE DURUMU")

    # Kontrol edilecek dizinler ve beklenen dosya turleri
    checks = [
        ("data/raw", [".csv"], "Ham Veri (RSS)"),
        ("data/processed", [".csv"], "Islenmi Veri"),
        ("data/features", [".pkl"], "Feature Dosyalari"),
        ("models", [".pkl"], "Egitilmis Modeller"),
        ("figures", [".png"], "Gorseller")
    ]

    for dir_name, extensions, description in checks:
        dir_path = root_path / dir_name

        if not dir_path.exists():
            print_status(f"{description}: Dizin yok", "WARN")
            continue

        files = []
        for ext in extensions:
            files.extend(list(dir_path.glob(f"*{ext}")))

        if files:
            print_status(f"{description}: {len(files)} dosya", "OK")
            for f in files[:3]:  # Ilk 3 dosyayi goster
                print(f"      - {f.name}")
            if len(files) > 3:
                print(f"      ... ve {len(files) - 3} dosya daha")
        else:
            print_status(f"{description}: Bos", "WARN")


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description="Proje Sifirlama ve Yeniden Olusturma Araci",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python3 reset_and_rebuild.py              # Sadece temizle
  python3 reset_and_rebuild.py --rebuild    # Temizle ve yeniden olustur
  python3 reset_and_rebuild.py --status     # Proje durumunu goster
  python3 reset_and_rebuild.py --keep-raw   # Ham veriyi koru
        """
    )

    parser.add_argument(
        '--rebuild', '-r',
        action='store_true',
        help='Temizledikten sonra yeniden olustur'
    )

    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Sadece proje durumunu goster'
    )

    parser.add_argument(
        '--keep-raw',
        action='store_true',
        help='Ham veriyi (data/raw) silme'
    )

    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Onay istemeden devam et'
    )

    args = parser.parse_args()

    root_path = get_project_root()

    print("\n" + "=" * 60)
    print("  FINANSAL DUYGU ANALIZI - PROJE YONETIM ARACI")
    print("=" * 60)
    print(f"  Proje Dizini: {root_path}")

    # Sadece durum goster
    if args.status:
        show_project_status(root_path)
        return

    # Onay iste
    if not args.yes:
        print("\n[!] UYARI: Bu islem tum uretilmis verileri silecek!")
        if args.keep_raw:
            print("    (Ham veri korunacak)")

        response = input("\nDevam etmek istiyor musunuz? (e/h): ")
        if response.lower() not in ['e', 'evet', 'y', 'yes']:
            print("\nIslem iptal edildi.")
            return

    # Temizle
    clean_project(root_path)

    # Yeniden olustur
    if args.rebuild:
        success = rebuild_project(root_path)
        if success:
            print("\n" + "=" * 60)
            print("  TAMAMLANDI! Proje sifirdan yeniden olusturuldu.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("  HATA! Yeniden olusturma basarisiz.")
            print("=" * 60)
            sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("  TAMAMLANDI! Proje temizlendi.")
        print("  Yeniden olusturmak icin: python3 reset_and_rebuild.py --rebuild")
        print("=" * 60)

    # Final durum
    show_project_status(root_path)


if __name__ == "__main__":
    main()
