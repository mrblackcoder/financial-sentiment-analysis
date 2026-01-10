#!/usr/bin/env python3
"""
Proje Temizleme ve ZIP Oluşturma Scripti

Bu script:
1. Gereksiz dosyaları ve AI kalıntılarını temizler
2. 3 ayrı ZIP dosyası oluşturur (her ekip üyesi için)
3. Sunum dosyalarını dahil eder

Kullanım: python create_submission_zips.py
"""

import os
import shutil
import zipfile
from pathlib import Path

# Proje dizini
PROJECT_DIR = Path('.')

# Ekip üyeleri bilgileri
TEAM_MEMBERS = [
    {'name': 'MEHMET_TAHA_BOYNIKOGLU', 'number': '2121251034'},
    {'name': 'MERVE_KEDERSIZ', 'number': '2221251045'},
    {'name': 'ELIF_HANDE_ARSLAN', 'number': '2121251021'},
]

# SİLİNECEK DOSYA VE KLASÖRLER (gereksiz + AI kalıntıları)
FILES_TO_DELETE = [
    # Python cache
    '__pycache__',
    '.pytest_cache',
    '*.pyc',
    '*.pyo',
    '.ipynb_checkpoints',
    
    # IDE ve editor
    '.vscode',
    '.idea',
    '*.swp',
    '*.swo',
    '.DS_Store',
    'Thumbs.db',
    
    # Virtual environment
    'venv',
    '.venv',
    'env',
    '.env',
    
    # AI kalıntıları ve geçici dosyalar
    '.aider*',
    '.cursor*',
    '.copilot*',
    'CLAUDE_*',
    'AI_*',
    '*_backup*',
    '*.bak',
    '*.tmp',
    '*.temp',
    '*~',
    
    # Log dosyaları
    '*.log',
    'logs/',
    
    # Test ve debug dosyaları
    'test_*.py',
    '*_test.py',
    'debug_*.py',
    'scratch_*.py',
    'temp_*.py',
    
    # Eski/kullanılmayan dosyalar
    'old_*',
    'unused_*',
    'deprecated_*',
    
    # ZIP dosyaları (yeniden oluşturacağız)
    '*.zip',
    
    # Jupyter notebook checkpoints
    '.ipynb_checkpoints/',
]

# TUTULACAK DOSYALAR (önemli proje dosyaları)
FILES_TO_KEEP = [
    # Ana scriptler
    'create_full_dataset.py',
    'train_and_evaluate.py',
    'demo_notebook.ipynb',
    'requirements.txt',
    'README.md',
    
    # Sunum dosyaları
    'KONUSMA_METNI.md',
    'SLAYT_ICERIKLERI.md',
    'SUNUM_REHBERI.md',
    'FINANCIAL_SENTIMENT_ANALYSIS_REPORT.pdf',
    
    # Kaynak kodlar
    'src/',
    
    # Veri (gerçek scraping verileri dahil)
    'data/',
    
    # Eğitilmiş modeller
    'models/',
    
    # Görseller
    'figures/',
]

def clean_directory(directory: Path):
    """Gereksiz dosyaları temizle"""
    deleted_count = 0
    
    for pattern in FILES_TO_DELETE:
        # Glob pattern ile eşleşen dosyaları bul
        if '*' in pattern:
            for item in directory.rglob(pattern):
                try:
                    if item.is_file():
                        item.unlink()
                        print(f"  [DELETED] {item}")
                        deleted_count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        print(f"  [DELETED] {item}/")
                        deleted_count += 1
                except Exception as e:
                    print(f"  [ERROR] {item}: {e}")
        else:
            # Tam eşleşme
            item = directory / pattern
            if item.exists():
                try:
                    if item.is_file():
                        item.unlink()
                        print(f"  [DELETED] {item}")
                        deleted_count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        print(f"  [DELETED] {item}/")
                        deleted_count += 1
                except Exception as e:
                    print(f"  [ERROR] {item}: {e}")
    
    # Alt dizinlerde de temizlik yap
    for subdir in directory.iterdir():
        if subdir.is_dir() and subdir.name not in ['.git', 'data', 'models', 'figures', 'src']:
            for pattern in ['__pycache__', '.ipynb_checkpoints', '.DS_Store']:
                for item in subdir.rglob(pattern):
                    try:
                        if item.is_file():
                            item.unlink()
                            deleted_count += 1
                        elif item.is_dir():
                            shutil.rmtree(item)
                            deleted_count += 1
                    except:
                        pass
    
    return deleted_count

def get_files_to_zip(directory: Path):
    """ZIP'e eklenecek dosyaların listesini döndür"""
    files_to_zip = []
    
    # Ana scriptler ve dosyalar
    important_files = [
        'create_full_dataset.py',
        'train_and_evaluate.py', 
        'demo_notebook.ipynb',
        'requirements.txt',
        'README.md',
        # Sunum dosyaları
        'KONUSMA_METNI.md',
        'SLAYT_ICERIKLERI.md',
        'SUNUM_REHBERI.md',
    ]
    
    for f in important_files:
        path = directory / f
        if path.exists():
            files_to_zip.append(path)
    
    # PDF raporlar ve sunumlar
    for pdf in directory.glob('*.pdf'):
        files_to_zip.append(pdf)
    
    # PowerPoint sunumlar
    for pptx in directory.glob('*.pptx'):
        files_to_zip.append(pptx)
    for ppt in directory.glob('*.ppt'):
        files_to_zip.append(ppt)
    
    # src klasörü
    src_dir = directory / 'src'
    if src_dir.exists():
        for f in src_dir.rglob('*.py'):
            if '__pycache__' not in str(f):
                files_to_zip.append(f)
    
    # data klasörü
    data_dir = directory / 'data'
    if data_dir.exists():
        # processed CSV'ler
        processed_dir = data_dir / 'processed'
        if processed_dir.exists():
            for f in processed_dir.glob('*.csv'):
                files_to_zip.append(f)
        # raw veriler
        raw_dir = data_dir / 'raw'
        if raw_dir.exists():
            for f in raw_dir.glob('*.csv'):
                files_to_zip.append(f)
        # features (vectorizer'lar)
        features_dir = data_dir / 'features'
        if features_dir.exists():
            for f in features_dir.glob('*.pkl'):
                files_to_zip.append(f)
    
    # models klasörü
    models_dir = directory / 'models'
    if models_dir.exists():
        for f in models_dir.glob('*.pkl'):
            files_to_zip.append(f)
    
    # figures klasörü
    figures_dir = directory / 'figures'
    if figures_dir.exists():
        for f in figures_dir.glob('*.png'):
            files_to_zip.append(f)
        for f in figures_dir.glob('*.jpg'):
            files_to_zip.append(f)
        for f in figures_dir.glob('*.jpeg'):
            files_to_zip.append(f)
    
    return files_to_zip

def create_zip(member: dict, directory: Path, files: list):
    """Bir ekip üyesi için ZIP oluştur"""
    zip_name = f"{member['number']}_{member['name']}.zip"
    zip_path = directory.parent / zip_name
    
    print(f"\n  Creating {zip_name}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Proje klasör adı
        folder_name = f"{member['number']}_{member['name']}"
        
        for file_path in files:
            # Relative path hesapla
            rel_path = file_path.relative_to(directory)
            # ZIP içindeki yol
            arcname = f"{folder_name}/{rel_path}"
            
            try:
                zf.write(file_path, arcname)
            except Exception as e:
                print(f"    [WARNING] Could not add {file_path}: {e}")
        
        # Dosya sayısını göster
        print(f"    Added {len(zf.namelist())} files")
    
    # ZIP boyutunu göster
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"    Size: {size_mb:.2f} MB")
    print(f"    Location: {zip_path}")
    
    return zip_path

def main():
    print("=" * 70)
    print("PROJE TEMİZLEME VE ZIP OLUŞTURMA")
    print("=" * 70)
    
    # 1. Temizlik
    print("\n[1/3] Gereksiz dosyaları temizliyorum...")
    deleted = clean_directory(PROJECT_DIR)
    print(f"\n  Toplam {deleted} dosya/klasör silindi")
    
    # 2. ZIP'e eklenecek dosyaları topla
    print("\n[2/3] ZIP'e eklenecek dosyaları topluyorum...")
    files = get_files_to_zip(PROJECT_DIR)
    print(f"  {len(files)} dosya bulundu")
    
    # Dosya listesini kategorilere göre göster
    print("\n  Dahil edilen dosyalar:")
    
    # Kategorize et
    scripts = [f for f in files if f.suffix == '.py']
    notebooks = [f for f in files if f.suffix == '.ipynb']
    docs = [f for f in files if f.suffix in ['.md', '.pdf', '.pptx', '.ppt']]
    data = [f for f in files if 'data' in str(f)]
    models = [f for f in files if 'models' in str(f)]
    figures = [f for f in files if 'figures' in str(f)]
    
    print(f"    Python Scripts: {len(scripts)}")
    print(f"    Notebooks: {len(notebooks)}")
    print(f"    Docs/Sunum: {len(docs)}")
    for d in docs:
        print(f"      - {d.name}")
    print(f"    Data files: {len(data)}")
    print(f"    Model files: {len(models)}")
    print(f"    Figure files: {len(figures)}")
    
    # 3. ZIP'leri oluştur
    print("\n[3/3] ZIP dosyalarını oluşturuyorum...")
    
    created_zips = []
    for member in TEAM_MEMBERS:
        zip_path = create_zip(member, PROJECT_DIR, files)
        created_zips.append(zip_path)
    
    # Özet
    print("\n" + "=" * 70)
    print("TAMAMLANDI!")
    print("=" * 70)
    print("\nOluşturulan ZIP dosyaları:")
    for i, (member, zip_path) in enumerate(zip(TEAM_MEMBERS, created_zips), 1):
        print(f"  {i}. {zip_path.name}")
        print(f"     İsim: {member['name'].replace('_', ' ')}")
        print(f"     Numara: {member['number']}")
    
    print(f"\nZIP dosyaları şu konumda: {PROJECT_DIR.parent.absolute()}")
    print("\n" + "=" * 70)
    print("ZIP İÇERİĞİ:")
    print("=" * 70)
    print("  ✓ demo_notebook.ipynb (Ana demo notebook)")
    print("  ✓ create_full_dataset.py (Dataset oluşturma)")
    print("  ✓ train_and_evaluate.py (Model eğitimi)")
    print("  ✓ requirements.txt (Bağımlılıklar)")
    print("  ✓ README.md (Proje açıklaması)")
    print("  ✓ KONUSMA_METNI.md (Sunum konuşma metni)")
    print("  ✓ SLAYT_ICERIKLERI.md (Slayt içerikleri)")
    print("  ✓ SUNUM_REHBERI.md (Sunum rehberi)")
    print("  ✓ *.pdf (PDF raporlar/sunumlar)")
    print("  ✓ *.pptx (PowerPoint sunumlar)")
    print("  ✓ src/ (Kaynak kodlar)")
    print("  ✓ data/processed/*.csv (İşlenmiş veriler)")
    print("  ✓ data/raw/*.csv (Ham veriler - RSS)")
    print("  ✓ data/features/*.pkl (Feature extractors)")
    print("  ✓ models/*.pkl (Eğitilmiş modeller)")
    print("  ✓ figures/*.png (Görseller)")
    print("\nTeslim için her ekip üyesi kendi ZIP dosyasını yükleyebilir.")

if __name__ == '__main__':
    main()
