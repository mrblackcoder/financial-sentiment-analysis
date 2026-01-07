"""Pre-upload checklist and file size verification"""

import os
from pathlib import Path

def human_readable_size(size):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def check_file_sizes(directory='.'):
    """Check for large files (>100MB)"""
    large_files = []
    total_size = 0
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common excludes
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
        
        for file in files:
            if file.startswith('.'):
                continue
                
            filepath = os.path.join(root, file)
            try:
                size = os.path.getsize(filepath)
                total_size += size
                
                if size > 100 * 1024 * 1024:  # 100MB
                    large_files.append((filepath, size))
            except:
                pass
    
    return large_files, total_size

def check_required_files():
    """Check if all required files exist"""
    required = [
        'README.md',
        'requirements.txt',
        '.gitignore',
        'create_full_dataset.py',
        'train_and_evaluate.py',
        'reset_and_rebuild.py',
        'src/data/real_scraper.py',
        'src/data/sentiment_labeler.py',
        'src/data/augmentation.py',
    ]
    
    missing = []
    for file in required:
        if not Path(file).exists():
            missing.append(file)
    
    return missing

def main():
    print("=" * 70)
    print("PRE-UPLOAD CHECKLIST")
    print("=" * 70)
    
    # Check required files
    print("\n[1] Checking required files...")
    missing = check_required_files()
    if missing:
        print("  ❌ Missing files:")
        for f in missing:
            print(f"     - {f}")
    else:
        print("  ✅ All required files present")
    
    # Check file sizes
    print("\n[2] Checking file sizes...")
    large_files, total_size = check_file_sizes()
    
    print(f"  Total repository size: {human_readable_size(total_size)}")
    
    if large_files:
        print("  ⚠️  Large files found (>100MB):")
        for filepath, size in large_files:
            print(f"     - {filepath}: {human_readable_size(size)}")
        print("\n  Consider adding these to .gitignore or using Git LFS")
    else:
        print("  ✅ No files >100MB (GitHub friendly)")
    
    # Check .gitignore
    print("\n[3] Checking .gitignore...")
    if Path('.gitignore').exists():
        print("  ✅ .gitignore exists")
    else:
        print("  ❌ .gitignore missing - create it!")
    
    # Check documentation
    print("\n[4] Checking documentation...")
    readme = Path('README.md')
    if readme.exists():
        size = readme.stat().st_size
        if size > 1000:  # At least 1KB
            print(f"  ✅ README.md exists ({human_readable_size(size)})")
        else:
            print("  ⚠️  README.md exists but seems too short")
    else:
        print("  ❌ README.md missing")
    
    # Final recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    recommendations = []
    
    if total_size > 100 * 1024 * 1024:  # >100MB total
        recommendations.append("Repository is large. Consider excluding data/features/*.pkl from Git.")
    
    if not Path('.gitignore').exists():
        recommendations.append("Create .gitignore file to exclude unnecessary files.")
    
    if missing:
        recommendations.append("Add missing required files before upload.")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("✅ Repository is ready for GitHub upload!")
    
    print("\n" + "=" * 70)
    print("Next step: bash upload_to_github.sh")
    print("=" * 70)

if __name__ == '__main__':
    main()
