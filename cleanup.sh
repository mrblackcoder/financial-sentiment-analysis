#!/bin/bash

echo "Cleaning up unnecessary files for GitHub upload..."

# Remove Mac system files
find . -name ".DS_Store" -delete
echo "✓ Removed .DS_Store files"

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
echo "✓ Removed Python cache files"

# Remove Jupyter checkpoints
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null
echo "✓ Removed Jupyter checkpoints"

# Remove old/backup files
rm -f *.bak *.tmp *.old 2>/dev/null
rm -f *~ 2>/dev/null
echo "✓ Removed backup files"

# Keep only necessary model files (best model)
# Remove individual model files, keep only the combined results
# (We'll regenerate them if needed)

echo ""
echo "Cleanup complete! Ready for GitHub."
