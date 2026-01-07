#!/bin/bash

echo "==============================================="
echo "Preparing project for GitHub upload..."
echo "==============================================="

# Step 1: Cleanup
echo ""
echo "[1/5] Cleaning unnecessary files..."
bash cleanup.sh

# Step 2: Initialize git (if not already)
if [ ! -d .git ]; then
    echo ""
    echo "[2/5] Initializing Git repository..."
    git init
    echo "✓ Git initialized"
else
    echo ""
    echo "[2/5] Git already initialized"
fi

# Step 3: Add files
echo ""
echo "[3/5] Adding files to Git..."
git add .
echo "✓ Files staged"

# Step 4: Commit
echo ""
echo "[4/5] Creating commit..."
read -p "Enter commit message (default: 'Initial commit - Financial Sentiment Analysis'): " commit_msg
commit_msg=${commit_msg:-"Initial commit - Financial Sentiment Analysis"}
git commit -m "$commit_msg"
echo "✓ Changes committed"

# Step 5: Instructions for GitHub
echo ""
echo "[5/5] GitHub Upload Instructions"
echo "==============================================="
echo ""
echo "MANUAL STEPS (do this on GitHub.com):"
echo "1. Go to https://github.com/new"
echo "2. Repository name: financial-sentiment-analysis"
echo "3. Description: ML & Deep Learning for Financial News Classification"
echo "4. Keep it PUBLIC (for portfolio)"
echo "5. Do NOT initialize with README (we have one)"
echo "6. Click 'Create repository'"
echo ""
echo "Then run these commands:"
echo ""
echo "  git remote add origin https://github.com/YOUR_USERNAME/financial-sentiment-analysis.git"
echo "  git branch -M main"
echo "  git push -u origin main"
echo ""
echo "==============================================="
echo "Project is ready for GitHub! 🚀"
echo "==============================================="
