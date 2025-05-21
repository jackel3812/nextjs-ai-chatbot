#!/bin/bash

# This script prepares and pushes Riley-AI to Hugging Face Spaces
# Make sure you've authenticated with Hugging Face CLI first: huggingface-cli login

# Check for required tools
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "Git not found. Please install Git first."
    exit 1
fi

# Variables (change these to match your Hugging Face username)
HF_USERNAME=$(huggingface-cli whoami 2>/dev/null || echo "YOUR_HF_USERNAME")  # Auto-detect or use placeholder
if [ "$HF_USERNAME" = "YOUR_HF_USERNAME" ]; then
    echo "Please replace YOUR_HF_USERNAME in this script with your actual Hugging Face username"
    echo "Or log in with: huggingface-cli login"
    exit 1
fi
SPACE_NAME="riley-ai"
SPACE_REPO="$HF_USERNAME/$SPACE_NAME"

# Check if Hugging Face CLI is installed and logged in
if ! command -v huggingface-cli &> /dev/null; then
    echo "Hugging Face CLI not found. Installing..."
    pip install huggingface_hub
fi

# Update username in huggingface.json
sed -i "s/USERNAME/$HF_USERNAME/g" huggingface.json

echo "=== Preparing Riley-AI for Hugging Face Spaces ==="
echo "This will create a Spaces repository at: https://huggingface.co/spaces/$SPACE_REPO"

# Add huggingface remote if it doesn't exist
if ! git remote | grep -q "huggingface"; then
    echo "Adding Hugging Face remote..."
    git remote add huggingface https://huggingface.co/spaces/$SPACE_REPO
fi

# Create a new branch for deployment
git checkout -b hf-deploy

# Push to Hugging Face Spaces
echo "=== Pushing to Hugging Face Spaces ==="
echo "This may take a few minutes..."
git push -f huggingface hf-deploy:main

echo "=== Deployment Complete ==="
echo "Your Riley-AI is now being built on Hugging Face Spaces."
echo "You can view the build status at: https://huggingface.co/spaces/$SPACE_REPO"
echo "Once the build is complete, your app will be available at: https://$HF_USERNAME-$SPACE_NAME.hf.space"

# Switch back to main branch
git checkout main
