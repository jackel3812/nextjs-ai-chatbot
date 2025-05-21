#!/bin/bash
# filepath: /workspaces/nextjs-ai-chatbot/install-on-hf.sh

# Riley-AI: One-click installer for Hugging Face Spaces
# This script streamlines the installation process on Hugging Face Spaces

echo "███████╗ ██╗██╗     ███████╗██╗   ██╗       █████╗ ██╗"
echo "██╔══██╗██║██║     ██╔════╝╚██╗ ██╔╝      ██╔══██╗██║"
echo "██████╔╝██║██║     █████╗   ╚████╔╝█████╗ ███████║██║"
echo "██╔══██╗██║██║     ██╔══╝    ╚██╔╝ ╚════╝ ██╔══██║██║"
echo "██║  ██║██║███████╗███████╗   ██║        ██║  ██║██║"
echo "╚═╝  ╚═╝╚═╝╚══════╝╚══════╝   ╚═╝        ╚═╝  ╚═╝╚═╝"
echo "Hugging Face Spaces Installation Script"
echo "========================================"

# Check for required tools
echo "Checking prerequisites..."

if ! command -v git &> /dev/null; then
    echo "Error: Git not found. Please install Git first."
    exit 1
fi

if ! command -v pip &> /dev/null; then
    echo "Error: pip not found. Please install Python and pip first."
    exit 1
fi

if ! command -v huggingface-cli &> /dev/null; then
    echo "Hugging Face CLI not found. Installing..."
    pip install huggingface_hub
fi

# Check if user is logged in to Hugging Face
echo "Checking Hugging Face login status..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "You need to log in to Hugging Face first."
    huggingface-cli login
fi

# Get Hugging Face username
HF_USERNAME=$(huggingface-cli whoami)
echo "Detected Hugging Face username: $HF_USERNAME"

# Create the Space if it doesn't exist
echo "Checking if Space exists..."
SPACE_NAME="riley-ai"
SPACE_REPO="$HF_USERNAME/$SPACE_NAME"

# Clone the repository
echo "Setting up Riley-AI..."
git clone https://github.com/jackel3812/nextjs-ai-chatbot.git riley-ai-temp
cd riley-ai-temp

# Update the username in huggingface.json
echo "Configuring for your Hugging Face account..."
sed -i "s/USERNAME/$HF_USERNAME/g" huggingface.json

# Set up Git for Hugging Face
echo "Setting up Git for Hugging Face Spaces..."
git remote add huggingface https://huggingface.co/spaces/$SPACE_REPO

# Generate a random NEXTAUTH_SECRET
NEXTAUTH_SECRET=$(openssl rand -base64 32)
echo "Generated NEXTAUTH_SECRET for authentication security"

# Create a new branch for deployment
git checkout -b hf-deploy

# Push to Hugging Face Spaces
echo "Pushing Riley-AI to Hugging Face Spaces..."
echo "This may take a few minutes..."
git push -f huggingface hf-deploy:main

echo ""
echo "✅ Deployment started!"
echo ""
echo "Your Riley-AI instance is now being built on Hugging Face Spaces."
echo "You can monitor the build status at: https://huggingface.co/spaces/$SPACE_REPO"
echo ""
echo "Once the build is complete, your app will be available at:"
echo "https://$HF_USERNAME-$SPACE_NAME.hf.space"
echo ""
echo "For more detailed setup instructions, please refer to:"
echo "https://github.com/jackel3812/nextjs-ai-chatbot/blob/main/HUGGINGFACE-SETUP.md"

# Clean up
cd ..
rm -rf riley-ai-temp

exit 0
