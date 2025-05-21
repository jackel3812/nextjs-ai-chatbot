#!/bin/bash

# Helper script to check Hugging Face Space environment and set up Riley-AI
# This script is intended to be run inside the Hugging Face Space environment

echo "🤖 Setting up Riley-AI components..."

# Check if we're in a Hugging Face Space environment
if [ -n "$SPACE_ID" ]; then
  echo "Detected Hugging Face Space environment: $SPACE_ID"
  
  # Set environment variables
  export NEXT_PUBLIC_BACKEND_URL="http://localhost:5000"
  
  # Create an .env.local file if it doesn't exist
  if [ ! -f .env.local ]; then
    echo "Creating .env.local file"
    cat > .env.local << EOL
NEXT_PUBLIC_BACKEND_URL=http://localhost:5000
NEXTAUTH_URL=${NEXTAUTH_URL:-http://localhost:3000}
NEXTAUTH_SECRET=${NEXTAUTH_SECRET:-$(openssl rand -base64 32)}
NODE_ENV=production
EOL
  fi
  
  # Ensure Python requirements are installed
  if [ -f riley_ai/requirements.txt ]; then
    echo "Installing Python requirements"
    pip install -r riley_ai/requirements.txt
  fi
  
  # Check if Python modules are functional
  echo "Verifying Python backend..."
  python -c "import sys; sys.path.append('riley_ai'); from riley_ai.app import app; print('✅ Python backend verification successful')" || echo "⚠️ Warning: Python backend verification failed"
  
  echo "✅ Riley-AI setup complete!"
  echo "Starting services with app-start.sh"
else
  echo "Not running in Hugging Face Space environment"
  echo "For local development, use ./start.sh instead"
fi
