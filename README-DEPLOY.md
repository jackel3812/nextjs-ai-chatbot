# Riley-AI

A Next.js chatbot with advanced AI capabilities powered by a modular Python backend.

## Features

- Modern Next.js frontend with React components
- Python backend with modular architecture
- Full conversation history with persistent memory
- Multiple chat modes (Inventor, Scientific analyst, etc.)
- Advanced reasoning and simulation capabilities
- Voice synthesis and recognition (planned)

## Architecture

- Frontend: Next.js 14+ with React
- Backend: Flask Python app with modular structure
- Database: SQLite for local storage

## Deployment

This app is deployed as a Docker container on Hugging Face Spaces.

## Setup

To run locally:
```bash
# Install dependencies
pip install -r riley_ai/requirements.txt
pnpm install

# Start both services
./app-start.sh
```
