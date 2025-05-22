# Riley AI

A streamlined AI assistant built with Flask and OpenAI's GPT.

## Structure
```
riley_ai/
├── app.py              # Main Flask application
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── jarvis/            # Core AI functionality
├── routes/            # API endpoints
├── templates/         # HTML templates
├── static/            # CSS, JS, and assets
└── instance/          # Database and instance-specific files
```

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   export FLASK_ENV="development"
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## API Endpoints
- `/` - Main chat interface
- `/api/chat` - Chat endpoint
- `/api/invent` - Invention generation endpoint
