#!/bin/bash

# Ensure virtual environment is activated if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

# Start the Flask application
flask run --host=0.0.0.0 --port=5000
