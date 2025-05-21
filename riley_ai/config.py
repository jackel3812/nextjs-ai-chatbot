# Configuration for Riley-Ai backend
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
    # Add API keys, environment variables, etc. here

# Example usage:
# from config import Config
# app.config.from_object(Config)
