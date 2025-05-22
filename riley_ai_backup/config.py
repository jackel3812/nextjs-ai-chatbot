from os import environ
from pathlib import Path

class Config:
    # Basic configuration
    SECRET_KEY = environ.get('SECRET_KEY', 'dev-key-please-change')
    BASE_DIR = Path(__file__).parent.absolute()
    
    # Flask settings
    FLASK_ENV = environ.get('FLASK_ENV', 'production')
    DEBUG = FLASK_ENV == 'development'
    
    # Server settings
    HOST = '0.0.0.0'
    PORT = int(environ.get('PORT', 5000))
    
    # Database settings
    SQLALCHEMY_DATABASE_URI = 'sqlite:///instance/riley.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # API Keys
    OPENAI_API_KEY = environ.get('OPENAI_API_KEY')
    
    # Cors settings
    CORS_ORIGINS = [
        'http://localhost:5000',
        'http://127.0.0.1:5000',
        environ.get('FRONTEND_URL', '*')
    ]
    
    # Riley AI Settings
    DEFAULT_AI_MODEL = "gpt-4"
    MAX_MEMORY_ENTRIES = 100
    ENABLE_VOICE = False  # Set to True to enable voice features
