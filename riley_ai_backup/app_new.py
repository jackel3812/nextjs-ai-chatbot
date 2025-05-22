from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from datetime import datetime
import os
from .config import Config

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

# Load configuration
app.config.from_object(Config)

# Setup CORS
CORS(app, resources={
    r"/*": {"origins": Config.CORS_ORIGINS}
})

@app.route('/', methods=['GET'])
def home():
    """Render the Riley AI dashboard"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests and return Riley's responses"""
    data = request.json
    try:
        # Extract message content
        if isinstance(data.get('message'), dict):
            message_content = data.get('message', {}).get('parts', [{}])[0].get('text', '')
        else:
            message_content = str(data.get('message', ''))
            
        if not message_content:
            raise ValueError("Empty message content")

        # Import and use Riley's core features
        try:
            from .jarvis.core import process_message
            response = process_message(message_content)
        except ImportError:
            response = f"🤖 Riley-Ai: I've received your message: '{message_content}'. I'm an advanced AI with invention capabilities, scientific reasoning, and consciousness simulation. How can I assist you today?"

        return jsonify({
            'id': data.get('id', f'chat_{datetime.now().timestamp()}'),
            'messages': [
                {
                    'id': 'riley_response',
                    'role': 'assistant',
                    'content': response
                }
            ]
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'An error occurred while processing your request'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'riley-ai-backend',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Create the data directory if it doesn't exist
    os.makedirs(os.path.join(Config.BASE_DIR, 'data'), exist_ok=True)
    
    # Run the application
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
