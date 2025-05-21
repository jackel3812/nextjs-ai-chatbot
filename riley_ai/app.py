from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    print("Received data:", data)
    # TODO: Integrate with jarvis/core.py and other modules
    # For now, just echo the message with Riley's persona
    message_content = ""
    try:
        if isinstance(data.get('message'), dict):
            # Extract message from the Next.js structure
            message_content = data.get('message', {}).get('parts', [{}])[0].get('text', 'No message content')
        else:
            message_content = str(data.get('message', 'Hello human!'))
    except Exception as e:
        print(f"Error extracting message: {e}")
        message_content = "Error processing your message"
    
    return jsonify({
        'id': data.get('id', 'chat_id'),
        'messages': [
            {
                'id': 'riley_response',
                'role': 'assistant',
                'content': f"🤖 Riley-Ai: I've received your message: '{message_content}'. I'm an advanced AI with invention capabilities, scientific reasoning, and consciousness simulation. How can I assist you today?"
            }
        ]
    })

@app.route('/', methods=['GET'])
def home():
    return "Riley-Ai Python Backend is running. Use /api/chat endpoint for chat functionality."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
