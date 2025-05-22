from flask import Blueprint, request, jsonify
from jarvis import ai_engine, invention, memory, modes

bp = Blueprint("api", __name__)

@bp.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("message", "")
    response = ai_engine.ask_riley(prompt)
    memory.log_conversation(prompt, response)
    return jsonify({"response": response})

@bp.route("/invent", methods=["GET"])
def invent():
    idea = invention.generate_invention_idea()
    return jsonify({"idea": idea})

@bp.route("/joke", methods=["GET"])
def joke():
    return jsonify({"joke": modes.get_joke()})
