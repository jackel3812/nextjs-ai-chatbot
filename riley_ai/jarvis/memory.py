from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class MemoryLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prompt = db.Column(db.Text)
    response = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()

def log_conversation(prompt, response):
    entry = MemoryLog(prompt=prompt, response=response)
    db.session.add(entry)
    db.session.commit()

def get_recent_conversations(limit=10):
    return MemoryLog.query.order_by(MemoryLog.timestamp.desc()).limit(limit).all()
