import openai
from config import Config

class AIEngine:
    def __init__(self):
        openai.api_key = Config.OPENAI_API_KEY

    def ask_riley(self, prompt, mode="default"):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"

ai_engine = AIEngine()
