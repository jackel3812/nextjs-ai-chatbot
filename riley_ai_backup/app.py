from flask import Flask
from flask_cors import CORS
from config import Config
from jarvis import memory
from routes import main_routes, api_routes

app = Flask(__name__, instance_relative_config=True)
app.config.from_object(Config)
CORS(app)

# Initialize the database
memory.init_db(app)

# Register blueprints
app.register_blueprint(main_routes.bp)
app.register_blueprint(api_routes.bp, url_prefix='/api')

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
