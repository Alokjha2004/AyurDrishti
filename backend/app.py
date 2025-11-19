from dotenv import load_dotenv

load_dotenv()
from flask import Flask, jsonify, request
from flask_cors import CORS
from routes.predict import predict_plant
from db import get_use_from_db, store_use_to_db
from gemini_fetch import get_use_from_gemini
from routes.chatbot import chatbot_bp

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Register Blueprint
app.register_blueprint(predict_plant)
app.register_blueprint(chatbot_bp)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Backend running!"})

# If frontend ever wants to fetch uses separately
@app.route("/api/get-plant-uses", methods=["POST"])
def fetch_uses():
    data = request.json
    sci = data.get("scientific_name")

    if not sci:
        return jsonify({"error": "scientific_name required"}), 400

    sci = sci.strip()

    uses = get_use_from_db(sci)
    if uses:
        return jsonify({
            "source": "database",
            "scientific_name": sci,
            "uses": uses
        })

    # Fetch from Gemini if not found
    uses = get_use_from_gemini(sci)
    if not uses:
        return jsonify({"error": "Gemini error"}), 500

    store_use_to_db(sci, uses)

    return jsonify({
        "source": "gemini",
        "scientific_name": sci,
        "uses": uses
    })

if __name__ == "__main__":
    app.run(debug=True)
