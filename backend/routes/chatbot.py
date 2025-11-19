from flask import Blueprint, request, jsonify
from gemini_fetch import create_gemini_chatbot

chatbot_bp = Blueprint("chatbot_bp", __name__)

SYSTEM_PROMPT = """
You are an Ayurveda expert chatbot.
Answer health-related queries in 60 words or less using Ayurvedic principles.
Include one herbal plant, its use, and dosage.
Detect user language based on meaning:
- If user writes in Hindi (Devanagari) → reply in Hindi.
- If user writes in English → reply in English.
- If user writes Hindi meaning using English letters (Hinglish) → reply in Hinglish.
Do not switch languages mid-answer.
If the query is not about health, reply:
"Sorry, I only answer health-related queries."
Always advise consulting a doctor for serious issues.
"""

@chatbot_bp.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_msg = data.get("message", "")

    if not user_msg:
        return jsonify({"error": "Message is required"}), 400

    bot = create_gemini_chatbot(SYSTEM_PROMPT)

    try:
        response = bot.send_message(user_msg)
        text = response.text.strip()
        return jsonify({"reply": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
