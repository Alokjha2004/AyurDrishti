import os
import google.generativeai as genai

print(f"🔑 Gemini API Key exists: {bool(os.getenv('GEMINI_API_KEY'))}")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_use_from_gemini(scientific_name):
    print(f"🤖 Calling Gemini for: '{scientific_name}'")
    prompt = f"List the medicinal uses of the plant {scientific_name}. Keep the answer short and clear."

    try:
        # ✅ Use gemini-2.5-flash (available in your list)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        text = response.text.strip()
        print(f"✅ Gemini Response: {text[:100]}...")
        return text

    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return None
    
def create_gemini_chatbot(system_prompt):
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=system_prompt
    )
    chat = model.start_chat(history=[])
    return chat
