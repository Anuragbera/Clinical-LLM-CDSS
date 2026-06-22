import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


def generate_chat_response(user_query, clinical_results, chat_history=None):

    # =========================
    # Build conversation memory
    # =========================
    history_text = ""
    if chat_history:
        for role, msg in chat_history[-5:]:
            history_text += f"{role}: {msg}\n"

    # =========================
    # Prompt
    # =========================
    prompt = f"""
    You are a clinical AI assistant chatbot.

    Clinical Results:
    {clinical_results}

    Conversation History:
    {history_text}

    User Question:
    {user_query}

    Your tasks:
    1. Answer the user's question clearly
    2. Based on clinical results, ask 1 relevant follow-up question

    Rules:
    - Keep it simple and human-like
    - Do not overwhelm
    - Ask ONLY one question
    - Focus on symptoms or lifestyle

    Format:
    Answer:
    <your answer>

    Follow-up Question:
    <your question>
    """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    response_text = response.json().get("response", "")

    # =========================
    # Safety Layer
    # =========================
    if "Severe" in str(clinical_results):
        return "⚠️ This appears serious. Please consult a doctor immediately.\n\n" + response_text

    return response_text