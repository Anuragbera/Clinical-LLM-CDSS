
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


def generate_explanation(clinical_output):
    prompt = f"""
    You are a clinical AI assistant.

    Explain the following medical analysis in a patient-friendly way.

    Clinical Output:
    {clinical_output}

    Structure your answer into:
    1. Condition Overview
    2. Possible Causes
    3. Severity Interpretation
    4. What This Means For The Patient
    5. Immediate Next Steps
    6. When to Seek Urgent Care (if applicable)

    Guidelines:
    - Be clear and simple
    - Avoid medical jargon
    - Be accurate but not alarming
    - Always suggest consulting a healthcare professional
    """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json().get("response", "")
