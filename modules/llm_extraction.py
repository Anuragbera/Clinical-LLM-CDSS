import re
import json
import requests
from modules.normalization import normalize_lab_keys

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


# 1️ LLM CALL (SYMPTOMS ONLY)

def call_llm(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    if "response" not in data:
        print("\n[LLM ERROR]:", data)
        return ""

    return data["response"]


def extract_json(text):
    try:
        match = re.search(r"\{.*}", text, re.DOTALL)
        if not match:
            return None

        json_str = match.group(0)

        # Remove trailing commas
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        return json.loads(json_str)

    except Exception as e:
        print("JSON parsing failed:", e)
        return None


# 2️ REGEX LAB EXTRACTION (STABLE)


def extract_labs_regex(text):
    text = text.lower().replace("\n", " ")

    patterns = {
        # ---------- ANEMIA ----------
        "HGB": r"(hemoglobin|hgb|hb)\s*(is)?\s*([\d.]+)",
        "FERRITTE": r"(ferritin|ferritte)\s*(is)?\s*([\d.]+)",
        "MCV": r"(mcv)\s*(is)?\s*([\d.]+)",
        "RBC": r"(rbc)\s*(is)?\s*([\d.]+)",
        "RDW": r"(rdw)\s*(is)?\s*([\d.]+)",
        "MCH": r"(mch)\s*(is)?\s*([\d.]+)",
        "MCHC": r"(mchc)\s*(is)?\s*([\d.]+)",
        "HCT": r"(hct|hematocrit)\s*(is)?\s*([\d.]+)",
        "B12": r"(vitamin b12|b12)\b[^\d]{0,10}(\d{1,3}(?:\.\d{1,2})?)\b",
        "FOLATE": r"(folate)[^\d]{0,20}([\d]+\.?\d*)",

        # ---------- LIVER ----------
        "Age": r"(age)\s*(is)?\s*([\d.]+)",
        "BMI": r"(bmi)\s*(is)?\s*([\d.]+)",
        "LFT - Total Bilirubin (mg/dL)": r"(total bilirubin)\s*(is)?\s*([\d.]+)",
        "LFT - Albumin (g/dL)": r"(albumin)\s*(is)?\s*([\d.]+)",
        "PT-INR": r"(inr)\s*(is)?\s*([\d.]+)",
        "LFT - SGOT (AST) (U/L)": r"(ast)\s*(is)?\s*([\d.]+)",
        "LFT - SGPT (ALT) (U/L)": r"(alt)\s*(is)?\s*([\d.]+)",
        "LFT - Alkaline Phosphatase (U/L)": r"(alp)\s*(is)?\s*([\d.]+)",
        "RFT - Urea (mg/dL)": r"(urea)\s*(is)?\s*([\d.]+)",
        "RFT - Creatinine (mg/dL)": r"(creatinine)\s*(is)?\s*([\d.]+)",
    }

    labs = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.groups()[-1]
            value = value.strip().rstrip(".,;")
            labs[key] = float(value)

    # ==========================
    # ASCITES
    # ==========================
    if "ascites" in text:
        if "mild" in text:
            labs["Ascites"] = "mild"
        elif "moderate" in text:
            labs["Ascites"] = "moderate"
        elif "severe" in text:
            labs["Ascites"] = "severe"
        else:
            labs["Ascites"] = "present"

    # ==========================
    # ENCEPHALOPATHY (ROBUST)
    # ==========================

    text_lower = text.lower()

    # Case 1: No encephalopathy
    if "no encephalopathy" in text_lower:
        labs["Encephalopathy"] = "none"

    else:
        # Match grade with flexible patterns
        match = re.search(r"grade[\s:\-]*([0-4])", text_lower)

        if match:
            labs["Encephalopathy"] = f"grade {match.group(1)}"

        else:
            # Optional: detect general mention
            if "encephalopathy" in text_lower:
                labs["Encephalopathy"] = "grade 1"  # assume mild if unspecified

    return labs


# ======================================
# LLM SYMPTOM EXTRACTION
# ======================================

def extract_symptoms_llm(text):
    cleaned = text.replace("\n", " ")

    prompt = f"""
Extract ONLY symptoms explicitly mentioned.
Do NOT infer anything.

Return STRICT JSON ONLY.
No explanation, no extra text.

Format:
{{"symptoms": ["symptom1", "symptom2"]}}

Text:
{cleaned}
"""

    raw = call_llm(prompt)
    parsed = extract_json(raw)

    # =========================================
    # CASE 1: LLM FAILED → GENERIC FALLBACK
    # =========================================
    if parsed is None:
        return fallback_symptom_extraction(text)

    symptoms = parsed.get("symptoms", [])

    # =========================================
    # NORMALIZE OUTPUT
    # =========================================
    clean_symptoms = []

    for s in symptoms:

        # dict → {"name": "fatigue"}
        if isinstance(s, dict):
            val = s.get("name") or s.get("symptom")
            if val:
                clean_symptoms.append(val.lower())

        # string → "fatigue"
        elif isinstance(s, str):
            clean_symptoms.append(s.lower())

    # remove duplicates
    clean_symptoms = list(set(clean_symptoms))

    # =========================================
    # IF LLM RETURNS EMPTY → FALLBACK
    # =========================================
    if not clean_symptoms:
        return fallback_symptom_extraction(text)

    return clean_symptoms


# =========================================
# GENERIC FALLBACK (NO HARDCODING)
# =========================================
def fallback_symptom_extraction(text):
    text = text.lower()

    # remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    words = text.split()

    # remove non-symptom words (basic filtering)
    stopwords = {
        "my", "is", "are", "was", "were", "and", "or",
        "the", "a", "an", "i", "feel", "have",
        "with", "of", "to"
    }

    # keep meaningful words only
    candidates = [
        w for w in words
        if w not in stopwords and len(w) > 3
    ]

    # remove numeric tokens
    candidates = [w for w in candidates if not w.isdigit()]

    # return unique words (approx symptoms)
    return list(set(candidates))


# ======================================
# MAIN FUNCTION
# ======================================
def extract_structured_data(user_text):
    # Extract labs (regex)
    raw_labs = extract_labs_regex(user_text)

    # Normalize + encode categorical safely
    labs_numeric, labs_display = normalize_lab_keys(raw_labs)

    # Extract symptoms via LLM
    symptoms = extract_symptoms_llm(user_text)

    # Remove ascites & encephalopathy duplication from symptoms
    clean_symptoms = []
    for s in symptoms:
        if "ascites" not in s.lower() and "encephalopathy" not in s.lower():
            clean_symptoms.append(s)

    # DEBUG PRINT (IMPORTANT)
    print("\n[DEBUG] Extracted Raw Labs:", raw_labs)
    print("[DEBUG] Normalized Numeric Labs:", labs_numeric)
    print("[DEBUG] Symptoms:", clean_symptoms)

    return {
        "labs": labs_display,  # human readable
        "labs_numeric": labs_numeric,  # numeric for ML
        "symptoms": clean_symptoms
    }

def extract_labs_from_ocr_llm(ocr_text):
    prompt = f"""
    You are a clinical lab extraction system.

    The report may contain:
    1. CBC values in sequence format
    2. Liver and biochemical values in key-value format

    ----------------------------------------
    IMPORTANT CBC RULE (VERY STRICT)
    ----------------------------------------
    If a sequence of numbers is present for CBC, map them EXACTLY in this order:

    [WBC, NE#, LY#, MO#, EO#, BA#, RBC, HGB, HCT, MCV, MCH, RDW]

    DO NOT shift values.
    DO NOT guess mapping.
    Use position-based assignment ONLY for CBC.

    ----------------------------------------
    LIVER + OTHER LAB RULE
    ----------------------------------------
    For other labs, extract using keyword matching:

    BILIRUBIN → LFT - Total Bilirubin (mg/dL)  
    SGOT/AST → LFT - SGOT (AST) (U/L)  
    SGPT/ALT → LFT - SGPT (ALT) (U/L)  
    ALP → LFT - Alkaline Phosphatase (U/L)  
    TOTAL PROTEIN → LFT - Total Protein (g/dL)  
    ALBUMIN → LFT - Albumin (g/dL)  
    INR → PT-INR  
    UREA → RFT - Urea (mg/dL)  
    CREATININE → RFT - Creatinine (mg/dL)  

    ----------------------------------------
    IRON + CBC RELATED
    ----------------------------------------
    FERRITIN → FERRITIN  
    FOLATE → FOLATE  
    B12 / Vitamin B12 → B12  

    ----------------------------------------
    DEMOGRAPHIC + CLINICAL
    ----------------------------------------
    AGE → Age  
    BMI → BMI  

    ASCITES → Ascites (none/mild/moderate/severe)  
    ENCEPHALOPATHY → Encephalopathy (none/grade 1/grade 2/grade 3)

    ----------------------------------------
    RETURN FORMAT (STRICT JSON ONLY)
    ----------------------------------------
    {{
      "WBC": float,
      "NE#": float,
      "LY#": float,
      "MO#": float,
      "EO#": float,
      "BA#": float,
      "RBC": float,
      "HGB": float,
      "HCT": float,
      "MCV": float,
      "MCH": float,
      "MCHC": float,
      "RDW": float,

      "PLT": float,
      "MPV": float,
      "PCT": float,
      "PDW": float,

      "FERRITIN": float,
      "FOLATE": float,
      "B12": float,

      "LFT - Total Bilirubin (mg/dL)": float,
      "LFT - SGOT (AST) (U/L)": float,
      "LFT - SGPT (ALT) (U/L)": float,
      "LFT - Alkaline Phosphatase (U/L)": float,
      "LFT - Total Protein (g/dL)": float,
      "LFT - Albumin (g/dL)": float,
      "PT-INR": float,
      "RFT - Urea (mg/dL)": float,
      "RFT - Creatinine (mg/dL)": float,

      "Age": float,
      "BMI": float,
      "Ascites": "none/mild/moderate/severe",
      "Encephalopathy": "none/grade 1/grade 2/grade 3"
    }}

    ----------------------------------------
    RULES
    ----------------------------------------
    - Do NOT hallucinate values
    - Do NOT shift CBC values
    - Ignore units
    - Extract only visible values
    - Return null if not present
    - Output ONLY JSON (no explanation)

    ----------------------------------------
    OCR TEXT:
    {ocr_text}
    """

    raw = call_llm(prompt)
    parsed = extract_json(raw)

    if parsed is None:
        return {}

    # remove nulls
    parsed = {k: v for k, v in parsed.items() if v is not None}

    return parsed