# ===============================
# MINIMUM REQUIRED FEATURES
# ===============================

ANEMIA_MINIMUM = [
    "HGB",
    "MCV",
    "RBC",
    "RDW",
    "FERRITTE"

]

ANEMIA_FULL = [
    "HGB",
    "HCT",
    "MCV",
    "MCH",
    "MCHC",
    "RDW",
    "RBC",
    "FERRITTE",
    "FOLATE",
    "B12"
]

LIVER_MINIMUM = [
    "LFT - Total Bilirubin (mg/dL)",
    "LFT - Albumin (g/dL)",
    "PT-INR",
    "Ascites",
    "Encephalopathy"
]

LIVER_FULL = [
    "Age",
    "BMI",
    "LFT - Total Bilirubin (mg/dL)",
    "LFT - SGOT (AST) (U/L)",
    "LFT - SGPT (ALT) (U/L)",
    "LFT - Alkaline Phosphatase (U/L)",
    "LFT - Total Protein (g/dL)",
    "LFT - Albumin (g/dL)",
    "PT-INR",
    "RFT - Urea (mg/dL)",
    "RFT - Creatinine (mg/dL)",
    "Ascites",
    "Encephalopathy"
]


# ===============================
# FEATURE CHECK FUNCTION
# ===============================

def check_required_features(labs, model_type):

    if model_type == "anemia":
        required = ANEMIA_MINIMUM

    elif model_type == "liver":
        required = LIVER_MINIMUM

    else:
        return True, []

    missing = [feature for feature in required if feature not in labs]

    if missing:
        return False, missing

    return True, []
