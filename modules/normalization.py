def normalize_lab_keys(labs):

    key_mapping = {
        "HGB": ["HEMOGLOBIN", "HGB"],
        "MCV": ["MCV"],
        "RBC": ["RBC"],
        "RDW": ["RDW"],
        "FERRITIN": ["FERRITIN"],
        "HCT": ["HCT"],
        "MCH": ["MCH"],
        "MCHC": ["MCHC"],
        "FOLATE": ["FOLATE"],
        "B12": ["VITAMIN B12", "B12"],

        "LFT - Total Bilirubin (mg/dL)": ["BILIRUBIN"],
        "LFT - SGOT (AST) (U/L)": ["SGOT"],
        "LFT - SGPT (ALT) (U/L)": ["SGPT"],
        "LFT - Alkaline Phosphatase (U/L)": ["ALP"],
        "LFT - Total Protein (g/dL)": ["TOTAL PROTEIN"],
        "LFT - Albumin (g/dL)": ["ALBUMIN"],
        "PT-INR": ["INR"],
        "RFT - Urea (mg/dL)": ["UREA"],
        "RFT - Creatinine (mg/dL)": ["CREATININE"],

        "Age": ["AGE"],
        "BMI": ["BMI"],
        "Ascites": ["ASCITES"],
        "Encephalopathy": ["ENCEPHALOPATHY"]
    }

    ASCITES_MAP = {
        "none": 0,
        "mild": 1,
        "moderate": 2,
        "severe": 3
    }

    ENCEPH_MAP = {
        "none": 0,
        "grade 1": 1,
        "grade 2": 2,
        "grade 3": 3
    }

    normalized = {}
    display_values = {}

    for key, value in labs.items():

        canonical_key = key

        #  FIXED MAPPING
        for target, aliases in key_mapping.items():
            if key in aliases or key == target:
                canonical_key = target
                break

        # ================= ASCITES =================
        if canonical_key == "Ascites":
            v = str(value).lower().strip()
            normalized[canonical_key] = ASCITES_MAP.get(v, 0)
            display_values[canonical_key] = v

        # ================= ENCEPH =================
        elif canonical_key == "Encephalopathy":
            v = str(value).lower().strip()
            normalized[canonical_key] = ENCEPH_MAP.get(v, 0)
            display_values[canonical_key] = v

        # ================= NUMERIC =================
        else:
            try:
                val = float(value)
                normalized[canonical_key] = val
                display_values[canonical_key] = val
            except:
                continue

    return normalized, display_values