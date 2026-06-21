class FeatureMapper:
    def __init__(self):
        # Maps the Fuzzy Match outputs to your exact Model Feature requirements
        self.mapping = {
            # ANEMIA MODEL KEYS
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

            # LIVER MODEL KEYS
            "LFT - Total Bilirubin (mg/dL)": ["BILIRUBIN", "TOTAL BILIRUBIN"],
            "LFT - SGOT (AST) (U/L)": ["AST", "SGOT"],
            "LFT - SGPT (ALT) (U/L)": ["ALT", "SGPT"],
            "LFT - Alkaline Phosphatase (U/L)": ["ALKALINE PHOSPHATASE", "ALP"],
            "LFT - Total Protein (g/dL)": ["TOTAL PROTEIN"],
            "LFT - Albumin (g/dL)": ["ALBUMIN"],
            "PT-INR": ["INR", "PT-INR"],
            "RFT - Urea (mg/dL)": ["UREA"],
            "RFT - Creatinine (mg/dL)": ["CREATININE"],

            "Age": ["AGE"],
            "BMI": ["BMI"],
            "Ascites": ["ASCITES"],
            "Encephalopathy": ["ENCEPHALOPATHY"]
        }

    def map_features(self, raw_data):
        structured = {}
        for raw_key, value in raw_data.items():
            for target, aliases in self.mapping.items():
                if raw_key in aliases:
                    structured[target] = value
                    break
        return structured