from modules.feature_requirements import check_required_features
from modules.predict import predict_anemia_multilabel, predict_disease
from modules.severity import anemia_severity, liver_severity


# =========================================
# DISEASE DETECTION (SMART ROUTER)
# =========================================
def disease_router(data):

    labs = data.get("labs_numeric", {})
    labs = {k.upper(): v for k, v in labs.items()}

    models = []

    # =========================
    # ANEMIA FEATURES
    # =========================
    anemia_features = [
        "HGB", "MCV", "RBC", "RDW",
        "FERRITIN", "HCT", "MCH",
        "FOLATE", "B12"
    ]

    # =========================
    # LIVER FEATURES
    # =========================
    liver_features = [
        "LFT - TOTAL BILIRUBIN (MG/DL)",
        "LFT - SGOT (AST) (U/L)",
        "LFT - SGPT (ALT) (U/L)",
        "LFT - ALKALINE PHOSPHATASE (U/L)",
        "LFT - TOTAL PROTEIN (G/DL)",
        "LFT - ALBUMIN (G/DL)",
        "PT-INR",
        "RFT - UREA (MG/DL)",
        "RFT - CREATININE (MG/DL)",
        "ASCITES",
        "ENCEPHALOPATHY"
    ]

    # =========================
    # COUNT FEATURES
    # =========================
    anemia_count = sum(1 for f in anemia_features if f in labs)
    liver_count = sum(1 for f in liver_features if f in labs)

    print("\n[ROUTER DEBUG]")
    print("Anemia feature count:", anemia_count)
    print("Liver feature count:", liver_count)

    # =========================
    # DECISION LOGIC (THRESHOLD)
    # =========================
    if anemia_count >= 3:
        models.append("anemia")

    if liver_count >= 3:
        models.append("liver")

    return models


# =========================================
# MAIN ROUTER
# =========================================
def run_router(data):

    labs = data.get("labs_numeric", {})
    symptoms = data.get("symptoms", [])

    print("\n[ROUTER DEBUG] Labs:", labs)
    print("[ROUTER DEBUG] Symptoms:", symptoms)

    detected_models = disease_router(data)

    if not detected_models:
        return {
            "status": "error",
            "message": "No disease model matched",
            "labs": labs,
            "symptoms": symptoms
        }

    results = {}

    # =========================================
    # LOOP THROUGH MODELS
    # =========================================
    for model in detected_models:

        valid, missing = check_required_features(labs, model)

        if not valid:
            results[model] = {
                "status": "error",
                "missing_features": missing
            }
            continue

        try:
            # =====================================
            # ANEMIA
            # =====================================
            if model == "anemia":

                from modules.feature_requirements import ANEMIA_FULL

                prediction = predict_anemia_multilabel(
                    labs,
                    feature_order=ANEMIA_FULL
                )

                severity, advice = anemia_severity(
                    labs,
                    prediction.get("predicted_labels", [])
                )

                results[model] = {
                    "status": "success",
                    "prediction": prediction,
                    "severity": severity,
                    "advice": advice
                }

            # =====================================
            # LIVER
            # =====================================
            elif model == "liver":

                prediction = predict_disease(
                    labs,
                    model_name="liver"
                )

                # 🔥 FIX: severity depends ONLY on prediction
                severity, advice = liver_severity(
                    prediction.get("prediction")
                )

                results[model] = {
                    "status": "success",
                    "prediction": prediction,
                    "severity": severity,
                    "advice": advice
                }

        except Exception as e:
            results[model] = {
                "status": "error",
                "message": str(e)
            }

    # =========================================
    # FINAL RESPONSE
    # =========================================
    return {
        "status": "success",
        "detected_models": detected_models,
        "labs_used": labs,
        "symptoms": symptoms,
        "results": results
    }