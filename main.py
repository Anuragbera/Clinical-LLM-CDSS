from modules.router import disease_router
from modules.feature_schema import ANEMIA_FEATURE_ORDER, LIVER_FEATURE_ORDER
from modules.feature_requirements import check_required_features
from modules.predict import (
    predict_anemia_multilabel,
    predict_disease,
    anemia_predict_proba_shap,
    liver_predict_proba_shap
)
from modules.severity import anemia_severity, liver_severity
from modules.llm_explanation import generate_explanation
from modules.explainability import get_shap_explanation
from modules.predict import LIVER_FEATURES

import numpy as np


def run_clinical_engine(structured_data):
    labs = structured_data.get("labs_numeric", {})

    # =========================
    #  NORMALIZE KEYS (VERY IMPORTANT)
    # =========================
    labs = {k.upper(): v for k, v in labs.items()}

    print("\n[DEBUG] LABS RECEIVED:", labs)

    results = {}

    models_to_run = disease_router(structured_data)

    for model in models_to_run:

        # =========================
        # AUTO-FILL MISSING FEATURES
        # =========================
        if model == "anemia":
            for f in ANEMIA_FEATURE_ORDER:
                if f not in labs:
                    labs[f] = np.nan  # or np.nan if your model supports it

        elif model == "liver":
            for f in LIVER_FEATURE_ORDER:
                if f not in labs:
                    labs[f] = np.nan

        # =========================
        # VALIDATION (NOW FLEXIBLE)
        # =========================
        valid, missing = check_required_features(labs, model)

        print(f"[DEBUG] Model: {model}")
        print(f"[DEBUG] Missing Features: {missing}")

        if not valid:
            results[model] = {
                "prediction": "Insufficient Data",
                "missing_features": missing
            }
            continue

        # ================= ANEMIA =================
        if model == "anemia":

            prediction_result = predict_anemia_multilabel(labs, ANEMIA_FEATURE_ORDER)

            severity, advice = anemia_severity(
                labs,
                prediction_result["predicted_labels"]
            )

            explanation = generate_explanation(prediction_result)

            # FILTER ONLY REQUIRED FEATURES
            filtered_labs = {f: labs.get(f, 0) for f in ANEMIA_FEATURE_ORDER}

            input_array = np.array([
                list(filtered_labs.values())
            ])

            shap_values = get_shap_explanation(
                model=anemia_predict_proba_shap,
                input_array=input_array,
                feature_names=ANEMIA_FEATURE_ORDER
            )

            results["anemia"] = {
                "probabilities": prediction_result["probabilities"],
                "predicted_labels": prediction_result["predicted_labels"],
                "severity": severity,
                "recommended_action": advice,
                "explanation": explanation,
                "shap_values": shap_values
            }

        # ================= LIVER =================
        elif model == "liver":

            prediction_result = predict_disease(labs, "liver")

            severity, advice = liver_severity(
                prediction_result["prediction"]
            )

            explanation = generate_explanation(prediction_result)

            input_array = np.array([
                [labs.get(f, 0) for f in LIVER_FEATURES]
            ])

            shap_values = get_shap_explanation(
                model=liver_predict_proba_shap,
                input_array=input_array,
                feature_names=LIVER_FEATURES
            )

            results["liver"] = {
                "prediction": prediction_result["prediction"],
                "confidence": prediction_result["confidence"],
                "severity": severity,
                "recommended_action": advice,
                "explanation": explanation,
                "shap_values": shap_values
            }

    return results
