import os
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd


# =========================================
# PATHS
# =========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


# =========================================
# LOAD ANEMIA
# =========================================
ANEMIA_IMPUTER = joblib.load(os.path.join(ARTIFACTS_DIR, "anemia_imputer.pkl"))
ANEMIA_SCALER = joblib.load(os.path.join(ARTIFACTS_DIR, "anemia_scaler.pkl"))

ANEMIA_LABELS = [
    "Iron_Label",
    "B12_Label",
    "Folate_Label",
    "HGB_Label"
]

ANEMIA_XGB = {
    label: joblib.load(os.path.join(MODEL_DIR, f"anemia_xgb_{label}.pkl"))
    for label in ANEMIA_LABELS
}

ANEMIA_META = {
    label: joblib.load(os.path.join(MODEL_DIR, f"anemia_meta_{label}.pkl"))
    for label in ANEMIA_LABELS
}

ANEMIA_MLP = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "anemia_mlp_multilabel.h5")
)


# =========================================
# LOAD LIVER
# =========================================
LIVER_XGB = joblib.load(os.path.join(MODEL_DIR, "liver_xgb.pkl"))
LIVER_META = joblib.load(os.path.join(MODEL_DIR, "liver_meta.pkl"))
LIVER_MLP = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "liver_mlp.h5")
)

LIVER_IMPUTER = joblib.load(os.path.join(ARTIFACTS_DIR, "liver_imputer.pkl"))
LIVER_SCALER = joblib.load(os.path.join(ARTIFACTS_DIR, "liver_scaler.pkl"))
LIVER_FEATURES = joblib.load(os.path.join(ARTIFACTS_DIR, "liver_features.pkl"))
LIVER_ENCODER = joblib.load(os.path.join(ARTIFACTS_DIR, "liver_label_encoder.pkl"))


CONFIDENCE_THRESHOLD = 0.5


# =========================================
# SHAP WRAPPER - ANEMIA
# =========================================
def anemia_predict_proba_shap(X):

    X = np.array(X)
    X_scaled = ANEMIA_SCALER.transform(X)

    mlp_probs = ANEMIA_MLP.predict(X_scaled, verbose=0)

    outputs = []

    for i in range(X_scaled.shape[0]):

        x = X_scaled[i].reshape(1, -1)

        xgb_prob = ANEMIA_XGB["Iron_Label"].predict_proba(x)[0][1]

        stacked = np.array([[xgb_prob, mlp_probs[i][0]]])

        final_prob = ANEMIA_META["Iron_Label"].predict_proba(stacked)[0][1]

        outputs.append(final_prob)

    return np.array(outputs).reshape(-1, 1)


# =========================================
#  SHAP WRAPPER - LIVER
# =========================================
def liver_predict_proba_shap(X):

    X = np.array(X)

    expected = len(LIVER_FEATURES)

    if X.shape[1] > expected:
        X = X[:, :expected]
    elif X.shape[1] < expected:
        padding = np.zeros((X.shape[0], expected - X.shape[1]))
        X = np.hstack([X, padding])

    X_scaled = LIVER_SCALER.transform(X)

    xgb_probs = LIVER_XGB.predict_proba(X_scaled)
    mlp_probs = LIVER_MLP.predict(X_scaled, verbose=0)

    stacked = np.hstack([xgb_probs, mlp_probs])
    final_probs = LIVER_META.predict_proba(stacked)

    return np.max(final_probs, axis=1).reshape(-1, 1)


# =========================================
# ANEMIA PREDICTION
# =========================================
def predict_anemia_multilabel(labs, feature_order):

    input_row = [labs.get(f, np.nan) for f in feature_order]

    input_df = pd.DataFrame([input_row], columns=feature_order)

    input_vector = ANEMIA_IMPUTER.transform(input_df)
    input_vector = ANEMIA_SCALER.transform(input_vector)

    mlp_probs = ANEMIA_MLP.predict(input_vector, verbose=0)[0]

    probabilities = {}

    for i, label in enumerate(ANEMIA_LABELS):

        xgb_prob = ANEMIA_XGB[label].predict_proba(input_vector)[0][1]

        stacked = np.array([[xgb_prob, mlp_probs[i]]])
        final_prob = ANEMIA_META[label].predict_proba(stacked)[0][1]

        clean_name = label.replace("_Label", "").replace("_", " ")
        probabilities[clean_name] = round(float(final_prob), 4)

    predicted = [
        name for name, prob in probabilities.items()
        if prob >= CONFIDENCE_THRESHOLD
    ]

    if not predicted:
        predicted = ["None Detected"]

    return {
        "probabilities": probabilities,
        "predicted_labels": predicted
    }


# =========================================
# LIVER PREDICTION
# =========================================
def predict_disease(labs, model_name):

    if model_name != "liver":
        raise ValueError("Unsupported model")

    input_row = []
    missing_features = []

    for f in LIVER_FEATURES:
        val = labs.get(f, np.nan)
        if pd.isna(val):
            missing_features.append(f)
        input_row.append(val)

    input_df = pd.DataFrame([input_row], columns=LIVER_FEATURES)

    input_vector = LIVER_IMPUTER.transform(input_df)
    input_vector = LIVER_SCALER.transform(input_vector)

    xgb_probs = LIVER_XGB.predict_proba(input_vector)
    mlp_probs = LIVER_MLP.predict(input_vector, verbose=0)

    stacked = np.hstack([xgb_probs, mlp_probs])
    final_probs = LIVER_META.predict_proba(stacked)

    prediction_index = int(np.argmax(final_probs))
    confidence = float(np.max(final_probs))

    predicted_label = LIVER_ENCODER.inverse_transform([prediction_index])[0]

    return {
        "prediction": f"Child-Pugh Class {predicted_label}",
        "confidence": round(confidence, 4),
        "missing_features": missing_features
    }