import os
import numpy as np
import pandas as pd
import joblib
import shap

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


MODEL_DIR = Path(r"C:\Users\User\Documents\Clinical_LLM\models")
ARTIFACT_DIR = Path(r"C:\Users\User\Documents\Clinical_LLM\artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def build_mlp(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_multiclass_model(df, feature_cols, target_col, model_name):

    if "Ascites" in df.columns:
        df["Ascites"] = df["Ascites"].map({
            "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3
        })

    if "Encephalopathy" in df.columns:
        df["Encephalopathy"] = df["Encephalopathy"].map({
            "None": 0, "Grade 1": 1, "Grade 2": 2, "Grade 3": 3
        })

    X = df[feature_cols]
    y = df[target_col]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n_classes = len(np.unique(y_encoded))
    oof_xgb = np.zeros((len(X), n_classes))
    oof_mlp = np.zeros((len(X), n_classes))

    # ======================
    # BASE MODELS (OOF)
    # ======================
    for train_idx, val_idx in skf.split(X, y_encoded):

        X_tr_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_tr = y_encoded[train_idx]

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_tr = scaler.fit_transform(imputer.fit_transform(X_tr_raw))
        X_val = scaler.transform(imputer.transform(X_val_raw))

        xgb = XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )
        xgb.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgb.predict_proba(X_val)

        mlp = build_mlp(X_tr.shape[1], n_classes)
        mlp.fit(X_tr, y_tr, epochs=40, batch_size=32, verbose=0)
        oof_mlp[val_idx] = mlp.predict(X_val, verbose=0)

    meta_X = np.hstack([oof_xgb, oof_mlp])

    # ======================
    # META MODEL (FIXED OOF)
    # ======================
    meta = LogisticRegression(max_iter=1000, multi_class="multinomial")

    meta_oof = np.zeros((len(X), n_classes))

    for train_idx, val_idx in skf.split(meta_X, y_encoded):
        meta.fit(meta_X[train_idx], y_encoded[train_idx])
        meta_oof[val_idx] = meta.predict_proba(meta_X[val_idx])

    cv_probs = meta_oof
    cv_preds = np.argmax(cv_probs, axis=1)

    print("\n===== CROSS VALIDATION PERFORMANCE (FIXED) =====")
    print("Accuracy:", accuracy_score(y_encoded, cv_preds))
    print("Precision:", precision_score(y_encoded, cv_preds, average="macro"))
    print("Recall:", recall_score(y_encoded, cv_preds, average="macro"))
    print("F1 Score:", f1_score(y_encoded, cv_preds, average="macro"))
    print("AUC:", roc_auc_score(y_encoded, cv_probs, multi_class="ovr"))
    print("Confusion Matrix:\n", confusion_matrix(y_encoded, cv_preds))

    print("\n===== CV PROBABILITIES (FIXED) =====")
    print(cv_probs[:5])

    # ======================
    # FINAL TRAINING
    # ======================
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_full = scaler.fit_transform(imputer.fit_transform(X))

    final_xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    final_xgb.fit(X_full, y_encoded)

    final_mlp = build_mlp(X_full.shape[1], n_classes)
    final_mlp.fit(X_full, y_encoded, epochs=40, batch_size=32, verbose=0)

    # ======================
    # TEST SET
    # ======================
    X_test_proc = scaler.transform(imputer.transform(X_test_eval))

    xgb_probs = final_xgb.predict_proba(X_test_proc)
    mlp_probs = final_mlp.predict(X_test_proc)

    stacked_test = np.hstack([xgb_probs, mlp_probs])

    meta.fit(meta_X, y_encoded)  # train full meta for test

    test_probs = meta.predict_proba(stacked_test)
    test_preds = np.argmax(test_probs, axis=1)

    print("\n===== TEST PERFORMANCE (FIXED) =====")
    print("Accuracy:", accuracy_score(y_test_eval, test_preds))
    print("Precision:", precision_score(y_test_eval, test_preds, average="macro"))
    print("Recall:", recall_score(y_test_eval, test_preds, average="macro"))
    print("F1 Score:", f1_score(y_test_eval, test_preds, average="macro"))
    print("AUC:", roc_auc_score(y_test_eval, test_probs, multi_class="ovr"))
    print("Confusion Matrix:\n", confusion_matrix(y_test_eval, test_preds))

    print("\n===== TEST PROBABILITIES (FIXED) =====")
    print(test_probs[:5])
    print(f"\n{model_name} training complete.")

# ======================
    # SHAP (ORIGINAL)
    # ======================
    try:
        explainer = shap.Explainer(final_xgb)
        shap_values = explainer(X_full[:500])
        shap.plots.bar(shap_values)
    except Exception as e:
        print("SHAP skipped:", e)

    print(f"\n{model_name} training complete.")


# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":

    print("Starting liver training pipeline...")

    df = pd.read_excel(
        r"C:\Users\User\Documents\Clinical_LLM\data\augmented_child_pugh_2000rows.xlsx"
    )

    print("Dataset Loaded:", df.shape)

    LIVER_FEATURES = [
        "Age","BMI",
        "LFT - Total Bilirubin (mg/dL)",
        "LFT - Albumin (g/dL)",
        "PT-INR",
        "LFT - SGOT (AST) (U/L)",
        "LFT - SGPT (ALT) (U/L)",
        "LFT - Alkaline Phosphatase (U/L)",
        "RFT - Urea (mg/dL)",
        "RFT - Creatinine (mg/dL)",
        "Ascites","Encephalopathy"
    ]

    train_multiclass_model(
        df=df,
        feature_cols=LIVER_FEATURES,
        target_col="Child-Pugh Class",
        model_name="liver"
    )