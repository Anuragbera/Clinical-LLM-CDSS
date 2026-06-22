import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

from modules.predict import (
    predict_anemia_multilabel,
    predict_disease
)

# ======================
# LOAD DATASET
# ======================
df_anemia = pd.read_excel(r"C:\Users\User\Documents\Clinical_LLM\data\anemia.xlsx")
df_liver = pd.read_excel(r"C:\Users\User\Documents\Clinical_LLM\data\augmented_child_pugh_2000rows.xlsx")

# 🔥 TEMP FIX: reduce size (remove later for full evaluation)
df_anemia = df_anemia.sample(300, random_state=42)
df_liver = df_liver.sample(300, random_state=42)

# ======================
# CREATE TRUE LABELS (ANEMIA)
# ======================
def create_labels(df):
    labels = []
    for i in range(len(df)):
        if df.loc[i, "FERRITTE"] < 30:
            labels.append("Iron")
        elif df.loc[i, "B12"] < 200:
            labels.append("B12")
        elif df.loc[i, "FOLATE"] < 4:
            labels.append("Folate")
        elif df.loc[i, "HGB"] < 12:
            labels.append("HGB")
        else:
            labels.append("None Detected")
    return labels

df_anemia["true_label"] = create_labels(df_anemia)

# ======================
# ANEMIA EVALUATION
# ======================
print("\n===== ANEMIA EVALUATION =====")

y_true = []
y_pred = []

FEATURES = [
    "HGB","HCT","MCV","MCH","MCHC",
    "RDW","RBC","FERRITTE","FOLATE","B12"
]

for i, (_, row) in enumerate(df_anemia.iterrows()):

    if i % 50 == 0:
        print(f"Processed {i} samples...")

    labs = {f: row[f] for f in FEATURES}

    result = predict_anemia_multilabel(labs, FEATURES)

    # ✅ FIX: choose highest probability label
    probs = result["probabilities"]
    pred = max(probs, key=probs.get)

    y_true.append(row["true_label"])
    y_pred.append(pred)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# ======================
# LIVER EVALUATION
# ======================
print("\n===== LIVER EVALUATION =====")

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

y_true = []
y_pred = []

for i, (_, row) in enumerate(df_liver.iterrows()):

    if i % 50 == 0:
        print(f"Processed {i} samples...")

    labs = {f: row[f] for f in LIVER_FEATURES}

    result = predict_disease(labs, "liver")

    pred = result["prediction"].split()[-1]

    y_true.append(str(row["Child-Pugh Class"]))
    y_pred.append(pred)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))