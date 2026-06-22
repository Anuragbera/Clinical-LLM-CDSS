import os
import joblib
import numpy as np
import pandas as pd

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

ANEMIA_FEATURES = [
    "HGB", "HCT", "MCV", "MCH", "MCHC",
    "RDW", "RBC", "FERRITTE", "FOLATE", "B12"
]

LABELS = [
    "Iron_Label", "B12_Label", "Folate_Label", "HGB_Label"
]

N_SPLITS = 5


def create_multilabel_targets(df):
    df["Iron_Label"] = (df["FERRITTE"] < 30).astype(int)
    df["B12_Label"] = (df["B12"] < 200).astype(int)
    df["Folate_Label"] = (df["FOLATE"] < 4).astype(int)
    df["HGB_Label"] = (df["HGB"] < 12).astype(int)
    return df


def train_multilabel_anemia(df):

    df = create_multilabel_targets(df)

    X = df[ANEMIA_FEATURES]
    Y = df[LABELS]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(X))

    joblib.dump(imputer, os.path.join(ARTIFACTS_DIR, "anemia_imputer.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "anemia_scaler.pkl"))

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y["Iron_Label"], random_state=42
    )

    xgb_models = {}
    meta_models = {}

    # ======================
    # TRAIN PER LABEL
    # ======================
    for i, label in enumerate(LABELS):

        print(f"\n===== {label} CROSS-VALIDATION =====")

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

        oof_stack = np.zeros((len(X_train), 2))

        cv_accs, cv_precs, cv_recs, cv_f1s, cv_aucs = [], [], [], [], []

        for train_idx, val_idx in skf.split(X_train, Y_train[label]):

            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr = Y_train[label].iloc[train_idx]
            y_val = Y_train[label].iloc[val_idx]

            # XGB
            xgb = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05
            )
            xgb.fit(X_tr, y_tr)

            xgb_probs_val = xgb.predict_proba(X_val)[:, 1]

            # MLP (use global model later, avoid retraining here)
            # temporary placeholder → we fill after training MLP
            # so skip here

            oof_stack[val_idx, 0] = xgb_probs_val

            # Metrics (using XGB only for CV baseline)
            preds = (xgb_probs_val > 0.5).astype(int)
            cv_accs.append(accuracy_score(y_val, preds))
            cv_precs.append(precision_score(y_val, preds))
            cv_recs.append(recall_score(y_val, preds))
            cv_f1s.append(f1_score(y_val, preds))
            cv_aucs.append(roc_auc_score(y_val, xgb_probs_val))

        print("Accuracy:", np.mean(cv_accs))
        print("Precision:", np.mean(cv_precs))
        print("Recall:", np.mean(cv_recs))
        print("F1:", np.mean(cv_f1s))
        print("AUC:", np.mean(cv_aucs))

        # Train final XGB
        final_xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05
        )
        final_xgb.fit(X_train, Y_train[label])
        xgb_models[label] = final_xgb

        joblib.dump(final_xgb, os.path.join(MODEL_DIR, f"anemia_xgb_{label}.pkl"))

    # ======================
    # MLP TRAIN (GLOBAL)
    # ======================
    mlp = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(4, activation="sigmoid")
    ])

    mlp.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    mlp.fit(X_train, Y_train.values, epochs=50, batch_size=32, verbose=1)

    mlp.save(os.path.join(MODEL_DIR, "anemia_mlp_multilabel.h5"))

    # ======================
    # META MODEL (FIXED OOF)
    # ======================
    for i, label in enumerate(LABELS):

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        oof_stack = np.zeros((len(X_train), 2))

        for train_idx, val_idx in skf.split(X_train, Y_train[label]):

            # XGB
            xgb = xgb_models[label]

            xgb_probs_val = xgb.predict_proba(X_train[val_idx])[:, 1]
            mlp_probs_val = mlp.predict(X_train[val_idx])[:, i]

            oof_stack[val_idx] = np.vstack([xgb_probs_val, mlp_probs_val]).T

        meta = LogisticRegression()
        meta.fit(oof_stack, Y_train[label])

        print("\n===== CV PROBABILITY OUTPUT =====")
        print("Label:", label)
        print("First 10 CV probabilities:")
        print(oof_stack[:10])

        print("Shape:", oof_stack.shape)
        print("Check first row sum (approx):", np.sum(oof_stack[0]))

        meta_models[label] = meta
        joblib.dump(meta, os.path.join(MODEL_DIR, f"anemia_meta_{label}.pkl"))

    # ======================
    # TEST PERFORMANCE (VALID)
    # ======================
    print("\n===== TEST PERFORMANCE (80:20) =====")

    for i, label in enumerate(LABELS):

        xgb_probs = xgb_models[label].predict_proba(X_test)[:, 1]
        mlp_probs = mlp.predict(X_test)[:, i]

        stacked = np.vstack([xgb_probs, mlp_probs]).T
        meta = meta_models[label]

        final_probs = meta.predict_proba(stacked)[:, 1]
        final_preds = (final_probs > 0.5).astype(int)

        print("\n===== TEST PROBABILITY OUTPUT =====")
        print("Label:", label)
        print("y_true (first 10):")
        print(Y_test[label].values[:10])

        print("\ny_pred_proba (first 10):")
        print(final_probs[:10])

        print("Shape:", final_probs.shape)

        print(f"\n===== {label} TEST PERFORMANCE =====")
        print("Accuracy:", accuracy_score(Y_test[label], final_preds))
        print("Precision:", precision_score(Y_test[label], final_preds))
        print("Recall:", recall_score(Y_test[label], final_preds))
        print("F1:", f1_score(Y_test[label], final_preds))
        print("AUC:", roc_auc_score(Y_test[label], final_probs))
        print("Confusion Matrix:\n", confusion_matrix(Y_test[label], final_preds))

        # VALID ROC
        fpr, tpr, _ = roc_curve(Y_test[label], final_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(Y_test[label], final_probs):.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"ROC Curve - {label}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    print("Starting training...")

    df = pd.read_excel(r"C:\Users\User\Documents\Clinical_LLM\data\anemia.xlsx")

    train_multilabel_anemia(df)