import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

FINAL_DIR = "features/optimal/final"
OUTPUT_DIR = "model_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_DIR = "features/optimal/final"
OUTPUT_DIR = "model_results"
PRED_DIR = "model_predictions"
MODEL_DIR = "saved_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LABEL_COLS = [
    "heel_lift", "toe_out", "toe_in", "weight_shift_to_toes",
    "knee_valgus", "knee_varus", "excessive_knee_travel",
    "butt_wink", "lumbar_extension", "hip_shift",
    "forward_lean", "chest_collapse",
    "shallow_depth", "excessive_depth",
    "weak_bracing", "poor_breathing",
    "dropping_too_fast", "unintentional_pause",
    "asymmetrical_pattern", "poor_stance",
    "good_technique",
]


def get_models():
    return {
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss"
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=50, subsample=0.9, colsample_bytree=0.9
        ),
        "RandomForest": RandomForestClassifier(n_estimators=300),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300)
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    return metrics, y_pred


def plot_confusion(y_test, y_pred, fault, model_name, out_dir):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{fault} — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cm_{fault}_{model_name}.png"), dpi=300)
    plt.close()


def plot_roc(model, X_test, y_test, fault, model_name, out_dir):
    if len(np.unique(y_test)) < 2:
        return  # cannot compute ROC for single-class faults

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC — {fault} — {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"roc_{fault}_{model_name}.png"), dpi=300)
    plt.close()


def process_variant(variant):
    print(f"\n=== Training models for variant: {variant} ===")

    df = pd.read_csv(f"{FINAL_DIR}/final_features_{variant}.csv")

    X = df.drop(columns=LABEL_COLS + ["session_id"], errors="ignore")
    # sessions = df["session_id"].values
    
    results = []
    pred_store = {}

    out_dir = os.path.join(OUTPUT_DIR, variant)
    os.makedirs(out_dir, exist_ok=True)

    for fault in LABEL_COLS:
        y = df[fault].values

        # Skip faults with no positive samples
        if len(np.unique(y)) < 2:
            print(f"[SKIP] Fault '{fault}' has only one class")
            continue

        # X_train, X_test, y_train, y_test, sess_train, sess_test = train_test_split(
        #     X, y, sessions, test_size=0.25, random_state=42, stratify=y
        # )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        models = get_models()

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            metrics, y_pred = evaluate_model(model, X_test, y_test)

            metrics.update({
                "variant": variant,
                "fault": fault,
                "model": model_name
            })

            results.append(metrics)

            # Save confusion matrix + ROC
            plot_confusion(y_test, y_pred, fault, model_name, out_dir)
            plot_roc(model, X_test, y_test, fault, model_name, out_dir)
            
            # Save model
            model_path = os.path.join(MODEL_DIR, f"{variant}_{fault}_{model_name}.pkl")
            joblib.dump(model, model_path)

            # Save predictions
            if model_name not in pred_store:
                pred_store[model_name] = {"y_true": [], "y_pred": [], "session_id": []}

            pred_store[model_name]["y_true"].extend(y_test.tolist())
            pred_store[model_name]["y_pred"].extend(y_pred.tolist())
            # pred_store[model_name]["session_id"].extend(sess_test.tolist())

    # Save prediction dictionary
    pred_path = os.path.join(PRED_DIR, f"{variant}_predictions.pkl")
    joblib.dump(pred_store, pred_path)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, "model_results.csv"), index=False)

    print(f"[OK] Saved → {out_dir}/model_results.csv")
    print(f"[OK] Saved predictions → {pred_path}")


def run_all():
    for fname in os.listdir(FINAL_DIR):
        if fname.startswith("final_features_") and fname.endswith(".csv"):
            variant = fname.replace("final_features_", "").replace(".csv", "")
            process_variant(variant)


if __name__ == "__main__":
    run_all()
