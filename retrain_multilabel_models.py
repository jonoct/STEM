import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, hamming_loss,
    f1_score, classification_report
)

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

FINAL_DIR = "features/optimal/final"
OUTPUT_DIR = "model_results_multilabel"
PRED_DIR = "model_predictions_multilabel"
MODEL_DIR = "saved_models_multilabel"

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


def get_base_models():
    return {
        "RandomForest Multi": RandomForestClassifier(n_estimators=300),
        "LightGBM Multi": LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=50, subsample=0.9, colsample_bytree=0.9
        ),
        "XGBoost Multi": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss"
        ),
        "MLP Multi": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300)
    }


def evaluate_multilabel(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "subset_accuracy": accuracy_score(y_test, y_pred),
        "hamming_loss": hamming_loss(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_test, y_pred, average="micro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    return metrics, y_pred


def process_variant(variant):
    print(f"\n=== Multi-label training for variant: {variant} ===")

    df = pd.read_csv(f"{FINAL_DIR}/final_features_{variant}.csv")

    X = df.drop(columns=LABEL_COLS + ["session_id"], errors="ignore")
    y = df[LABEL_COLS]
    
    # Only use the below once you get more data from different people
    
    # sessions = df["session_id"].values

    # X_train, X_test, y_train, y_test, sess_train, sess_test = train_test_split(
    #     X, y, sessions, test_size=0.25, random_state=42
    # )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    results = []
    pred_store = {}

    for model_name, base_model in get_base_models().items():
        print(f"Training {model_name}...")

        model = MultiOutputClassifier(base_model)
        model.fit(X_train, y_train)

        metrics, y_pred = evaluate_multilabel(model, X_test, y_test)

        metrics.update({
            "variant": variant,
            "model": model_name
        })

        results.append(metrics)

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{variant}_{model_name}.pkl")
        joblib.dump(model, model_path)

        # Save predictions
        pred_store[model_name] = {
            "y_true": y_test.values,
            "y_pred": y_pred
            # ,
            # "session_id": sess_test
        }

        # Save per-fault classification report
        report = classification_report(
            y_test, y_pred, target_names=LABEL_COLS, zero_division=0
        )
        with open(f"{OUTPUT_DIR}/{variant}_{model_name}_report.txt", "w") as f:
            f.write(report)

    # Save prediction dictionary
    pred_path = os.path.join(PRED_DIR, f"{variant}_multilabel_predictions.pkl")
    joblib.dump(pred_store, pred_path)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/{variant}_multilabel_results.csv", index=False)

    print(f"[OK] Saved → {OUTPUT_DIR}/{variant}_multilabel_results.csv")
    print(f"[OK] Saved predictions → {pred_path}")


def run_all():
    for fname in os.listdir(FINAL_DIR):
        if fname.startswith("final_features_") and fname.endswith(".csv"):
            variant = fname.replace("final_features_", "").replace(".csv", "")
            process_variant(variant)

if __name__ == "__main__":
    run_all()
