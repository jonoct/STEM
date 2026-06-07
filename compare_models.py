# compare_models.py
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, hamming_loss, accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

ML_DATA_FOLDER = "./ml_data"
MODELS_FOLDER = "./models"
RESULTS_FOLDER = "./results"

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


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.drop(columns=LABEL_COLS + ["session_id", "rep_start", "rep_end"])
    y = df[LABEL_COLS]
    sessions = df["session_id"].values
    return X.values, y.values, sessions


def session_aware_split(X, y, sessions, seed=42):
    unique_sessions = np.unique(sessions)

    train_sess, test_sess = train_test_split(unique_sessions, test_size=0.2, random_state=seed)
    train_sess, val_sess = train_test_split(train_sess, test_size=0.2, random_state=seed)

    def mask(sess_ids):
        return np.isin(sessions, sess_ids)

    train_mask = mask(train_sess)
    val_mask = mask(val_sess)
    test_mask = mask(test_sess)

    return (
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        X[test_mask], y[test_mask],
        train_sess, val_sess, test_sess
    )


def evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n=== Training {name} ===")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    h_loss = hamming_loss(y_test, y_pred)
    subset_acc = accuracy_score(y_test, y_pred)

    print(f"{name} Macro F1: {macro_f1:.4f}")
    print(f"{name} Hamming Loss: {h_loss:.4f}")
    print(f"{name} Subset Accuracy: {subset_acc:.4f}")
    print(f"{name} Train Time: {train_time:.2f}s")

    return {
        "name": name,
        "macro_f1": macro_f1,
        "hamming_loss": h_loss,
        "subset_acc": subset_acc,
        "train_time": train_time,
        "model": model,
        "y_pred": y_pred,
    }


def run_for_dataset(dataset_path):
    base = os.path.basename(dataset_path)
    variant = base[len("squat_ml_dataset_"):-len(".csv")]

    print(f"\n=== DATASET: {variant} ===")

    X, y, sessions = load_dataset(dataset_path)

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        train_sess, val_sess, test_sess
    ) = session_aware_split(X, y, sessions)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model_list = [
        ("XGBoost Baseline", OneVsRestClassifier(XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", tree_method="hist", n_jobs=-1
        ))),
        ("LightGBM", OneVsRestClassifier(LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            num_leaves=50, subsample=0.9,
            colsample_bytree=0.9, n_jobs=-1
        ))),
        ("Random Forest", OneVsRestClassifier(RandomForestClassifier(
            n_estimators=500, max_depth=None, n_jobs=-1
        ))),
        ("MLP Neural Network", OneVsRestClassifier(MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            learning_rate_init=0.001,
            max_iter=300
        )))
    ]

    results = []
    preds_bundle = {}

    for name, model in model_list:
        r = evaluate_model(name, model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test)
        results.append({
            "model": name,
            "data_type": variant,
            "macro_f1": r["macro_f1"],
            "hamming_loss": r["hamming_loss"],
            "subset_acc": r["subset_acc"],
            "train_time": r["train_time"],
        })
        preds_bundle[name] = {
            "y_true": y_test,
            "y_pred": r["y_pred"],
        }

    # pick best
    best_idx = np.argmax([r["macro_f1"] for r in results])
    best_name, best_model = model_list[best_idx][0], model_list[best_idx][1]

    print(f"\nBest model for {variant}: {best_name}")

    os.makedirs(MODELS_FOLDER, exist_ok=True)
    model_path = os.path.join(MODELS_FOLDER, f"best_{variant}.pkl")
    joblib.dump(
        {"model": best_model, "scaler": scaler, "label_cols": LABEL_COLS},
        model_path
    )
    print(f"[OK] Saved best model to: {model_path}")

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    preds_path = os.path.join(RESULTS_FOLDER, f"{variant}_predictions.pkl")
    joblib.dump(preds_bundle, preds_path)
    print(f"[OK] Saved predictions to: {preds_path}")

    return pd.DataFrame(results)


def run_all_datasets():
    all_metrics = []

    for fname in os.listdir(ML_DATA_FOLDER):
        if fname.startswith("squat_ml_dataset_") and fname.endswith(".csv"):
            df = run_for_dataset(os.path.join(ML_DATA_FOLDER, fname))
            all_metrics.append(df)

    if all_metrics:
        metrics_all = pd.concat(all_metrics, ignore_index=True)
        out_csv = os.path.join(RESULTS_FOLDER, "all_dataset_metrics.csv")
        metrics_all.to_csv(out_csv, index=False)
        print(f"\n[OK] Saved all metrics to: {out_csv}")


if __name__ == "__main__":
    run_all_datasets()
