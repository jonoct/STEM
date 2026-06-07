# per_fault_analysis.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

ML_DATA_FOLDER = "./ml_data"
RESULTS_FOLDER = "./results/per_fault_analysis"

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
    return X, y.values, sessions, df


def session_split(X, y_fault, sessions, seed=42):
    unique_sessions = np.unique(sessions)

    train_sess, test_sess = train_test_split(unique_sessions, test_size=0.2, random_state=seed)
    train_sess, val_sess = train_test_split(train_sess, test_size=0.2, random_state=seed)

    def mask(sess_ids):
        return np.isin(sessions, sess_ids)

    train_mask = mask(train_sess)
    val_mask = mask(val_sess)
    test_mask = mask(test_sess)

    return (
        X.iloc[train_mask], y_fault[train_mask],
        X.iloc[val_mask], y_fault[val_mask],
        X.iloc[test_mask], y_fault[test_mask],
        train_sess, val_sess, test_sess
    )


def build_models():
    return {
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", tree_method="hist", n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            num_leaves=50, subsample=0.9,
            colsample_bytree=0.9, n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=None, n_jobs=-1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            learning_rate_init=0.001,
            max_iter=300
        ),
    }


def shap_plot(model, X_sample, feature_names, out_path):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[SHAP] Skipping SHAP: {e}")


def permutation_importance_mlp(model, X_test_s, y_test, feature_names, out_path):
    result = permutation_importance(
        model, X_test_s, y_test,
        scoring="accuracy",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    importances = result.importances_mean
    idx = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(6, 5))
    sns.barplot(
        x=importances[idx],
        y=[feature_names[i] for i in idx],
        orient="h"
    )
    plt.title("MLP Permutation Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_fault_for_variant(variant, dataset_path):
    print(f"\n=== VARIANT: {variant} ===")

    out_dir = os.path.join(RESULTS_FOLDER, variant)
    os.makedirs(out_dir, exist_ok=True)

    X, y, sessions, df_raw = load_dataset(dataset_path)
    feature_names = X.columns

    results = []

    for fault_idx, fault in enumerate(LABEL_COLS):
        print(f"\n--- Fault: {fault} ---")

        y_fault = y[:, fault_idx]

        (
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            train_sess, val_sess, test_sess
        ) = session_split(X, y_fault, sessions)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        fault_dir = os.path.join(out_dir, fault)
        os.makedirs(fault_dir, exist_ok=True)

        models = build_models()

        for model_name, model in models.items():
            print(f"  Training {model_name}...")

            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            macro_f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
            acc = accuracy_score(y_test, y_pred)

            try:
                roc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
            except:
                roc = np.nan

            results.append({
                "variant": variant,
                "fault": fault,
                "model": model_name,
                "macro_f1": macro_f1,
                "accuracy": acc,
                "roc_auc": roc,
            })

            # SHAP for tree models
            if model_name in ["XGBoost", "LightGBM", "RandomForest"]:
                shap_path = os.path.join(fault_dir, f"{model_name}_shap.png")
                shap_plot(model, X_test_s[:300], feature_names, shap_path)

            # Permutation importance for MLP
            if model_name == "MLP":
                perm_path = os.path.join(fault_dir, f"{model_name}_perm.png")
                permutation_importance_mlp(model, X_test_s, y_test, feature_names, perm_path)

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(out_dir, "fault_results.csv"), index=False)
    print(f"[OK] Saved results for {variant}")


def run_all_variants():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    for fname in os.listdir(ML_DATA_FOLDER):
        if fname.startswith("squat_ml_dataset_") and fname.endswith(".csv"):
            variant = fname[len("squat_ml_dataset_"):-len(".csv")]
            path = os.path.join(ML_DATA_FOLDER, fname)
            run_fault_for_variant(variant, path)


if __name__ == "__main__":
    run_all_variants()
