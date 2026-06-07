import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

ML_DATA_DIR = "ml_data"
OUTPUT_DIR = "features/optimal"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def select_features_for_fault(X, y_fault, feature_names, model_name, top_k=20):
    selected = set()

    # 1. Mutual Information
    try:
        mi = mutual_info_classif(X, y_fault, discrete_features=False)
        idx = np.argsort(mi)[::-1][:top_k]
        selected.update(feature_names[idx])
    except:
        pass

    # 2. ANOVA F-score
    try:
        fvals, _ = f_classif(X, y_fault)
        idx = np.argsort(fvals)[::-1][:top_k]
        selected.update(feature_names[idx])
    except:
        pass

    # 3. Model-based importance
    if model_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss"
        )
    elif model_name == "LightGBM":
        model = LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=50, subsample=0.9, colsample_bytree=0.9
        )
    elif model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=300)
    else:
        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300)

    model.fit(X, y_fault)

    # Tree models
    if model_name in ["XGBoost", "LightGBM", "RandomForest"]:
        try:
            importances = model.feature_importances_
            idx = np.argsort(importances)[::-1][:top_k]
            selected.update(feature_names[idx])
        except:
            pass

    # MLP: permutation importance
    if model_name == "MLP":
        try:
            result = permutation_importance(
                model, X, y_fault, scoring="accuracy",
                n_repeats=10, random_state=42
            )
            idx = np.argsort(result.importances_mean)[::-1][:top_k]
            selected.update(feature_names[idx])
        except:
            pass

    return list(selected)


def process_variant(variant):
    ml_path = f"{ML_DATA_DIR}/squat_ml_dataset_{variant}.csv"
    if not os.path.exists(ml_path):
        print(f"[SKIP] No ML dataset for variant '{variant}'")
        return

    print(f"\n=== Processing variant: {variant} ===")

    df = pd.read_csv(ml_path)

    X = df.drop(columns=LABEL_COLS + ["session_id"], errors="ignore")
    y = df[LABEL_COLS]
    feature_names = np.array(X.columns)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    all_selected = set()

    for fault_idx, fault in enumerate(LABEL_COLS):
        y_fault = y[fault].values

        for model_name in ["XGBoost", "LightGBM", "RandomForest", "MLP"]:
            feats = select_features_for_fault(Xs, y_fault, feature_names, model_name, top_k=20)
            all_selected.update(feats)

    optimal_df = df[list(all_selected) + LABEL_COLS + ["session_id"]]

    out_path = f"{OUTPUT_DIR}/features_optimal_{variant}.csv"
    optimal_df.to_csv(out_path, index=False)

    print(f"[OK] Saved optimal features → {out_path}")


def run_all():
    for fname in os.listdir(ML_DATA_DIR):
        if fname.startswith("squat_ml_dataset_") and fname.endswith(".csv"):
            variant = fname.replace("squat_ml_dataset_", "").replace(".csv", "")
            process_variant(variant)


if __name__ == "__main__":
    run_all()
