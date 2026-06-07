# stats_tests.py
import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

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

def mcnemar_per_fault(y_true, y_a, y_b):
    results = []

    for i, label in enumerate(LABEL_COLS):
        both_correct = raw_correct = kal_correct = both_wrong = 0

        for t, r, k in zip(y_true[:, i], y_a[:, i], y_b[:, i]):
            if r == t and k == t:
                both_correct += 1
            elif r == t and k != t:
                raw_correct += 1
            elif r != t and k == t:
                kal_correct += 1
            else:
                both_wrong += 1

        table = [[both_correct, raw_correct],
                 [kal_correct, both_wrong]]

        p = mcnemar(table, exact=True).pvalue

        results.append({
            "fault": label,
            "p_value": p
        })

    return pd.DataFrame(results)

def wilcoxon_macro_f1(a_metrics, b_metrics):
    a = a_metrics["macro_f1"].values
    b = b_metrics["macro_f1"].values
    stat, p = wilcoxon(a, b)
    return stat, p

def cohen_d(a, b):
    diff = b - a
    return diff.mean() / diff.std()

def run_stats():
    metrics_path = os.path.join(RESULTS_FOLDER, "all_dataset_metrics.csv")
    if not os.path.exists(metrics_path):
        print("[ERROR] all_dataset_metrics.csv not found. Run compare_models.py first.")
        return

    metrics = pd.read_csv(metrics_path)
    variants = metrics["data_type"].unique()

    # pairwise comparisons
    for i in range(len(variants)):
        for j in range(i+1, len(variants)):
            va, vb = variants[i], variants[j]
            print(f"\n=== {va} vs {vb} ===")

            a_metrics = metrics[metrics["data_type"] == va]
            b_metrics = metrics[metrics["data_type"] == vb]

            stat, p = wilcoxon_macro_f1(a_metrics, b_metrics)
            d = cohen_d(a_metrics["macro_f1"].values,
                        b_metrics["macro_f1"].values)
            print(f"Wilcoxon macro-F1 p={p:.4f}, Cohen's d={d:.3f}")

            # per-model, per-fault McNemar (using XGBoost Baseline only, for example)
            a_pred_path = os.path.join(RESULTS_FOLDER, f"{va}_predictions.pkl")
            b_pred_path = os.path.join(RESULTS_FOLDER, f"{vb}_predictions.pkl")
            if not (os.path.exists(a_pred_path) and os.path.exists(b_pred_path)):
                print("[WARN] Missing prediction files, skipping McNemar.")
                continue

            a_preds = joblib.load(a_pred_path)
            b_preds = joblib.load(b_pred_path)

            model_name = "XGBoost Baseline"
            if model_name not in a_preds or model_name not in b_preds:
                print("[WARN] Model not found in predictions, skipping McNemar.")
                continue

            y_true = a_preds[model_name]["y_true"]
            y_a = a_preds[model_name]["y_pred"]
            y_b = b_preds[model_name]["y_pred"]

            df_m = mcnemar_per_fault(y_true, y_a, y_b)
            out_csv = os.path.join(RESULTS_FOLDER, f"mcnemar_{va}_vs_{vb}.csv")
            df_m.to_csv(out_csv, index=False)
            print(f"[OK] Saved McNemar per-fault to: {out_csv}")

if __name__ == "__main__":
    run_stats()
