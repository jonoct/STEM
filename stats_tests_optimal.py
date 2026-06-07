# stats_tests.py
import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

RESULTS_FOLDER = "./results/optimal"
PRED_SINGLE_FOLDER = "./model_predictions"
PRED_MULTI_FOLDER = "./model_predictions_multilabel"

# Set these explicitly
BASELINE_VARIANT = "raw_smoothed"
BEST_VARIANT = "airpods_acc"

# If you know the best multi-label model name, set it here
BEST_MULTI_MODEL_NAME = "LightGBM Multi"

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
    """Run McNemar test per fault."""
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
            "both_correct": both_correct,
            "raw_correct_only": raw_correct,
            "best_correct_only": kal_correct,
            "both_wrong": both_wrong,
            "p_value": p
        })

    return pd.DataFrame(results)


def wilcoxon_macro_f1(a_metrics, b_metrics):
    """Wilcoxon signed-rank test on macro-F1."""
    a = a_metrics["f1_macro"].values
    b = b_metrics["f1_macro"].values
    stat, p = wilcoxon(a, b)
    return stat, p


def cohen_d(a, b):
    """Effect size."""
    diff = b - a
    return diff.mean() / (diff.std() + 1e-9)


def load_predictions(variant):
    """Load both single-label and multi-label predictions."""
    preds = {}

    # Single-label
    single_path = os.path.join(PRED_SINGLE_FOLDER, f"{variant}_predictions.pkl")
    if os.path.exists(single_path):
        preds.update(joblib.load(single_path))

    # Multi-label
    multi_path = os.path.join(PRED_MULTI_FOLDER, f"{variant}_multilabel_predictions.pkl")
    if os.path.exists(multi_path):
        preds.update(joblib.load(multi_path))

    return preds


def run_stats():
    metrics_path = os.path.join(RESULTS_FOLDER, "all_dataset_metrics.csv")
    if not os.path.exists(metrics_path):
        print("[ERROR] all_dataset_metrics.csv not found.")
        return

    metrics = pd.read_csv(metrics_path)

    # Split metrics into single-label and multi-label
    single_label_metrics = metrics[metrics["fault"] != "ALL"]
    multi_label_metrics  = metrics[metrics["fault"] == "ALL"]

    # -----------------------------
    # SINGLE-LABEL: baseline vs best variant
    # -----------------------------
    a_single = single_label_metrics[single_label_metrics["data_type"] == BASELINE_VARIANT]
    b_single = single_label_metrics[single_label_metrics["data_type"] == BEST_VARIANT]

    single_stats_rows = []

    if len(a_single) > 0 and len(b_single) > 0:
        stat, p = wilcoxon_macro_f1(a_single, b_single)
        d = cohen_d(a_single["f1_macro"].values, b_single["f1_macro"].values)

        single_stats_rows.append({
            "baseline_variant": BASELINE_VARIANT,
            "best_variant": BEST_VARIANT,
            "level": "single_label",
            "wilcoxon_stat": stat,
            "wilcoxon_p": p,
            "cohen_d": d,
            "n_faults": len(a_single)
        })

        print(f"[Single-label] {BASELINE_VARIANT} vs {BEST_VARIANT}: "
              f"Wilcoxon p={p:.4f}, d={d:.3f}")
    else:
        print("[Single-label] No comparable rows, skipping.")

    # -----------------------------
    # MULTI-LABEL: baseline vs best variant
    # -----------------------------
    a_multi = multi_label_metrics[multi_label_metrics["data_type"] == BASELINE_VARIANT]
    b_multi = multi_label_metrics[multi_label_metrics["data_type"] == BEST_VARIANT]

    if len(a_multi) > 0 and len(b_multi) > 0:
        stat, p = wilcoxon_macro_f1(a_multi, b_multi)
        d = cohen_d(a_multi["f1_macro"].values, b_multi["f1_macro"].values)

        single_stats_rows.append({
            "baseline_variant": BASELINE_VARIANT,
            "best_variant": BEST_VARIANT,
            "level": "multi_label",
            "wilcoxon_stat": stat,
            "wilcoxon_p": p,
            "cohen_d": d,
            "n_faults": len(a_multi)
        })

        print(f"[Multi-label] {BASELINE_VARIANT} vs {BEST_VARIANT}: "
              f"Wilcoxon p={p:.4f}, d={d:.3f}")
    else:
        print("[Multi-label] No comparable rows, skipping.")

    # Save Wilcoxon + Cohen's d summary
    stats_out = os.path.join(RESULTS_FOLDER, f"stats_wilcoxon_cohen_{BASELINE_VARIANT}_vs_{BEST_VARIANT}.csv")
    pd.DataFrame(single_stats_rows).to_csv(stats_out, index=False)
    print(f"[OK] Saved Wilcoxon/Cohen's d summary to: {stats_out}")

    # -----------------------------
    # MCNEMAR: best multi-label model only
    # -----------------------------
    a_preds = load_predictions(BASELINE_VARIANT)
    b_preds = load_predictions(BEST_VARIANT)

    if BEST_MULTI_MODEL_NAME not in a_preds or BEST_MULTI_MODEL_NAME not in b_preds:
        print(f"[WARN] {BEST_MULTI_MODEL_NAME} not found in predictions for one of the variants.")
        return

    print(f"Running McNemar for best multi-label model: {BEST_MULTI_MODEL_NAME}")

    y_true = a_preds[BEST_MULTI_MODEL_NAME]["y_true"]
    y_a = a_preds[BEST_MULTI_MODEL_NAME]["y_pred"]
    y_b = b_preds[BEST_MULTI_MODEL_NAME]["y_pred"]

    y_true = np.asarray(y_true)
    y_a = np.asarray(y_a)
    y_b = np.asarray(y_b)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_a.ndim == 1:
        y_a = y_a.reshape(-1, 1)
    if y_b.ndim == 1:
        y_b = y_b.reshape(-1, 1)

    # Require multi-label shape
    if y_a.shape[1] == 1 or y_b.shape[1] == 1:
        print(f"[SKIP] McNemar not applicable: {BEST_MULTI_MODEL_NAME} appears single-label.")
        return

    df_m = mcnemar_per_fault(y_true, y_a, y_b)
    out_csv = os.path.join(
        RESULTS_FOLDER,
        f"mcnemar_{BASELINE_VARIANT}_vs_{BEST_VARIANT}_{BEST_MULTI_MODEL_NAME.replace(' ', '_')}.csv"
    )
    df_m.to_csv(out_csv, index=False)
    print(f"[OK] Saved McNemar per-fault to: {out_csv}")


if __name__ == "__main__":
    run_stats()
