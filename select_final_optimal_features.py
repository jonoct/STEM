import os
import pandas as pd

ML_DATA_DIR = "ml_data"
STABILITY_DIR = "features/stability"
CONSENSUS_DIR = "features/consensus"
OUTPUT_DIR = "features/optimal/final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Tunable thresholds ===
STABILITY_THRESHOLD = 0.01
FAULT_COVERAGE_THRESHOLD = 3
MODEL_COVERAGE_THRESHOLD = 2

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


def load_consensus(variant):
    path = os.path.join(CONSENSUS_DIR, variant, "consensus_features.csv")
    if not os.path.exists(path):
        print(f"[SKIP] No consensus features for variant '{variant}'")
        return None
    return pd.read_csv(path)


def load_stability(variant):
    base = os.path.join(STABILITY_DIR, variant)
    stab_path = os.path.join(base, "feature_stability.csv")
    fault_path = os.path.join(base, "feature_fault_heatmap.csv")
    model_path = os.path.join(base, "feature_model_heatmap.csv")

    if not os.path.exists(stab_path):
        print(f"[SKIP] No stability data for variant '{variant}'")
        return None, None, None

    return (
        pd.read_csv(stab_path),
        pd.read_csv(fault_path, index_col=0),
        pd.read_csv(model_path, index_col=0),
    )


def select_final_features(variant):
    print(f"\n=== Selecting final optimal features for variant: {variant} ===")

    consensus_df = load_consensus(variant)
    stability_df, fault_heatmap, model_heatmap = load_stability(variant)

    if consensus_df is None or stability_df is None:
        return

    final_features = []

    for _, row in stability_df.iterrows():
        feat = row["feature"]
        stability = row["stability"]

        fault_count = fault_heatmap[feat].sum() if feat in fault_heatmap.columns else 0
        model_count = model_heatmap[feat].sum() if feat in model_heatmap.columns else 0

        if (
            stability >= STABILITY_THRESHOLD and
            fault_count >= FAULT_COVERAGE_THRESHOLD and
            model_count >= MODEL_COVERAGE_THRESHOLD
        ):
            final_features.append(feat)

    # Load ML dataset
    ml_path = os.path.join(ML_DATA_DIR, f"squat_ml_dataset_{variant}.csv")
    df = pd.read_csv(ml_path)

    # Filter to final features + labels + session_id
    keep_cols = [c for c in final_features if c in df.columns]
    keep_cols += LABEL_COLS
    keep_cols += ["session_id"]

    final_df = df[keep_cols]

    out_path = os.path.join(OUTPUT_DIR, f"final_features_{variant}.csv")
    final_df.to_csv(out_path, index=False)

    print(f"[OK] Saved final optimal features → {out_path}")


def run_all():
    for variant in os.listdir(CONSENSUS_DIR):
        if os.path.isdir(os.path.join(CONSENSUS_DIR, variant)):
            select_final_features(variant)


if __name__ == "__main__":
    run_all()
