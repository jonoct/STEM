import os
import pandas as pd

STABILITY_DIR = "features/stability"
OUTPUT_DIR = "features/consensus"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Tunable thresholds ===
STABILITY_THRESHOLD = 0.01          # feature must appear in ≥10% of top-K slots
FAULT_COVERAGE_THRESHOLD = 1        # feature must appear in ≥3 faults
MODEL_COVERAGE_THRESHOLD = 0        # feature must appear in ≥2 models


def load_stability_files(variant):
    base = os.path.join(STABILITY_DIR, variant)

    stability_path = os.path.join(base, "feature_stability.csv")
    fault_heatmap_path = os.path.join(base, "feature_fault_heatmap.csv")
    model_heatmap_path = os.path.join(base, "feature_model_heatmap.csv")

    if not os.path.exists(stability_path):
        print(f"[SKIP] No stability data for variant '{variant}'")
        return None, None, None

    stability_df = pd.read_csv(stability_path)
    fault_heatmap = pd.read_csv(fault_heatmap_path, index_col=0)
    model_heatmap = pd.read_csv(model_heatmap_path, index_col=0)

    return stability_df, fault_heatmap, model_heatmap


def select_consensus_features(variant):
    print(f"\n=== Selecting consensus features for variant: {variant} ===")

    stability_df, fault_heatmap, model_heatmap = load_stability_files(variant)
    if stability_df is None:
        return

    # === Apply thresholds ===
    selected = []

    for _, row in stability_df.iterrows():
        feat = row["feature"]
        stability = row["stability"]
        print(feat)
        print(fault_heatmap.index)
        # Fault coverage
        fault_count = fault_heatmap[feat].sum() if feat in fault_heatmap.columns else 0

        # Model coverage
        model_count = model_heatmap[feat].sum() if feat in model_heatmap.columns else 0

        # Apply consensus rules
        if (
            stability >= STABILITY_THRESHOLD and
            fault_count >= FAULT_COVERAGE_THRESHOLD and
            model_count >= MODEL_COVERAGE_THRESHOLD
        ):
            selected.append([feat, stability, fault_count, model_count])

    consensus_df = pd.DataFrame(
        selected,
        columns=["feature", "stability", "fault_coverage", "model_coverage"]
    ).sort_values("stability", ascending=False)

    # === Save output ===
    out_dir = os.path.join(OUTPUT_DIR, variant)
    os.makedirs(out_dir, exist_ok=True)

    consensus_df.to_csv(os.path.join(out_dir, "consensus_features.csv"), index=False)

    print(f"[OK] Saved consensus features → features/consensus/{variant}/consensus_features.csv")


def run_all():
    for variant in os.listdir(STABILITY_DIR):
        if os.path.isdir(os.path.join(STABILITY_DIR, variant)):
            select_consensus_features(variant)


if __name__ == "__main__":
    run_all()
