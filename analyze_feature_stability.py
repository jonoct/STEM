import os
import pandas as pd
from collections import defaultdict

ANALYSIS_DIR = "features/analysis"
OUTPUT_DIR = "features/stability"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K = 20  # how many top features per ranking file to consider


def load_rankings_for_variant(variant):
    variant_dir = os.path.join(ANALYSIS_DIR, variant)
    if not os.path.exists(variant_dir):
        print(f"[SKIP] No analysis directory for variant '{variant}'")
        return None

    ranking_files = [
        f for f in os.listdir(variant_dir)
        if f.startswith("rankings_") and f.endswith(".csv")
    ]

    return [os.path.join(variant_dir, f) for f in ranking_files]


def analyze_variant(variant):
    print(f"\n=== Analyzing feature stability for variant: {variant} ===")

    ranking_files = load_rankings_for_variant(variant)
    if ranking_files is None:
        return

    feature_counts = defaultdict(int)
    feature_fault_counts = defaultdict(lambda: defaultdict(int))
    feature_model_counts = defaultdict(lambda: defaultdict(int))

    total_possible = 0

    for path in ranking_files:
        df = pd.read_csv(path)

        # Extract fault + model from filename
        fname = os.path.basename(path)
        parts = fname.replace("rankings_", "").replace(".csv", "").split("_")
        fault = parts[0]
        model = parts[1]

        # Sort by score descending
        df_sorted = df.sort_values("score", ascending=False)

        # Take top_k features
        top_features = df_sorted["feature"].head(TOP_K).tolist()

        for feat in top_features:
            feature_counts[feat] += 1
            feature_fault_counts[feat][fault] += 1
            feature_model_counts[feat][model] += 1

        total_possible += TOP_K

    # Build stability table
    stability_rows = []
    for feat, count in feature_counts.items():
        stability = count / total_possible
        stability_rows.append([feat, count, stability])

    stability_df = pd.DataFrame(stability_rows, columns=["feature", "count", "stability"])
    stability_df = stability_df.sort_values("stability", ascending=False)

    # Save stability table
    out_dir = os.path.join(OUTPUT_DIR, variant)
    os.makedirs(out_dir, exist_ok=True)

    stability_df.to_csv(os.path.join(out_dir, "feature_stability.csv"), index=False)

    # Save per-fault heatmap CSV
    fault_heatmap = pd.DataFrame(feature_fault_counts).fillna(0).astype(int)
    fault_heatmap.to_csv(os.path.join(out_dir, "feature_fault_heatmap.csv"))

    # Save per-model heatmap CSV
    model_heatmap = pd.DataFrame(feature_model_counts).fillna(0).astype(int)
    model_heatmap.to_csv(os.path.join(out_dir, "feature_model_heatmap.csv"))

    print(f"[OK] Stability analysis complete for variant '{variant}'")


def run_all():
    for variant in os.listdir(ANALYSIS_DIR):
        if os.path.isdir(os.path.join(ANALYSIS_DIR, variant)):
            analyze_variant(variant)


if __name__ == "__main__":
    run_all()
