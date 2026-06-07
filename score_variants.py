import os
import pandas as pd
import numpy as np

COMPARISON_FOLDER = "./results/comparisons"
BASELINE = "raw_smoothed"

# ---------------------------------------------------------
# SCORING WEIGHTS (you can tune these if needed)
# ---------------------------------------------------------
W_MULTI_F1 = 20
W_MULTI_SUBSET = 10
W_SINGLE_MEAN_F1 = 20
W_SINGLE_FAULT_IMPROVEMENTS = 10

W_INTERP_DIFFICULTY = 5
W_INTERP_PURITY = 5
W_INTERP_FEATURE_ATTRIB = 5   # placeholder if you want to add later

W_BIOMECH = 5                 # qualitative tie-breaker (optional)
# ---------------------------------------------------------


def load_variant_results(variant):
    """Load the three comparison files for a given variant."""
    folder = os.path.join(COMPARISON_FOLDER, f"{variant}_vs_{BASELINE}")

    single_path = os.path.join(folder, "single_label_comparison.csv")
    multi_path = os.path.join(folder, "multi_label_comparison.csv")
    interp_path = os.path.join(folder, "interpretability_comparison.csv")

    if not (os.path.exists(single_path) and os.path.exists(multi_path) and os.path.exists(interp_path)):
        print(f"[WARN] Missing comparison files for {variant}. Skipping.")
        return None

    single = pd.read_csv(single_path)
    multi = pd.read_csv(multi_path)
    interp = pd.read_csv(interp_path)

    return single, multi, interp


def score_variant(single, multi, interp):
    """Compute a 100-point score for a variant."""
    score = 0

    # -----------------------------
    # PERFORMANCE (60%)
    # -----------------------------
    # Multi-label global metrics
    score += W_MULTI_F1 * float(multi["delta_f1_macro"].iloc[0])
    score += W_MULTI_SUBSET * float(multi["delta_subset_acc"].iloc[0])

    # Single-label mean improvement
    score += W_SINGLE_MEAN_F1 * single["delta_f1"].mean()

    # Fraction of faults improved
    frac_improved = (single["delta_f1"] > 0).sum() / len(single)
    score += W_SINGLE_FAULT_IMPROVEMENTS * frac_improved

    # -----------------------------
    # INTERPRETABILITY (15%)
    # -----------------------------
    # Difficulty (lower is better)
    score += W_INTERP_DIFFICULTY * (-interp["delta_difficulty"].mean())

    # Cluster purity (higher is better)
    score += W_INTERP_PURITY * interp["delta_purity"].mean()

    # Placeholder for feature attribution improvements
    # (You can add a metric later if desired)
    score += 0

    # -----------------------------
    # BIOMECHANICAL PLAUSIBILITY (5%)
    # -----------------------------
    # Optional: you can manually add +5 for variants that show
    # cleaner SHAP, PCA loadings, or more plausible feature correlations.
    # For now, leave as 0.
    score += 0

    return score


def run_scoring():
    variants = [
        d.replace(f"_vs_{BASELINE}", "")
        for d in os.listdir(COMPARISON_FOLDER)
        if d.endswith(f"_vs_{BASELINE}")
    ]

    results = []

    for variant in variants:
        loaded = load_variant_results(variant)
        if loaded is None:
            continue

        single, multi, interp = loaded
        score = score_variant(single, multi, interp)

        results.append({
            "variant": variant,
            "score": score,
            "delta_f1_macro": float(multi["delta_f1_macro"].iloc[0]),
            "delta_subset_acc": float(multi["delta_subset_acc"].iloc[0]),
            "mean_single_delta_f1": single["delta_f1"].mean(),
            "faults_improved": (single["delta_f1"] > 0).sum(),
            "mean_delta_difficulty": interp["delta_difficulty"].mean(),
            "mean_delta_purity": interp["delta_purity"].mean(),
        })

    df = pd.DataFrame(results).sort_values("score", ascending=False)
    print("\n=== VARIANT RANKING ===")
    print(df.to_string(index=False))

    df.to_csv("./results/comparisons/variant_scores.csv", index=False)
    print("\nSaved ranking to results/comparisons/variant_scores.csv")


if __name__ == "__main__":
    run_scoring()
