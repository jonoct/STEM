import os
import pandas as pd

RESULTS_FOLDER = "./results/optimal"
MULTI_FOLDER = "./model_results_multilabel"

baseline_variant = "raw_smoothed"

def best_per_fault(df):
    return (df.sort_values("macro_f1", ascending=False)
              .groupby("fault", as_index=False)
              .first())

def compare_single_label(baseline_variant, new_variant):
    base = pd.read_csv(f"{RESULTS_FOLDER}/{baseline_variant}/metrics_summary.csv")
    new  = pd.read_csv(f"{RESULTS_FOLDER}/{new_variant}/metrics_summary.csv")

    base_best = best_per_fault(base)
    new_best  = best_per_fault(new)

    merged = base_best.merge(
        new_best,
        on="fault",
        suffixes=("_base", "_new")
    )

    merged["delta_f1"] = merged["macro_f1_new"] - merged["macro_f1_base"]
    merged["delta_acc"] = merged["accuracy_new"] - merged["accuracy_base"]

    return merged

def compare_multilabel(baseline_variant, new_variant):
    base = pd.read_csv(f"{MULTI_FOLDER}/{baseline_variant}_multilabel_results.csv")
    new  = pd.read_csv(f"{MULTI_FOLDER}/{new_variant}_multilabel_results.csv")

    base_best = base.sort_values("f1_macro", ascending=False).iloc[0]
    new_best  = new.sort_values("f1_macro", ascending=False).iloc[0]

    return pd.DataFrame([{
        "variant_base": baseline_variant,
        "model_base": base_best["model"],
        "f1_macro_base": base_best["f1_macro"],
        "subset_acc_base": base_best["subset_accuracy"],

        "variant_new": new_variant,
        "model_new": new_best["model"],
        "f1_macro_new": new_best["f1_macro"],
        "subset_acc_new": new_best["subset_accuracy"],

        "delta_f1_macro": new_best["f1_macro"] - base_best["f1_macro"],
        "delta_subset_acc": new_best["subset_accuracy"] - base_best["subset_accuracy"],
    }])

def compare_interpretability(baseline_variant, new_variant):
    diff_base = pd.read_csv(f"{RESULTS_FOLDER}/{baseline_variant}/fault_difficulty_ranking.csv")
    diff_new  = pd.read_csv(f"{RESULTS_FOLDER}/{new_variant}/fault_difficulty_ranking.csv")

    pur_base = pd.read_csv(f"{RESULTS_FOLDER}/{baseline_variant}/cluster_purity_metrics.csv")
    pur_new  = pd.read_csv(f"{RESULTS_FOLDER}/{new_variant}/cluster_purity_metrics.csv")

    merged = diff_base.merge(diff_new, on="fault", suffixes=("_base", "_new"))
    merged = merged.merge(pur_base, on="fault")
    merged = merged.merge(pur_new, on="fault", suffixes=("_pur_base", "_pur_new"))

    merged["delta_difficulty"] = merged["difficulty_new"] - merged["difficulty_base"]
    merged["delta_purity"] = merged["purity_pur_new"] - merged["purity_pur_base"]

    return merged

def compare_variant_to_baseline(new_variant):
    out_dir = f"./results/comparisons/{new_variant}_vs_{baseline_variant}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Comparing {new_variant} vs {baseline_variant} ===")

    # Single-label
    single = compare_single_label(baseline_variant, new_variant)
    single.to_csv(f"{out_dir}/single_label_comparison.csv", index=False)

    # Multi-label
    multi = compare_multilabel(baseline_variant, new_variant)
    multi.to_csv(f"{out_dir}/multi_label_comparison.csv", index=False)

    # Interpretability
    interpret = compare_interpretability(baseline_variant, new_variant)
    interpret.to_csv(f"{out_dir}/interpretability_comparison.csv", index=False)

    print(f"Saved comparison results to: {out_dir}")


def run_all():
    variants = [
        v for v in os.listdir(RESULTS_FOLDER)
        if os.path.isdir(os.path.join(RESULTS_FOLDER, v))
        and v != baseline_variant
    ]

    for v in variants:
        compare_variant_to_baseline(v)

if __name__ == "__main__":
    run_all()
