import os
import pandas as pd

RESULTS_FOLDER = "./results/optimal"

baseline_variant = "raw_smoothed"   # whatever name you used
new_variant = "airpods_acc"         # example

# Load per-fault metrics
base_metrics = pd.read_csv(os.path.join(RESULTS_FOLDER, baseline_variant, "model_results.csv"))
new_metrics  = pd.read_csv(os.path.join(RESULTS_FOLDER, new_variant, "model_results.csv"))

# Keep best model per fault (by f1_macro)
def best_per_fault(df):
    return (df.sort_values("f1_macro", ascending=False)
              .groupby("fault", as_index=False)
              .first())

base_best = best_per_fault(base_metrics)
new_best  = best_per_fault(new_metrics)

merged = base_best.merge(
    new_best,
    on="fault",
    suffixes=("_base", "_new")
)

merged["delta_f1_macro"] = merged["f1_macro_new"] - merged["f1_macro_base"]
merged["delta_accuracy"] = merged["accuracy_new"] - merged["accuracy_base"]

merged.to_csv(os.path.join(RESULTS_FOLDER, f"compare_single_{baseline_variant}_vs_{new_variant}.csv"), index=False)

ML_RESULTS_FOLDER = "./model_results_multilabel"

base_multi = pd.read_csv(os.path.join(ML_RESULTS_FOLDER, f"{baseline_variant}_multilabel_results.csv"))
new_multi  = pd.read_csv(os.path.join(ML_RESULTS_FOLDER, f"{new_variant}_multilabel_results.csv"))

# Take best model per variant by f1_macro
base_best_multi = base_multi.sort_values("f1_macro", ascending=False).iloc[0]
new_best_multi  = new_multi.sort_values("f1_macro", ascending=False).iloc[0]

summary = pd.DataFrame([{
    "variant_base": baseline_variant,
    "model_base": base_best_multi["model"],
    "f1_macro_base": base_best_multi["f1_macro"],
    "subset_acc_base": base_best_multi["subset_accuracy"],
    "variant_new": new_variant,
    "model_new": new_best_multi["model"],
    "f1_macro_new": new_best_multi["f1_macro"],
    "subset_acc_new": new_best_multi["subset_accuracy"],
    "delta_f1_macro": new_best_multi["f1_macro"] - base_best_multi["f1_macro"],
    "delta_subset_acc": new_best_multi["subset_accuracy"] - base_best_multi["subset_accuracy"],
}])

summary.to_csv(os.path.join(RESULTS_FOLDER, f"compare_multi_{baseline_variant}_vs_{new_variant}.csv"), index=False)
