import os
import pandas as pd

SINGLE_DIR = "model_results"
MULTI_DIR = "model_results_multilabel"
OUT_PATH = "results/optimal/all_dataset_metrics.csv"

def rebuild_all_metrics():
    rows = []

    # -----------------------------
    # SINGLE-LABEL RESULTS
    # -----------------------------
    for variant in os.listdir(SINGLE_DIR):
        variant_dir = os.path.join(SINGLE_DIR, variant)
        if not os.path.isdir(variant_dir):
            continue

        csv_path = os.path.join(variant_dir, "model_results.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df["data_type"] = variant
        rows.append(df)

    # -----------------------------
    # MULTI-LABEL RESULTS
    # -----------------------------
    for fname in os.listdir(MULTI_DIR):
        if fname.endswith("_multilabel_results.csv"):
            variant = fname.replace("_multilabel_results.csv", "")
            csv_path = os.path.join(MULTI_DIR, fname)

            df = pd.read_csv(csv_path)
            df["fault"] = "ALL"
            df["data_type"] = variant
            rows.append(df)

    # -----------------------------
    # MERGE + SAVE
    # -----------------------------
    if rows:
        df_all = pd.concat(rows, ignore_index=True)
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        df_all.to_csv(OUT_PATH, index=False)
        print(f"[OK] Rebuilt metrics → {OUT_PATH}")
    else:
        print("[ERROR] No metrics found.")

if __name__ == "__main__":
    rebuild_all_metrics()
