import os
import pandas as pd

def count_reps(features_folder):
    results = []

    for fname in os.listdir(features_folder):
        if not fname.startswith("features_"):
            continue

        # e.g. features_1779582973_raw.csv
        parts = fname.split("_")
        session_id = parts[1]
        condition = fname.replace(".csv", "").replace(session_id, "").replace("features__", "")

        df = pd.read_csv(os.path.join(features_folder, fname))
        rep_count = len(df)

        results.append({
            "session_id": session_id,
            "condition": condition,
            "rep_count": rep_count
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = count_reps("./features")
    df.to_csv("rep_counts.csv", index=False)
    print(df)
