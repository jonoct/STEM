import os
import pandas as pd
import feature_extraction as fe

def build_ml_dataset(fused_folder, out_csv, label_dict=None):
    """
    Build a master ML dataset from all fused CSVs in a folder.
    
    label_dict: optional dict mapping session_id -> dict of labels
        Example:
        {
            "1779514418": {"heel_lift": 1, "knee_valgus": 0, ...},
            "1779512826": {"heel_lift": 0, "knee_valgus": 1, ...},
        }
    """

    all_rows = []

    for fname in os.listdir(fused_folder):
        if not fname.endswith("_fused.csv"):
            continue

        session_id = fname.split("_")[-2]  # e.g. merged_pose_IMU_1779514418_fused.csv
        fused_path = os.path.join(fused_folder, fname)

        print(f"Processing session {session_id}")

        # Extract rep-level features
        rep_df = fe.extract_features_from_fused_csv(
            fused_csv=fused_path,
            out_csv=None  # we don't need per-session output
        )

        # Add session_id
        rep_df["session_id"] = session_id

        # Add labels if provided
        if label_dict and session_id in label_dict:
            for label_name, label_value in label_dict[session_id].items():
                rep_df[label_name] = label_value

        all_rows.append(rep_df)

    # Combine all sessions
    if len(all_rows) == 0:
        raise RuntimeError("No fused CSVs found in folder")

    master_df = pd.concat(all_rows, ignore_index=True)

    # Save
    master_df.to_csv(out_csv, index=False)
    print(f"Saved ML dataset to: {out_csv}")

    return master_df

label_dict = {
    "1779582973": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 0,
        "butt_wink": 1,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 0,
        "chest_collapse": 0,
        "shallow_depth": 0,
        "excessive_depth": 1,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 0,
        "good_technique": 0
    },
    "1779583096": {
        "heel_lift": 1,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 1,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 1,
        "butt_wink": 1,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 1,
        "chest_collapse": 0,
        "shallow_depth": 0,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 0,
        "good_technique": 0
    },
    "1779583231": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 1,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 0,
        "butt_wink": 1,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 0,
        "chest_collapse": 0,
        "shallow_depth": 0,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 1,
        "good_technique": 0
    },
    "1779583350": {
        "heel_lift": 0,
        "toe_out": 1,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 0,
        "butt_wink": 1,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 1,
        "chest_collapse": 0,
        "shallow_depth": 0,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 1,
        "good_technique": 0
    },
    "1779583499": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 1,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 1,
        "butt_wink": 1,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 1,
        "chest_collapse": 0,
        "shallow_depth": 1,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 0,
        "good_technique": 0
    },
    "1779583658": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 1,
        "knee_varus": 0,
        "excessive_knee_travel": 1,
        "butt_wink": 1,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 0,
        "chest_collapse": 0,
        "shallow_depth": 0,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 1,
        "good_technique": 0
    },
    "1779583797": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 1,
        "excessive_knee_travel": 0,
        "butt_wink": 1,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 0,
        "chest_collapse": 0,
        "shallow_depth": 0,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 1,
        "good_technique": 0
    },
    "1779583966": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 1,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 1,
        "butt_wink": 1,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 1,
        "chest_collapse": 1,
        "shallow_depth": 0,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 0,
        "good_technique": 0
    },
    "1779584093": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 0,
        "butt_wink": 1,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 0,
        "chest_collapse": 0,
        "shallow_depth": 0,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 0,
        "good_technique": 0
    },
    "1779584434": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 0,
        "butt_wink": 1,
        "lumbar_extension": 1,
        "hip_shift": 0,
        "forward_lean": 1,
        "chest_collapse": 1,
        "shallow_depth": 1,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 0,
        "good_technique": 0
    },
    "1779584592": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 0,
        "butt_wink": 1,
        "lumbar_extension": 0,
        "hip_shift": 1,
        "forward_lean": 0,
        "chest_collapse": 0,
        "shallow_depth": 0,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 1,
        "poor_stance": 1,
        "good_technique": 0
    },
    "1779584721": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 0,
        "butt_wink": 1,
        "lumbar_extension": 0,
        "hip_shift": 1,
        "forward_lean": 0,
        "chest_collapse": 0,
        "shallow_depth": 0,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 1,
        "poor_stance": 1,
        "good_technique": 0
    }
    
    ,
    "1780214416": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 0,
        "butt_wink": 0,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 1,
        "chest_collapse": 0,
        "shallow_depth": 1,
        "excessive_depth": 0,
        "weak_bracing": 1,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 0,
        "good_technique": 0
    },
    "1780214499": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 0,
        "butt_wink": 0,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 1,
        "chest_collapse": 1,
        "shallow_depth": 1,
        "excessive_depth": 0,
        "weak_bracing": 1,
        "poor_breathing": 1,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 0,
        "good_technique": 0
    },
    "1780214625": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 0,
        "butt_wink": 0,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 0,
        "chest_collapse": 0,
        "shallow_depth": 1,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 0,
        "good_technique": 0
    },
    "1780214702": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 0,
        "butt_wink": 1,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 1,
        "chest_collapse": 0,
        "shallow_depth": 0,
        "excessive_depth": 1,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 1,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 0,
        "good_technique": 0
    },
    "1780214797": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 0,
        "butt_wink": 1,
        "lumbar_extension": 0,
        "hip_shift": 1,
        "forward_lean": 0,
        "chest_collapse": 0,
        "shallow_depth": 0,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 1,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 1,
        "poor_stance": 0,
        "good_technique": 0
    },
    "1780214945": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 0,
        "butt_wink": 0,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 0,
        "chest_collapse": 0,
        "shallow_depth": 0,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 0,
        "good_technique": 1
    },
    "1780215032": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 0,
        "butt_wink": 0,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 0,
        "chest_collapse": 0,
        "shallow_depth": 1,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 1,
        "unintentional_pause": 0,
        "asymmetrical_pattern": 0,
        "poor_stance": 0,
        "good_technique": 0
    },
    "1780215193": {
        "heel_lift": 0,
        "toe_out": 0,
        "toe_in": 0,
        "weight_shift_to_toes": 0,
        "knee_valgus": 0,
        "knee_varus": 0,
        "excessive_knee_travel": 0,
        "butt_wink": 0,
        "lumbar_extension": 0,
        "hip_shift": 0,
        "forward_lean": 0,
        "chest_collapse": 0,
        "shallow_depth": 0,
        "excessive_depth": 0,
        "weak_bracing": 0,
        "poor_breathing": 0,
        "dropping_too_fast": 0,
        "unintentional_pause": 1,
        "asymmetrical_pattern": 0,
        "poor_stance": 0,
        "good_technique": 0
    }
}

def build_ml_dataset_variant(features_folder, suffix, out_csv, label_dict):
    all_rows = []

    for fname in os.listdir(features_folder):
        if not "features" in fname:
            continue
        if not fname.endswith(f"{suffix}.csv"):
            continue
        print(fname)
        # e.g. features_1779582973_raw.csv → session_id = 1779582973
        parts = fname.split("_")
        session_id = parts[1]

        path = os.path.join(features_folder, fname)
        rep_df = pd.read_csv(path)
        rep_df["session_id"] = session_id

        if label_dict and session_id in label_dict:
            for label_name, label_value in label_dict[session_id].items():
                rep_df[label_name] = label_value

        all_rows.append(rep_df)

    if not all_rows:
        raise RuntimeError(f"No feature files found for suffix {suffix}")

    master_df = pd.concat(all_rows, ignore_index=True)
    master_df.to_csv(out_csv, index=False)
    print(f"Saved ML dataset to: {out_csv}")
    return master_df

# master_df = build_ml_dataset(
#     fused_folder=r"C:\Users\joono\OneDrive\Desktop\1. Post Graduate Diploma in Computer and Information Science\STEM Research Methods\Data Stitching Test",
#     out_csv="squat_ml_dataset.csv",
#     label_dict=label_dict
# )

if __name__ == "__main__":
    fused_folder = r"C:\Users\joono\OneDrive\Desktop\1. Post Graduate Diploma in Computer and Information Science\STEM Research Methods\Data Stitching Test\features"

    build_ml_dataset_variant(fused_folder, "_raw", r"ml_data\squat_ml_dataset_raw.csv", label_dict)
    build_ml_dataset_variant(fused_folder, "_raw_smoothed", r"ml_data\squat_ml_dataset_raw_smoothed.csv", label_dict)

    build_ml_dataset_variant(fused_folder, "_watch_gyro", r"ml_data\squat_ml_dataset_watch_gyro.csv", label_dict)
    build_ml_dataset_variant(fused_folder, "_watch_acc", r"ml_data\squat_ml_dataset_watch_acc.csv", label_dict)

    build_ml_dataset_variant(fused_folder, "_airpods_gyro", r"ml_data\squat_ml_dataset_airpods_gyro.csv", label_dict)
    build_ml_dataset_variant(fused_folder, "_airpods_acc", r"ml_data\squat_ml_dataset_airpods_acc.csv", label_dict)

    build_ml_dataset_variant(fused_folder, "_combined_gyro", r"ml_data\squat_ml_dataset_combined_gyro.csv", label_dict)
    build_ml_dataset_variant(fused_folder, "_combined_acc", r"ml_data\squat_ml_dataset_combined_acc.csv", label_dict)
    build_ml_dataset_variant(fused_folder, "_combined_mixed", r"ml_data\squat_ml_dataset_combined_mixed.csv", label_dict)

    build_ml_dataset_variant(fused_folder, "_watch_quat", r"ml_data\squat_ml_dataset_watch_quat.csv", label_dict)
    build_ml_dataset_variant(fused_folder, "_airpods_quat", r"ml_data\squat_ml_dataset_airpods_quat.csv", label_dict)
    build_ml_dataset_variant(fused_folder, "_combined_quat", r"ml_data\squat_ml_dataset_combined_quat.csv", label_dict)


# fused_folder=r"C:\Users\joono\OneDrive\Desktop\1. Post Graduate Diploma in Computer and Information Science\STEM Research Methods\Data Stitching Test\features"

# build_ml_dataset_variant(fused_folder, "_raw", r"ml_data\squat_ml_dataset_raw.csv", label_dict)
# build_ml_dataset_variant(fused_folder, "_raw_smoothed", r"ml_data\squat_ml_dataset_raw_smoothed.csv", label_dict)

# build_ml_dataset_variant(fused_folder, "_watch_gyro", r"ml_data\squat_ml_dataset_watch_gyro.csv", label_dict)
# build_ml_dataset_variant(fused_folder, "_watch_acc", r"ml_data\squat_ml_dataset_watch_acc.csv", label_dict)

# build_ml_dataset_variant(fused_folder, "_airpods_gyro", r"ml_data\squat_ml_dataset_airpods_gyro.csv", label_dict)
# build_ml_dataset_variant(fused_folder, "_airpods_acc", r"ml_data\squat_ml_dataset_airpods_acc.csv", label_dict)

# build_ml_dataset_variant(fused_folder, "_combined_gyro", r"ml_data\squat_ml_dataset_combined_gyro.csv", label_dict)
# build_ml_dataset_variant(fused_folder, "_combined_acc", r"ml_data\squat_ml_dataset_combined_acc.csv", label_dict)
# build_ml_dataset_variant(fused_folder, "_combined_mixed", r"ml_data\squat_ml_dataset_combined_mixed.csv", label_dict)

# build_ml_dataset_variant(fused_folder, "_watch_quat", r"ml_data\squat_ml_dataset_watch_quat.csv", label_dict)
# build_ml_dataset_variant(fused_folder, "_airpods_quat", r"ml_data\squat_ml_dataset_airpods_quat.csv", label_dict)
# build_ml_dataset_variant(fused_folder, "_combined_quat", r"ml_data\squat_ml_dataset_combined_quat.csv", label_dict)
