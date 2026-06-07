import os
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from scipy.interpolate import interp1d
import mediapipe as mp

# -----------------------------
# MediaPipe Pose setup
# -----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# Build a dict: index -> landmark name (e.g. 0 -> 'NOSE')
POSE_LANDMARK_NAMES: Dict[int, str] = {
    lm.value: lm.name for lm in mp_pose.PoseLandmark
}


# -----------------------------
# File grouping by shared ID
# -----------------------------
def find_session_groups(root: str) -> Dict[str, Dict[str, str]]:
    """
    Scan a directory for files that share a numeric ID suffix, e.g.:

      video_1779512826.mov
      watchData_1779512826.csv
      session_30901_videoTimestamps_1779512826.csv
      phoneIMU_1779512826.csv
      airpodsIMU_1779512826.csv

    Returns a dict: session_id -> {role: path}
    """
    root_path = Path(root)
    patterns = [
        "video_*.mov",
        "video_*.mp4",
        "watchData_*.csv",
        "session_*_videoTimestamps_*.csv",
        "phoneIMU_*.csv",
        "airpodsIMU_*.csv",
    ]

    files = []
    for pat in patterns:
        files.extend(root_path.glob(pat))

    groups: Dict[str, Dict[str, str]] = {}

    for f in files:
        name = f.name

        # Extract the numeric ID at the end before extension
        # e.g. "video_1779512826.mov" -> "1779512826"
        stem = f.stem
        parts = stem.split("_")
        session_id = parts[-1]

        if session_id not in groups:
            groups[session_id] = {}

        if name.startswith("video_"):
            groups[session_id]["video"] = str(f)
        elif name.startswith("watchData_"):
            groups[session_id]["watch"] = str(f)
        elif name.startswith("session_") and "videoTimestamps" in name:
            groups[session_id]["session"] = str(f)
        elif name.startswith("phoneIMU_"):
            groups[session_id]["phone"] = str(f)
        elif name.startswith("airpodsIMU_"):
            groups[session_id]["airpods"] = str(f)

    return groups


# -----------------------------
# Pose extraction + annotated video
# -----------------------------
def extract_pose_from_video(
    video_path: str,
    pose_csv_out: str,
    annotated_video_out: str,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> None:
    """
    Run MediaPipe Pose on a video, save:
      - CSV with named landmarks
      - Annotated video with landmarks overlaid
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(annotated_video_out, fourcc, fps, (width, height))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    rows = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Compute video time in seconds
        t_video = frame_idx / fps

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # Draw landmarks on a copy for visualization
        annotated_frame = frame_bgr.copy()

        if results.pose_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2),
            )

            # Extract landmarks into a flat dict
            lm_row = {
                "frameIndex": frame_idx,
                "t_video": t_video,
            }

            for idx, lm in enumerate(results.pose_landmarks.landmark):
                name = POSE_LANDMARK_NAMES.get(idx, f"LM_{idx}")
                lm_row[f"{name}_x"] = lm.x
                lm_row[f"{name}_y"] = lm.y
                lm_row[f"{name}_z"] = lm.z
                lm_row[f"{name}_visibility"] = lm.visibility

        else:
            # No pose detected: still keep frame/time, NaNs for landmarks
            lm_row = {
                "frameIndex": frame_idx,
                "t_video": t_video,
            }
            for idx in range(len(POSE_LANDMARK_NAMES)):
                name = POSE_LANDMARK_NAMES.get(idx, f"LM_{idx}")
                lm_row[f"{name}_x"] = np.nan
                lm_row[f"{name}_y"] = np.nan
                lm_row[f"{name}_z"] = np.nan
                lm_row[f"{name}_visibility"] = np.nan

        rows.append(lm_row)

        # Write annotated frame
        out_writer.write(annotated_frame)

        frame_idx += 1

    cap.release()
    out_writer.release()
    pose.close()

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(pose_csv_out, index=False)
    print(f"Saved pose CSV: {pose_csv_out}")
    print(f"Saved annotated video: {annotated_video_out}")


# -----------------------------
# Alignment helpers
# -----------------------------
def read_optional_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def align_pose_with_sensors(
    pose_csv: str,
    session_csv: str,
    watch_csv: Optional[str],
    phone_csv: Optional[str],
    airpods_csv: Optional[str],
    merged_out: str,
) -> None:

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    pose_df = pd.read_csv(pose_csv)
    session_df = pd.read_csv(session_csv)
    watch_df = read_optional_csv(watch_csv)
    phone_df = read_optional_csv(phone_csv)
    airpods_df = read_optional_csv(airpods_csv)

    # ---------------------------------------------------------
    # 1. Normalize timestamps
    # ---------------------------------------------------------

    # Video timestamps → master timeline
    video_start = session_df["globalTime"].iloc[0]
    session_df["t_global"] = session_df["globalTime"] - video_start

    # Add t_global to pose_df using frameIndex
    pose_df = pose_df.merge(
        session_df[["frameIndex", "t_global"]],
        on="frameIndex",
        how="left"
    )

    # Normalize phone IMU
    if phone_df is not None:
        phone_df["t_global"] -= phone_df["t_global"].iloc[0]

    # Normalize AirPods IMU
    if airpods_df is not None:
        airpods_df["t_global"] -= airpods_df["t_global"].iloc[0]

    # Normalize watch IMU
    if watch_df is not None:
        watch_df["t_global"] -= watch_df["t_global"].iloc[0]

    # ---------------------------------------------------------
    # 2. Interpolate IMU streams to video timeline
    # ---------------------------------------------------------
    t_ref = session_df["t_global"].values  # one per frameIndex

    def interp_df(df, drop_cols):
        df = df.sort_values("t_global")
        cols = [c for c in df.columns if c not in drop_cols]
        out = {}
        for c in cols:
            f = interp1d(df["t_global"], df[c], bounds_error=False, fill_value="extrapolate")
            out[c] = f(t_ref)
        return pd.DataFrame(out)

    phone_interp = interp_df(phone_df, ["t_global", "t_local"]) if phone_df is not None else None
    airpods_interp = interp_df(airpods_df, ["t_global", "t_local"]) if airpods_df is not None else None
    watch_interp = interp_df(watch_df, ["t_global", "t_watch"]) if watch_df is not None else None

    # ---------------------------------------------------------
    # 3. Combine everything using frameIndex
    # ---------------------------------------------------------
    combined = pd.concat(
        [
            session_df[["frameIndex", "t_global"]],
            pose_df.drop(columns=["t_global"]),
            phone_interp.add_prefix("phone_") if phone_interp is not None else None,
            airpods_interp.add_prefix("airpods_") if airpods_interp is not None else None,
            watch_interp.add_prefix("watch_") if watch_interp is not None else None,
        ],
        axis=1
    )

    combined.to_csv(merged_out, index=False)
    print(f"Saved merged dataset: {merged_out}")



# -----------------------------
# Batch processing entry point
# -----------------------------
def process_all_sessions(root: str, output_root: Optional[str] = None) -> None:
    """
    For each session ID:
      1) Extract pose landmarks + annotated video
      2) Align pose with session + IMUs
    """
    if output_root is None:
        output_root = root

    groups = find_session_groups(root)
    print(f"Found {len(groups)} session groups")

    for session_id, files in groups.items():
        video_path = files.get("video")
        session_path = files.get("session")

        if video_path is None or session_path is None:
            print(f"[{session_id}] Missing video or session file; skipping")
            continue

        watch_path = files.get("watch")
        phone_path = files.get("phone")
        airpods_path = files.get("airpods")

        # Output paths
        pose_csv_out = str(Path(output_root) / fr"pose_landmarks\poseLandmarks_{session_id}.csv")
        annotated_video_out = str(Path(output_root) / fr"annotated_videos\video_annotated_{session_id}.mp4")
        merged_out = str(Path(output_root) / fr"merged\merged_pose_IMU_{session_id}.csv")
        merged_pose_out = str(Path(output_root) / fr"merged\merged_pose_{session_id}.csv")

        print(f"\n=== Session {session_id} ===")
        print(f"Video:   {video_path}")
        print(f"Session: {session_path}")
        print(f"Watch:   {watch_path}")
        print(f"Phone:   {phone_path}")
        print(f"AirPods: {airpods_path}")

        # 1) Pose extraction + annotated video
        extract_pose_from_video(
            video_path=video_path,
            pose_csv_out=pose_csv_out,
            annotated_video_out=annotated_video_out,
        )

        # 2) Alignment
        align_pose_with_sensors(
            pose_csv=pose_csv_out,
            session_csv=session_path,
            watch_csv=watch_path,
            phone_csv=phone_path,
            airpods_csv=airpods_path,
            merged_out=merged_out,
        )
        
        # 3) Pose only Alignment
        align_pose_with_sensors(
            pose_csv=pose_csv_out,
            session_csv=session_path,
            watch_csv=None,
            phone_csv=None,
            airpods_csv=None,
            merged_out=merged_pose_out,
        )
        
        from plot_alignment import plot_alignment
        
        plot_alignment(
            merged_csv=merged_out,
            out_path=fr"alignment_plots\alignment_{session_id}.png"
        )
        
        import kalman_multi_joint as kmj
        import feature_extraction as fe
        
        # 1) RAW features (no Kalman) → need joint angles first
        raw_df = pd.read_csv(merged_pose_out)
        raw_df = kmj.extract_joint_angles(raw_df)  # adds *_angle_pose_rad columns
        
        raw_with_angles = fr"variants\merged_pose_IMU_{session_id}_raw_angles.csv"
        raw_df.to_csv(raw_with_angles, index=False)
        
        fe.extract_features_from_fused_csv(
            fused_csv=raw_with_angles,
            out_csv=fr"features\features_{session_id}_raw.csv"
        )
        
        # RAW SMOOTHED
        raw_df = pd.read_csv(merged_pose_out)
        raw_df = kmj.extract_joint_angles(raw_df)
        raw_df = fe.smooth_pose_angles(raw_df)
        
        raw_smoothed_path = fr"variants\merged_pose_IMU_{session_id}_raw_smoothed.csv"
        raw_df.to_csv(raw_smoothed_path, index=False)
        
        fe.extract_features_from_fused_csv(
            fused_csv=raw_smoothed_path,
            out_csv=fr"features\features_{session_id}_raw_smoothed.csv"
        )

        # 2) Watch-only fusion
        kmj.run_multi_joint_kalman(
            merged_csv=merged_out,
            out_csv=fr"variants\merged_pose_IMU_{session_id}_fused_watch.csv",
            mode="watch"
        )
        fe.extract_features_from_fused_csv(
            fused_csv=fr"variants\merged_pose_IMU_{session_id}_fused_watch.csv",
            out_csv=fr"features\features_{session_id}_watch.csv"
        )
        
        # 3) AirPods-only fusion
        kmj.run_multi_joint_kalman(
            merged_csv=merged_out,
            out_csv=fr"variants\merged_pose_IMU_{session_id}_fused_airpods.csv",
            mode="airpods"
        )
        fe.extract_features_from_fused_csv(
            fused_csv=fr"variants\merged_pose_IMU_{session_id}_fused_airpods.csv",
            out_csv=fr"features\features_{session_id}_airpods.csv"
        )
        
        # 4) Combined watch + AirPods fusion
        kmj.run_multi_joint_kalman(
            merged_csv=merged_out,
            out_csv=fr"variants\merged_pose_IMU_{session_id}_fused_watch_airpods.csv",
            mode="combined"
        )
        fe.extract_features_from_fused_csv(
            fused_csv=fr"variants\merged_pose_IMU_{session_id}_fused_watch_airpods.csv",
            out_csv=fr"features\features_{session_id}_watch_airpods.csv"
        )
        
        # WATCH (gyro) – existing
        kmj.run_multi_joint_kalman(
            merged_csv=merged_out,
            out_csv=fr"variants\merged_pose_IMU_{session_id}_fused_watch_gyro.csv",
            mode="watch_gyro"
        )
        fe.extract_features_from_fused_csv(
            fused_csv=fr"variants\merged_pose_IMU_{session_id}_fused_watch_gyro.csv",
            out_csv=fr"features\features_{session_id}_watch_gyro.csv"
        )
        
        # WATCH (acc)
        kmj.run_multi_joint_kalman(
            merged_csv=merged_out,
            out_csv=fr"variants\merged_pose_IMU_{session_id}_fused_watch_acc.csv",
            mode="watch_acc"
        )
        fe.extract_features_from_fused_csv(
            fused_csv=fr"variants\merged_pose_IMU_{session_id}_fused_watch_acc.csv",
            out_csv=fr"features\features_{session_id}_watch_acc.csv"
        )
        
        # AIRPODS (gyro)
        kmj.run_multi_joint_kalman(
            merged_csv=merged_out,
            out_csv=fr"variants\merged_pose_IMU_{session_id}_fused_airpods_gyro.csv",
            mode="airpods_gyro"
        )
        fe.extract_features_from_fused_csv(
            fused_csv=fr"variants\merged_pose_IMU_{session_id}_fused_airpods_gyro.csv",
            out_csv=fr"features\features_{session_id}_airpods_gyro.csv"
        )
        
        # AIRPODS (acc)
        kmj.run_multi_joint_kalman(
            merged_csv=merged_out,
            out_csv=fr"variants\merged_pose_IMU_{session_id}_fused_airpods_acc.csv",
            mode="airpods_acc"
        )
        fe.extract_features_from_fused_csv(
            fused_csv=fr"variants\merged_pose_IMU_{session_id}_fused_airpods_acc.csv",
            out_csv=fr"features\features_{session_id}_airpods_acc.csv"
        )
        
        # COMBINED (gyro)
        kmj.run_multi_joint_kalman(
            merged_csv=merged_out,
            out_csv=fr"variants\merged_pose_IMU_{session_id}_fused_combined_gyro.csv",
            mode="combined_gyro"
        )
        fe.extract_features_from_fused_csv(
            fused_csv=fr"variants\merged_pose_IMU_{session_id}_fused_combined_gyro.csv",
            out_csv=fr"features\features_{session_id}_combined_gyro.csv"
        )
        
        # COMBINED (acc)
        kmj.run_multi_joint_kalman(
            merged_csv=merged_out,
            out_csv=fr"variants\merged_pose_IMU_{session_id}_fused_combined_acc.csv",
            mode="combined_acc"
        )
        fe.extract_features_from_fused_csv(
            fused_csv=fr"variants\merged_pose_IMU_{session_id}_fused_combined_acc.csv",
            out_csv=fr"features\features_{session_id}_combined_acc.csv"
        )
        
        # COMBINED (mixed: watch gyro + airpods acc)
        kmj.run_multi_joint_kalman(
            merged_csv=merged_out,
            out_csv=fr"variants\merged_pose_IMU_{session_id}_fused_combined_mixed.csv",
            mode="combined_mixed"
        )
        fe.extract_features_from_fused_csv(
            fused_csv=fr"variants\merged_pose_IMU_{session_id}_fused_combined_mixed.csv",
            out_csv=fr"features\features_{session_id}_combined_mixed.csv"
        )
        
        # WATCH (quat) 
        kmj.run_multi_joint_kalman(
            merged_csv=merged_out,
            out_csv=fr"variants\merged_pose_IMU_{session_id}_fused_watch_quat.csv",
            mode="watch_quat"
        )
        fe.extract_features_from_fused_csv(
            fused_csv=fr"variants\merged_pose_IMU_{session_id}_fused_watch_quat.csv",
            out_csv=fr"features\features_{session_id}_watch_quat.csv"
        )
        
        # AIRPODS quaternion
        kmj.run_multi_joint_kalman(
            merged_csv=merged_out,
            out_csv=fr"variants\merged_pose_IMU_{session_id}_fused_airpods_quat.csv",
            mode="airpods_quat"
        )
        fe.extract_features_from_fused_csv(
            fused_csv=fr"variants\merged_pose_IMU_{session_id}_fused_airpods_quat.csv",
            out_csv=fr"features\features_{session_id}_airpods_quat.csv"
        )
        
        # COMBINED quaternion
        kmj.run_multi_joint_kalman(
            merged_csv=merged_out,
            out_csv=fr"variants\merged_pose_IMU_{session_id}_fused_combined_quat.csv",
            mode="airpods_quat"
        )
        fe.extract_features_from_fused_csv(
            fused_csv=fr"variants\merged_pose_IMU_{session_id}_fused_combined_quat.csv",
            out_csv=fr"features\features_{session_id}_combined_quat.csv"
        )

if __name__ == "__main__":
    # Change this to the folder where your files live
    data_root = r".\Input Data"
    output_root = r".\"
    process_all_sessions(root=data_root, output_root=output_root)