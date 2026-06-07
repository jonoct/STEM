import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

# Savitzky–Golay smoothing (window must be odd and < len)
def smooth(x, win=11, poly=3):
    win = min(win, len(x) - (1 - len(x) % 2))
    if win < 5:
        return x
    return savgol_filter(x, window_length=win, polyorder=poly)

def smooth_pose_angles(df):
    df = df.copy()
    for joint in ["ankle", "knee", "hip", "trunk"]:
        col = f"{joint}_angle_pose_rad"
        if col in df.columns:
            df[f"{joint}_angle_pose_smoothed"] = smooth(df[col].values)
    return df


def preprocess_signals(df):
    """
    Prepare smoothed signals for segmentation.
    Returns:
      t, knee_ang_s, hip_ang_s, imu_omega_s
    """
    t = df["t_global"].values

    knee = df["knee_angle_fused_rad"].values
    hip = df["hip_angle_fused_rad"].values

    # IMU angular velocity for segmentation (trunk or hip)
    if "trunk_omega_fused" in df.columns:
        imu_omega = df["trunk_omega_fused"].values
    elif "hip_omega_fused" in df.columns:
        imu_omega = df["hip_omega_fused"].values
    else:
        imu_omega = np.zeros_like(knee)

    # Savitzky–Golay smoothing (window must be odd and < len)
    def smooth(x, win=11, poly=3):
        win = min(win, len(x) - (1 - len(x) % 2))  # ensure odd and <= len
        if win < 5:
            return x
        return savgol_filter(x, window_length=win, polyorder=poly)

    knee_s = smooth(knee)
    hip_s = smooth(hip)
    imu_s = smooth(imu_omega)

    return t, knee_s, hip_s, imu_s

def segment_squat_reps(t, hip_vel, vel_thresh=0.1, min_rep_time=0.5):
    """
    Segment squat reps using fused hip vertical velocity.
    Returns list of (start_idx, bottom_idx, end_idx).
    Make more robust by:
        Lowering threshold - vel_thresh = 0.03
        Adding smoothing - hip_vel_smooth = hip_vel.rolling(5, center=True).mean()
        Requiring a minimum descent depth by checking the knee angle change -
           if knee_angle[i] - knee_angle[start] > min_depth_angle:
                bottom = i           
    """

    reps = []
    n = len(hip_vel)

    state = "idle"
    start = bottom = None

    for i in range(1, n):
        v = hip_vel[i]

        if state == "idle":
            if v < -vel_thresh:  # start descending
                start = i
                state = "descending"

        elif state == "descending":
            if v > -vel_thresh:  # reached bottom
                bottom = i
                state = "ascending"

        elif state == "ascending":
            if v < vel_thresh:  # reached top
                end = i
                if t[end] - t[start] > min_rep_time:
                    reps.append((start, bottom, end))
                state = "idle"

    return reps

def segment_squat_reps_zero_cross(t, vel, min_rep_time=0.4):
    reps = []
    n = len(vel)

    # Find zero-crossings
    zero_crossings = np.where(np.diff(np.sign(vel)) != 0)[0]

    # Pair them up: descent zero-cross → ascent zero-cross
    for i in range(0, len(zero_crossings)-1, 2):
        start = zero_crossings[i]
        bottom = zero_crossings[i+1]

        # Find end: next zero-cross
        if i+2 < len(zero_crossings):
            end = zero_crossings[i+2]
        else:
            continue

        if t[end] - t[start] > min_rep_time:
            reps.append((start, bottom, end))

    return reps

def segment_squat_reps_angle(t, knee_angle, min_depth_rad=0.20, min_rep_time=0.4):
    """
    Robust squat rep segmentation using knee angle minima/maxima.
    - Finds bottoms as local minima of knee angle
    - Finds tops as local maxima before and after each bottom
    - Ensures minimum depth and minimum rep duration
    Returns list of (start_idx, bottom_idx, end_idx)
    """

    reps = []

    # 1. Find bottoms (local minima)
    bottoms, _ = find_peaks(-knee_angle, distance=10)

    # 2. Find tops (local maxima)
    tops, _ = find_peaks(knee_angle, distance=10)

    if len(bottoms) == 0 or len(tops) < 2:
        return reps

    for b in bottoms:
        # Find nearest top BEFORE bottom
        tops_before = tops[tops < b]
        if len(tops_before) == 0:
            continue
        start = tops_before[-1]

        # Find nearest top AFTER bottom
        tops_after = tops[tops > b]
        if len(tops_after) == 0:
            continue
        end = tops_after[0]

        # Depth check
        depth = knee_angle[start] - knee_angle[b]
        if depth < min_depth_rad:
            continue

        # Duration check
        if t[end] - t[start] < min_rep_time:
            continue

        reps.append((start, b, end))

    return reps

def segment_squat_reps_unified(
    t,
    knee_angle,
    hip_angle,
    imu_omega,
    min_rep_time=0.4,
    merge_window=0.25,
):
    """
    Unified rep segmentation using:
      - knee angle minima/maxima
      - hip angle minima/maxima (backup)
      - IMU omega zero-crossings (support)
      - adaptive depth and duration thresholds
    Returns list of (start_idx, bottom_idx, end_idx).
    """

    n = len(t)
    reps = []

    # --- 1) Candidate bottoms from knee & hip ---
    knee_bottoms, _ = find_peaks(-knee_angle, distance=10)
    hip_bottoms, _ = find_peaks(-hip_angle, distance=10)

    bottoms_all = np.sort(np.concatenate([knee_bottoms, hip_bottoms]))
    if len(bottoms_all) == 0:
        return reps

    # --- 2) Merge bottoms within merge_window seconds ---
    merged_bottoms = []
    current_group = [bottoms_all[0]]

    for idx in bottoms_all[1:]:
        if t[idx] - t[current_group[-1]] <= merge_window:
            current_group.append(idx)
        else:
            merged_bottoms.append(int(np.median(current_group)))
            current_group = [idx]
    merged_bottoms.append(int(np.median(current_group)))

    # --- 3) Tops (maxima) for knee & hip ---
    knee_tops, _ = find_peaks(knee_angle, distance=10)
    hip_tops, _ = find_peaks(hip_angle, distance=10)
    tops_all = np.sort(np.concatenate([knee_tops, hip_tops]))

    if len(tops_all) < 2:
        return reps

    # --- 4) Adaptive depth threshold (based on knee angle) ---
    depths = []
    for b in merged_bottoms:
        tops_before = tops_all[tops_all < b]
        tops_after = tops_all[tops_all > b]
        if len(tops_before) == 0 or len(tops_after) == 0:
            continue
        start = tops_before[-1]
        end = tops_after[0]
        depth = knee_angle[start] - knee_angle[b]
        if depth > 0:
            depths.append(depth)

    if len(depths) == 0:
        return reps

    depth_thresh = np.percentile(depths, 20)  # adaptive
    depth_thresh = max(depth_thresh, 0.10)    # minimum floor

    # --- 5) Build reps with quality checks ---
    for b in merged_bottoms:
        tops_before = tops_all[tops_all < b]
        tops_after = tops_all[tops_all > b]
        if len(tops_before) == 0 or len(tops_after) == 0:
            continue

        start = tops_before[-1]
        end = tops_after[0]

        # Duration check
        if t[end] - t[start] < min_rep_time:
            continue

        # Depth check
        depth = knee_angle[start] - knee_angle[b]
        if depth < depth_thresh:
            continue

        # IMU support: require some movement around bottom
        window_idx = (t >= t[b] - 0.2) & (t <= t[b] + 0.2)
        if np.sum(window_idx) > 3:
            imu_var = np.var(imu_omega[window_idx])
            if imu_var < 1e-5:  # almost no motion → likely noise
                continue

        reps.append((start, b, end))

    return reps

def extract_rep_features(df, start, bottom, end):
    rep = df.iloc[start:end+1]

    # Convert global indices → local indices
    local_bottom = bottom - start
    local_end = end - start

    features = {}

    # --- Temporal ---
    features["rep_duration"] = rep["t_global"].iloc[local_end] - rep["t_global"].iloc[0]
    features["eccentric_time"] = rep["t_global"].iloc[local_bottom] - rep["t_global"].iloc[0]
    features["concentric_time"] = rep["t_global"].iloc[local_end] - rep["t_global"].iloc[local_bottom]

    # --- Joint angle ranges ---
    for side in ["right", "left"]:
        for joint in ["ankle", "knee", "hip", "trunk"]:
            col = f"{side}_{joint}_angle_fused_rad" if f"{side}_{joint}_angle_fused_rad" in rep.columns else f"{side}_{joint}_angle_pose_rad"
            ang = rep[col]
            features[f"{side}_{joint}_angle_min"] = ang.min()
            features[f"{side}_{joint}_angle_max"] = ang.max()
            features[f"{side}_{joint}_angle_range"] = ang.max() - ang.min()

    # # --- Watch IMU features ---
    # if "watch_ua_y" in rep.columns:
    #     acc = rep["watch_ua_y"]
    #     features["watch_acc_y_peak"] = acc.max()
    #     features["watch_acc_y_rms"] = np.sqrt(np.mean(acc**2))
    #     features["watch_acc_y_jerk"] = np.mean(np.abs(np.diff(acc)))
    
    # if "watch_ua_x" in rep.columns:
    #     acc = rep["watch_ua_x"]
    #     features["watch_acc_x_peak"] = acc.max()
    #     features["watch_acc_x_rms"] = np.sqrt(np.mean(acc**2))
    #     features["watch_acc_x_jerk"] = np.mean(np.abs(np.diff(acc)))
    
    # if "watch_ua_z" in rep.columns:
    #     acc = rep["watch_ua_z"]
    #     features["watch_acc_z_peak"] = acc.max()
    #     features["watch_acc_z_rms"] = np.sqrt(np.mean(acc**2))
    #     features["watch_acc_z_jerk"] = np.mean(np.abs(np.diff(acc)))
    
    watch_cols = [c for c in rep.columns if "watch" in c]
    airpods_cols = [c for c in rep.columns if "airpods" in c]
    
    for col in watch_cols:
        if "omega" in col:
            continue
        series = rep[col]
        features[f"{col}_mean"] = series.mean()
        features[f"{col}_std"] = series.std()
        features[f"{col}_min"] = series.min()
        features[f"{col}_max"] = series.max()
        features[f"{col}_range"] = series.max() - series.min()
        features[f"{col}_rms"] = np.sqrt(np.mean(series**2))
        features[f"{col}_jerk"] = np.mean(np.abs(np.diff(series)))
    
    for col in airpods_cols:
        if "omega" in col:
            continue
        series = rep[col]
        features[f"{col}_mean"] = series.mean()
        features[f"{col}_std"] = series.std()
        features[f"{col}_min"] = series.min()
        features[f"{col}_max"] = series.max()
        features[f"{col}_range"] = series.max() - series.min()
        features[f"{col}_rms"] = np.sqrt(np.mean(series**2))
        features[f"{col}_jerk"] = np.mean(np.abs(np.diff(series)))


    # --- Stability / control ---
    if "trunk_omega_fused" in rep.columns:
        omega = rep["trunk_omega_fused"]
    elif "watch_rot_x" in rep.columns:
        omega = rep["watch_rot_x"]
    else:
        omega = np.zeros(len(rep))
    
    features["trunk_omega_var"] = np.var(omega)
    features["trunk_omega_rms"] = np.sqrt(np.mean(omega**2))

    # --- Asymmetry (if right side exists) ---
    if "RIGHT_KNEE_angle_fused_rad" in rep.columns:
        left_knee = rep["knee_angle_fused_rad"]
        right_knee = rep["RIGHT_KNEE_angle_fused_rad"]
        features["knee_asymmetry"] = np.mean(np.abs(left_knee - right_knee))

    return features

def extract_features_from_fused_csv(fused_csv, out_csv):
    df = pd.read_csv(fused_csv)

    # 1. Segment reps
    # reps = segment_squat_reps(
    #     t=df["t_global"].values,
    #     hip_vel=df["hip_omega_fused"].values  # vertical hip velocity proxy
    # )
    
    # reps = segment_squat_reps_zero_cross(
    #     t=df["t_global"].values,
    #     vel=df["hip_omega_fused"].values  # vertical hip velocity proxy
    # )
    
    if "right_knee_angle_fused_rad" in df.columns:
        knee_col = "right_knee_angle_fused_rad"
    elif "right_knee_angle_pose_smoothed" in df.columns:
        knee_col = "righ_knee_angle_pose_smoothed"
    elif "right_knee_angle_pose_rad" in df.columns:
        knee_col = "right_knee_angle_pose_rad"
    print(knee_col)
    reps = segment_squat_reps_angle(
        t=df["t_global"].values,
        knee_angle=df[knee_col].values
    )
     

    # # preprocess signals
    # t, knee_s, hip_s, imu_s = preprocess_signals(df)

    # # unified segmentation
    # reps = segment_squat_reps_unified(
    #     t=t,
    #     knee_angle=knee_s,
    #     hip_angle=hip_s,
    #     imu_omega=imu_s,
    # )

    print(f"Detected {len(reps)} squat reps")

    # 2. Extract features per rep
    all_features = []
    for (start, bottom, end) in reps:
        feats = extract_rep_features(df, start, bottom, end)
        feats["rep_start"] = start
        feats["rep_end"] = end
        all_features.append(feats)

    # 3. Save dataset
    out_df = pd.DataFrame(all_features)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved rep-level feature dataset to: {out_csv}")

    return out_df