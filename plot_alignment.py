import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_alignment(merged_csv, out_path=None, smooth_window=5):
    """
    Visualize alignment between:
      - Pose landmarks
      - Watch IMU
      - Phone IMU
      - AirPods IMU
    All plotted against t_global.

    Produces FOUR versions:
      1. Original (full scale)
      2. Percentile clipped (scaled)
      3. Z-score normalized IMU
      4. Min-max normalized IMU
    """

    df = pd.read_csv(merged_csv)

    if "t_global" not in df.columns:
        raise ValueError("Merged CSV must contain 't_global' column")

    t = df["t_global"].values

    # Optional smoothing
    def smooth(x):
        if smooth_window <= 1:
            return x
        return pd.Series(x).rolling(smooth_window, min_periods=1, center=True).mean().values

    # -----------------------------
    # Select representative signals
    # -----------------------------
    pose_cols = [c for c in df.columns if "_x" in c and any(j in c for j in ["HIP", "KNEE", "ANKLE"])]
    pose_cols = pose_cols[:6]

    # watch_cols = [c for c in df.columns if c.startswith("watch_") and any(k in c for k in ["ua", "rot", "grav"])]
    watch_cols = [c for c in df.columns if c.startswith("watch_") and any(k in c for k in ["quat"])]    
    watch_cols = watch_cols[:6]

    phone_cols = [c for c in df.columns if c.startswith("phone_") and any(k in c for k in ["acc", "gyro", "grav"])]
    phone_cols = phone_cols[:6]

    airpods_cols = [c for c in df.columns if c.startswith("airpods_") and any(k in c for k in ["acc", "gyro"])]
    # airpods_cols = [c for c in df.columns if c.startswith("airpods_") and any(k in c for k in ["qx", "qy", "qz", "qw"])]
    airpods_cols = airpods_cols[:6]

    # =========================================================
    # 1. ORIGINAL PLOT
    # =========================================================
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle("Sensor & Pose Alignment Over Time", fontsize=18)

    # Pose
    ax = axes[0]
    for col in pose_cols:
        ax.plot(t, smooth(df[col].values), label=col)
    ax.set_ylabel("Pose")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Watch
    ax = axes[1]
    for col in watch_cols:
        ax.plot(t, smooth(df[col].values), label=col)
    ax.set_ylabel("Watch IMU")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Phone
    ax = axes[2]
    for col in phone_cols:
        ax.plot(t, smooth(df[col].values), label=col)
    ax.set_ylabel("Phone IMU")
    ax.legend(fontsize=8)
    ax.grid(True)

    # AirPods
    ax = axes[3]
    for col in airpods_cols:
        ax.plot(t, smooth(df[col].values), label=col)
    ax.set_ylabel("AirPods IMU")
    ax.set_xlabel("t_global (s)")
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")
    else:
        plt.show()

    # =========================================================
    # 2. PERCENTILE CLIPPED PLOT
    # =========================================================
    def clip_percentile(series, low=1, high=99):
        lo = np.percentile(series, low)
        hi = np.percentile(series, high)
        return np.clip(series, lo, hi)

    fig2, axes2 = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig2.suptitle("Sensor & Pose Alignment (Percentile Scaled)", fontsize=18)

    # Pose unchanged
    ax = axes2[0]
    for col in pose_cols:
        ax.plot(t, smooth(df[col].values), label=col)
    ax.set_ylabel("Pose")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Watch clipped
    ax = axes2[1]
    for col in watch_cols:
        ax.plot(t, smooth(clip_percentile(df[col].values)), label=col)
    ax.set_ylabel("Watch (scaled)")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Phone clipped
    ax = axes2[2]
    for col in phone_cols:
        ax.plot(t, smooth(clip_percentile(df[col].values)), label=col)
    ax.set_ylabel("Phone (scaled)")
    ax.legend(fontsize=8)
    ax.grid(True)

    # AirPods clipped
    ax = axes2[3]
    for col in airpods_cols:
        ax.plot(t, smooth(clip_percentile(df[col].values)), label=col)
    ax.set_ylabel("AirPods (scaled)")
    ax.set_xlabel("t_global (s)")
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.tight_layout()

    if out_path:
        scaled_path = out_path.replace(".png", "_scaled.png")
        plt.savefig(scaled_path, dpi=200)
        print(f"Saved: {scaled_path}")
    else:
        plt.show()

    # =========================================================
    # 3. Z-SCORE NORMALIZED PLOT
    # =========================================================
    def zscore(series):
        return (series - np.mean(series)) / (np.std(series) + 1e-8)

    fig3, axes3 = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig3.suptitle("Sensor & Pose Alignment (Z-Score Normalized)", fontsize=18)

    # Pose unchanged
    ax = axes3[0]
    for col in pose_cols:
        ax.plot(t, smooth(df[col].values), label=col)
    ax.set_ylabel("Pose")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Watch z-score
    ax = axes3[1]
    for col in watch_cols:
        ax.plot(t, smooth(zscore(df[col].values)), label=col)
    ax.set_ylabel("Watch (z-score)")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Phone z-score
    ax = axes3[2]
    for col in phone_cols:
        ax.plot(t, smooth(zscore(df[col].values)), label=col)
    ax.set_ylabel("Phone (z-score)")
    ax.legend(fontsize=8)
    ax.grid(True)

    # AirPods z-score
    ax = axes3[3]
    for col in airpods_cols:
        ax.plot(t, smooth(zscore(df[col].values)), label=col)
    ax.set_ylabel("AirPods (z-score)")
    ax.set_xlabel("t_global (s)")
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.tight_layout()

    if out_path:
        z_path = out_path.replace(".png", "_zscore.png")
        plt.savefig(z_path, dpi=200)
        print(f"Saved: {z_path}")
    else:
        plt.show()

    # =========================================================
    # 4. MIN-MAX NORMALIZED PLOT
    # =========================================================
    def minmax(series):
        mn = np.min(series)
        mx = np.max(series)
        return (series - mn) / (mx - mn + 1e-8)

    fig4, axes4 = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig4.suptitle("Sensor & Pose Alignment (Min-Max Normalized)", fontsize=18)

    # Pose unchanged
    ax = axes4[0]
    for col in pose_cols:
        ax.plot(t, smooth(df[col].values), label=col)
    ax.set_ylabel("Pose")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Watch min-max
    ax = axes4[1]
    for col in watch_cols:
        ax.plot(t, smooth(minmax(df[col].values)), label=col)
    ax.set_ylabel("Watch (min-max)")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Phone min-max
    ax = axes4[2]
    for col in phone_cols:
        ax.plot(t, smooth(minmax(df[col].values)), label=col)
    ax.set_ylabel("Phone (min-max)")
    ax.legend(fontsize=8)
    ax.grid(True)

    # AirPods min-max
    ax = axes4[3]
    for col in airpods_cols:
        ax.plot(t, smooth(minmax(df[col].values)), label=col)
    ax.set_ylabel("AirPods (min-max)")
    ax.set_xlabel("t_global (s)")
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.tight_layout()

    if out_path:
        mm_path = out_path.replace(".png", "_minmax.png")
        plt.savefig(mm_path, dpi=200)
        print(f"Saved: {mm_path}")
    else:
        plt.show()

# plot_alignment(
#     merged_csv="merged_pose_IMU_1779582973.csv",
#     out_path="alignment_1779582973_new.png"
# )