import numpy as np
import pandas as pd

def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_to_omega(df, prefix, dt):
    """
    prefix: 'watch_quat' or 'airpods_q'
    Produces new columns:
      prefix + '_omega_x'
      prefix + '_omega_y'
      prefix + '_omega_z'
    """

    qw = df[f"{prefix}w"].values
    qx = df[f"{prefix}x"].values
    qy = df[f"{prefix}y"].values
    qz = df[f"{prefix}z"].values

    omegas = []

    for i in range(len(df)-1):
        q1 = np.array([qw[i], qx[i], qy[i], qz[i]])
        q2 = np.array([qw[i+1], qx[i+1], qy[i+1], qz[i+1]])

        dq = (q2 - q1) / dt
        q_inv = quat_conjugate(q1)

        omega_quat = 2 * quat_multiply(q_inv, dq)
        # ignore scalar part
        omegas.append(omega_quat[1:])

    # pad last sample
    omegas.append(omegas[-1])

    omegas = np.array(omegas)
    df[f"{prefix}_omega_x"] = omegas[:,0]
    df[f"{prefix}_omega_y"] = omegas[:,1]
    df[f"{prefix}_omega_z"] = omegas[:,2]

    return df

def choose_imu_source(df, mode):
    """
    mode options:
      - 'watch'
      - 'watch_gyro'
      - 'watch_acc'
      - 'airpods'
      - 'airpods_gyro'
      - 'airpods_acc'
      - 'combined'
      - 'combined_gyro'
      - 'combined_acc'
      - 'combined_mixed'  (watch gyro + airpods acc)
      - 'watch_quat'      (quaternion-derived omega, TODO)
    Returns dict joint -> imu_column
    """

    if mode == "watch":
        return {
            "ankle": "watch_rot_z",
            "knee":  "watch_rot_y",
            "hip":   "watch_rot_y",
            "trunk": "watch_rot_x",
        }

    if mode == "airpods":
        return {
            "ankle": "airpods_gyroZ",
            "knee":  "airpods_gyroY",
            "hip":   "airpods_gyroY",
            "trunk": "airpods_gyroX",
        }

    if mode == "combined":
        # average watch + airpods when both exist
        out = {}
        mapping = {
            "ankle": ("watch_rot_z", "airpods_gyroZ"),
            "knee":  ("watch_rot_y", "airpods_gyroY"),
            "hip":   ("watch_rot_y", "airpods_gyroY"),
            "trunk": ("watch_rot_x", "airpods_gyroX"),
        }

        for joint, (wcol, acol) in mapping.items():
            if wcol in df.columns and acol in df.columns:
                comb_col = f"{joint}_omega_combined"
                df[comb_col] = (df[wcol].fillna(0) + df[acol].fillna(0)) / 2.0
                out[joint] = comb_col
            elif wcol in df.columns:
                out[joint] = wcol
            elif acol in df.columns:
                out[joint] = acol
            else:
                # nothing available for this joint
                out[joint] = None

        return out

    if mode == "watch_gyro":
        return {
            "ankle": "watch_rot_z",
            "knee":  "watch_rot_y",
            "hip":   "watch_rot_y",
            "trunk": "watch_rot_x",
        }

    if mode == "watch_acc":
        return {
            "ankle": "watch_ua_z",
            "knee":  "watch_ua_y",
            "hip":   "watch_ua_y",
            "trunk": "watch_ua_x",
        }

    if mode == "airpods_gyro":
        return {
            "ankle": "airpods_gyroZ",
            "knee":  "airpods_gyroY",
            "hip":   "airpods_gyroY",
            "trunk": "airpods_gyroX",
        }

    if mode == "airpods_acc":
        return {
            "ankle": "airpods_accZ",
            "knee":  "airpods_accY",
            "hip":   "airpods_accY",
            "trunk": "airpods_accX",
        }

    if mode == "combined_gyro":
        out = {}
        mapping = {
            "ankle": ("watch_rot_z", "airpods_gyroZ"),
            "knee":  ("watch_rot_y", "airpods_gyroY"),
            "hip":   ("watch_rot_y", "airpods_gyroY"),
            "trunk": ("watch_rot_x", "airpods_gyroX"),
        }
        for joint, (wcol, acol) in mapping.items():
            if wcol in df.columns and acol in df.columns:
                comb_col = f"{joint}_omega_combined_gyro"
                df[comb_col] = (df[wcol].fillna(0) + df[acol].fillna(0)) / 2.0
                out[joint] = comb_col
            elif wcol in df.columns:
                out[joint] = wcol
            elif acol in df.columns:
                out[joint] = acol
            else:
                out[joint] = None
        return out

    if mode == "combined_acc":
        out = {}
        mapping = {
            "ankle": ("watch_ua_z", "airpods_accZ"),
            "knee":  ("watch_ua_y", "airpods_accY"),
            "hip":   ("watch_ua_y", "airpods_accY"),
            "trunk": ("watch_ua_x", "airpods_accX"),
        }
        for joint, (wcol, acol) in mapping.items():
            if wcol in df.columns and acol in df.columns:
                comb_col = f"{joint}_omega_combined_acc"
                df[comb_col] = (df[wcol].fillna(0) + df[acol].fillna(0)) / 2.0
                out[joint] = comb_col
            elif wcol in df.columns:
                out[joint] = wcol
            elif acol in df.columns:
                out[joint] = acol
            else:
                out[joint] = None
        return out

    if mode == "combined_mixed":
        # watch gyro + airpods acc
        out = {}
        mapping = {
            "ankle": ("watch_rot_z", "airpods_accZ"),
            "knee":  ("watch_rot_y", "airpods_accY"),
            "hip":   ("watch_rot_y", "airpods_accY"),
            "trunk": ("watch_rot_x", "airpods_accX"),
        }
        for joint, (wcol, acol) in mapping.items():
            if wcol in df.columns and acol in df.columns:
                comb_col = f"{joint}_omega_combined_mixed"
                df[comb_col] = (df[wcol].fillna(0) + df[acol].fillna(0)) / 2.0
                out[joint] = comb_col
            elif wcol in df.columns:
                out[joint] = wcol
            elif acol in df.columns:
                out[joint] = acol
            else:
                out[joint] = None
        return out

    if mode == "watch_quat":
        return {
            "ankle": "watch_quat__omega_z",
            "knee":  "watch_quat__omega_y",
            "hip":   "watch_quat__omega_y",
            "trunk": "watch_quat__omega_x",
        }
    
    if mode == "airpods_quat":
        return {
            "ankle": "airpods_q_omega_z",
            "knee":  "airpods_q_omega_y",
            "hip":   "airpods_q_omega_y",
            "trunk": "airpods_q_omega_x",
        }
    
    if mode == "combined_quat":
        out = {}
        mapping = {
            "ankle": ("watch_quat__omega_z", "airpods_q_omega_z"),
            "knee":  ("watch_quat__omega_y", "airpods_q_omega_y"),
            "hip":   ("watch_quat__omega_y", "airpods_q_omega_y"),
            "trunk": ("watch_quat__omega_x", "airpods_q_omega_x"),
        }
        for joint, (wcol, acol) in mapping.items():
            if wcol in df.columns and acol in df.columns:
                comb = f"{joint}_omega_combined_quat"
                df[comb] = (df[wcol] + df[acol]) / 2
                out[joint] = comb
            elif wcol in df.columns:
                out[joint] = wcol
            else:
                out[joint] = acol
        return out


    raise ValueError("Invalid IMU fusion mode")

def compute_2d_angle(a, b, c):
    """
    Angle at point b formed by points a-b-c in 2D (x,y).
    Returns angle in radians.
    """
    ba = a - b
    bc = c - b
    ba /= (np.linalg.norm(ba) + 1e-8)
    bc /= (np.linalg.norm(bc) + 1e-8)
    cosang = np.clip(np.dot(ba, bc), -1.0, 1.0)
    return np.arccos(cosang)

class AngleKalman:
    """
    State x = [angle, angular_velocity]^T
    Measurements:
      - angle from pose
      - angular velocity from IMU
    """

    def __init__(self, dt, q_angle=1e-4, q_omega=1e-3, r_angle=1e-3, r_omega=1e-2):
        self.dt = dt

        self.F = np.array([[1, dt],
                           [0, 1]])

        self.Q = np.array([[q_angle, 0],
                           [0, q_omega]])

        self.H_angle = np.array([[1, 0]])
        self.H_omega = np.array([[0, 1]])

        self.R_angle = np.array([[r_angle]])
        self.R_omega = np.array([[r_omega]])

        self.x = np.zeros((2, 1))
        self.P = np.eye(2)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, H, R):
        z = np.array([[z]])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

    def step(self, angle_meas=None, omega_meas=None):
        self.predict()

        if angle_meas is not None and np.isfinite(angle_meas):
            self.update(angle_meas, self.H_angle, self.R_angle)

        if omega_meas is not None and np.isfinite(omega_meas):
            self.update(omega_meas, self.H_omega, self.R_omega)

        return float(self.x[0]), float(self.x[1])

def extract_joint_angles(df):
    sides = {
        "right": {
            "hip": "RIGHT_HIP",
            "knee": "RIGHT_KNEE",
            "ankle": "RIGHT_ANKLE",
            "foot": "RIGHT_FOOT_INDEX",
            "shoulder": "RIGHT_SHOULDER"
        },
        "left": {
            "hip": "LEFT_HIP",
            "knee": "LEFT_KNEE",
            "ankle": "LEFT_ANKLE",
            "foot": "LEFT_FOOT_INDEX",
            "shoulder": "LEFT_SHOULDER"
        }
    }

    for side, lm in sides.items():
        ankle_angles = []
        knee_angles = []
        hip_angles = []
        trunk_angles = []

        for _, row in df.iterrows():
            hip = np.array([row[f"{lm['hip']}_x"], row[f"{lm['hip']}_y"]])
            knee = np.array([row[f"{lm['knee']}_x"], row[f"{lm['knee']}_y"]])
            ankle = np.array([row[f"{lm['ankle']}_x"], row[f"{lm['ankle']}_y"]])
            foot = np.array([row[f"{lm['foot']}_x"], row[f"{lm['foot']}_y"]])
            shoulder = np.array([row[f"{lm['shoulder']}_x"], row[f"{lm['shoulder']}_y"]])

            ankle_angles.append(compute_2d_angle(knee, ankle, foot))
            knee_angles.append(compute_2d_angle(hip, knee, ankle))
            hip_angles.append(compute_2d_angle(shoulder, hip, knee))

            vec = shoulder - hip
            vertical = np.array([0, -1])
            vec /= (np.linalg.norm(vec) + 1e-8)
            trunk_angles.append(np.arccos(np.clip(np.dot(vec, vertical), -1, 1)))

        df[f"{side}_ankle_angle_pose_rad"] = ankle_angles
        df[f"{side}_knee_angle_pose_rad"] = knee_angles
        df[f"{side}_hip_angle_pose_rad"] = hip_angles
        df[f"{side}_trunk_angle_pose_rad"] = trunk_angles

    return df

# -----------------------------
#  KALMAN FUSION FOR BOTH SIDES
# -----------------------------
def kalman_fuse_all_joints(df, mode):
    t = df["t_global"].values
    dt = np.median(np.diff(t))

    imu_cols = choose_imu_source(df, mode)

    for side in ["right", "left"]:
        for joint in ["ankle", "knee", "hip", "trunk"]:
            pose_col = f"{side}_{joint}_angle_pose_rad"

            if pose_col not in df.columns:
                continue

            imu_col = imu_cols.get(joint)
            if imu_col not in df.columns:
                print(f"Warning: IMU column {imu_col} missing for {side} {joint}")
                continue

            kf = AngleKalman(dt)
            fused_angle = []
            fused_omega = []

            for ang, omg in zip(df[pose_col], df[imu_col]):
                a_f, w_f = kf.step(angle_meas=ang, omega_meas=omg)
                fused_angle.append(a_f)
                fused_omega.append(w_f)

            df[f"{side}_{joint}_angle_fused_rad"] = fused_angle
            df[f"{side}_{joint}_omega_fused"] = fused_omega

    return df

def run_multi_joint_kalman(merged_csv, out_csv, mode="watch"):
    df = pd.read_csv(merged_csv)

    df = extract_joint_angles(df)
    
    # compute dt
    t = df["t_global"].values
    dt = np.median(np.diff(t))
    
    # WATCH quaternion → omega
    if "watch_quat_w" in df.columns:
        df = quat_to_omega(df, prefix="watch_quat_", dt=dt)
    
    # AIRPODS quaternion → omega
    if "airpods_qw" in df.columns:
        df = quat_to_omega(df, prefix="airpods_q", dt=dt)
    
    df = kalman_fuse_all_joints(df, mode)

    df.to_csv(out_csv, index=False)
    print(f"Saved multi-joint Kalman fused dataset to: {out_csv}")

    return df

# run_multi_joint_kalman(
#     merged_csv=r"merged/merged_pose_IMU_1779582973.csv",
#     out_csv="merged_pose_IMU_1779514418_fused.csv"
# )
