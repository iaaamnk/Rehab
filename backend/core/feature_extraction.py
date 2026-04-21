"""
Biomechanical feature extraction for both Kinect (batch training) and
MediaPipe (live inference) skeletons.
"""

import numpy as np

# ── Angle utility ──────────────────────────────────────────────────────
def calculate_angle(v1, v2):
    """Angle (degrees) between two 3‑D vectors."""
    v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


# ── Joint index maps ──────────────────────────────────────────────────
KINECT_JOINTS = {
    "spine_base": 0, "spine_mid": 1, "neck": 2, "head": 3,
    "shoulder_left": 4, "elbow_left": 5, "wrist_left": 6, "hand_left": 7,
    "shoulder_right": 8, "elbow_right": 9, "wrist_right": 10, "hand_right": 11,
    "hip_left": 12, "knee_left": 13, "ankle_left": 14, "foot_left": 15,
    "hip_right": 16, "knee_right": 17, "ankle_right": 18, "foot_right": 19,
    "spine_shoulder": 20, "hand_tip_left": 21, "thumb_left": 22,
    "hand_tip_right": 23, "thumb_right": 24,
}

MEDIAPIPE_LANDMARKS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
}

# Canonical feature vector order (shared by training and inference).
FEATURE_NAMES = [
    "right_shoulder_angle",
    "left_shoulder_angle",
    "trunk_lean",
    "right_elbow_angle",
    "trunk_lateral_lean",
    "shoulder_height_diff",
]


# ── Kinect (batch) ────────────────────────────────────────────────────
def extract_frame_features_kinect(joints):
    """
    Parameters
    ----------
    joints : ndarray (25, 3)

    Returns
    -------
    dict  keyed by FEATURE_NAMES
    """
    k = KINECT_JOINTS

    r_shoulder = joints[k["shoulder_right"]]
    r_elbow    = joints[k["elbow_right"]]
    r_wrist    = joints[k["wrist_right"]]
    r_hip      = joints[k["hip_right"]]
    l_shoulder = joints[k["shoulder_left"]]
    l_elbow    = joints[k["elbow_left"]]
    l_hip      = joints[k["hip_left"]]
    spine_sh   = joints[k["spine_shoulder"]]
    spine_base = joints[k["spine_base"]]

    # 1. Right shoulder abduction
    right_shoulder_angle = calculate_angle(r_hip - r_shoulder, r_elbow - r_shoulder)

    # 2. Left shoulder abduction
    left_shoulder_angle = calculate_angle(l_hip - l_shoulder, l_elbow - l_shoulder)

    # 3. Trunk lean (sagittal — forward/backward tilt)
    spine_vec = spine_sh - spine_base
    trunk_lean = calculate_angle(spine_vec, np.array([0.0, 1.0, 0.0]))

    # 4. Right elbow flexion
    right_elbow_angle = calculate_angle(r_shoulder - r_elbow, r_wrist - r_elbow)

    # 5. Trunk lateral lean (frontal plane)
    mid_sh = (r_shoulder + l_shoulder) / 2
    mid_hp = (r_hip + l_hip) / 2
    trunk_frontal = mid_sh - mid_hp
    trunk_frontal_proj = np.array([trunk_frontal[0], trunk_frontal[1], 0.0])
    trunk_lateral_lean = calculate_angle(trunk_frontal_proj, np.array([0.0, 1.0, 0.0]))

    # 6. Shoulder height difference
    shoulder_height_diff = float(r_shoulder[1] - l_shoulder[1])

    return {
        "right_shoulder_angle": right_shoulder_angle,
        "left_shoulder_angle": left_shoulder_angle,
        "trunk_lean": trunk_lean,
        "right_elbow_angle": right_elbow_angle,
        "trunk_lateral_lean": trunk_lateral_lean,
        "shoulder_height_diff": shoulder_height_diff,
    }


def extract_features_batch(positions):
    """
    Parameters
    ----------
    positions : ndarray (num_frames, 25, 3)

    Returns
    -------
    ndarray (num_frames, len(FEATURE_NAMES))
    """
    out = []
    for frame in positions:
        d = extract_frame_features_kinect(frame)
        out.append([d[n] for n in FEATURE_NAMES])
    return np.array(out, dtype=np.float64)


# ── MediaPipe (live) ──────────────────────────────────────────────────
class KinematicFeatureExtractor:
    """Frame‑by‑frame feature extractor for MediaPipe Pose (33 landmarks)."""

    def __init__(self, calibration_frames=30):
        self.calibration_frames_required = calibration_frames
        self.calibration_buffer = []
        self.is_calibrated = False
        self.baseline_trunk_lean = 0.0

    def process_frame(self, landmarks):
        """
        Parameters
        ----------
        landmarks : list of 33 dicts  [{'x','y','z','visibility'}, ...]

        Returns
        -------
        dict  (superset of FEATURE_NAMES + legacy keys for backward compat)
        """
        if not landmarks or len(landmarks) < 33:
            return None

        try:
            def lm(idx):
                return np.array([landmarks[idx]["x"],
                                 landmarks[idx]["y"],
                                 landmarks[idx]["z"]])

            m = MEDIAPIPE_LANDMARKS
            r_shoulder = lm(m["right_shoulder"])
            r_elbow    = lm(m["right_elbow"])
            r_wrist    = lm(m["right_wrist"])
            r_hip      = lm(m["right_hip"])
            l_shoulder = lm(m["left_shoulder"])
            l_elbow    = lm(m["left_elbow"])
            l_hip      = lm(m["left_hip"])

            # 1–2. Shoulder angles
            right_shoulder_angle = calculate_angle(r_hip - r_shoulder, r_elbow - r_shoulder)
            left_shoulder_angle  = calculate_angle(l_hip - l_shoulder, l_elbow - l_shoulder)

            # 3. Trunk lean
            mid_sh = (r_shoulder + l_shoulder) / 2
            mid_hp = (r_hip + l_hip) / 2
            spine  = mid_sh - mid_hp
            vertical = np.array([0.0, -1.0, 0.0])  # MediaPipe Y points down
            trunk_lean = calculate_angle(spine, vertical)

            # 4. Right elbow angle
            right_elbow_angle = calculate_angle(r_shoulder - r_elbow, r_wrist - r_elbow)

            # 5. Trunk lateral lean
            trunk_frontal = np.array([spine[0], spine[1], 0.0])
            trunk_lateral_lean = calculate_angle(trunk_frontal, np.array([0.0, -1.0, 0.0]))

            # 6. Shoulder height diff
            shoulder_height_diff = float(r_shoulder[1] - l_shoulder[1])

            features = {
                "right_shoulder_angle": right_shoulder_angle,
                "left_shoulder_angle": left_shoulder_angle,
                "trunk_lean": trunk_lean,
                "right_elbow_angle": right_elbow_angle,
                "trunk_lateral_lean": trunk_lateral_lean,
                "shoulder_height_diff": shoulder_height_diff,
                # Legacy keys kept for backward compatibility
                "absolute_trunk_lean": trunk_lean,
                "relative_trunk_lean": 0.0,
            }

            # ── Calibration ──
            if not self.is_calibrated:
                self.calibration_buffer.append(trunk_lean)
                if len(self.calibration_buffer) >= self.calibration_frames_required:
                    self.baseline_trunk_lean = float(np.mean(self.calibration_buffer))
                    self.is_calibrated = True

            if self.is_calibrated:
                features["relative_trunk_lean"] = trunk_lean - self.baseline_trunk_lean

            return features

        except Exception:
            return None
