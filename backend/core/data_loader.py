"""
MotionDataLoader — loads the Kinect v2 motion capture data from data_new/.

Data format per trial directory:
  - Joint_Positions.csv : raw 3D positions, 3 columns (x, y, z) per row.
    Every 25 consecutive rows correspond to one frame (25 Kinect joints).
  - Labels.csv          : one integer label per frame (binary).

Subjects  H01‑H10 = healthy,  P01‑P09 = patients.
"""

import os
import numpy as np
from scipy.signal import butter, filtfilt


class MotionDataLoader:
    NUM_JOINTS = 25  # Kinect v2 skeleton

    def __init__(self, data_root=None, fps=30):
        if data_root is None:
            data_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data_new",
            )
        self.data_root = data_root
        self.fps = fps

    # ------------------------------------------------------------------
    # Single trial
    # ------------------------------------------------------------------
    def load_trial(self, subject_id, exercise_name):
        """
        Returns
        -------
        positions : ndarray (num_frames, 25, 3) or None
        labels    : ndarray (num_frames,)        or None
        """
        trial_dir = os.path.join(self.data_root, subject_id, exercise_name)
        pos_path = os.path.join(trial_dir, "Joint_Positions.csv")
        lbl_path = os.path.join(trial_dir, "Labels.csv")

        if not os.path.exists(pos_path):
            return None, None

        raw = np.loadtxt(pos_path, delimiter=",")
        num_frames = raw.shape[0] // self.NUM_JOINTS
        positions = raw[: num_frames * self.NUM_JOINTS].reshape(
            num_frames, self.NUM_JOINTS, 3
        )

        labels = None
        if os.path.exists(lbl_path):
            labels = np.loadtxt(lbl_path, dtype=int)
            min_len = min(num_frames, len(labels))
            positions = positions[:min_len]
            labels = labels[:min_len]

        return positions, labels

    # ------------------------------------------------------------------
    # All trials
    # ------------------------------------------------------------------
    def load_all_trials(self):
        """
        Returns
        -------
        all_positions : list[ndarray]  each (N, 25, 3)
        all_labels    : list[ndarray]  each (N,)
        trial_info    : list[dict]
        """
        all_positions, all_labels, trial_info = [], [], []

        for subject_id in sorted(os.listdir(self.data_root)):
            subject_dir = os.path.join(self.data_root, subject_id)
            if not os.path.isdir(subject_dir):
                continue
            for exercise in sorted(os.listdir(subject_dir)):
                exercise_dir = os.path.join(subject_dir, exercise)
                if not os.path.isdir(exercise_dir):
                    continue

                positions, labels = self.load_trial(subject_id, exercise)
                if positions is not None:
                    all_positions.append(positions)
                    all_labels.append(labels)
                    trial_info.append(
                        {
                            "subject": subject_id,
                            "exercise": exercise,
                            "num_frames": len(positions),
                            "is_patient": subject_id.startswith("P"),
                        }
                    )
                    print(
                        f"  Loaded {subject_id}/{exercise}: "
                        f"{len(positions)} frames, "
                        f"label dist {np.bincount(labels)}"
                    )

        return all_positions, all_labels, trial_info

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------
    def apply_lowpass_filter(self, positions, cutoff=6.0, order=4):
        """Zero‑phase Butterworth lowpass filter on each joint trajectory."""
        if positions.shape[0] < 15:
            return positions

        nyq = 0.5 * self.fps
        b, a = butter(order, cutoff / nyq, btype="low", analog=False)

        smoothed = np.zeros_like(positions)
        for j in range(self.NUM_JOINTS):
            for c in range(3):
                smoothed[:, j, c] = filtfilt(b, a, positions[:, j, c])
        return smoothed
