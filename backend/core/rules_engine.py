"""
Clinical rules engine with session‑level scoring (ROM, Stability, Quality).
"""

import numpy as np


class RulesEngine:
    # ── clinical thresholds (degrees) ──
    MAX_TRUNK_LEAN = 20.0
    MAX_RELATIVE_LEAN = 10.0
    MAX_LATERAL_LEAN = 15.0

    def __init__(self):
        self._reset_session()

    # ------------------------------------------------------------------
    # Per‑frame evaluation
    # ------------------------------------------------------------------
    def evaluate_frame(self, features: dict) -> dict:
        alerts: list[str] = []
        is_compensatory = False

        trunk   = features.get("trunk_lean", features.get("absolute_trunk_lean", 0))
        rel     = features.get("relative_trunk_lean", 0)
        lateral = features.get("trunk_lateral_lean", 0)
        sh_ang  = features.get("right_shoulder_angle", 0)

        # Track session history
        self.frame_count += 1
        self.trunk_history.append(trunk)
        self.shoulder_history.append(sh_ang)
        self.max_rom = max(self.max_rom, sh_ang)

        # Rule checks
        if rel > self.MAX_RELATIVE_LEAN:
            alerts.append(
                f"Excessive forward lean from baseline (+{rel:.1f}°). "
                "Keep your back straight."
            )
            is_compensatory = True

        if trunk > self.MAX_TRUNK_LEAN:
            alerts.append(
                f"Severe trunk lean ({trunk:.1f}°). "
                "Please straighten your posture."
            )
            is_compensatory = True

        if lateral > self.MAX_LATERAL_LEAN:
            alerts.append(
                f"Lateral lean detected ({lateral:.1f}°). "
                "Keep your trunk centred."
            )
            is_compensatory = True

        if is_compensatory:
            self.compensatory_frames += 1

        return {
            "is_compensatory_rule_based": is_compensatory,
            "alerts": alerts,
        }

    # ------------------------------------------------------------------
    # Session‑level clinical scores  (all 0‑100)
    # ------------------------------------------------------------------
    def get_session_scores(self) -> dict:
        if self.frame_count == 0:
            return dict(rom_score=0, stability_score=0, quality_score=0, total_score=0)

        # ROM score — functional shoulder abduction ≈ 120 ° baseline
        rom_score = min(100.0, (self.max_rom / 120.0) * 100)

        # Stability score — lower trunk‑lean variance → higher score
        if len(self.trunk_history) > 1:
            variance = float(np.var(self.trunk_history))
            stability_score = max(0.0, 100.0 - variance * 5)
        else:
            stability_score = 100.0

        # Quality score — % of non‑compensatory frames
        quality_score = (
            (self.frame_count - self.compensatory_frames) / self.frame_count
        ) * 100

        total_score = rom_score * 0.3 + stability_score * 0.3 + quality_score * 0.4

        return {
            "rom_score": round(rom_score, 1),
            "stability_score": round(stability_score, 1),
            "quality_score": round(quality_score, 1),
            "total_score": round(total_score, 1),
        }

    # ------------------------------------------------------------------
    def reset(self):
        self._reset_session()

    def _reset_session(self):
        self.frame_count = 0
        self.compensatory_frames = 0
        self.max_rom = 0.0
        self.trunk_history: list[float] = []
        self.shoulder_history: list[float] = []
