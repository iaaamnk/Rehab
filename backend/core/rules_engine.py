"""
Clinical rules engine with session‑level scoring (ROM, Stability, Quality).
"""

import numpy as np


class RulesEngine:
    # ── clinical thresholds (degrees) ──
    MAX_TRUNK_LEAN = 20.0
    MAX_RELATIVE_LEAN = 10.0
    MAX_LATERAL_LEAN = 15.0

    # ~30 seconds rolling window at ~15 fps
    WINDOW_SIZE = 450

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
        sh_ang  = max(features.get("right_shoulder_angle", 0), features.get("left_shoulder_angle", 0))

        # Track session history
        self.frame_count += 1
        self.trunk_history.append(trunk)
        self.rel_history.append(rel)
        self.lateral_history.append(lateral)
        self.shoulder_history.append(sh_ang)

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

        self.compensatory_history.append(is_compensatory)

        # Rolling window
        if len(self.trunk_history) > self.WINDOW_SIZE:
            self.trunk_history.pop(0)
            self.rel_history.pop(0)
            self.lateral_history.pop(0)
            self.shoulder_history.pop(0)
            self.compensatory_history.pop(0)

        return {
            "is_compensatory_rule_based": is_compensatory,
            "alerts": alerts,
        }

    # ------------------------------------------------------------------
    # Session‑level clinical scores  (all 0‑100)
    # ------------------------------------------------------------------
    def get_session_scores(self) -> dict:
        if not self.trunk_history:
            return dict(rom_score=0, stability_score=0, quality_score=0, total_score=0)

        # ROM score — functional shoulder abduction ≈ 120 ° baseline
        max_rom = max(self.shoulder_history)
        rom_score = min(100.0, (max_rom / 120.0) * 100)

        # Stability score — lower trunk‑lean variance → higher score
        if len(self.trunk_history) > 1:
            variance = float(np.var(self.trunk_history))
            stability_score = max(0.0, 100.0 - variance * 5)
        else:
            stability_score = 100.0

        # Quality score — % of non‑compensatory frames
        comp_count = sum(self.compensatory_history)
        quality_score = ((len(self.compensatory_history) - comp_count) / len(self.compensatory_history)) * 100

        total_score = rom_score * 0.3 + stability_score * 0.3 + quality_score * 0.4

        return {
            "rom_score": round(rom_score, 1),
            "stability_score": round(stability_score, 1),
            "quality_score": round(quality_score, 1),
            "total_score": round(total_score, 1),
        }

    # ------------------------------------------------------------------
    # Aggregated session-wide overall feedback & posture tips
    # ------------------------------------------------------------------
    def get_overall_feedback(self) -> list[dict]:
        if not self.trunk_history:
            return []

        feedback = []
        total_frames = len(self.rel_history)

        # 1. Forward Lean
        if total_frames > 0:
            excess_rel_frames = [r for r in self.rel_history if r > self.MAX_RELATIVE_LEAN]
            rel_excess_pct = (len(excess_rel_frames) / total_frames) * 100
            
            if rel_excess_pct > 20.0:
                avg_excess_rel = float(sum(excess_rel_frames) / len(excess_rel_frames)) if excess_rel_frames else 0.0
                feedback.append({
                    "category": "Forward Lean",
                    "status": "needs_improvement",
                    "message": f"Frequent forward trunk lean detected (+{avg_excess_rel:.1f}° above baseline in {rel_excess_pct:.0f}% of frames).",
                    "tip": "Focus on keeping your chest up and shoulders back. Avoid leaning forward as you lift your arm."
                })
            else:
                feedback.append({
                    "category": "Forward Lean",
                    "status": "excellent",
                    "message": "Good sagittal posture maintained.",
                    "tip": "Keep maintaining this upright trunk alignment."
                })

        # 2. Lateral Lean
        if total_frames > 0:
            lat_excess_count = sum(1 for l in self.lateral_history if abs(l) > self.MAX_LATERAL_LEAN)
            lat_excess_pct = (lat_excess_count / total_frames) * 100
            
            if lat_excess_pct > 15.0:
                max_lat = float(max(abs(l) for l in self.lateral_history)) if self.lateral_history else 0.0
                feedback.append({
                    "category": "Lateral Lean",
                    "status": "needs_improvement",
                    "message": f"Significant side-to-side trunk sway detected (max lateral lean of {max_lat:.1f}°).",
                    "tip": "Engage your core muscles to keep your trunk centered. Avoid tilting to the side to help lift your arm."
                })
            else:
                feedback.append({
                    "category": "Lateral Lean",
                    "status": "excellent",
                    "message": "Excellent lateral stability.",
                    "tip": "Great work keeping your trunk level and centered."
                })

        # 3. Stability
        if len(self.trunk_history) > 1:
            variance = float(np.var(self.trunk_history))
            if variance > 4.0:
                feedback.append({
                    "category": "Stability",
                    "status": "needs_improvement",
                    "message": f"Trunk movement is slightly unsteady (posture variance: {variance:.1f}).",
                    "tip": "Perform the exercise slowly and focus on core stability. Control the motion both on the way up and down."
                })
            else:
                feedback.append({
                    "category": "Stability",
                    "status": "excellent",
                    "message": "Steady posture control throughout.",
                    "tip": "Smooth and controlled movement. Keep maintaining this steady pace."
                })
        else:
            feedback.append({
                "category": "Stability",
                "status": "excellent",
                "message": "Initializing stability metrics.",
                "tip": "Keep performing steady movements to establish stability scores."
            })

        # 4. Range of Motion
        if self.shoulder_history:
            max_rom = float(max(self.shoulder_history))
            if max_rom < 85.0:
                feedback.append({
                    "category": "Range of Motion",
                    "status": "needs_improvement",
                    "message": f"Limited shoulder range of motion (Max: {max_rom:.1f}°).",
                    "tip": "Try to raise your arm a bit higher if you can do so comfortably. Avoid pushing through joint pain."
                })
            else:
                feedback.append({
                    "category": "Range of Motion",
                    "status": "excellent",
                    "message": f"Strong range of motion reached ({max_rom:.1f}°).",
                    "tip": "Excellent movement amplitude. Continue working to sustain this range of motion."
                })

        return feedback

    # ------------------------------------------------------------------
    def reset(self):
        self._reset_session()

    def _reset_session(self):
        self.frame_count = 0
        self.trunk_history: list[float] = []
        self.rel_history: list[float] = []
        self.lateral_history: list[float] = []
        self.shoulder_history: list[float] = []
        self.compensatory_history: list[bool] = []

