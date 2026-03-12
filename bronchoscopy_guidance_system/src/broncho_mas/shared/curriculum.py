from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set


@dataclass(frozen=True)
class LandmarkInfo:
    """
    Teaching-friendly landmark spec.

    recommended_angles is a sequence of "access orientations" (NOT a scan path).
    Each element describes:
      - angle: e.g., "90° right", "45° left", "0°"
      - purpose: why you hold that access orientation (find landmark / access lobe)
      - note (optional): short rule like "defer from access angle to inspect segments"
    """
    landmark_id: str
    recommended_angles: List[Dict[str, str]]
    recognition_cue: str


class CurriculumEngine:
    """
    Deterministic curriculum + navigation planning engine.

    Manager consumes the plan; LLM only verbalizes it.
    """

    def __init__(self, visit_order: Sequence[str]):
        self.visit_order = [str(x).upper() for x in visit_order]

    # ---------------- core decision ----------------
    def next_airway(self, reached: Set[str]) -> str:
        reached_u = {str(x).upper() for x in (reached or set())}
        for a in self.visit_order:
            if a not in reached_u:
                return a
        return ""

    def _is_unknown_airway(self, airway: str) -> bool:
        a = (airway or "").strip().upper()
        if not a:
            return True
        # allow carina and main bronchi labels from simulator
        if a in {"CARINA", "TRACHEA", "RMB", "LMB", "RIGHT_MAIN_BRONCHUS", "LEFT_MAIN_BRONCHUS"}:
            return False
        return a not in set(self.visit_order)

    # ---------------- teaching primitives ----------------
    @staticmethod
    def neutral_pose() -> Dict[str, str]:
        # JoVE definition: straight arm/wrist; suction button forward; steering lever neutral
        return {
            "angle": "0°",
            "name": "neutral",
            "definition": "arm and wrist straight; suction button facing forward; steering lever centered",
        }

    @staticmethod
    def inspect_rule_text() -> str:
        return (
            "Inspection rule: once the landmark view is obtained, deviate from the access angle to inspect individual segments; "
            "then return to the access angle to re-anchor."
        )

    @staticmethod
    def angles_to_text(angles: List[Dict[str, str]]) -> str:
        """
        Render access orientations into a single teaching-friendly line for micro-steps.
        """
        parts = []
        for item in angles:
            angle = item.get("angle", "").strip()
            purpose = item.get("purpose", "").strip()
            if angle and purpose:
                parts.append(f"{angle} ({purpose})")
            elif angle:
                parts.append(angle)
        return " → ".join(parts) if parts else ""

    def landmark_for_airway(self, next_airway: str) -> LandmarkInfo:
        """
        Approximate Four-Landmarks mapping.
        Angles are *access orientations*; scanning segments should deviate from these angles.
        """
        a = (next_airway or "").strip().upper()

        # Landmark 2: RUL (RB1-3)
        if a.startswith("RB") and a in {"RB1", "RB2", "RB3"}:
            return LandmarkInfo(
                landmark_id="L2_RUL",
                recommended_angles=[
                    {
                        "angle": "90° right",
                        "purpose": "access right upper lobe (Mercedes sign view)",
                        "note": self.inspect_rule_text(),
                    }
                ],
                recognition_cue="Mercedes sign / trifurcation at the right upper lobe",
            )

        # Landmark 3: RML + RLL pathway via bronchus intermedius
        if a.startswith("RB"):
            return LandmarkInfo(
                landmark_id="L3_RML_RLL",
                recommended_angles=[
                    {
                        "angle": "45° right",
                        "purpose": "access bronchus intermedius / middle lobe pathway",
                        "note": self.inspect_rule_text(),
                    },
                    {
                        "angle": "0°",
                        "purpose": "neutral re-anchor to continue into lower lobe (segments 7–10)",
                        "note": "Use Landmark 1 (carina) reset if orientation is lost.",
                    },
                ],
                recognition_cue="Bronchus intermedius: longer airway with sequential segmental openings along the lumen",
            )

        # Landmark 4: Left side access sequence
        if a.startswith("LB"):
            return LandmarkInfo(
                landmark_id="L4_LEFT",
                recommended_angles=[
                    {
                        "angle": "90° left",
                        "purpose": "access left upper lobe + lingula",
                        "note": self.inspect_rule_text(),
                    },
                    {
                        "angle": "45° left",
                        "purpose": "access segment 6 (opposite lingula)",
                        "note": self.inspect_rule_text(),
                    },
                    {
                        "angle": "0°",
                        "purpose": "neutral to access left lower lobe (segments 8–10)",
                        "note": "Use Landmark 1 (carina) reset if orientation is lost.",
                    },
                ],
                recognition_cue="Left main bronchus takeoff: more horizontal course than the right; stable bifurcation landmarks",
            )

        # Fallback: Landmark 1 carina
        return LandmarkInfo(
            landmark_id="L1_CARINA",
            recommended_angles=[
                {
                    "angle": "0°",
                    "purpose": "neutral reference at carina (Landmark 1)",
                    "note": "If orientation is lost, return here to reorient.",
                }
            ],
            recognition_cue="Carina bifurcation; symmetric right/left main bronchi",
        )

    def should_reorient(self, current_airway: str, student_question: str = "", stagnating: bool = False) -> bool:
        q = (student_question or "").lower()
        q_lost = any(k in q for k in ["lost", "where am i", "confused", "i can't", "don't know", "dont know", "迷路", "在哪"])
        return self._is_unknown_airway(current_airway) or stagnating or q_lost

    def make_plan(
        self,
        current_airway: str,
        reached: Set[str],
        student_question: str = "",
        *,
        is_back: bool = False,
        back_streak: int = 0,
        stagnating: bool = False,
    ) -> Dict[str, Any]:
        curr = (current_airway or "").strip().upper()
        nxt = self.next_airway(reached)

        # Done
        if not nxt:
            return {
                "mode": "done",
                "current_airway": curr,
                "next_airway": "",
                "anchor_landmark": "L1_CARINA",
                "recommended_angle": self.neutral_pose(),
                "recognition_cue": "carina bifurcation",
                "micro_steps": ["Action: Stop. Check: all curriculum targets reached; end the procedure."],
                "why": "All curriculum targets reached.",
            }

        # Backtracking / safety mode
        if is_back or back_streak >= 2:
            return {
                "mode": "backtrack",
                "current_airway": curr,
                "next_airway": nxt,
                "anchor_landmark": "L1_CARINA",
                "recommended_angle": self.neutral_pose(),
                "recognition_cue": "carina bifurcation; symmetric right/left main bronchi",
                "micro_steps": [
                    "Action: Withdraw 1–2 cm to widen the field of view. Check: lumen looks wider and more centered (less wall contact).",
                    "Action: Re-center the lumen (keep the dark airway lumen in the middle). Check: the ring-shaped lumen stays concentric, not sliding off-screen.",
                    "Action: Re-anchor at Landmark 1 (carina) and reset to neutral (0°): arm/wrist straight, suction button forward, steering lever neutral. "
                    "Check: carina centered; both main bronchi open symmetrically.",
                ],
                "why": "Backtracking detected; prioritize safety and re-orientation before continuing.",
            }

        # Reorientation if uncertain
        if self.should_reorient(curr, student_question=student_question, stagnating=stagnating):
            tgt = self.landmark_for_airway(nxt)
            access_line = self.angles_to_text(tgt.recommended_angles)

            return {
                "mode": "reorient",
                "current_airway": curr,
                "next_airway": nxt,
                "anchor_landmark": "L1_CARINA",
                "recommended_angle": self.neutral_pose(),
                "recognition_cue": "carina bifurcation; symmetric main bronchi",
                "micro_steps": [
                    "Action: Re-anchor at Landmark 1 (carina) and reset to neutral (0°): arm/wrist straight, suction button forward, steering lever neutral. "
                    "Check: carina centered; right and left main bronchi open symmetrically.",
                    f"Action: Use the access orientation(s): {access_line}. Check: the target view becomes stable before advancing.",
                    f"Action: Identify the landmark cue: {tgt.recognition_cue}. Check: cue persists for 1–2 seconds without drifting.",
                    f"Action: Advance into {nxt} in small increments. Check: lumen stays centered; avoid scraping the wall.",
                ],
                "why": "Re-anchor at carina (Landmark 1), then continue structured progress using landmark access orientations.",
            }

        # Normal advance
        tgt = self.landmark_for_airway(nxt)
        access_line = self.angles_to_text(tgt.recommended_angles)

        return {
            "mode": "advance",
            "current_airway": curr,
            "next_airway": nxt,
            "anchor_landmark": tgt.landmark_id,
            # keep key name for manager compatibility, but content is now structured
            "recommended_angle": {
                "access_orientations": tgt.recommended_angles,
                "neutral_definition": self.neutral_pose(),
            },
            "recognition_cue": tgt.recognition_cue,
            "micro_steps": [
                f"Action: Set the access orientation: {access_line}. Check: view is stable and lumen remains centered.",
                f"Action: Confirm landmark cue: {tgt.recognition_cue}. Check: cue remains consistent for 1–2 seconds.",
                f"Action: Inspect segments by deviating from the access angle, then re-anchor to the access angle. Check: no segments are skipped.",
                f"Action: Advance into {nxt}. Check: lumen stays centered; minimal wall contact.",
            ],
            "why": "Structured progress: next unvisited segment.",
        }