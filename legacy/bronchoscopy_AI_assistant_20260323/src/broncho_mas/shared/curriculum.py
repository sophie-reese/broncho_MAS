from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set


@dataclass(frozen=True)
class LandmarkInfo:
    """
    Teaching-friendly landmark spec.

    recommended_angles is a sequence of access orientations, not a scan path.
    """
    landmark_id: str
    recommended_angles: List[Dict[str, str]]
    recognition_cue: str


class CurriculumEngine:
    """
    Slim curriculum engine.

    Responsibilities:
    - keep visit order
    - compute next_airway from reached regions
    - provide landmark/access teaching info for a chosen target

    Non-responsibilities:
    - do NOT infer current state from prompt text
    - do NOT guess airway state from natural language
    - do NOT decide low-level control from simulator signals
    """

    DEFAULT_KNOWN_ANCHORS = {
        "CARINA",
        "TRACHEA",
        "RMB",
        "LMB",
        "RIGHT_MAIN_BRONCHUS",
        "LEFT_MAIN_BRONCHUS",
    }

    def __init__(self, visit_order: Sequence[str]):
        self.visit_order = [str(x).upper() for x in visit_order]
        self.visit_order_set = set(self.visit_order)

    # ---------------- basic sequencing ----------------
    def normalize_reached(self, reached: Sequence[str] | Set[str] | None) -> List[str]:
        if not reached:
            return []
        out: List[str] = []
        seen = set()
        for item in reached:
            key = str(item).strip().upper()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def next_airway(self, reached: Set[str] | Sequence[str] | None) -> str:
        reached_u = set(self.normalize_reached(reached))
        for airway in self.visit_order:
            if airway not in reached_u:
                return airway
        return ""

    def coverage_ratio(self, reached: Set[str] | Sequence[str] | None) -> float:
        reached_u = set(self.normalize_reached(reached))
        total = len(self.visit_order)
        if total == 0:
            return 0.0
        covered = len([a for a in self.visit_order if a in reached_u])
        return round(covered / total, 4)

    def progress_snapshot(self, reached: Set[str] | Sequence[str] | None) -> Dict[str, Any]:
        reached_list = self.normalize_reached(reached)
        return {
            "reached": reached_list,
            "next_airway": self.next_airway(reached_list),
            "reached_count": len([a for a in self.visit_order if a in set(reached_list)]),
            "total": len(self.visit_order),
            "coverage_ratio": self.coverage_ratio(reached_list),
        }

    # ---------------- airway helpers ----------------
    def is_known_airway(self, airway: str) -> bool:
        a = (airway or "").strip().upper()
        if not a:
            return False
        return a in self.visit_order_set or a in self.DEFAULT_KNOWN_ANCHORS

    # ---------------- teaching primitives ----------------
    @staticmethod
    def neutral_pose() -> Dict[str, str]:
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
        parts = []
        for item in angles:
            angle = item.get("angle", "").strip()
            purpose = item.get("purpose", "").strip()
            if angle and purpose:
                parts.append(f"{angle} ({purpose})")
            elif angle:
                parts.append(angle)
        return " → ".join(parts) if parts else ""

    # ---------------- landmarks ----------------
    def landmark_for_airway(self, airway: str) -> LandmarkInfo:
        """
        Map a target airway to teaching-friendly landmark/access info.
        """
        a = (airway or "").strip().upper()

        if a in {"RB1", "RB2", "RB3"}:
            return LandmarkInfo(
                landmark_id="L2_RUL",
                recommended_angles=[
                    {
                        "angle": "90° right",
                        "purpose": "access right upper lobe",
                        "note": self.inspect_rule_text(),
                    }
                ],
                recognition_cue="Mercedes sign / trifurcation at the right upper lobe",
            )

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
                        "purpose": "neutral re-anchor to continue into lower lobe",
                        "note": "Use Landmark 1 (carina) reset if orientation is lost.",
                    },
                ],
                recognition_cue="Bronchus intermedius: longer airway with sequential segmental openings along the lumen",
            )

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
                        "purpose": "access segment 6",
                        "note": self.inspect_rule_text(),
                    },
                    {
                        "angle": "0°",
                        "purpose": "neutral to access left lower lobe",
                        "note": "Use Landmark 1 (carina) reset if orientation is lost.",
                    },
                ],
                recognition_cue="Left main bronchus takeoff: more horizontal course than the right; stable bifurcation landmarks",
            )

        return LandmarkInfo(
            landmark_id="L1_CARINA",
            recommended_angles=[
                {
                    "angle": "0°",
                    "purpose": "neutral reference at carina",
                    "note": "If orientation is lost, return here to reorient.",
                }
            ],
            recognition_cue="Carina bifurcation; symmetric right/left main bronchi",
        )

    # ---------------- plan generation ----------------
    def make_plan(
        self,
        current_airway: str,
        reached: Set[str] | Sequence[str] | None,
        student_question: str = "",
        *,
        requested_next_airway: str = "",
        is_back: bool = False,
        back_streak: int = 0,
        stagnating: bool = False,
    ) -> Dict[str, Any]:
        """
        Build a deterministic teaching plan from already-known state.

        Important: this function assumes current_airway / reached / requested_next_airway
        were provided by upstream state. It does not try to infer them from text.
        """
        curr = (current_airway or "").strip().upper()
        reached_u = set(self.normalize_reached(reached))
        requested = (requested_next_airway or "").strip().upper()

        nxt = requested or self.next_airway(reached_u)

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

        # Safety/backtrack mode is still useful, but it uses upstream flags only.
        if is_back or back_streak >= 2:
            return {
                "mode": "backtrack",
                "current_airway": curr,
                "next_airway": nxt,
                "anchor_landmark": "L1_CARINA",
                "recommended_angle": self.neutral_pose(),
                "recognition_cue": "carina bifurcation; symmetric right/left main bronchi",
                "micro_steps": [
                    "Action: Withdraw 1–2 cm to widen the field of view. Check: lumen looks wider and more centered.",
                    "Action: Re-center the lumen. Check: the lumen stays concentric rather than sliding toward the wall.",
                    "Action: Re-anchor at Landmark 1 (carina) and reset to neutral. Check: carina centered; both main bronchi open symmetrically.",
                ],
                "why": "Backtracking detected; prioritize safety and re-orientation before continuing.",
            }

        # Reorientation is driven by explicit upstream state, not by text guessing.
        if stagnating or not self.is_known_airway(curr):
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
                    "Action: Re-anchor at Landmark 1 (carina) and reset to neutral. Check: carina centered; right and left main bronchi open symmetrically.",
                    f"Action: Use the access orientation(s): {access_line}. Check: the target view becomes stable before advancing.",
                    f"Action: Identify the landmark cue: {tgt.recognition_cue}. Check: cue persists for 1–2 seconds without drifting.",
                    f"Action: Advance into {nxt} in small increments. Check: lumen stays centered; avoid scraping the wall.",
                ],
                "why": "Known target exists, but current orientation is uncertain; reset and re-anchor first.",
            }

        tgt = self.landmark_for_airway(nxt)
        access_line = self.angles_to_text(tgt.recommended_angles)

        return {
            "mode": "advance",
            "current_airway": curr,
            "next_airway": nxt,
            "anchor_landmark": tgt.landmark_id,
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
            "why": "Structured progress to the next planned airway.",
        }
