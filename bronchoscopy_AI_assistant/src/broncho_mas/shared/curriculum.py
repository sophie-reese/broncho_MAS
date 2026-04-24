from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set

from . import (
    airway_path_to_root,
    classify_anatomical_relationship,
    get_airway_info,
    nearest_shared_ancestor,
    normalize_airway_code,
)

DEFAULT_AIRWAY_VISIT_ORDER: tuple[str, ...] = (
    "RB1", "RB2", "RB3", "RB4", "RB5", "RB6", "RB7", "RB8", "RB9", "RB10",
    "LB1+2", "LB3", "LB4", "LB5", "LB6", "LB8", "LB9", "LB10",
)


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

    FAMILY_MEMBERS = {
        "right_upper_family": ["RB1", "RB2", "RB3"],
        "right_upper_lobe": ["RB1", "RB2", "RB3"],
        "right_middle_family": ["RB4", "RB5"],
        "right_middle_lobe": ["RB4", "RB5"],
        "right_lower_family": ["RB6", "RB7", "RB8", "RB9", "RB10"],
        "right_lower_lobe": ["RB6", "RB7", "RB8", "RB9", "RB10"],
        "left_upper_family": ["LB1+2", "LB3"],
        "left_upper_division": ["LB1+2", "LB3"],
        "lingula_family": ["LB4", "LB5"],
        "lingula": ["LB4", "LB5"],
        "left_lower_family": ["LB6", "LB8", "LB9", "LB10"],
        "left_lower_lobe": ["LB6", "LB8", "LB9", "LB10"],
    }

    FAMILY_LABELS = {
        "right_upper_family": "right upper family",
        "right_upper_lobe": "right upper lobe",
        "right_middle_family": "right middle family",
        "right_middle_lobe": "right middle lobe",
        "right_lower_family": "right lower family",
        "right_lower_lobe": "right lower lobe",
        "left_upper_family": "left upper family",
        "left_upper_division": "left upper division",
        "lingula_family": "lingula",
        "lingula": "lingula",
        "left_lower_family": "left lower family",
        "left_lower_lobe": "left lower lobe",
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

    def family_for_airway(self, airway: str) -> str:
        a = (airway or "").strip().upper()
        info = get_airway_info(a)
        if info and info.family:
            return info.family
        for family, members in self.FAMILY_MEMBERS.items():
            if a in members:
                return family
        return ""

    def family_members(self, family_name: str) -> List[str]:
        return list(self.FAMILY_MEMBERS.get(str(family_name or "").strip(), []))

    def family_label(self, family_name: str) -> str:
        return self.FAMILY_LABELS.get(str(family_name or "").strip(), "")

    def family_complete(self, reached: Set[str] | Sequence[str] | None, family_name: str) -> bool:
        members = self.family_members(family_name)
        if not members:
            return False
        reached_u = set(self.normalize_reached(reached))
        return all(member in reached_u for member in members)

    def right_lung_complete(self, reached: Set[str] | Sequence[str] | None) -> bool:
        reached_u = set(self.normalize_reached(reached))
        right_targets = [a for a in self.visit_order if a.startswith("RB")]
        return bool(right_targets) and all(a in reached_u for a in right_targets)

    def session_complete(self, reached: Set[str] | Sequence[str] | None) -> bool:
        reached_u = set(self.normalize_reached(reached))
        return bool(self.visit_order) and all(a in reached_u for a in self.visit_order)

    def route_to_airway(self, airway: str, current_airway: str = "") -> List[str]:
        a = normalize_airway_code(airway)
        curr = normalize_airway_code(current_airway)
        if not a:
            return []

        target_path = list(reversed(airway_path_to_root(a)))
        target_path = [node for node in target_path if node != "TRACHEA"]
        if not target_path:
            return [a]
        if not curr:
            return target_path
        if curr == a:
            return [a]

        shared = nearest_shared_ancestor(curr, a)
        if shared and shared in target_path:
            shared_index = target_path.index(shared)
            # Include the shared local landmark when moving between sibling or
            # same-family branches; exclude it when the scope is already there.
            if curr == shared:
                return target_path[shared_index + 1 :] or [a]
            return target_path[shared_index:] or [a]

        relationship = classify_anatomical_relationship(curr, a)
        if relationship in {"cross_main_bronchus", "cross_side", "unknown"}:
            return target_path

        return [a]

    def transition_context(self, current_airway: str, target_airway: str) -> Dict[str, str]:
        curr = normalize_airway_code(current_airway)
        target = normalize_airway_code(target_airway)
        current_info = get_airway_info(curr)
        target_info = get_airway_info(target)
        relationship = classify_anatomical_relationship(curr, target)
        shared = nearest_shared_ancestor(curr, target)

        if relationship in {"sibling", "same_family"}:
            transition_type = "local_sibling"
        elif relationship in {"regional_branch_change", "ancestor_target", "descendant_target"}:
            transition_type = "regional_reanchor"
        elif relationship in {"cross_main_bronchus", "unknown"}:
            transition_type = "global_reanchor"
        else:
            transition_type = "advance"

        return {
            "current_code": curr,
            "target_code": target,
            "current_label": current_info.label if current_info else curr,
            "target_label": target_info.label if target_info else target,
            "current_side": current_info.side if current_info else "",
            "target_side": target_info.side if target_info else "",
            "relationship": relationship,
            "shared_ancestor": shared,
            "shared_ancestor_label": get_airway_info(shared).label if shared and get_airway_info(shared) else shared,
            "transition_type": transition_type,
        }

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
                recognition_cue="the right upper-lobe trifurcation",
            )

        if a.startswith("RB"):
            return LandmarkInfo(
                landmark_id="L3_RIGHT_MIDDLE_LOWER",
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
                landmark_id="L4_LEFT_MAIN",
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
                recognition_cue="Left lung entry: stable left-sided orientation organizing upper lobe + lingula, segment 6, and lower lobe",
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
            recognition_cue="the carina with both main bronchi in view",
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
        transition = self.transition_context(curr, nxt)
        route = self.route_to_airway(nxt, current_airway=curr)
        family = self.family_for_airway(nxt)
        family_label = self.family_label(family)

        if not nxt:
            return {
                "mode": "done",
                "current_airway": curr,
                "next_airway": "",
                "route": [],
                "airway_family": "",
                "family_label": "",
                "transition_type": "done",
                "anatomy_context": transition,
                "reanchor_target": "",
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
                "route": route,
                "airway_family": family,
                "family_label": family_label,
                "transition_type": "global_reanchor",
                "anatomy_context": transition,
                "reanchor_target": "CARINA",
                "anchor_landmark": "L1_CARINA",
                "recommended_angle": self.neutral_pose(),
                "recognition_cue": "the carina with both main bronchi in view",
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
                "route": route,
                "airway_family": family,
                "family_label": family_label,
                "transition_type": "global_reanchor",
                "anatomy_context": transition,
                "reanchor_target": "CARINA",
                "anchor_landmark": "L1_CARINA",
                "recommended_angle": self.neutral_pose(),
                "recognition_cue": "the carina with both main bronchi in view",
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
            "route": route,
            "airway_family": family,
            "family_label": family_label,
            "transition_type": transition.get("transition_type", "advance"),
            "anatomy_context": transition,
            "reanchor_target": transition.get("shared_ancestor", ""),
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
