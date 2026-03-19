from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from ..shared.curriculum import CurriculumEngine

# Legacy prompt parsing support only. Main path should use parse_state().
_CURRENT_PATTERNS = [
    re.compile(r"current airway\s*[:=]\s*([A-Za-z0-9+\-]+)", re.IGNORECASE),
    re.compile(r"current region\s*[:=]\s*([A-Za-z0-9+\-]+)", re.IGNORECASE),
    re.compile(r'anatomical_position\s*[:=]\s*"?([A-Za-z0-9+\-]+)"?', re.IGNORECASE),
    re.compile(r'currently at\s+"?([A-Za-z0-9+\-]+)"?', re.IGNORECASE),
]

_REQUESTED_NEXT_PATTERNS = [
    re.compile(r"next airway\s*[:=]\s*([A-Za-z0-9+\-]+)", re.IGNORECASE),
    re.compile(r"requested next airway\s*[:=]\s*([A-Za-z0-9+\-]+)", re.IGNORECASE),
    re.compile(r'next lumen to be explored is\s*"?(?:the\s+)?([A-Za-z0-9+\-]+)"?', re.IGNORECASE),
]

_NAV_TARGET_PATTERNS = [
    re.compile(r"target region\s*[:=]\s*([A-Za-z0-9+\-]+)", re.IGNORECASE),
    re.compile(r"target airway\s*[:=]\s*([A-Za-z0-9+\-]+)", re.IGNORECASE),
    re.compile(r"navigation target is ['\"]?([A-Za-z0-9+\-]+)['\"]?", re.IGNORECASE),
]

_REACHED_LIST_PATTERNS = [
    re.compile(r"reached_regions(?:\(last\))?\s*[:=]\s*(\[[^\]]*\])", re.IGNORECASE),
    re.compile(r"regions_seen\s*[:=]\s*(\[[^\]]*\])", re.IGNORECASE),
    re.compile(r"the following areas have been inspected:\s*([^\.]+)", re.IGNORECASE),
]

_MISSING_LIST_PATTERNS = [
    re.compile(r"missing_regions\s*[:=]\s*(\[[^\]]*\])", re.IGNORECASE),
    re.compile(r"waiting_regions\s*[:=]\s*(\[[^\]]*\])", re.IGNORECASE),
    re.compile(r"the following areas are waited to be examed:\s*([^\.]+)", re.IGNORECASE),
]


@dataclass
class RealtimeContext:
    current_airway: str = ""
    requested_next_airway: str = ""
    navigation_target: str = ""
    reached_regions: List[str] | None = None
    missing_regions: List[str] | None = None
    target_not_visible: bool = False
    needs_backtrack: bool = False
    needs_encouragement: bool = False
    wall_contact_risk: bool = False
    raw_current: str = ""
    raw_previous: str = ""
    raw_student_q: str = ""


class RealtimeInstructorEngine:
    """Deterministic Layer-2 engine for short, interruptible bronchoscopy guidance."""

    def __init__(self, curriculum_engine: CurriculumEngine):
        self.curriculum_engine = curriculum_engine
        visit_order = getattr(curriculum_engine, "visit_order", None) or getattr(curriculum_engine, "airway_order", None) or getattr(curriculum_engine, "AIRWAY_VISIT_ORDER", [])
        self._allowed_airways = set(visit_order or [])

    # ---------------- main path ----------------
    def parse_state(self, state: Dict[str, Any]) -> RealtimeContext:
        state = dict(state or {})

        current_airway = self._clean_airway(state.get("current_airway"))
        requested_next_airway = self._clean_airway(
            state.get("requested_next_airway") or state.get("target_airway")
        )
        navigation_target = self._clean_airway(state.get("target_airway"))

        reached_regions = self._normalize_region_list(state.get("reached_regions") or [])
        missing_regions = self._normalize_region_list(state.get("missing_regions") or [])

        target_visible_raw = state.get("is_target_visible", None)
        target_not_visible = False if target_visible_raw is None else (not bool(target_visible_raw))

        return RealtimeContext(
            current_airway=current_airway,
            requested_next_airway=requested_next_airway,
            navigation_target=navigation_target,
            reached_regions=reached_regions,
            missing_regions=missing_regions,
            target_not_visible=target_not_visible,
            needs_backtrack=bool(state.get("backtracking", False)),
            needs_encouragement=bool(state.get("needs_encouragement", False)),
            wall_contact_risk=bool(state.get("wall_contact_risk", False)),
            raw_current=str(state.get("current_situation") or "").strip(),
            raw_previous=str(state.get("previous_msgs") or "").strip(),
            raw_student_q=str(state.get("student_question") or "").strip(),
        )

    # ---------------- legacy fallback ----------------
    def parse_context(self, current: str, previous_msgs: str = "", student_q: str = "") -> RealtimeContext:
        current = current or ""
        previous_msgs = previous_msgs or ""
        student_q = student_q or ""

        nav_target = self._extract_from_patterns(_NAV_TARGET_PATTERNS, current)
        ctx = RealtimeContext(
            current_airway=self._extract_from_patterns(_CURRENT_PATTERNS, current),
            requested_next_airway=self._extract_from_patterns(_REQUESTED_NEXT_PATTERNS, current),
            navigation_target=nav_target,
            reached_regions=self._extract_region_list(_REACHED_LIST_PATTERNS, current),
            missing_regions=self._extract_region_list(_MISSING_LIST_PATTERNS, current),
            target_not_visible=self._target_not_visible(current),
            needs_backtrack=self._needs_backtrack(current, nav_target),
            needs_encouragement=("encouraged" in current.lower() or "more than usual time" in current.lower()),
            wall_contact_risk=("pink blur" in current.lower() or "wall-facing" in current.lower() or "wall contact" in current.lower()),
            raw_current=current.strip(),
            raw_previous=previous_msgs.strip(),
            raw_student_q=student_q.strip(),
        )
        if not ctx.reached_regions:
            ctx.reached_regions = []
        if not ctx.missing_regions:
            ctx.missing_regions = []
        return ctx

    def build_plan(self, ctx: RealtimeContext) -> Dict[str, Any]:
        reached = {str(x).upper() for x in (ctx.reached_regions or []) if str(x).strip()}
        current_airway = (ctx.current_airway or "").upper()
        requested = (ctx.requested_next_airway or "").upper()
        navigation_target = (ctx.navigation_target or "").upper()

        target_airway = (
            navigation_target
            or requested
            or self.curriculum_engine.next_airway(reached)
            or self._first_valid(ctx.missing_regions)
            or current_airway
            or "RB1"
        )

        landmark = self.curriculum_engine.landmark_for_airway(target_airway)
        neutral = self.curriculum_engine.neutral_pose()
        recommended = {
            "access_orientations": getattr(landmark, "recommended_angles", []) or [],
            "neutral_definition": neutral,
        }

        if ctx.needs_backtrack:
            mode = "backtrack"
            anchor = "L1_CARINA"
            cue = "carina bifurcation; symmetric right and left main bronchi"
            recommended = neutral
            micro_steps = [
                "Back out slowly.",
                "Keep the lumen centered.",
                "Stop when the carina is centered.",
            ]
        elif ctx.target_not_visible:
            mode = "locate"
            anchor = getattr(landmark, "landmark_id", "") or (current_airway if current_airway else "L1_CARINA")
            cue = self._normalize_cue(getattr(landmark, "recognition_cue", "") or "the next opening")
            first_angle = "0°"
            angles = getattr(landmark, "recommended_angles", []) or []
            if angles and isinstance(angles[0], dict):
                first_angle = str(angles[0].get("angle", "0°"))
            side = "counter-clockwise" if (target_airway.startswith("LB") or target_airway == "LMB") else "clockwise"
            micro_steps = [
                "Hold neutral at the anchor.",
                f"Rotate {side} toward {first_angle}.",
                f"Look for {cue}.",
            ]
        else:
            mode = "advance"
            anchor = getattr(landmark, "landmark_id", "") or "L1_CARINA"
            cue = self._normalize_cue(getattr(landmark, "recognition_cue", "") or "the next lumen")
            micro_steps = [
                "Keep the lumen centered.",
                f"Follow the landmark cue: {cue}.",
                f"Advance toward {target_airway} in small increments.",
            ]

        return {
            "mode": mode,
            "current_airway": current_airway,
            "next_airway": target_airway,
            "anchor_landmark": anchor,
            "recognition_cue": cue,
            "recommended_angle": recommended,
            "micro_steps": micro_steps,
            "why": "Realtime deterministic plan inside MAS runtime.",
        }

    def build_guidance(self, ctx: RealtimeContext, plan: Dict[str, Any]) -> Dict[str, Any]:
        target = str(plan.get("next_airway", "")).upper()
        mode = str(plan.get("mode", "reorient")).lower()
        cue = self._normalize_cue(str(plan.get("recognition_cue", "")).strip())
        current = str(ctx.current_airway or "").upper()

        if mode == "backtrack":
            lines = ["Back out slowly.", "Center the carina."]
            if ctx.wall_contact_risk:
                lines = ["Ease back off the wall.", "Re-center on the carina."]
        elif mode == "locate":
            if current == "CARINA" and target.startswith("LB"):
                lines = ["Hold at the carina.", "Rotate counter-clockwise to find LMB."]
            elif current == "CARINA" and target.startswith("RB"):
                lines = ["Hold at the carina.", "Rotate clockwise toward the right upper-lobe entry."]
            else:
                lines = ["Hold center.", f"Look for {self._short_cue(cue)}."]
        else:
            if current and target and current == target:
                lines = ["Good. Keep the lumen centered.", f"Inspect {target} carefully."]
            elif current == "LMB" and target.startswith("LB"):
                lines = ["Good. Keep the lumen centered.", f"Advance toward {target} in small steps."]
            elif current == "CARINA" and target.startswith("LB"):
                lines = ["Hold at the carina.", "Rotate counter-clockwise and find LMB."]
            elif current == "CARINA" and target.startswith("RB"):
                lines = ["Hold at the carina.", "Rotate clockwise toward the right upper-lobe entry."]
            else:
                lines = ["Keep the lumen centered.", f"Advance toward {target or 'the next opening'}."]
                if cue:
                    lines.append(f"Look for {self._short_cue(cue)}.")

        if ctx.needs_encouragement and lines:
            lines[0] = self._prefix_encouragement(lines[0])

        lines = self._compress(lines)
        utterance = " ".join(lines).strip()
        if self._is_repetitive(utterance, ctx.raw_previous):
            utterance = lines[-1] if lines else "Hold center."

        return {
            "utterance": utterance,
            "needs_visual_guidance": self._student_requested_visual(ctx.raw_student_q),
        }

    def build_statistics(self, ctx: RealtimeContext, plan: Dict[str, Any]) -> Dict[str, Any]:
        if ctx.needs_backtrack:
            trend = "stable"
            issue = "Needs controlled withdrawal to the carina"
            focus = "back out slowly and re-center the lumen"
        elif ctx.target_not_visible:
            trend = "stable"
            issue = "Target lumen not yet visualized"
            focus = "keep the view centered while searching"
        elif ctx.needs_encouragement:
            trend = "improving"
            issue = "Progress is slow but direction is correct"
            focus = "use smaller, steadier movements"
        else:
            trend = "stable"
            issue = "Maintain orientation while advancing"
            focus = "keep the lumen centered"
        notes = f"mode={plan.get('mode','')}; current={ctx.current_airway or 'unknown'}; next={plan.get('next_airway','')}"
        return {
            "trend": trend,
            "likely_issue": issue,
            "coach_focus_next": focus,
            "notes": notes,
        }

    # ---------------- helpers ----------------
    def _clean_airway(self, value: Any) -> str:
        x = str(value or "").strip().upper()
        if x in {"", "NONE", "NULL", "UNKNOWN"}:
            return ""
        if self._allowed_airways and x not in self._allowed_airways and x not in {"CARINA", "TRACHEA", "RMB", "LMB"}:
            return ""
        return x

    def _normalize_region_list(self, values: Any) -> List[str]:
        if not isinstance(values, (list, tuple, set)):
            return []
        cleaned: List[str] = []
        for item in values:
            x = self._clean_airway(item)
            if x and x not in cleaned:
                cleaned.append(x)
        return cleaned

    def _extract_from_patterns(self, patterns: List[re.Pattern[str]], text: str) -> str:
        for rx in patterns:
            m = rx.search(text or "")
            if m:
                value = str(m.group(1)).strip().upper()
                if value not in {"", "NONE", "NULL", "UNKNOWN"}:
                    return value
        return ""

    def _extract_region_list(self, patterns: List[re.Pattern[str]], text: str) -> List[str]:
        for rx in patterns:
            m = rx.search(text or "")
            if not m:
                continue
            chunk = str(m.group(1)).strip()
            parsed = self._parse_listish(chunk)
            cleaned: List[str] = []
            for item in parsed:
                x = str(item).strip().upper()
                if not x or x in {"NONE", "NULL", "UNKNOWN"}:
                    continue
                if self._allowed_airways and x not in self._allowed_airways:
                    continue
                if x not in cleaned:
                    cleaned.append(x)
            if cleaned:
                return cleaned
        return []

    @staticmethod
    def _parse_listish(chunk: str) -> List[str]:
        chunk = (chunk or "").strip()
        if not chunk:
            return []
        if chunk.startswith("[") and chunk.endswith("]"):
            try:
                obj = ast.literal_eval(chunk)
                if isinstance(obj, list):
                    return [str(x) for x in obj]
            except Exception:
                pass
        return [part.strip() for part in chunk.split(",") if part.strip()]

    @staticmethod
    def _target_not_visible(current: str) -> bool:
        low = (current or "").lower()
        return (
            "target visible: false" in low
            or "is_target_visible: false" in low
            or "target_not_visible: true" in low
            or "not visible" in low
        )

    @staticmethod
    def _needs_backtrack(current: str, nav_target: str) -> bool:
        low = (current or "").lower()
        nav = (nav_target or "").lower()
        return (
            "backtracking: true" in low
            or "needs_backtrack: true" in low
            or "back out" in low
            or nav in {"back", "backtrack", "carina"}
        )

    def _first_valid(self, values: List[str] | None) -> str:
        for v in values or []:
            x = str(v).strip().upper()
            if not x:
                continue
            if self._allowed_airways and x not in self._allowed_airways:
                continue
            return x
        return ""

    @staticmethod
    def _normalize_cue(cue: str) -> str:
        cue = (cue or "").strip().rstrip(".")
        if not cue:
            return "the next lumen"
        cue = cue.replace("/", " or ")
        cue = re.sub(r"\s+", " ", cue).strip(" ,;:")
        return cue or "the next lumen"


    @classmethod
    def _short_cue(cls, cue: str) -> str:
        cue = cls._normalize_cue(cue)
        if not cue:
            return "the next lumen"
        return cue.rstrip(".")

    @staticmethod
    def _prefix_encouragement(line: str) -> str:
        if line.lower().startswith(("good", "nice", "great")):
            return line
        return f"Good. {line}"

    @staticmethod
    def _student_requested_visual(student_q: str) -> bool:
        q = (student_q or "").lower()
        return any(k in q for k in ["diagram", "picture", "image", "show me", "visual"])

    @staticmethod
    def _compress(lines: List[str], max_lines: int = 2) -> List[str]:
        cleaned: List[str] = []
        bad_tail_words = {
            "and", "or", "to", "the", "a", "an", "with",
            "toward", "towards", "into", "at", "of", "for",
        }

        for ln in lines:
            ln = re.sub(r"\s+", " ", str(ln or "").strip())
            if not ln:
                continue
            ln = ln.replace(" and then ", ". ").replace(" then ", ". ")
            words = ln.split()
            if len(words) > 12:
                ln = " ".join(words[:12]).rstrip(",;:")
                while ln.split() and ln.split()[-1].lower() in bad_tail_words:
                    ln = " ".join(ln.split()[:-1]).rstrip(",;:")
            if ln and ln[-1] not in ".!?":
                ln += "."
            cleaned.append(ln)

        return cleaned[:max_lines] or ["Hold center."]

    @staticmethod
    def _is_repetitive(current_utt: str, previous_msgs: str) -> bool:
        current_utt = (current_utt or "").strip().lower()
        previous_msgs = (previous_msgs or "").strip().lower()
        if not current_utt or not previous_msgs:
            return False
        return current_utt in previous_msgs[-300:]
