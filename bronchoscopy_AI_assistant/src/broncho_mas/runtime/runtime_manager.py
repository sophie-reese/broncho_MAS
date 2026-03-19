from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ..shared.event_signal import EventSignalEngine
try:
    from ..shared.logging_utils import RunLogger
except Exception:  # pragma: no cover
    from logging_utils import RunLogger  # type: ignore

try:
    from ..shared.directional_hint_builder import DirectionalHintBuilder, attach_directional_hint
except Exception:  # pragma: no cover
    from directional_hint_builder import DirectionalHintBuilder, attach_directional_hint  # type: ignore

ANCHOR_AIRWAYS: Set[str] = {"CARINA", "LMB", "RMB", "TRACHEA"}


def _safe_coverage_ratio(reached_count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(reached_count / total, 4)


class _SimpleLLM:
    is_fallback_backend = True

    def generate(self, prompt: str) -> str:
        return ""

    def __call__(self, messages: Any) -> str:
        return ""


class RuntimeManager:
    AIRWAY_VISIT_ORDER: List[str] = [
        "RB1", "RB2", "RB3", "RB4", "RB5", "RB6", "RB7", "RB8", "RB9", "RB10",
        "LB1+2", "LB3", "LB4", "LB5", "LB6", "LB8", "LB9", "LB10",
    ]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-27B",
        llm: Optional[Any] = None,
        logger: Optional[RunLogger] = None,
    ):
        self.model_name = model_name
        self._last_ui_text = ""
        self._iteration = 0

        from ..shared.curriculum import CurriculumEngine
        from .realtime_engine import RealtimeInstructorEngine

        self.curriculum = CurriculumEngine(self.AIRWAY_VISIT_ORDER)
        self.rt_engine = RealtimeInstructorEngine(self.curriculum)
        self.model = llm or self._build_model()
        self.directional_builder = DirectionalHintBuilder()

        self.event_engine = EventSignalEngine()
        self.logger = logger or RunLogger(
            log_root=os.environ.get("BRONCHO_LOG_ROOT") or None,
            session_id=os.environ.get("BRONCHO_SESSION_ID") or None,
            pipeline="runtime",
        )
        self.logger.write_meta({
            "manager": "RuntimeManager",
            "provider": os.environ.get("BRONCHO_PROVIDER", ""),
            "model": os.environ.get("BRONCHO_MODEL", self.model_name),
            "fallback_backend": bool(getattr(self.model, "is_fallback_backend", False)),
        })

    def _build_model(self) -> Any:
        try:
            os.environ.setdefault("BRONCHO_MODEL", self.model_name)
            os.environ.setdefault("BRONCHO_PROVIDER", os.environ.get("BRONCHO_PROVIDER", "hf"))
            from ..shared.model_selector import create_model  # type: ignore
            model = create_model(None)
            print(
                f"[runtime] provider={os.environ.get('BRONCHO_PROVIDER', '')} "
                f"model={os.environ.get('BRONCHO_MODEL', '')}"
            )
            return model
        except Exception as exc:
            print(f"[runtime] model init failed: {exc}")
            return _SimpleLLM()

    def _extract_block(self, text: str, tag: str) -> str:
        if not text:
            return ""
        m = re.search(rf"{re.escape(tag)}\s*:(.*?)(?=\n\s*[A-Z_]+\s*:|\Z)", str(text), flags=re.S)
        return (m.group(1) if m else "").strip()

    def _parse_prompt_or_payload(self, payload: Any) -> Tuple[Dict[str, Any], str]:
        if isinstance(payload, dict):
            state = self._normalize_runtime_payload(payload)
            return state, "payload"

        prompt_text = str(payload or "")
        state = self._normalize_legacy_prompt(prompt_text)
        return state, "legacy_prompt"

    def _normalize_runtime_payload(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        raw_payload = dict(raw or {})
        raw_payload.setdefault("airway_visit_order", list(self.AIRWAY_VISIT_ORDER))

        reached_regions = raw_payload.get("reached_regions") or raw_payload.get("regions_seen") or []
        if not isinstance(reached_regions, list):
            try:
                reached_regions = list(reached_regions)
            except Exception:
                reached_regions = []
        reached_regions = [str(x).strip().upper() for x in reached_regions if str(x).strip()]

        control_triplet = raw_payload.get("m_jointsVelRel")
        if not isinstance(control_triplet, (list, tuple)) or len(control_triplet) < 3:
            control_triplet = raw_payload.get("joints_vel_rel")
        if not isinstance(control_triplet, (list, tuple)) or len(control_triplet) < 3:
            control_triplet = raw_payload.get("control_triplet")
        if not isinstance(control_triplet, (list, tuple)) or len(control_triplet) < 3:
            control_triplet = [0.0, 0.0, 0.0]
        try:
            control_triplet = [float(control_triplet[0]), float(control_triplet[1]), float(control_triplet[2])]
        except Exception:
            control_triplet = [0.0, 0.0, 0.0]

        current_airway = (
            raw_payload.get("anatomical_position")
            or raw_payload.get("current_airway")
            or raw_payload.get("current_region")
            or ""
        )
        target_airway = (
            raw_payload.get("current_target")
            or raw_payload.get("target_airway")
            or raw_payload.get("target_region")
            or raw_payload.get("next_destination")
            or raw_payload.get("requested_next_airway")
            or ""
        )

        current_airway = str(current_airway).strip().upper()
        target_airway = str(target_airway).strip().upper()

        return {
            "raw_payload": raw_payload,
            "prompt_text": str(raw_payload.get("llm_prompt_text") or "").strip(),
            "previous_msgs": str(
                raw_payload.get("previous_msgs")
                or raw_payload.get("history")
                or raw_payload.get("llm_history")
                or ""
            ).strip(),
            "student_question": str(
                raw_payload.get("student_question")
                or raw_payload.get("student_q")
                or raw_payload.get("question")
                or ""
            ).strip(),
            "soft_prompt": str(raw_payload.get("soft_prompt") or "").strip(),
            "need_llm": bool(raw_payload.get("need_llm", False)),
            "llm_reason": str(raw_payload.get("llm_reason") or "").strip(),
            "phase": str(raw_payload.get("phase") or "").strip(),
            "current_airway": current_airway,
            "target_airway": target_airway,
            "requested_next_airway": str(raw_payload.get("requested_next_airway") or "").strip().upper(),
            "reached_regions": reached_regions,
            "just_reached": bool(raw_payload.get("just_reached", False)),
            "backtracking": bool(raw_payload.get("backtracking", False)),
            "drift_detected": bool(raw_payload.get("drift_detected", False)),
            "is_centered": bool(raw_payload.get("is_centered", False)),
            "is_stable": bool(raw_payload.get("is_stable", False)),
            "is_target_visible": bool(raw_payload.get("is_target_visible", False)),
            "wall_contact_risk": bool(raw_payload.get("wall_contact_risk", False)),
            "need_recenter": bool(raw_payload.get("need_recenter", False)),
            "control_triplet": control_triplet,
            "robot_joints": raw_payload.get("robot_joints") or [0, 0, 0],
            "bounding_boxes": raw_payload.get("bounding_boxes") or [],
            "mode": str(raw_payload.get("mode") or "").strip(),
            "visualization_mode": str(raw_payload.get("visualization_mode") or "").strip(),
        }

    def _normalize_legacy_prompt(self, prompt_text: str) -> Dict[str, Any]:
        current_situation = self._extract_block(prompt_text, "CURRENT_SITUATION")
        previous_msgs = self._extract_block(prompt_text, "PREVIOUS_MSGS")
        student_question = self._extract_block(prompt_text, "STUDENT_QUESTION")

        reached_list = self._extract_reached_regions(current_situation)
        current_airway = self._extract_current_airway(current_situation)
        target_airway = self._extract_target_hint(current_situation)
        control_triplet = self._extract_control_triplet(current_situation, {}) or [0.0, 0.0, 0.0]

        return {
            "raw_payload": {},
            "prompt_text": prompt_text,
            "previous_msgs": previous_msgs,
            "student_question": student_question,
            "soft_prompt": "",
            "need_llm": False,
            "llm_reason": "",
            "phase": "",
            "current_airway": current_airway,
            "target_airway": target_airway,
            "requested_next_airway": "",
            "reached_regions": reached_list,
            "just_reached": False,
            "backtracking": False,
            "drift_detected": False,
            "is_centered": False,
            "is_stable": False,
            "is_target_visible": False,
            "wall_contact_risk": False,
            "need_recenter": False,
            "control_triplet": control_triplet,
            "robot_joints": [0, 0, 0],
            "bounding_boxes": [],
            "mode": "",
            "visualization_mode": "",
        }

    def _build_current_situation_from_state(self, state: Dict[str, Any]) -> str:
        parts: List[str] = []
        mapping = [
            ("phase", "Phase"),
            ("current_airway", "Current airway"),
            ("target_airway", "Target airway"),
            ("requested_next_airway", "Requested next airway"),
            ("soft_prompt", "Soft prompt"),
            ("llm_reason", "LLM reason"),
        ]
        for key, label in mapping:
            val = state.get(key)
            if val not in (None, ""):
                parts.append(f"{label}: {val}")

        bool_mapping = [
            ("need_llm", "Need LLM"),
            ("backtracking", "Backtracking"),
            ("drift_detected", "Drift detected"),
            ("is_centered", "Is centered"),
            ("is_stable", "Is stable"),
            ("is_target_visible", "Target visible"),
            ("wall_contact_risk", "Wall contact risk"),
            ("need_recenter", "Need recenter"),
        ]
        for key, label in bool_mapping:
            parts.append(f"{label}: {str(bool(state.get(key, False))).lower()}")

        parts.append(f"reached_regions: {json.dumps(state.get('reached_regions', []))}")
        parts.append(f"control_triplet: {json.dumps(state.get('control_triplet', [0.0, 0.0, 0.0]))}")
        return "\n".join(parts).strip()

    def _compose_prompt(self, current_situation: str, previous_msgs: str, student_question: str) -> str:
        prompt = f"CURRENT_SITUATION: {current_situation}".strip()
        if previous_msgs:
            prompt += f"\n\nPREVIOUS_MSGS: {previous_msgs}"
        if student_question:
            prompt += f"\n\nSTUDENT_QUESTION: {student_question}"
        return prompt

    def _extract_reached_regions(self, current_situation: str) -> List[str]:
        text = str(current_situation or "")
        allowed = set(self.AIRWAY_VISIT_ORDER)
        out: List[str] = []
        patterns = [
            r"reached_regions\(last\)\s*[:=]\s*(\[[^\]]*\])",
            r"REACHED_REGIONS\s*[:=]\s*(\[[^\]]*\])",
            r"reached_regions\s*[:=]\s*(\[[^\]]*\])",
            r"timeline_reached\s*[:=]\s*(\[[^\]]*\])",
            r"regions_seen\s*[:=]\s*(\[[^\]]*\])",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.I)
            if not m:
                continue
            chunk = m.group(1)
            for parser in (json.loads, ast.literal_eval):
                try:
                    val = parser(chunk)
                    if isinstance(val, list):
                        for item in val:
                            x = str(item).strip().upper()
                            if x in allowed and x not in out:
                                out.append(x)
                        return out
                except Exception:
                    pass
        return out

    def _extract_current_airway(self, current_situation: str) -> str:
        text = str(current_situation or "")
        allowed = set(self.AIRWAY_VISIT_ORDER) | ANCHOR_AIRWAYS
        patterns = [
            r"anatomical_position\s*[:=]\s*\"?([A-Za-z0-9+\-]+)\"?",
            r"current airway\s*[:=]\s*([A-Za-z0-9+\-]+)",
            r"current region\s*[:=]\s*([A-Za-z0-9+\-]+)",
            r'currently at\s+\"?([A-Za-z0-9+\-]+)\"?',
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.I)
            if not m:
                continue
            x = m.group(1).strip().upper()
            if x in {"", "NONE", "NULL", "UNKNOWN"}:
                return ""
            if x in allowed:
                return x
        return ""

    def _extract_target_hint(self, current_situation: str) -> str:
        text = str(current_situation or "")
        allowed = set(self.AIRWAY_VISIT_ORDER)
        patterns = [
            r"target region\s*[:=]\s*([A-Za-z0-9+\-]+)",
            r"target airway\s*[:=]\s*([A-Za-z0-9+\-]+)",
            r"navigation target is ['\"]?([A-Za-z0-9+\-]+)['\"]?",
            r"next lumen to be explored\s*(?:is)?\s*[:=]?\s*['\"]?([A-Za-z0-9+\-]+)['\"]?",
            r"requested next airway\s*[:=]\s*([A-Za-z0-9+\-]+)",
            r"current_target\s*[:=]\s*([A-Za-z0-9+\-]+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.I)
            if not m:
                continue
            x = m.group(1).strip().upper()
            if x in allowed:
                return x
        return ""

    def _extract_control_triplet(self, current_situation: str, raw_payload: Dict[str, Any]) -> Optional[List[float]]:
        candidates = [
            raw_payload.get("m_jointsVelRel"),
            raw_payload.get("joints_vel_rel"),
            raw_payload.get("control_triplet"),
        ]
        for val in candidates:
            if isinstance(val, (list, tuple)) and len(val) >= 3:
                try:
                    return [float(val[0]), float(val[1]), float(val[2])]
                except Exception:
                    pass

        text = str(current_situation or "")
        patterns = [
            r"m_jointsVelRel\s*[:=]\s*(\[[^\]]+\])",
            r"joints_vel_rel\s*[:=]\s*(\[[^\]]+\])",
            r"control_triplet\s*[:=]\s*(\[[^\]]+\])",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.I)
            if not m:
                continue
            chunk = m.group(1)
            for parser in (json.loads, ast.literal_eval):
                try:
                    val = parser(chunk)
                    if isinstance(val, (list, tuple)) and len(val) >= 3:
                        return [float(val[0]), float(val[1]), float(val[2])]
                except Exception:
                    pass
        return None

    def _needs_visual_guidance(self, student_question: str) -> bool:
        q = str(student_question or "").lower()
        return any(k in q for k in ["picture", "diagram", "image", "show me", "draw", "where is", "point", "visual"])

    def _sentence_split(self, text: str) -> List[str]:
        raw = re.split(r"(?<=[.!?])\s+|\n+", str(text or "").strip())
        out: List[str] = []
        for s in raw:
            x = re.sub(r"\s+", " ", s).strip()
            if x:
                out.append(x)
        return out

    def _normalize_landmark_language(self, text: str) -> str:
        x = str(text or "")
        x = re.sub(r"\bL\d+_[A-Z0-9+]+\b", "", x)
        x = re.sub(r"\bto\s*90°\s*right\b", "to the right", x, flags=re.I)
        x = re.sub(r"\bto\s*90°\s*left\b", "to the left", x, flags=re.I)
        x = re.sub(r"\b90°\s*right\b", "right", x, flags=re.I)
        x = re.sub(r"\b90°\s*left\b", "left", x, flags=re.I)
        x = re.sub(r"\b\d+\s*degrees?\b", "", x, flags=re.I)
        x = re.sub(r"\btrifurcation\b", "Mercedes sign", x, flags=re.I)
        x = re.sub(r"\banchor landmark\b", "view", x, flags=re.I)
        x = re.sub(r"\bHold neutral at\b", "Hold steady at", x, flags=re.I)
        x = re.sub(r"\bRotate clockwise to the right\b", "Turn right", x, flags=re.I)
        x = re.sub(r"\bRotate counter-clockwise to the left\b", "Turn left", x, flags=re.I)
        x = re.sub(r"\bRotate clockwise\b", "Turn right", x, flags=re.I)
        x = re.sub(r"\bRotate counter-clockwise\b", "Turn left", x, flags=re.I)
        x = re.sub(r"\bReacquire\b", "Find", x, flags=re.I)
        x = re.sub(r"\bLocate\b", "Find", x, flags=re.I)
        x = re.sub(r"\bidentify\b", "find", x, flags=re.I)
        x = re.sub(r"\s+", " ", x).strip(" ,;")
        return x

    def _sanitize_micro_step(self, text: str) -> str:
        x = str(text or "").strip()
        if not x:
            return ""
        x = self._normalize_landmark_language(x)
        replacements = [
            ("Hold neutral", "Hold steady"),
            ("Advance toward", "Advance into"),
            ("in small increments", "slowly"),
            ("while keeping the lumen centered", ""),
            ("recognition cue", "landmark"),
        ]
        for a, b in replacements:
            x = x.replace(a, b)
        x = re.sub(r"\bexact\b", "", x, flags=re.I)
        x = re.sub(r"\bprecisely\b", "", x, flags=re.I)
        x = re.sub(r"\s+", " ", x).strip(" ,;")
        return x

    def _clinicalize_ui_text(self, text: str) -> str:
        x = self._normalize_landmark_language(text)
        banned_patterns = [
            r"\bL\d+_[A-Z0-9+]+\b",
            r"\b90°\b",
            r"\banchor landmark\b",
        ]
        for pat in banned_patterns:
            x = re.sub(pat, "", x, flags=re.I)
        x = re.sub(r"\s+", " ", x).strip(" ,;")
        return x

    def _trim_line_words(self, text: str, max_words: int = 18) -> str:
        x = re.sub(r"\s+", " ", str(text or "").strip())
        if not x:
            return ""
        words = x.split()
        if len(words) <= max_words:
            return x.rstrip(" ,;:")
        cut = " ".join(words[:max_words]).rstrip(" ,;:")
        if cut and cut[-1] not in ".!?":
            cut += "."
        return cut

    def _compress_utterance(self, text: str) -> str:
        x = self._normalize_landmark_language(text)
        lines = self._sentence_split(x)
        if not lines:
            x = re.sub(r"\s+", " ", x).strip()
            return self._trim_line_words(x, max_words=18) if x else ""
        clean: List[str] = []
        for line in lines:
            y = re.sub(r"\s+", " ", line).strip(" ,;")
            if not y:
                continue
            clean.append(self._trim_line_words(y, max_words=18))
            if len(clean) == 2:
                break
        return " ".join(clean).strip()

    def _short_ui_text(self, text: str) -> str:
        x = re.sub(r"\s+", " ", str(text or "").strip())
        if not x:
            return ""
        parts = self._sentence_split(x)
        if not parts:
            return x
        candidate = " ".join(parts[:2]).strip()
        words = candidate.split()
        if len(words) > 28:
            candidate = " ".join(words[:28]).rstrip(",;: ")
            while candidate.endswith(("and", "or", "to", "the", "a", "an", "with", "toward", "towards", "into")):
                candidate = " ".join(candidate.split()[:-1]).rstrip(",;: ")
            if candidate and candidate[-1] not in ".!?":
                candidate += "."
        return candidate

    def _extract_text_response(self, resp: Any) -> str:
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp.strip()
        if isinstance(resp, dict):
            for key in ("utterance", "text", "content", "response", "output"):
                val = resp.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        content = getattr(resp, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if txt:
                        parts.append(str(txt))
                else:
                    txt = getattr(item, "text", None)
                    if txt:
                        parts.append(str(txt))
            return "\n".join(parts).strip()
        text = getattr(resp, "text", None)
        if isinstance(text, str):
            return text.strip()
        return str(resp).strip()

    def _lookup_landmark(self, airway: str) -> Dict[str, Any]:
        info = self.curriculum.landmark_for_airway(airway or "RB1")
        return {
            "landmark_id": getattr(info, "landmark_id", ""),
            "recommended_angles": getattr(info, "recommended_angles", []),
            "recognition_cue": getattr(info, "recognition_cue", ""),
        }

    def _build_authoritative_plan(
        self,
        state: Dict[str, Any],
        current_situation: str,
        previous_msgs: str,
        student_question: str,
        current_airway: str,
        reached_set: Set[str],
        target_hint: str,
        landmark_hint: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            ctx = self.rt_engine.parse_state({
                "current_airway": current_airway,
                "requested_next_airway": state.get("requested_next_airway", "") or target_hint,
                "target_airway": target_hint,
                "reached_regions": list(reached_set),
                "missing_regions": [],
                "is_target_visible": bool(state.get("is_target_visible", False)),
                "backtracking": bool(state.get("backtracking", False)),
                "wall_contact_risk": bool(state.get("wall_contact_risk", False)),
                "needs_encouragement": False,
                "previous_msgs": previous_msgs,
                "student_question": student_question,
                "current_situation": current_situation,
            })
            plan = self.rt_engine.build_plan(ctx)
            if isinstance(plan, dict):
                if target_hint and not plan.get("next_airway"):
                    plan["next_airway"] = target_hint
                if current_airway and not plan.get("current_airway"):
                    plan["current_airway"] = current_airway
                if landmark_hint.get("recognition_cue") and not plan.get("recognition_cue"):
                    plan["recognition_cue"] = landmark_hint["recognition_cue"]
                return plan
        except Exception:
            pass

        plan = self.curriculum.make_plan(
            current_airway=str(current_airway or "").upper(),
            reached={str(x).upper() for x in reached_set},
            student_question=student_question,
            requested_next_airway=target_hint,
            is_back=bool(state.get("backtracking", False)),
            stagnating=(not bool(state.get("is_target_visible", False))),
        )
        if not isinstance(plan, dict):
            plan = {}
        if target_hint:
            plan["next_airway"] = target_hint
        if current_airway and not plan.get("current_airway"):
            plan["current_airway"] = current_airway
        if landmark_hint.get("recognition_cue") and not plan.get("recognition_cue"):
            plan["recognition_cue"] = landmark_hint["recognition_cue"]
        return plan

    def _build_deterministic_guidance(
        self,
        state: Dict[str, Any],
        current_situation: str,
        previous_msgs: str,
        student_question: str,
        auth_plan_json: Dict[str, Any],
    ) -> str:
        directional = (auth_plan_json or {}).get("directional_hint", {})
        primary_action = str(directional.get("primary_action") or "").strip()
        secondary_action = str(directional.get("secondary_action") or "").strip()
        if primary_action:
            openers = ["Okay.", "Good.", "Stay with it.", "That's better."]
            opener = openers[self._iteration % len(openers)]
            spoken: List[str] = [opener, self.directional_builder._coachify(primary_action)]
            if secondary_action:
                spoken.append(self.directional_builder._coachify(secondary_action))
            return self._compress_utterance(" ".join(spoken[:3]).strip())

        try:
            ctx = self.rt_engine.parse_state({
                "current_airway": state.get("current_airway", ""),
                "requested_next_airway": state.get("requested_next_airway", "") or auth_plan_json.get("next_airway", ""),
                "target_airway": auth_plan_json.get("next_airway", ""),
                "reached_regions": state.get("reached_regions", []),
                "missing_regions": [],
                "is_target_visible": bool(state.get("is_target_visible", False)),
                "backtracking": bool(state.get("backtracking", False)),
                "wall_contact_risk": bool(state.get("wall_contact_risk", False)),
                "needs_encouragement": False,
                "previous_msgs": previous_msgs,
                "student_question": student_question,
                "current_situation": current_situation,
            })
            guidance = self.rt_engine.build_guidance(ctx, auth_plan_json)
            if isinstance(guidance, dict):
                return self._compress_utterance(guidance.get("utterance", ""))
        except Exception:
            pass

        steps = (auth_plan_json or {}).get("micro_steps") or []
        if isinstance(steps, list) and steps:
            compact = []
            for step in steps[:2]:
                t = self._sanitize_micro_step(step).rstrip(".")
                if t:
                    compact.append(t)
            if compact:
                return self._compress_utterance(". ".join(compact) + ".")

        target = str((auth_plan_json or {}).get("next_airway", "")).strip()
        cue = self._normalize_landmark_language(str((auth_plan_json or {}).get("recognition_cue", "")).strip())
        if target and cue:
            return self._compress_utterance(f"Advance to {target}. Look for {cue}.")
        if target:
            return self._compress_utterance(f"Advance to {target} slowly.")
        return "Keep the lumen centered. Move slowly."

    def _llm_verbalize(
        self,
        current_situation: str,
        previous_msgs: str,
        student_question: str,
        auth_plan_json: Dict[str, Any],
    ) -> str:
        try:
            from . import build_runtime_instructor_prompt  # type: ignore
            prompt = build_runtime_instructor_prompt(
                plan=auth_plan_json,
                current=current_situation,
                previous_msgs=previous_msgs,
                student_q=student_question,
            )
        except Exception:
            prompt = (
                "ROLE: Real-time bronchoscopy instructor\n"
                "TASK: Rewrite the deterministic plan into 1 or 2 short spoken teaching sentences.\n"
                "RULES:\n"
                "- Keep the clinical intent of PLAN_JSON.\n"
                "- Do not copy internal landmark codes.\n"
                "- Do not use exact engineering angles.\n"
                "- Prefer clinical airway language over pattern nicknames.\n"
                "- Keep it under 16 words per sentence.\n"
                "- Sound like a calm human instructor speaking live.\n"
                "- Include at most one action and one airway cue.\n\n"
                f"PLAN_JSON:\n{json.dumps(auth_plan_json, ensure_ascii=True)}\n\n"
                f"CURRENT_SITUATION:\n{current_situation}\n\n"
                f"PREVIOUS_MSGS:\n{previous_msgs}\n\n"
                f"STUDENT_QUESTION:\n{student_question}\n"
            )

        directional = (auth_plan_json or {}).get("directional_hint", {})
        if directional:
            prompt += "\n\nDIRECTIONAL_HINT:\n" + json.dumps(directional, ensure_ascii=True)

        raw = ""
        try:
            if hasattr(self.model, "generate"):
                raw = self._extract_text_response(self.model.generate(prompt))
        except Exception:
            raw = ""

        if not raw and callable(self.model):
            candidate_messages = [
                [{"role": "user", "content": prompt}],
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                prompt,
            ]
            for messages in candidate_messages:
                try:
                    raw = self._extract_text_response(self.model(messages))
                    if raw:
                        break
                except Exception:
                    continue

        return self._compress_utterance(raw)

    def _guidance_is_safe(self, llm_text: str, plan: Dict[str, Any], deterministic_text: str) -> bool:
        llm_norm = str(llm_text or "").upper()
        target = str((plan or {}).get("next_airway", "")).upper()
        if not llm_text.strip():
            return False
        if len(self._sentence_split(llm_text)) > 2:
            return False
        if len(llm_text.split()) > 32:
            return False
        airway_mentions = set(re.findall(r"\b(?:RB\d+|LB\d(?:\+\d)?|LMB|RMB|CARINA|TRACHEA)\b", llm_norm))
        if target and any(x not in {target, "LMB", "RMB", "CARINA", "TRACHEA"} for x in airway_mentions):
            return False
        return True

    def _compact_statistics(self, current_situation: str, current_airway: str, next_airway: str, auth_plan: Dict[str, Any]) -> Dict[str, Any]:
        s = str(current_situation or "").lower()
        mode = str((auth_plan or {}).get("mode", "")).lower()
        cue = self._normalize_landmark_language(str((auth_plan or {}).get("recognition_cue", "")).strip())

        if mode == "backtrack":
            return {
                "trend": "stable",
                "likely_issue": "lost orientation or unsafe advance",
                "coach_focus_next": "withdraw to the carina and re-center",
                "teaching_point": "Reset orientation before advancing again.",
                "notes": f"current={current_airway or 'unknown'}; target={next_airway or 'unknown'}; mode=backtrack.",
            }
        if mode == "locate" or "not visible" in s:
            return {
                "trend": "stable",
                "likely_issue": "target lumen not yet visualized",
                "coach_focus_next": "hold center and identify the next opening",
                "teaching_point": (f"Use the landmark cue: {cue}." if cue else "Do not advance until the lumen is identified."),
                "notes": f"current={current_airway or 'unknown'}; target={next_airway or 'unknown'}; mode=locate.",
            }
        return {
            "trend": "stable",
            "likely_issue": "normal navigation",
            "coach_focus_next": "advance with the lumen centered",
            "teaching_point": (f"Confirm {cue}." if cue else "Advance only with a centered view."),
            "notes": f"current={current_airway or 'unknown'}; target={next_airway or 'unknown'}; mode={mode or 'advance'}.",
        }

    def _build_statepacket(
        self,
        prompt: str,
        current_situation: str,
        previous_msgs: str,
        student_question: str,
        reached_list: List[str],
        curriculum_progress: Dict[str, Any],
        landmark_hint: Dict[str, Any],
        auth_plan_json: Dict[str, Any],
        raw_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "schema": "statepacket.runtime.v2",
            "prompt": prompt,
            "current_situation": current_situation,
            "previous_msgs": previous_msgs,
            "student_question": student_question,
            "reached_regions": [str(x).upper() for x in reached_list],
            "curriculum_progress": curriculum_progress,
            "landmark_hint": landmark_hint,
            "auth_plan": auth_plan_json,
            "raw_payload": raw_payload,
            "meta": {
                "manager": "RuntimeManager",
                "provider": os.environ.get("BRONCHO_PROVIDER", ""),
                "model": os.environ.get("BRONCHO_MODEL", self.model_name),
                "fallback_backend": bool(getattr(self.model, "is_fallback_backend", False)),
            },
        }

    def _run_from_state(self, state: Dict[str, Any], source_mode: str) -> Dict[str, Any]:
        prompt = state.get("prompt_text", "").strip()
        previous_msgs = state.get("previous_msgs", "")
        student_question = state.get("student_question", "")
        current_situation = self._build_current_situation_from_state(state)

        reached_list = [str(x).upper() for x in state.get("reached_regions", [])]
        reached_set: Set[str] = set(reached_list)

        current_airway = str(state.get("current_airway", "")).strip().upper()
        target_hint = str(state.get("target_airway", "")).strip().upper()

        deterministic_next = self.curriculum.next_airway(reached_set)
        next_airway = target_hint or deterministic_next or ""
        landmark_hint = self._lookup_landmark(next_airway or current_airway or "RB1")

        auth_plan_json = self._build_authoritative_plan(
            state=state,
            current_situation=current_situation,
            previous_msgs=previous_msgs,
            student_question=student_question,
            current_airway=current_airway,
            reached_set=reached_set,
            target_hint=next_airway,
            landmark_hint=landmark_hint,
        )

        auth_next = str(auth_plan_json.get("next_airway", "")).strip().upper() or next_airway
        if auth_next:
            next_airway = auth_next

        curriculum_progress = {
            "reached": reached_list,
            "next_airway": next_airway,
            "coverage_ratio": _safe_coverage_ratio(len(reached_list), len(self.AIRWAY_VISIT_ORDER)),
            "reached_count": len(reached_list),
            "total": len(self.AIRWAY_VISIT_ORDER),
        }

        raw_payload = dict(state.get("raw_payload") or {})
        event_frame = dict(raw_payload)
        event_frame.setdefault("anatomical_position", current_airway)
        event_frame.setdefault("current_target", target_hint or next_airway)
        event_frame.setdefault("next_destination", next_airway)
        event_frame.setdefault("reached_regions", reached_list)
        event_frame.setdefault("just_reached", bool(state.get("just_reached", False)))

        event_packet = self.event_engine.step(event_frame, history=None, plan=auth_plan_json)

        control_triplet = state.get("control_triplet") or [0.0, 0.0, 0.0]
        try:
            auth_plan_json = attach_directional_hint(
                auth_plan_json,
                m_jointsVelRel=control_triplet,
                event_flag=event_packet.get("event_flag") if isinstance(event_packet, dict) else raw_payload.get("event_flag"),
                builder=self.directional_builder,
            )
        except Exception as exc:
            print(f"[runtime] directional hint failed: {exc}")

        deterministic_ui = self._build_deterministic_guidance(
            state=state,
            current_situation=current_situation,
            previous_msgs=previous_msgs,
            student_question=student_question,
            auth_plan_json=auth_plan_json,
        )

        llm_ui = ""
        if not getattr(self.model, "is_fallback_backend", False):
            llm_ui = self._llm_verbalize(
                current_situation=current_situation,
                previous_msgs=previous_msgs,
                student_question=student_question,
                auth_plan_json=auth_plan_json,
            )

        ui_text = deterministic_ui
        if self._guidance_is_safe(llm_ui, auth_plan_json, deterministic_ui):
            ui_text = llm_ui
        if not ui_text:
            ui_text = "Keep the lumen centered. Advance slowly."

        utterance_full = self._clinicalize_ui_text(ui_text)
        ui_text = self._short_ui_text(utterance_full)
        self._last_ui_text = utterance_full

        statistics = self._compact_statistics(current_situation, current_airway, next_airway, auth_plan_json)

        statepacket = self._build_statepacket(
            prompt=prompt or self._compose_prompt(current_situation, previous_msgs, student_question),
            current_situation=current_situation,
            previous_msgs=previous_msgs,
            student_question=student_question,
            reached_list=reached_list,
            curriculum_progress=curriculum_progress,
            landmark_hint=landmark_hint,
            auth_plan_json=auth_plan_json,
            raw_payload=raw_payload,
        )

        result = {
            "ui_text": ui_text,
            "instructor": ui_text,
            "utterance_full": utterance_full,
            "deterministic_utterance_full": deterministic_ui,
            "llm_utterance_full": llm_ui,
            "needs_visual_guidance": self._needs_visual_guidance(student_question),
            "curriculum_progress": curriculum_progress,
            "landmark_hint": landmark_hint,
            "plan_json": auth_plan_json,
            "statistics": statistics,
            "statepacket": statepacket,
            "event_packet": event_packet,
            "raw": {
                "source_mode": source_mode,
                "prompt": prompt,
                "current_situation": current_situation,
                "used_fallback_backend": bool(getattr(self.model, "is_fallback_backend", False)),
                "deterministic_ui": deterministic_ui,
                "llm_ui": llm_ui,
                "control_triplet": control_triplet,
                "directional_hint": auth_plan_json.get("directional_hint"),
                "normalized_state": state,
            },
        }

        self._iteration += 1
        self.logger.log_step(
            step_index=self._iteration,
            state=event_frame,
            control=event_packet if isinstance(event_packet, dict) else {},
            guidance={
                "ui_text": ui_text,
                "utterance_full": utterance_full,
                "deterministic_utterance_full": deterministic_ui,
                "llm_utterance_full": llm_ui,
                "needs_visual_guidance": self._needs_visual_guidance(student_question),
            },
            plan=auth_plan_json,
            statistics=statistics,
            extra={
                "source_mode": source_mode,
                "prompt": prompt,
                "current_situation": current_situation,
                "current_airway": current_airway,
                "target_airway": next_airway,
                "reached_regions": reached_list,
                "curriculum_progress": curriculum_progress,
                "landmark_hint": landmark_hint,
                "statepacket": statepacket,
                "llm_called": bool(llm_ui),
                "used_fallback_backend": bool(getattr(self.model, "is_fallback_backend", False)),
                "control_triplet": control_triplet,
                "directional_hint": auth_plan_json.get("directional_hint"),
                "normalized_state": state,
            },
        )
        return result

    def run(self, payload: Any) -> Dict[str, Any]:
        state, source_mode = self._parse_prompt_or_payload(payload)
        return self._run_from_state(state, source_mode)

    def step(self, payload: Any) -> Dict[str, Any]:
        return self.run(payload)

    def __call__(self, payload: Any) -> Dict[str, Any]:
        return self.run(payload)
