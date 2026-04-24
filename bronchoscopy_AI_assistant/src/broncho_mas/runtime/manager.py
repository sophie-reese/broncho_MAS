from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ..shared.event_signal import EventSignalEngine
from ..shared.state_normalization import (
    DEFAULT_ANCHOR_AIRWAYS,
    build_current_situation_from_state,
    extract_block,
    extract_m_joints_vel_rel,
    extract_current_airway,
    extract_reached_regions,
    extract_target_hint,
    normalize_legacy_prompt,
    normalize_runtime_payload,
)
from ..shared.utterance_postprocess import (
    clinicalize_ui_text,
    compress_utterance,
    normalize_landmark_language,
    sanitize_micro_step,
    sentence_split,
    short_ui_text,
    trim_line_words,
)
try:
    from ..shared.logging_utils import RunLogger
except Exception:  # pragma: no cover
    from logging_utils import RunLogger  # type: ignore

try:
    from ..shared.directional_hint_builder import DirectionalHintBuilder, attach_directional_hint
except Exception:  # pragma: no cover
    from directional_hint_builder import DirectionalHintBuilder, attach_directional_hint  # type: ignore

ANCHOR_AIRWAYS: Set[str] = set(DEFAULT_ANCHOR_AIRWAYS)


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
    PIPELINE_NAME = "runtime"
    MANAGER_NAME = "RuntimeManager"
    STATEPACKET_SCHEMA = "statepacket.runtime.v2"

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
            pipeline=self.PIPELINE_NAME,
        )
        self.logger.write_meta({
            "manager": self.MANAGER_NAME,
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
                f"[{self.PIPELINE_NAME}] provider={os.environ.get('BRONCHO_PROVIDER', '')} "
                f"model={os.environ.get('BRONCHO_MODEL', '')}"
            )
            return model
        except Exception as exc:
            print(f"[{self.PIPELINE_NAME}] model init failed: {exc}")
            return _SimpleLLM()

    def _extract_block(self, text: str, tag: str) -> str:
        return extract_block(text, tag)

    def _parse_prompt_or_payload(self, payload: Any) -> Tuple[Dict[str, Any], str]:
        if isinstance(payload, dict):
            state = self._normalize_runtime_payload(payload)
            return state, "payload"

        prompt_text = str(payload or "")
        state = self._normalize_legacy_prompt(prompt_text)
        return state, "legacy_prompt"

    def _normalize_runtime_payload(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return normalize_runtime_payload(raw, self.AIRWAY_VISIT_ORDER)

    def _normalize_legacy_prompt(self, prompt_text: str) -> Dict[str, Any]:
        return normalize_legacy_prompt(prompt_text, self.AIRWAY_VISIT_ORDER, anchors=ANCHOR_AIRWAYS)

    def _build_current_situation_from_state(self, state: Dict[str, Any]) -> str:
        return build_current_situation_from_state(state)

    def _compose_prompt(self, current_situation: str, previous_msgs: str, student_question: str) -> str:
        prompt = f"CURRENT_SITUATION: {current_situation}".strip()
        if previous_msgs:
            prompt += f"\n\nPREVIOUS_MSGS: {previous_msgs}"
        if student_question:
            prompt += f"\n\nSTUDENT_QUESTION: {student_question}"
        return prompt

    def _extract_reached_regions(self, current_situation: str) -> List[str]:
        return extract_reached_regions(current_situation, allowed=self.AIRWAY_VISIT_ORDER)

    def _extract_current_airway(self, current_situation: str) -> str:
        return extract_current_airway(current_situation, allowed_airways=self.AIRWAY_VISIT_ORDER, anchors=ANCHOR_AIRWAYS)

    def _extract_target_hint(self, current_situation: str) -> str:
        return extract_target_hint(current_situation, allowed_airways=self.AIRWAY_VISIT_ORDER)

    def _extract_m_jointsVelRel(self, current_situation: str, raw_payload: Dict[str, Any]) -> Optional[List[float]]:
        return extract_m_joints_vel_rel(current_situation, raw_payload)

    def _needs_visual_guidance(self, student_question: str) -> bool:
        q = str(student_question or "").lower()
        return any(k in q for k in ["picture", "diagram", "image", "show me", "draw", "where is", "point", "visual"])

    def _sentence_split(self, text: str) -> List[str]:
        return sentence_split(text)

    def _normalize_landmark_language(self, text: str) -> str:
        return normalize_landmark_language(text)

    def _sanitize_micro_step(self, text: str) -> str:
        return sanitize_micro_step(text)

    def _clinicalize_ui_text(self, text: str) -> str:
        return clinicalize_ui_text(text)

    def _trim_line_words(self, text: str, max_words: int = 18) -> str:
        return trim_line_words(text, max_words=max_words)

    def _compress_utterance(self, text: str) -> str:
        return compress_utterance(text)

    def _short_ui_text(self, text: str) -> str:
        return short_ui_text(text)

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
            "schema": self.STATEPACKET_SCHEMA,
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
                "manager": self.MANAGER_NAME,
                "pipeline": self.PIPELINE_NAME,
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

        m_jointsVelRel = state.get("m_jointsVelRel") or [0.0, 0.0, 0.0]
        try:
            auth_plan_json = attach_directional_hint(
                auth_plan_json,
                m_jointsVelRel=m_jointsVelRel,
                event_flag=event_packet.get("event_flag") if isinstance(event_packet, dict) else raw_payload.get("event_flag"),
                builder=self.directional_builder,
            )
        except Exception as exc:
            print(f"[{self.PIPELINE_NAME}] directional hint failed: {exc}")

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
                "m_jointsVelRel": m_jointsVelRel,
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
                "m_jointsVelRel": m_jointsVelRel,
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
