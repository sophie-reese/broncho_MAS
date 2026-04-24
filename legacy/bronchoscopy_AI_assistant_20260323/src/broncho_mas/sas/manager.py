from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Set, Tuple

from ..shared.curriculum import CurriculumEngine
from ..shared.directional_hint_builder import DirectionalHintBuilder
from ..shared.event_signal import EventSignalEngine
from ..shared.logging_utils import RunLogger
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
    sentence_split,
    short_ui_text,
    relaxed_ui_text,
)
from .skills import (
    curriculum_skill,
    directional_skill,
    guidance_skill,
    question_router_skill,
    qa_skill,
    statistics_skill,
)


def build_sas_instructor_prompt(
    *,
    current: str,
    previous_msgs: str,
    student_q: str,
    plan: Dict[str, Any],
    state: Dict[str, Any],
) -> str:
    plan_json = json.dumps(plan, ensure_ascii=True)
    current_airway = plan.get("current_airway", "Unknown")
    next_airway = plan.get("next_airway", "Unknown")
    cue = plan.get("recognition_cue", "")
    directional = plan.get("directional_hint", {})

    state_snapshot = {
        "phase": state.get("phase", ""),
        "current_airway": state.get("current_airway", ""),
        "target_airway": state.get("target_airway", ""),
        "requested_next_airway": state.get("requested_next_airway", ""),
        "reached_regions": state.get("reached_regions", []),
        "need_llm": bool(state.get("need_llm", False)),
        "llm_reason": state.get("llm_reason", ""),
        "soft_prompt": state.get("soft_prompt", ""),
        "is_centered": bool(state.get("is_centered", False)),
        "is_stable": bool(state.get("is_stable", False)),
        "is_target_visible": bool(state.get("is_target_visible", False)),
        "drift_detected": bool(state.get("drift_detected", False)),
        "wall_contact_risk": bool(state.get("wall_contact_risk", False)),
        "need_recenter": bool(state.get("need_recenter", False)),
        "m_jointsVelRel": state.get("m_jointsVelRel", [0.0, 0.0, 0.0]),
    }

    return (
        "ROLE: Single-agent bronchoscopy coach with structured skills\n"
        "CONTEXT: You receive structured runtime state from an upstream online caller.\n"
        "TASK: Convert the authoritative plan into calm, bedside coaching for the next move.\n\n"
        "SKILLS:\n"
        "1. curriculum_skill: follow the deterministic curriculum plan.\n"
        "2. directional_skill: prioritize grounded control-direction cues when present.\n"
        "3. safety_skill: favor re-centering and conservative movement when state suggests uncertainty.\n\n"
        "GROUNDING RULES:\n"
        "1. Follow PLAN_JSON exactly for the intended airway and safety intent.\n"
        "2. Use the structured state as the source of truth when it conflicts with free text.\n"
        "3. If DIRECTIONAL_HINT is present, use it as the main motion cue.\n"
        "4. Do not invent anatomy, targets, or unsafe extra steps.\n"
        "5. Keep the final answer to one or two short spoken sentences.\n"
        "6. Do not use internal landmark IDs or engineering-angle phrasing.\n"
        "7. Prefer supportive, operational language suitable for live TTS.\n\n"
        "OUTPUT RULES:\n"
        "- Return plain text only.\n"
        "- Maximum 16 words per sentence.\n"
        "- Mention at most one airway target and one landmark cue.\n\n"
        f"PLAN_JSON:\n{plan_json}\n\n"
        f"STRUCTURED_STATE:\n{json.dumps(state_snapshot, ensure_ascii=True)}\n\n"
        f"CURRENT_SITUATION:\n{current.strip()}\n\n"
        f"PREVIOUS_MSGS:\n{previous_msgs.strip()}\n\n"
        f"STUDENT_QUESTION:\n{student_q.strip()}\n\n"
        f"TARGET: {current_airway} -> {next_airway}\n"
        f"CUE: {cue}\n"
        f"DIRECTIONAL_HINT: {json.dumps(directional, ensure_ascii=True)}\n\n"
        "ACTION:\n"
        "Return the spoken guidance now."
    )


class _SimpleLLM:
    is_fallback_backend = True

    def generate(self, prompt: str) -> str:
        return ""

    def __call__(self, messages: Any) -> str:
        return ""


class SASManager:
    PIPELINE_NAME = "sas"
    MANAGER_NAME = "SASManager"
    STATEPACKET_SCHEMA = "statepacket.sas.v1"
    AIRWAY_VISIT_ORDER: List[str] = [
        "RB1", "RB2", "RB3", "RB4", "RB5", "RB6", "RB7", "RB8", "RB9", "RB10",
        "LB1+2", "LB3", "LB4", "LB5", "LB6", "LB8", "LB9", "LB10",
    ]

    def __init__(self, model_name: str = "Qwen/Qwen3.5-27B", llm: Any = None, logger: Any = None):
        self.model_name = model_name
        self._last_ui_text = ""
        self._iteration = 0
        self.curriculum = CurriculumEngine(self.AIRWAY_VISIT_ORDER)
        self.model = llm or self._build_model()
        self.directional_builder = DirectionalHintBuilder()
        self.event_engine = EventSignalEngine()
        self.logger = logger or RunLogger(
            log_root=os.environ.get("BRONCHO_LOG_ROOT") or None,
            session_id=os.environ.get("BRONCHO_SESSION_ID") or None,
            pipeline=self.PIPELINE_NAME,
        )
        self._sas_state_context: Dict[str, Any] = {}
        self.logger.write_meta(
            {
                "schema": "broncho.meta.v2",
                "manager": self.MANAGER_NAME,
                "pipeline": self.PIPELINE_NAME,
                "design": "single_agent_skills",
                "skills": [
                    "curriculum_skill",
                    "directional_skill",
                    "guidance_skill",
                    "question_router_skill",
                    "qa_skill",
                    "statistics_skill",
                ],
                "provider": os.environ.get("BRONCHO_PROVIDER", ""),
                "model": os.environ.get("BRONCHO_MODEL", self.model_name),
                "fallback_backend": bool(getattr(self.model, "is_fallback_backend", False)),
            }
        )

    def _build_model(self) -> Any:
        try:
            os.environ.setdefault("BRONCHO_MODEL", self.model_name)
            os.environ.setdefault("BRONCHO_PROVIDER", os.environ.get("BRONCHO_PROVIDER", "hf"))
            from ..shared.model_selector import create_model

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
            return self._normalize_runtime_payload(payload), "payload"
        return self._normalize_legacy_prompt(str(payload or "")), "legacy_prompt"

    def _normalize_runtime_payload(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return normalize_runtime_payload(raw, self.AIRWAY_VISIT_ORDER)

    def _normalize_legacy_prompt(self, prompt_text: str) -> Dict[str, Any]:
        return normalize_legacy_prompt(prompt_text, self.AIRWAY_VISIT_ORDER, anchors=DEFAULT_ANCHOR_AIRWAYS)

    def _extract_reached_regions(self, current_situation: str) -> List[str]:
        return extract_reached_regions(current_situation, allowed=self.AIRWAY_VISIT_ORDER)

    def _extract_current_airway(self, current_situation: str) -> str:
        return extract_current_airway(current_situation, allowed_airways=self.AIRWAY_VISIT_ORDER, anchors=DEFAULT_ANCHOR_AIRWAYS)

    def _extract_target_hint(self, current_situation: str) -> str:
        return extract_target_hint(current_situation, allowed_airways=self.AIRWAY_VISIT_ORDER)

    def _extract_m_jointsVelRel(self, current_situation: str) -> List[float]:
        return extract_m_joints_vel_rel(current_situation)

    def _build_current_situation_from_state(self, state: Dict[str, Any]) -> str:
        return build_current_situation_from_state(state)

    def _compose_prompt(self, current_situation: str, previous_msgs: str, student_question: str) -> str:
        prompt = f"CURRENT_SITUATION: {current_situation}".strip()
        if previous_msgs:
            prompt += f"\n\nPREVIOUS_MSGS: {previous_msgs}"
        if student_question:
            prompt += f"\n\nSTUDENT_QUESTION: {student_question}"
        return prompt

    def _extract_text_response(self, resp: Any) -> str:
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp.strip()
        if isinstance(resp, dict):
            for key in ("text", "content"):
                value = resp.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, list):
                    chunks: List[str] = []
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            chunks.append(item.strip())
                        elif isinstance(item, dict):
                            t = item.get("text") or item.get("content")
                            if isinstance(t, str) and t.strip():
                                chunks.append(t.strip())
                    if chunks:
                        return " ".join(chunks).strip()
        for attr in ("content", "text"):
            value = getattr(resp, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                chunks: List[str] = []
                for item in value:
                    if isinstance(item, str) and item.strip():
                        chunks.append(item.strip())
                    elif isinstance(item, dict):
                        t = item.get("text") or item.get("content")
                        if isinstance(t, str) and t.strip():
                            chunks.append(t.strip())
                    else:
                        t = getattr(item, "text", None) or getattr(item, "content", None)
                        if isinstance(t, str) and t.strip():
                            chunks.append(t.strip())
                if chunks:
                    return " ".join(chunks).strip()
        message = getattr(resp, "message", None)
        if message is not None:
            msg_content = getattr(message, "content", None)
            if isinstance(msg_content, str) and msg_content.strip():
                return msg_content.strip()
        return ""

    def _sentence_split(self, text: str) -> List[str]:
        return sentence_split(text)

    def _normalize_landmark_language(self, text: str) -> str:
        return normalize_landmark_language(text)

    def _compress_utterance(self, text: str) -> str:
        return compress_utterance(text)

    def _clinicalize_ui_text(self, text: str) -> str:
        return clinicalize_ui_text(text)

    def _short_ui_text(self, text: str) -> str:
        return short_ui_text(text)

    def _looks_like_structured_object(self, text: str) -> bool:
        t = str(text or "").strip()
        bad_markers = [
            "ChatMessage(",
            "ChatCompletion(",
            "tool_calls=",
            "raw=",
            "role='assistant'",
            'role="assistant"',
        ]
        return any(marker in t for marker in bad_markers)

    def _llm_verbalize(
        self,
        current_situation: str,
        previous_msgs: str,
        student_question: str,
        auth_plan_json: Dict[str, Any],
    ) -> str:
        prompt = build_sas_instructor_prompt(
            plan=auth_plan_json,
            current=current_situation,
            previous_msgs=previous_msgs,
            student_q=student_question,
            state=self._sas_state_context,
        )

        self._last_llm_debug_repr = ""
        raw = ""
        try:
            if hasattr(self.model, "generate"):
                resp = self.model.generate(prompt)
                self._last_llm_debug_repr = repr(resp)
                raw = self._extract_text_response(resp)
        except Exception:
            raw = ""

        if not raw and callable(self.model):
            for messages in (
                [{"role": "user", "content": prompt}],
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                prompt,
            ):
                try:
                    resp = self.model(messages)
                    self._last_llm_debug_repr = repr(resp)
                    raw = self._extract_text_response(resp)
                    if raw:
                        break
                except Exception:
                    continue

        return compress_utterance(raw)

    def _guidance_is_safe(self, llm_text: str, plan: Dict[str, Any]) -> bool:
        llm_norm = str(llm_text or "").upper()
        target = str((plan or {}).get("next_airway", "")).upper()
        if not llm_text.strip():
            return False
        if self._looks_like_structured_object(llm_text):
            return False
        if len(self._sentence_split(llm_text)) > 2:
            return False
        if len(llm_text.split()) > 40:
            return False
        airway_mentions = set(re.findall(r"\b(?:RB\d+|LB\d(?:\+\d)?|LMB|RMB|CARINA|TRACHEA)\b", llm_norm))
        if target and any(item not in {target, "LMB", "RMB", "CARINA", "TRACHEA"} for item in airway_mentions):
            return False
        return True

    def _needs_visual_guidance(self, student_question: str) -> bool:
        q = str(student_question or "").lower()
        return any(token in q for token in ["picture", "diagram", "image", "show me", "draw", "where is", "point", "visual"])

    def _qa_allowed(self, *, state: Dict[str, Any], question_mode: str) -> bool:
        if question_mode == "none":
            return False
        if bool(state.get("wall_contact_risk", False)) or bool(state.get("need_recenter", False)) or bool(state.get("drift_detected", False)):
            return False
        return True

    def _lookup_landmark(self, airway: str) -> Dict[str, Any]:
        info = self.curriculum.landmark_for_airway(airway or "RB1")
        return {
            "landmark_id": getattr(info, "landmark_id", ""),
            "recommended_angles": getattr(info, "recommended_angles", []),
            "recognition_cue": getattr(info, "recognition_cue", ""),
        }

    def _build_statepacket(
        self,
        *,
        prompt: str,
        current_situation: str,
        previous_msgs: str,
        student_question: str,
        reached_list: List[str],
        curriculum_progress: Dict[str, Any],
        landmark_hint: Dict[str, Any],
        auth_plan_json: Dict[str, Any],
        raw_payload: Dict[str, Any],
        skills_used: List[str],
        question_mode: str,
        qa_utterance_full: str,
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
                "skills_used": skills_used,
                "question_mode": question_mode,
                "qa_utterance_full": qa_utterance_full,
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
        next_airway = target_hint or self.curriculum.next_airway(reached_set) or ""
        landmark_hint = self._lookup_landmark(next_airway or current_airway or "RB1")

        auth_plan_json = curriculum_skill(
            curriculum=self.curriculum,
            state=state,
            reached_set=reached_set,
            current_airway=current_airway,
            target_hint=next_airway,
        )
        next_airway = str(auth_plan_json.get("next_airway", "")).strip().upper() or next_airway

        curriculum_progress = {
            "reached": reached_list,
            "next_airway": next_airway,
            "coverage_ratio": round(len(reached_list) / max(len(self.AIRWAY_VISIT_ORDER), 1), 4),
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
        state["event_packet"] = event_packet if isinstance(event_packet, dict) else {}
        directional_hint = directional_skill(
            builder=self.directional_builder,
            m_jointsVelRel=state.get("m_jointsVelRel") or [0.0, 0.0, 0.0],
            event_packet=event_packet,
        )
        auth_plan_json["directional_hint"] = directional_hint

        question_mode = question_router_skill(state=state, plan=auth_plan_json)
        auth_plan_json["question_mode"] = question_mode

        skills_used = ["curriculum_skill", "directional_skill", "guidance_skill", "statistics_skill", "question_router_skill"]

        deterministic_ui = guidance_skill(
            state=state,
            plan=auth_plan_json,
            directional_hint=directional_hint,
        )

        llm_ui = ""
        qa_ui = ""
        if not getattr(self.model, "is_fallback_backend", False):
            llm_ui = self._llm_verbalize(
                current_situation=current_situation,
                previous_msgs=previous_msgs,
                student_question=student_question,
                auth_plan_json=auth_plan_json,
            )
            if self._qa_allowed(state=state, question_mode=question_mode):
                qa_ui = qa_skill(
                    model=self.model,
                    state=state,
                    plan=auth_plan_json,
                    current_situation=current_situation,
                    previous_msgs=previous_msgs,
                    question_mode=question_mode,
                    fallback_guidance=deterministic_ui or llm_ui,
                )
                if qa_ui:
                    skills_used.append("qa_skill")

        has_question = bool(str(student_question or "").strip()) and question_mode != "none"
        if has_question:
            if self._qa_allowed(state=state, question_mode=question_mode):
                if qa_ui and self._guidance_is_safe(qa_ui, auth_plan_json):
                    chosen_ui = qa_ui
                else:
                    chosen_ui = deterministic_ui or (llm_ui if self._guidance_is_safe(llm_ui, auth_plan_json) else "")
            else:
                chosen_ui = deterministic_ui or (llm_ui if self._guidance_is_safe(llm_ui, auth_plan_json) else "")
        else:
            chosen_ui = llm_ui if self._guidance_is_safe(llm_ui, auth_plan_json) else deterministic_ui

        if not chosen_ui:
            chosen_ui = "Keep the lumen centered. Advance slowly."

        utterance_full = clinicalize_ui_text(chosen_ui)
        if bool(state.get("wall_contact_risk", False)) or bool(state.get("need_recenter", False)) or bool(state.get("drift_detected", False)):
            ui_text = short_ui_text(utterance_full, max_words=20)
        else:
            ui_text = relaxed_ui_text(utterance_full, max_words=42)
        self._last_ui_text = utterance_full

        statistics = statistics_skill(
            current_situation=current_situation,
            current_airway=current_airway,
            next_airway=next_airway,
            plan=auth_plan_json,
        )

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
            skills_used=skills_used,
            question_mode=question_mode,
            qa_utterance_full=qa_ui,
        )

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
                "qa_utterance_full": qa_ui,
                "needs_visual_guidance": self._needs_visual_guidance(student_question),
                "question_mode": question_mode,
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
                "llm_debug_repr": getattr(self, "_last_llm_debug_repr", ""),
                "used_fallback_backend": bool(getattr(self.model, "is_fallback_backend", False)),
                "m_jointsVelRel": state.get("m_jointsVelRel") or [0.0, 0.0, 0.0],
                "directional_hint": directional_hint,
                "normalized_state": state,
                "skills_used": skills_used,
                "question_mode": question_mode,
                "qa_utterance_full": qa_ui,
            },
        )

        return {
            "ui_text": ui_text,
            "instructor": ui_text,
            "utterance_full": utterance_full,
            "deterministic_utterance_full": deterministic_ui,
            "llm_utterance_full": llm_ui,
            "qa_utterance_full": qa_ui,
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
                "qa_ui": qa_ui,
                "llm_debug_repr": getattr(self, "_last_llm_debug_repr", ""),
                "m_jointsVelRel": state.get("m_jointsVelRel") or [0.0, 0.0, 0.0],
                "directional_hint": directional_hint,
                "normalized_state": state,
                "skills_used": skills_used,
                "question_mode": question_mode,
                "qa_utterance_full": qa_ui,
            },
        }

    def run(self, payload: Any) -> Dict[str, Any]:
        state, source_mode = self._parse_prompt_or_payload(payload)
        self._sas_state_context = dict(state or {})
        return self._run_from_state(state, source_mode)

    def step(self, payload: Any) -> Dict[str, Any]:
        return self.run(payload)

    def __call__(self, payload: Any) -> Dict[str, Any]:
        return self.run(payload)


SingleAgentManager = SASManager
