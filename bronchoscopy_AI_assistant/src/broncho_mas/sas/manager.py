from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..shared.curriculum import CurriculumEngine, DEFAULT_AIRWAY_VISIT_ORDER
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
    light_cleanup_ui_text,
)
from .skill_registry import SKILL_REGISTRY
from .skill_policy import SkillPolicyCompiler, SkillPolicyLoader
from .skills import (
    SkillResult,
    build_skill_record,
    mark_landmark_as_taught,
    merge_teaching_with_guidance,
    realize_frame_response,
    safety_risk,
)


@dataclass
class SessionMilestones:
    opening_sent: bool = False
    right_lung_complete_announced: bool = False
    session_complete_announced: bool = False
    auto_report_written: bool = False
    auto_report_in_progress: bool = False


@dataclass
class TurnCounters:
    llm_calls: int = 0
    iteration: int = 0


@dataclass
class TurnState:
    session_context: Dict[str, Any] = field(default_factory=dict)
    last_ui_text: str = ""
    last_llm_debug_repr: str = ""


@dataclass
class TurnLogEntry:
    turn_index: int
    selected_skill: str
    action_intent: str
    target_airway: str
    support_mode: str
    utterance_summary: str


class TurnLog:
    def __init__(self, maxlen: int = 4) -> None:
        self._entries: deque[TurnLogEntry] = deque(maxlen=maxlen)

    def record(self, entry: TurnLogEntry) -> None:
        self._entries.append(entry)

    def to_prompt_context(self) -> str:
        entries = list(self._entries)[-2:]
        if not entries:
            return ""
        lines = [
            (
                f"turn={entry.turn_index}; "
                f"skill={entry.selected_skill}; "
                f"intent={entry.action_intent or 'none'}; "
                f"target={entry.target_airway or 'unknown'}; "
                f"support={entry.support_mode or 'none'}; "
                f"summary={entry.utterance_summary or 'none'}"
            )
            for entry in entries
        ]
        return " | ".join(lines)


@dataclass
class SkillSelection:
    winner: SkillResult
    guidance: SkillResult
    support: SkillResult
    teaching: SkillResult
    qa: SkillResult
    all_records: List[Dict[str, Any]]


class SkillDispatcher:
    def __init__(self, execute_skill) -> None:
        self._execute_skill = execute_skill

    @staticmethod
    def _inactive_skill_result(skill: str, reason: str) -> SkillResult:
        return SkillResult(
            skill=skill,
            active=False,
            priority=0.0,
            reason=reason,
            data={},
            utterance="",
        )

    @staticmethod
    def _pick_winner(candidates: List[SkillResult]) -> SkillResult:
        active = [candidate for candidate in candidates if getattr(candidate, "active", False)]
        if not active:
            raise ValueError("No active skill candidates available")
        order = {"landmark_teaching_skill": 0, "qa_skill": 1, "guidance_skill": 2, "support_skill": 3}
        return sorted(
            active,
            key=lambda candidate: (-float(getattr(candidate, "priority", 0.0)), order.get(getattr(candidate, "skill", ""), 99)),
        )[0]

    def dispatch(
        self,
        *,
        state: Dict[str, Any],
        plan: Dict[str, Any],
        visit_order: tuple[str, ...],
        model: Any,
        current_situation: str,
        previous_msgs: str,
        previous_msgs_raw: str,
        policy_text: str,
        run_teaching: bool,
        run_qa: bool,
        run_support: bool,
    ) -> SkillSelection:
        guidance = self._execute_skill(
            "guidance_skill",
            state=state,
            plan=plan,
            directional_hint=dict(plan.get("directional_hint") or {}),
            model=model,
            current_situation=current_situation,
            previous_msgs=previous_msgs,
            policy_text=policy_text,
        )
        deterministic_guidance = str(guidance.data.get("deterministic_utterance") or "")

        support = self._execute_skill(
            "support_skill",
            state=state,
            plan=plan,
            previous_msgs=previous_msgs_raw,
        ) if run_support else self._inactive_skill_result("support_skill", "support suppressed before dispatch")

        teaching = self._execute_skill(
            "landmark_teaching_skill",
            state=state,
            plan=plan,
            model=model,
            current_situation=current_situation,
            previous_msgs=previous_msgs,
            policy_text=policy_text,
        ) if run_teaching else self._inactive_skill_result("landmark_teaching_skill", "teaching suppressed before dispatch")

        qa = self._execute_skill(
            "qa_skill",
            state=state,
            plan=plan,
            fallback_guidance=deterministic_guidance,
            qa_allowed=run_qa,
            visit_order=visit_order,
            model=model,
            current_situation=current_situation,
            previous_msgs=previous_msgs,
            policy_text=policy_text,
        ) if run_qa else self._inactive_skill_result("qa_skill", "qa suppressed before dispatch")

        all_records = [
            guidance.to_dict(),
            support.to_dict(),
            teaching.to_dict(),
            qa.to_dict(),
        ]
        winner = self._pick_winner([teaching, qa, guidance, support])
        return SkillSelection(
            winner=winner,
            guidance=guidance,
            support=support,
            teaching=teaching,
            qa=qa,
            all_records=all_records,
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
        "ROLE: Real-time bronchoscopy instructor.\n"
        "CONTEXT: You are coaching a trainee live from structured runtime state.\n"
        "TASK: Turn the authoritative plan into natural spoken coaching for the very next move.\n\n"
        "GROUNDING RULES:\n"
        "1. Follow PLAN_JSON exactly for action, target airway, and safety intent.\n"
        "2. Use STRUCTURED_STATE as the source of truth if free text is vague.\n"
        "3. If DIRECTIONAL_HINT is present, treat it as the main movement cue.\n"
        "4. Do not invent anatomy, targets, landmarks, or extra procedural steps.\n"
        "5. Do not add knob, bend, rotate, pull-back, or advance actions unless supported by the plan or directional hint.\n"
        "6. If the plan only supports one grounded action, keep it to that one action.\n"
        "7. Keep safety first: when uncertain, prefer centering, holding, or conservative movement.\n\n"
        "SPOKEN STYLE:\n"
        "- Sound like a friendly and inspiring instructor speaking beside the trainee in real time.\n"
        "- Use warm, natural, operational language.\n"
        "- A brief supportive opener is allowed when appropriate.\n"
        "- Prefer direct spoken verbs such as turn, pull back, hold, keep, bend, advance, find, or steady.\n"
        "- Do not sound like a report, checklist, or textbook.\n"
        "- Do not use internal landmark IDs or engineering-angle phrasing.\n"
        "- Treat PREVIOUS_STEP_SUMMARY only as short context; do not echo its wording.\n"
        "- Focus on what is new in this step rather than repeating the last phrasing.\n\n"
        "OUTPUT RULES:\n"
        "- Return plain text only.\n"
        "- Use one or two spoken sentences.\n"
        "- Keep the utterance easy to say aloud.\n"
        "- Prefer one primary action and at most one visual or positional cue.\n"
        "- Do not chain a long sequence of new actions.\n\n"
        f"PLAN_JSON:\n{plan_json}\n\n"
        f"STRUCTURED_STATE:\n{json.dumps(state_snapshot, ensure_ascii=True)}\n\n"
        f"CURRENT_SITUATION:\n{current.strip()}\n\n"
        f"PREVIOUS_STEP_SUMMARY:\n{previous_msgs.strip()}\n\n"
        f"STUDENT_QUESTION:\n{student_q.strip()}\n\n"
        f"TARGET: {current_airway} -> {next_airway}\n"
        f"CUE: {cue}\n"
        f"DIRECTIONAL_HINT: {json.dumps(directional, ensure_ascii=True)}\n\n"
        "ACTION:\n"
        "Return the spoken guidance now."
    )


def build_sas_instructor_prompt_with_policy(
    *,
    current: str,
    previous_msgs: str,
    student_q: str,
    plan: Dict[str, Any],
    state: Dict[str, Any],
    skill_policy_text: str,
) -> str:
    base_prompt = build_sas_instructor_prompt(
        plan=plan,
        current=current,
        previous_msgs=previous_msgs,
        student_q=student_q,
        state=state,
    )
    return (
        "SYSTEM_POLICY_SOURCE: These policy notes were compiled from skill MD files.\n"
        "Apply them as execution constraints for this call.\n\n"
        f"{skill_policy_text.strip()}\n\n"
        f"{base_prompt}"
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
    AIRWAY_VISIT_ORDER: List[str] = list(DEFAULT_AIRWAY_VISIT_ORDER)

    def __init__(self, model_name: str = "Qwen/Qwen3.5-27B", llm: Any = None, logger: Any = None, policy_loader: SkillPolicyLoader | None = None):
        self.model_name = model_name
        self.visit_order = tuple(self.AIRWAY_VISIT_ORDER)
        self.milestones = SessionMilestones()
        self.turn_counters = TurnCounters()
        self.turn_state = TurnState()
        self.turn_log = TurnLog()
        self.curriculum = CurriculumEngine(self.visit_order)
        self.model = llm or self._build_model()
        self.directional_builder = DirectionalHintBuilder()
        self.event_engine = EventSignalEngine()
        self.dispatcher = SkillDispatcher(self._execute_skill)
        self.logger = logger or RunLogger(
            log_root=os.environ.get("BRONCHO_LOG_ROOT") or None,
            session_id=os.environ.get("BRONCHO_SESSION_ID") or None,
            pipeline=self.PIPELINE_NAME,
        )
        self.skills = SKILL_REGISTRY
        self.policy_loader = policy_loader or SkillPolicyLoader(base_dir=Path(__file__).resolve().parent / "policies")
        self.policy_compiler = SkillPolicyCompiler(loader=self.policy_loader)
        self._runtime_policy_cache = ""
        self._auto_report_reason = ""
        self._auto_report_paths: Dict[str, str] = {}
        self._last_session_report: Dict[str, Any] = {}
        self.logger.write_meta(
            {
                "schema": "broncho.meta.v2",
                "manager": self.MANAGER_NAME,
                "pipeline": self.PIPELINE_NAME,
                "design": "single_agent_skills",
                "skills": list(self.skills.keys()),
                "provider": os.environ.get("BRONCHO_PROVIDER", ""),
                "model": os.environ.get("BRONCHO_MODEL", self.model_name),
                "fallback_backend": bool(getattr(self.model, "is_fallback_backend", False)),
            }
        )

    def __enter__(self) -> "SASManager":
        return self

    def __exit__(self, *_: Any) -> None:
        self._finalize_session_report_at_exit()

    def close(self) -> None:
        self._finalize_session_report_at_exit()

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

    def _parse_prompt_or_payload(self, payload: Any) -> Tuple[Dict[str, Any], str]:
        if isinstance(payload, dict):
            return normalize_runtime_payload(payload, self.AIRWAY_VISIT_ORDER), "payload"
        return normalize_legacy_prompt(str(payload or ""), self.AIRWAY_VISIT_ORDER, anchors=DEFAULT_ANCHOR_AIRWAYS), "legacy_prompt"

    def _build_current_situation_from_state(self, state: Dict[str, Any]) -> str:
        return build_current_situation_from_state(state)

    def _inherit_session_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(state or {})
        previous = dict(self.turn_state.session_context or {})
        carry_keys = (
            "landmark_teaching_history",
            "last_landmark_taught",
            "last_landmark_taught_turn",
            "last_acknowledged_waypoint",
            "last_acknowledged_waypoint_turn",
            "last_acknowledged_destination",
            "last_acknowledged_destination_turn",
            "session_turn_index",
        )
        for key in carry_keys:
            if key not in merged or merged.get(key) in (None, "", [], {}):
                if previous.get(key) not in (None, "", [], {}):
                    merged[key] = previous.get(key)
        return merged

    def _compose_prompt(self, current_situation: str, previous_msgs: str, student_question: str) -> str:
        prompt = f"CURRENT_SITUATION: {current_situation}".strip()
        if previous_msgs:
            prompt += f"\n\nPREVIOUS_STEP_SUMMARY: {previous_msgs}"
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

    def _get_runtime_skill_policy(self) -> str:
        if not self._runtime_policy_cache:
            self._runtime_policy_cache = self.policy_compiler.compile_runtime_guidance_policy()
        return self._runtime_policy_cache

    def _canonical_next_airway(self, *, reached_set: Set[str], target_hint: str) -> str:
        hint = str(target_hint or "").strip().upper()
        canonical = str(self.curriculum.next_airway(reached_set) or "").strip().upper()
        if not hint:
            return canonical
        if hint in reached_set and canonical:
            return canonical
        try:
            order = list(self.AIRWAY_VISIT_ORDER)
            if canonical and hint in order and canonical in order and order.index(hint) < order.index(canonical):
                return canonical
        except Exception:
            pass
        return hint or canonical

    def _should_force_global_backtrack(
        self,
        *,
        state: Dict[str, Any],
        current_airway: str,
        next_airway: str,
        raw_current_target: str,
    ) -> bool:
        upstream_back = str(raw_current_target or "").strip().upper() == "BACK" or bool(state.get("backtracking", False))
        if not upstream_back:
            return False
        current = str(current_airway or "").strip().upper()
        target = str(next_airway or "").strip().upper()
        if not current or not target:
            return True
        if not self.curriculum.is_known_airway(current):
            return True
        transition = self.curriculum.transition_context(current, target)
        transition_type = str(transition.get("transition_type") or "").strip().lower()
        if bool(state.get("wall_contact_risk", False)) or bool(state.get("need_recenter", False)):
            return True
        return transition_type == "global_reanchor"

    def _build_curriculum_plan(
        self,
        *,
        state: Dict[str, Any],
        reached_set: Set[str],
        current_airway: str,
        target_hint: str,
        raw_current_target: str = "",
    ) -> Dict[str, Any]:
        resolved_next = self._canonical_next_airway(reached_set=reached_set, target_hint=target_hint)
        transition = self.curriculum.transition_context(current_airway, resolved_next)
        transition_type = str(transition.get("transition_type") or "").strip().lower()
        force_global_backtrack = self._should_force_global_backtrack(
            state=state,
            current_airway=current_airway,
            next_airway=resolved_next,
            raw_current_target=raw_current_target,
        )
        smart_stagnating = (not bool(state.get("is_target_visible", False))) and (
            (not self.curriculum.is_known_airway(current_airway)) or transition_type == "global_reanchor"
        )
        plan = self.curriculum.make_plan(
            current_airway=current_airway,
            reached=reached_set,
            student_question=str(state.get("student_question") or "").strip(),
            requested_next_airway=resolved_next,
            is_back=force_global_backtrack,
            stagnating=smart_stagnating,
        )
        if isinstance(plan, dict):
            plan["resolved_next_airway"] = resolved_next
            plan["resolved_backtracking"] = force_global_backtrack
            plan["upstream_backtracking"] = bool(state.get("backtracking", False)) or str(raw_current_target or "").strip().upper() == "BACK"
        return plan

    def _build_directional_hint(
        self,
        *,
        m_jointsVelRel: List[float],
        event_packet: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.directional_builder.build(
            m_jointsVelRel,
            event_flag=event_packet.get("flag") if isinstance(event_packet, dict) else None,
        ).to_dict()

    def _merge_support_utterance(self, *, support_text: str, main_text: str, preserve_main_teaching: bool = False) -> str:
        support = self._finalize_surface_text(support_text)
        main = self._finalize_surface_text(main_text, preserve_teaching=preserve_main_teaching)
        if not support:
            return main
        if not main:
            return support
        low_main = main.lower()
        low_support = support.lower()
        if low_main.startswith(low_support):
            return main
        if low_support in {"good.", "nice."} and len(sentence_split(main)) >= 2:
            return main
        if low_support in {"good.", "nice.", "nice and steady. keep going.", "easy. re-center first.", "that’s okay. re-center and try again.", "that’s okay. re-center and try again from the carina."}:
            if low_main.startswith(("good.", "nice.", "easy.")):
                return main
        if low_support.startswith("that’s okay.") or low_support.startswith("that's okay."):
            return main
        return self._finalize_surface_text(f"{support} {main}", preserve_teaching=preserve_main_teaching)

    def _reset_turn_counters(self) -> None:
        self.turn_counters.llm_calls = 0

    def _maybe_realize_candidate(
        self,
        *,
        candidate: SkillResult,
        state: Dict[str, Any],
        current_situation: str,
        previous_msgs: str,
        policy_text: str,
    ) -> Tuple[str, bool, str]:
        deterministic = str(getattr(candidate, "deterministic_text", "") or candidate.utterance or "").strip()
        frame = dict(getattr(candidate, "frame", {}) or candidate.data.get("frame") or {})
        wants_realization = bool(getattr(candidate, "wants_realization", False))
        if not wants_realization or not frame:
            return deterministic, False, "realization_not_requested"
        if self.turn_counters.llm_calls >= 1:
            return deterministic, False, "llm_budget_exhausted"
        self.turn_counters.llm_calls += 1
        realized = realize_frame_response(
            model=self.model,
            state=state,
            frame=frame,
            current_situation=current_situation,
            previous_msgs=previous_msgs,
            policy_text=policy_text,
        )
        if self._candidate_realization_is_valid(realized, deterministic=deterministic, frame=frame):
            return realized, True, "realization_valid"
        return deterministic, False, "realization_invalid_fallback"

    def _candidate_realization_is_valid(self, text: str, *, deterministic: str, frame: Dict[str, Any]) -> bool:
        cleaned = str(text or "").strip()
        if not cleaned:
            return False
        if self._looks_like_structured_object(cleaned):
            return False
        mode = str((frame or {}).get("mode") or "").lower()
        if mode == "guidance" and not self._guidance_is_safe(cleaned, frame):
            return False
        if len(sentence_split(cleaned)) > int((frame or {}).get("mode") == "teaching") + 2:
            return False
        if len(cleaned.split()) > 60:
            return False
        return True

    def _llm_verbalize(
        self,
        current_situation: str,
        previous_msgs: str,
        student_question: str,
        auth_plan_json: Dict[str, Any],
    ) -> str:
        policy_text = self._get_runtime_skill_policy()
        prompt = build_sas_instructor_prompt_with_policy(
            plan=auth_plan_json,
            current=current_situation,
            previous_msgs=previous_msgs,
            student_q=student_question,
            state=self.turn_state.session_context,
            skill_policy_text=policy_text,
        )

        self.turn_state.last_llm_debug_repr = ""
        raw = ""
        try:
            if hasattr(self.model, "generate"):
                resp = self.model.generate(prompt)
                self.turn_state.last_llm_debug_repr = repr(resp)
                raw = self._extract_text_response(resp)
        except Exception:
            raw = ""

        if not raw and callable(self.model):
            for messages in (
                [
                    {"role": "system", "content": policy_text},
                    {"role": "user", "content": prompt},
                ],
                [{"role": "user", "content": prompt}],
                prompt,
            ):
                try:
                    resp = self.model(messages)
                    self.turn_state.last_llm_debug_repr = repr(resp)
                    raw = self._extract_text_response(resp)
                    if raw:
                        break
                except Exception:
                    continue

        raw = re.sub(r"\s+", " ", str(raw or "")).strip()
        return compress_utterance(raw)

    def _guidance_is_safe(self, llm_text: str, plan: Dict[str, Any]) -> bool:
        llm_norm = str(llm_text or "").upper()
        target = str((plan or {}).get("next_airway", "")).upper()
        if not llm_text.strip():
            return False
        if self._looks_like_structured_object(llm_text):
            return False
        if len(sentence_split(llm_text)) > 2:
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

    def _finalize_surface_text(self, text: str, preserve_teaching: bool = False) -> str:
        raw = str(text or "").strip()
        raw = re.sub(r"That['’]s okay\.?(?:\s+)?", "", raw, flags=re.I).strip()
        teaching_markers = ("Remember:", "A simple guide:")
        keep_teaching = preserve_teaching or any(marker in raw for marker in teaching_markers)
        cleaned = raw if keep_teaching else clinicalize_ui_text(raw)
        cleaned = light_cleanup_ui_text(cleaned)
        cleaned = re.sub(r"That['’]s okay\.?(?:\s+)?", "", cleaned, flags=re.I)
        return re.sub(r"\s+", " ", cleaned).strip(" ,")

    def _qa_allowed(self, *, state: Dict[str, Any], question_mode: str) -> bool:
        if not str(state.get("student_question") or "").strip():
            return False
        if str(question_mode or "").strip().lower() in {"none", "fragment_unclear", "other"}:
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

    def _build_opening_utterance(self) -> str:
        return (
            "Hello, I’m Sam. We’ll do the bronchoscopy together today. "
            "Keep the handle upright and the view centered. "
            "Use the knob to move the tip up and down, rotate the scope to turn, and move gently forward or back. "
            "I’ll guide you step by step."
        )

    def _build_right_lung_complete_utterance(self) -> str:
        return "Nice work. We’ve finished the right lung. Let’s move on to the left side."

    def _build_session_complete_utterance(self) -> str:
        return "Excellent work. We’ve completed the full bronchoscopy route. Now let’s look at your session report."

    def _milestone_override(self, *, reached_set: Set[str], next_airway: str, student_question: str) -> str:
        if reached_set:
            self.milestones.opening_sent = True
        if self.curriculum.session_complete(reached_set):
            if not self.milestones.session_complete_announced:
                self.milestones.session_complete_announced = True
                return self._build_session_complete_utterance()
            return ""
        if self.curriculum.right_lung_complete(reached_set) and str(next_airway or "").upper().startswith("LB"):
            if not self.milestones.right_lung_complete_announced:
                self.milestones.right_lung_complete_announced = True
                return self._build_right_lung_complete_utterance()
        if not self.milestones.opening_sent and not reached_set and not str(student_question or "").strip():
            self.milestones.opening_sent = True
            return self._build_opening_utterance()
        return ""

    def _select_response_result(self, guidance_result: Any, qa_result: Any) -> Any:
        return qa_result if getattr(qa_result, "active", False) else guidance_result

    def _select_reporting_stage(self, *, payload: Any = None) -> str:
        return "report"

    def _execute_skill(self, skill_name: str, **kwargs: Any) -> Any:
        spec = self.skills[skill_name]
        result = spec.backend(**kwargs)
        if isinstance(result, dict) and {"skill", "active", "priority", "reason", "data", "utterance"}.issubset(result.keys()):
            return SkillResult(
                skill=str(result.get("skill") or skill_name),
                active=bool(result.get("active", False)),
                priority=float(result.get("priority", 0.0)),
                reason=str(result.get("reason") or ""),
                data=dict(result.get("data") or {}),
                utterance=str(result.get("utterance") or ""),
                frame=dict(result.get("frame") or {}),
                deterministic_text=str(result.get("deterministic_text") or ""),
                wants_realization=bool(result.get("wants_realization", False)),
                constraints=dict(result.get("constraints") or {}),
                debug_reason=str(result.get("debug_reason") or ""),
            )
        return result

    def _merge_event_signals_into_state(self, state: Dict[str, Any], event_packet: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(event_packet, dict):
            return state
        merged = dict(state)
        merged["event_packet"] = event_packet
        if event_packet.get("soft_prompt") and not str(merged.get("soft_prompt") or "").strip():
            merged["soft_prompt"] = str(event_packet.get("soft_prompt") or "").strip()
        if event_packet.get("reason") and not str(merged.get("llm_reason") or "").strip():
            merged["llm_reason"] = str(event_packet.get("reason") or "").strip()
        merged["need_llm"] = bool(merged.get("need_llm", False) or event_packet.get("llm_trigger_flag", False))
        merged["llm_trigger_flag"] = bool(merged.get("llm_trigger_flag", False) or event_packet.get("llm_trigger_flag", False))
        return merged

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
        state: Dict[str, Any],
        skills_used: List[str],
        question_mode: str,
        selected_frame_mode: str,
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
            "landmark_teaching_history": list(state.get("landmark_teaching_history") or []),
            "last_landmark_taught": str(state.get("last_landmark_taught") or ""),
            "last_landmark_taught_turn": int(state.get("last_landmark_taught_turn") or 0),
            "last_acknowledged_waypoint": str(state.get("last_acknowledged_waypoint") or ""),
            "last_acknowledged_waypoint_turn": int(state.get("last_acknowledged_waypoint_turn") or 0),
            "last_acknowledged_destination": str(state.get("last_acknowledged_destination") or ""),
            "last_acknowledged_destination_turn": int(state.get("last_acknowledged_destination_turn") or 0),
            "session_turn_index": int(state.get("session_turn_index") or 0),
            "meta": {
                "manager": self.MANAGER_NAME,
                "pipeline": self.PIPELINE_NAME,
                "provider": os.environ.get("BRONCHO_PROVIDER", ""),
                "model": os.environ.get("BRONCHO_MODEL", self.model_name),
                "fallback_backend": bool(getattr(self.model, "is_fallback_backend", False)),
                "skills_used": skills_used,
                "question_mode": question_mode,
                "selected_frame_mode": selected_frame_mode,
                "qa_utterance_full": qa_utterance_full,
            },
        }

    def _run_from_state(self, state: Dict[str, Any], source_mode: str) -> Dict[str, Any]:
        self._reset_turn_counters()
        state = self._inherit_session_state(state)
        state["session_turn_index"] = int(state.get("session_turn_index") or 0) + 1
        prompt = state.get("prompt_text", "").strip()
        previous_msgs_raw = state.get("previous_msgs", "")
        previous_msgs = self.turn_log.to_prompt_context()
        student_question = state.get("student_question", "")

        reached_list = [str(x).upper() for x in state.get("reached_regions", [])]
        reached_set: Set[str] = set(reached_list)
        current_airway = str(state.get("current_airway", "")).strip().upper()
        target_hint = str(state.get("target_airway", "")).strip().upper()
        raw_payload = dict(state.get("raw_payload") or {})
        raw_current_target = str(raw_payload.get("current_target") or "").strip().upper()
        next_airway = self._canonical_next_airway(reached_set=reached_set, target_hint=target_hint)
        landmark_hint = self._lookup_landmark(next_airway or current_airway or "RB1")

        auth_plan_json = self._build_curriculum_plan(
            state=state,
            reached_set=reached_set,
            current_airway=current_airway,
            target_hint=next_airway,
            raw_current_target=raw_current_target,
        )
        next_airway = str(auth_plan_json.get("next_airway", "")).strip().upper() or next_airway
        resolved_backtracking = bool(auth_plan_json.get("resolved_backtracking", False)) or str(auth_plan_json.get("mode") or "").strip().lower() == "backtrack"
        state["backtracking"] = resolved_backtracking
        state["target_airway"] = next_airway
        current_situation = self._build_current_situation_from_state(state)

        curriculum_progress = {
            "reached": reached_list,
            "next_airway": next_airway,
            "coverage_ratio": round(len(reached_list) / max(len(self.AIRWAY_VISIT_ORDER), 1), 4),
            "reached_count": len(reached_list),
            "total": len(self.AIRWAY_VISIT_ORDER),
        }

        event_frame = dict(raw_payload)
        waypoint_target = raw_current_target if raw_current_target not in {"", "BACK", next_airway} else ""
        effective_event_target = "back" if resolved_backtracking else next_airway
        event_frame.setdefault("anatomical_position", current_airway)
        event_frame["current_target"] = effective_event_target
        event_frame["training_target"] = next_airway
        event_frame["waypoint_target"] = waypoint_target
        event_frame["next_destination"] = next_airway
        event_frame.setdefault("reached_regions", reached_list)
        event_frame.setdefault("just_reached", bool(state.get("just_reached", False)))

        event_packet = self.event_engine.step(event_frame, history=None, plan=auth_plan_json)
        state = self._merge_event_signals_into_state(
            state,
            event_packet if isinstance(event_packet, dict) else {},
        )
        state["training_target"] = next_airway
        state["waypoint_target"] = waypoint_target
        event_packet = state.get("event_packet", {})
        policy_text = self._get_runtime_skill_policy()
        directional_hint = self._build_directional_hint(
            m_jointsVelRel=state.get("m_jointsVelRel") or [0.0, 0.0, 0.0],
            event_packet=event_packet,
        )
        auth_plan_json["directional_hint"] = directional_hint
        question_mode = "question_present" if str(state.get("student_question") or "").strip() else "none"
        selection = self.dispatcher.dispatch(
            state=state,
            plan=auth_plan_json,
            model=self.model,
            current_situation=current_situation,
            previous_msgs=previous_msgs,
            previous_msgs_raw=previous_msgs_raw,
            policy_text=policy_text,
            run_teaching=not safety_risk(state),
            run_qa=self._qa_allowed(state=state, question_mode=question_mode),
            run_support=True,
            visit_order=self.visit_order,
        )

        guidance_result = selection.guidance
        support_result = selection.support
        landmark_teaching_result = selection.teaching
        qa_result = selection.qa
        skill_records = list(selection.all_records)
        selected_result = selection.winner
        selected_skill = selected_result.to_dict()
        selected_frame = dict(selected_result.data.get("frame") or {})
        guidance_frame = dict(guidance_result.data.get("frame") or {})
        landmark_teaching_frame = dict(landmark_teaching_result.data.get("frame") or {}) if landmark_teaching_result.active else None
        qa_frame = dict(qa_result.data.get("frame") or {}) if qa_result.active else None
        selected_frame_mode = str(
            selected_result.data.get("frame_mode")
            or (selected_frame.get("mode") if isinstance(selected_frame, dict) else "")
            or ("qa" if qa_result.active else "teaching" if landmark_teaching_result.active else "guidance")
        )

        qa_question_mode = str(qa_result.data.get("question_mode") or (qa_frame.get("question_mode") if isinstance(qa_frame, dict) else "") or "none")
        deterministic_ui = str(guidance_result.data.get("deterministic_utterance") or "")

        selected_surface_raw, selected_realized, selected_debug_reason = self._maybe_realize_candidate(
            candidate=selected_result,
            state=state,
            current_situation=current_situation,
            previous_msgs=previous_msgs,
            policy_text=policy_text,
        )
        selected_result.utterance = selected_surface_raw
        selected_result.data["realized"] = bool(selected_realized)
        selected_result.debug_reason = selected_debug_reason
        selected_skill = selected_result.to_dict()
        for idx, record in enumerate(skill_records):
            if str(record.get("skill") or "") == selected_result.skill:
                skill_records[idx] = selected_skill
                break

        guidance_realized = bool(selected_result.skill == "guidance_skill" and selected_realized)
        qa_realized = bool(selected_result.skill == "qa_skill" and selected_realized)

        guidance_text = selected_surface_raw if selected_result.skill == "guidance_skill" else str(guidance_result.deterministic_text or guidance_result.utterance or "")
        teaching_text = selected_surface_raw if selected_result.skill == "landmark_teaching_skill" else str(landmark_teaching_result.deterministic_text or landmark_teaching_result.utterance or "")
        qa_text = selected_surface_raw if selected_result.skill == "qa_skill" else str(qa_result.deterministic_text or qa_result.utterance or "")

        guidance_ui = self._finalize_surface_text(guidance_text.strip())
        teaching_ui = self._finalize_surface_text(teaching_text.strip(), preserve_teaching=landmark_teaching_result.active)
        deterministic_ui = self._finalize_surface_text(str(selected_result.deterministic_text or deterministic_ui).strip(), preserve_teaching=selected_result is landmark_teaching_result)
        support_ui = self._finalize_surface_text(str(support_result.deterministic_text or support_result.utterance or "").strip())
        qa_ui = self._finalize_surface_text(qa_text.strip())
        support_mode = str((support_result.data or {}).get("support_mode") or "")
        support_allowed = bool(getattr(support_result, "active", False))
        if landmark_teaching_result.active and support_mode not in {"arrival_feedback", "acknowledge_progress"}:
            support_allowed = False
        if bool(state.get("just_reached", False)):
            support_allowed = False
        if support_mode == "encourage_progress" and selected_result.skill == "guidance_skill":
            support_allowed = False
        main_surface = str(selected_result.utterance or "").strip()
        if qa_result.active and guidance_ui:
            qa_surface = str(qa_ui or "").strip()
            if guidance_ui.lower() not in qa_surface.lower():
                main_surface = f"{qa_surface} {guidance_ui}".strip()
            else:
                main_surface = qa_surface

        chosen_ui = self._merge_support_utterance(
            support_text=support_ui if support_allowed else "",
            main_text=main_surface,
            preserve_main_teaching=(selected_result is landmark_teaching_result),
        )

        if not chosen_ui:
            chosen_ui = deterministic_ui or "Keep the lumen centered. Advance slowly."

        milestone_ui = self._milestone_override(
            reached_set=reached_set,
            next_airway=next_airway,
            student_question=student_question,
        )
        milestone_override_active = bool(milestone_ui)
        if milestone_ui:
            chosen_ui = milestone_ui

        skills_used = ["guidance_skill"]
        if support_result.active:
            skills_used.append("support_skill")
        if landmark_teaching_result.active:
            skills_used.append("landmark_teaching_skill")
        if qa_result.active:
            skills_used.append("qa_skill")
        skills_used.append("statistics_skill")

        llm_ui = chosen_ui if selected_realized else ""
        assert self.turn_counters.llm_calls <= 1, "llm_calls_this_turn exceeded hard per-turn limit"
        utterance_full = chosen_ui
        if milestone_override_active:
            ui_text = utterance_full
        elif landmark_teaching_result.active:
            ui_text = relaxed_ui_text(utterance_full, max_words=60, max_sentences=3)
        elif bool(state.get("wall_contact_risk", False)) or bool(state.get("need_recenter", False)) or bool(state.get("drift_detected", False)):
            ui_text = short_ui_text(utterance_full, max_words=20)
        else:
            ui_text = relaxed_ui_text(utterance_full, max_words=42)
        ui_text = self._finalize_surface_text(ui_text, preserve_teaching=selected_result is landmark_teaching_result)
        self.turn_state.last_ui_text = utterance_full

        if landmark_teaching_result.active:
            state["last_landmark_taught"] = str(landmark_teaching_result.data.get("landmark_id") or "")
            state["last_landmark_taught_turn"] = int(state.get("session_turn_index") or 0)
            state = mark_landmark_as_taught(
                state,
                str(landmark_teaching_result.data.get("landmark_id") or ""),
            )
        if bool(state.get("just_reached", False)):
            if current_airway in {"RMB", "LMB", "BI", "CARINA", "RUL", "RML", "RLL", "LUL", "LLL"}:
                state["last_acknowledged_waypoint"] = current_airway
                state["last_acknowledged_waypoint_turn"] = int(state.get("session_turn_index") or 0)
            elif re.fullmatch(r"(?:RB\d+|LB\d(?:\+\d)?)", current_airway):
                state["last_acknowledged_destination"] = current_airway
                state["last_acknowledged_destination_turn"] = int(state.get("session_turn_index") or 0)

        statistics = self._execute_skill(
            "statistics_skill",
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
            state=state,
            skills_used=skills_used,
            question_mode=qa_question_mode,
            selected_frame_mode=selected_frame_mode,
            qa_utterance_full=qa_ui,
        )

        self.turn_counters.iteration += 1
        self.logger.log_step(
            step_index=self.turn_counters.iteration,
            state=event_frame,
            control=event_packet if isinstance(event_packet, dict) else {},
            guidance={
                "ui_text": ui_text,
                "utterance_full": utterance_full,
                "deterministic_utterance_full": deterministic_ui,
                "llm_utterance_full": llm_ui,
                "support_utterance_full": support_ui,
                "qa_utterance_full": qa_ui,
                "teaching_utterance_full": teaching_ui,
                "needs_visual_guidance": self._needs_visual_guidance(student_question),
                "question_mode": qa_question_mode,
                "selected_frame_mode": selected_frame_mode,
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
                "llm_calls_this_turn": int(self.turn_counters.llm_calls),
                "llm_debug_repr": self.turn_state.last_llm_debug_repr,
                "used_fallback_backend": bool(getattr(self.model, "is_fallback_backend", False)),
                "m_jointsVelRel": state.get("m_jointsVelRel") or [0.0, 0.0, 0.0],
                "directional_hint": directional_hint,
                "normalized_state": state,
                "skills_used": skills_used,
                "skill_records": skill_records,
                "selected_skill": selected_skill,
                "support_utterance_full": support_ui,
                "support_mode": support_result.data.get("support_mode", "none"),
                "question_mode": qa_question_mode,
                "qa_utterance_full": qa_ui,
                "guidance_utterance_full": guidance_ui,
                "teaching_utterance_full": teaching_ui,
                "guidance_realized": guidance_realized,
                "qa_realized": qa_realized,
                "selected_frame_mode": selected_frame_mode,
                "guidance_frame": guidance_frame,
                "landmark_teaching_frame": landmark_teaching_frame,
                "qa_frame": qa_frame,
                "selected_frame": selected_frame,
            },
        )

        selected_intent = str(
            selected_result.data.get("frame", {}).get("intent")
            or selected_frame.get("intent")
            or selected_result.data.get("frame_mode")
            or ""
        ).strip()
        self.turn_log.record(
            TurnLogEntry(
                turn_index=int(state.get("session_turn_index") or self.turn_counters.iteration),
                selected_skill=str(selected_result.skill or ""),
                action_intent=selected_intent,
                target_airway=next_airway,
                support_mode=support_mode,
                utterance_summary=str(utterance_full or "").strip()[:60],
            )
        )
        self.turn_state.session_context = dict(state or {})

        return {
            "ui_text": ui_text,
            "instructor": ui_text,
            "utterance_full": utterance_full,
            "deterministic_utterance_full": deterministic_ui,
            "llm_utterance_full": llm_ui,
            "support_utterance_full": support_ui,
            "qa_utterance_full": qa_ui,
            "guidance_utterance_full": guidance_ui,
            "teaching_utterance_full": teaching_ui,
            "needs_visual_guidance": self._needs_visual_guidance(student_question),
            "curriculum_progress": curriculum_progress,
            "landmark_hint": landmark_hint,
            "plan_json": auth_plan_json,
            "statistics": statistics,
            "statepacket": statepacket,
            "event_packet": event_packet,
            "frame": selected_frame,
            "skill_records": skill_records,
            "selected_skill": selected_skill,
            "guidance_frame": guidance_frame,
            "qa_frame": qa_frame,
            "raw": {
                "source_mode": source_mode,
                "prompt": prompt,
                "current_situation": current_situation,
                "used_fallback_backend": bool(getattr(self.model, "is_fallback_backend", False)),
                "deterministic_ui": deterministic_ui,
                "llm_ui": llm_ui,
                "qa_ui": qa_ui,
                "guidance_ui": guidance_ui,
                "teaching_ui": teaching_ui,
                "llm_debug_repr": self.turn_state.last_llm_debug_repr,
                "m_jointsVelRel": state.get("m_jointsVelRel") or [0.0, 0.0, 0.0],
                "directional_hint": directional_hint,
                "normalized_state": state,
                "skills_used": skills_used,
                "skill_records": skill_records,
                "selected_skill": selected_skill,
                "question_mode": qa_question_mode,
                "qa_utterance_full": qa_ui,
                "guidance_realized": guidance_realized,
                "qa_realized": qa_realized,
                "selected_frame_mode": selected_frame_mode,
                "guidance_frame": guidance_frame,
                "landmark_teaching_frame": landmark_teaching_frame,
                "qa_frame": qa_frame,
                "selected_frame": selected_frame,
            },
        }

    def _report_output_dir(self) -> Path:
        run_dir = getattr(self.logger, "run_dir", None)
        if run_dir:
            return Path(run_dir)
        env_dir = os.environ.get("BRONCHO_RUN_DIR") or os.environ.get("BRONCHO_RECORDING_DIR")
        if env_dir:
            return Path(env_dir)
        return Path.cwd()

    def _persist_session_report(self, *, report_payload: Dict[str, Any], reason: str) -> Dict[str, str]:
        output_dir = self._report_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        text_path = output_dir / "session_report.txt"
        json_path = output_dir / "session_report.json"

        report_text = str(report_payload.get("report_text") or "").strip()
        text_path.write_text((report_text + "\n") if report_text else "", encoding="utf-8")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema": "broncho.session_report.v1",
                    "reason": reason,
                    "session_id": getattr(self.logger, "session_id", os.environ.get("BRONCHO_SESSION_ID", "")),
                    "run_dir": str(output_dir),
                    "manager": self.MANAGER_NAME,
                    "pipeline": self.PIPELINE_NAME,
                    "iteration": self.turn_counters.iteration,
                    "report": report_payload,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        paths = {"text": str(text_path), "json": str(json_path)}
        self.milestones.auto_report_written = True
        self._auto_report_reason = reason
        self._auto_report_paths = paths
        self._last_session_report = dict(report_payload or {})
        return paths

    def _maybe_auto_generate_session_report(self, *, reason: str, payload: Any = None, force: bool = False) -> Optional[Dict[str, Any]]:
        if self.milestones.auto_report_in_progress:
            return None
        if self.milestones.auto_report_written and not force:
            return None
        if payload is None and not self.turn_state.session_context:
            return None

        self.milestones.auto_report_in_progress = True
        try:
            return self.generate_session_report(payload=payload, _save_reason=reason)
        except Exception as exc:
            append_error = getattr(self.logger, "append_error", None)
            if callable(append_error):
                append_error({
                    "event_type": "auto_report_error",
                    "reason": reason,
                    "error": str(exc),
                })
            return None
        finally:
            self.milestones.auto_report_in_progress = False

    def _finalize_session_report_at_exit(self) -> None:
        if self.milestones.auto_report_written:
            return
        if self.turn_counters.iteration <= 0 and not self.turn_state.session_context:
            return
        self._maybe_auto_generate_session_report(reason="process_exit")

    def run(self, payload: Any) -> Dict[str, Any]:
        state, source_mode = self._parse_prompt_or_payload(payload)
        self.turn_state.session_context = dict(state or {})
        result = self._run_from_state(state, source_mode)
        try:
            reached = list(result.get("curriculum_progress", {}).get("reached") or [])
            if self.curriculum.session_complete(reached):
                auto_report = self._maybe_auto_generate_session_report(reason="session_complete")
                if auto_report:
                    result["session_report_text"] = auto_report.get("report_text", "")
                    result["session_report_paths"] = auto_report.get("saved_paths", {})
        except Exception:
            pass
        return result

    def step(self, payload: Any) -> Dict[str, Any]:
        return self.run(payload)

    def generate_session_report(
        self,
        payload: Any = None,
        *,
        session_metrics: Optional[Dict[str, Any]] = None,
        teach_line: Optional[str] = None,
        sp_score: Optional[float] = None,
        _save_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        state: Dict[str, Any]
        report_stage = self._select_reporting_stage(payload=payload)
        if payload is None:
            state = dict(self.turn_state.session_context or {})
            source_mode = "cached_state"
        else:
            state, source_mode = self._parse_prompt_or_payload(payload)
            self.turn_state.session_context = dict(state or {})

        current_situation = self._build_current_situation_from_state(state) if state else ""
        reached_list = [str(x).upper() for x in state.get("reached_regions", [])]
        reached_set: Set[str] = set(reached_list)
        current_airway = str(state.get("current_airway", "")).strip().upper()
        target_hint = str(state.get("target_airway", "")).strip().upper()
        next_airway = target_hint or self.curriculum.next_airway(reached_set) or ""

        auth_plan_json = self._build_curriculum_plan(
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
            "student_questions": state.get("student_questions", 0),
        }

        metrics = dict(session_metrics or {})
        if "duration_seconds" not in metrics:
            metrics["duration_seconds"] = state.get("duration_seconds")
        if "backtrack_ratio" not in metrics:
            metrics["backtrack_ratio"] = state.get("backtrack_ratio", "N/A")
        if "student_questions" not in metrics:
            metrics["student_questions"] = state.get("student_questions", 0)

        if sp_score is None:
            expected_prefix = self.AIRWAY_VISIT_ORDER[: len(reached_list)]
            prefix_hits = sum(1 for i, airway in enumerate(expected_prefix) if i < len(reached_list) and reached_list[i] == airway)
            sp_score = round(prefix_hits / max(len(reached_list), 1), 4) if reached_list else 0.0

        report_payload = self._execute_skill(
            "reporting_skill",
            allowed_reached=reached_list,
            visit_order=self.AIRWAY_VISIT_ORDER,
            curriculum_progress=curriculum_progress,
            session_metrics=metrics,
            sp_score=float(sp_score),
            teach_line=teach_line,
        )

        skill_record = build_skill_record(
            skill="reporting_skill",
            active=True,
            priority=0.3,
            reason="end-of-session report requested",
            data=report_payload,
        )

        self.logger.log_step(
            step_index=self.turn_counters.iteration + 1,
            state=state.get("raw_payload", {}) if isinstance(state, dict) else {},
            control={},
            guidance={
                "ui_text": "",
                "utterance_full": report_payload.get("report_text", ""),
                "deterministic_utterance_full": report_payload.get("report_text", ""),
                "llm_utterance_full": "",
                "qa_utterance_full": "",
                "needs_visual_guidance": False,
                "question_mode": "report",
                "selected_frame_mode": report_stage,
            },
            plan=auth_plan_json,
            statistics={
                "report_generated": True,
            },
            extra={
                "source_mode": source_mode,
                "current_situation": current_situation,
                "skills_used": ["reporting_skill"],
                "report_stage": report_stage,
                "skill_records": [skill_record],
                "selected_skill": skill_record,
                "report": report_payload,
            },
        )

        saved_paths = self._persist_session_report(
            report_payload=report_payload,
            reason=str(_save_reason or "generate_session_report"),
        )

        return {
            "report_text": report_payload.get("report_text", ""),
            "report": report_payload,
            "skill_records": [skill_record],
            "selected_skill": skill_record,
            "curriculum_progress": curriculum_progress,
            "session_metrics": metrics,
            "plan_json": auth_plan_json,
            "saved_paths": saved_paths,
        }

    def __call__(self, payload: Any) -> Dict[str, Any]:
        return self.run(payload)


SingleAgentManager = SASManager
