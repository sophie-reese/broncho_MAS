from __future__ import annotations

from typing import Any, Dict, List

from ...shared import (
    LANDMARK_CARDS,
    get_landmark_card,
    local_navigation_cue,
    normalize_landmark_id,
)
from ...shared.utterance_postprocess import compress_utterance, light_cleanup_ui_text
from .base import BaseSkill, Plan, SkillResult, State, build_skill_record
from .guidance import _fallback_guidance, _recognition_cue
from .realization import _build_frame, _join_lines, deterministic_frame_text
from .utterance_helpers import _airway_token, merge_teaching_with_guidance, safety_risk

SUPPORTED_LANDMARKS = set(LANDMARK_CARDS.keys())

# Local compatibility aliases so older upstream labels still resolve to the new cards.
LEGACY_LANDMARK_ALIASES: Dict[str, str] = {
    "TRACHEA": "L1_CARINA",
    "CARINA": "L1_CARINA",
    "L1": "L1_CARINA",
    "MERCEDES": "L2_RUL",
    "MERCEDES_SIGN": "L2_RUL",
    "RIGHT_UPPER_LOBE": "L2_RUL",
    "L2": "L2_RUL",
    "BI": "L3_RIGHT_MIDDLE_LOWER",
    "BRONCHUS_INTERMEDIUS": "L3_RIGHT_MIDDLE_LOWER",
    "RML": "L3_RIGHT_MIDDLE_LOWER",
    "RLL": "L3_RIGHT_MIDDLE_LOWER",
    "RIGHT_MIDDLE_AND_LOWER": "L3_RIGHT_MIDDLE_LOWER",
    "L3_RIGHT_MIDDLE_LOWER": "L3_RIGHT_MIDDLE_LOWER",
    "L3_RIGHT_ML_LL": "L3_RIGHT_MIDDLE_LOWER",
    "L3": "L3_RIGHT_MIDDLE_LOWER",
    "LMB": "L4_LEFT_MAIN",
    "LEFT": "L4_LEFT_MAIN",
    "LEFT_SIDE": "L4_LEFT_MAIN",
    "LEFT_MAIN_BRONCHUS": "L4_LEFT_MAIN",
    "L4_LEFT": "L4_LEFT_MAIN",
    "L4_LEFT_MAIN": "L4_LEFT_MAIN",
    "LEFT_LUNG": "L4_LEFT_MAIN",
    "LUL": "L4_LEFT_MAIN",
    "LINGULA": "L4_LEFT_MAIN",
    "LLL": "L4_LEFT_MAIN",
    "L4": "L4_LEFT_MAIN",
}


def _normalize_landmark_token(value: Any) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")


def _canonicalize_landmark_value(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    candidates: List[str] = []
    token = _normalize_landmark_token(raw)
    spaced = token.replace("_", " ")

    for candidate in (raw, raw.upper(), raw.lower(), token, spaced, spaced.lower()):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        canonical = normalize_landmark_id(candidate)
        if canonical in SUPPORTED_LANDMARKS:
            return canonical
        legacy = LEGACY_LANDMARK_ALIASES.get(_normalize_landmark_token(candidate), "")
        if legacy in SUPPORTED_LANDMARKS:
            return legacy
    return ""


def resolve_landmark_id(*, state: State, plan: Plan) -> str:
    candidates: List[Any] = [
        state.get("validated_landmark"),
        state.get("landmark_id"),
        state.get("current_landmark"),
        state.get("teaching_landmark"),
        plan.get("anchor_landmark") if isinstance(plan, dict) else None,
        plan.get("landmark") if isinstance(plan, dict) else None,
        state.get("current_airway"),
    ]
    for raw in candidates:
        canonical = _canonicalize_landmark_value(raw)
        if canonical:
            return canonical
    return ""


def _coerce_landmark_history(raw: Any) -> List[str]:
    if raw is None:
        return []
    out: List[str] = []
    if isinstance(raw, (list, tuple, set)):
        values = list(raw)
    elif isinstance(raw, dict):
        values = [k for k, v in raw.items() if v]
    else:
        values = [raw]

    for value in values:
        canonical = _canonicalize_landmark_value(value)
        if canonical:
            out.append(canonical)
    return out


def get_taught_landmarks(state: State) -> List[str]:
    for key in (
        "taught_landmarks",
        "landmark_teaching_history",
        "teaching_history",
        "session_taught_landmarks",
    ):
        history = _coerce_landmark_history(state.get(key))
        if history:
            return history
    return []


def should_fire_landmark_teaching(*, state: State, landmark_id: str) -> bool:
    if not landmark_id or landmark_id not in SUPPORTED_LANDMARKS:
        return False
    if safety_risk(state):
        return False
    if bool(str(state.get("student_question") or "").strip()):
        return False
    if landmark_id in set(get_taught_landmarks(state)):
        return False
    turn_index = int(state.get("session_turn_index") or 0)
    if (
        _canonicalize_landmark_value(state.get("last_landmark_taught")) == landmark_id
        and (turn_index - int(state.get("last_landmark_taught_turn") or -10_000)) <= 8
    ):
        return False

    observed_candidates = [
        state.get("validated_landmark"),
        state.get("landmark_id"),
        state.get("current_landmark"),
        state.get("teaching_landmark"),
        state.get("current_airway"),
    ]
    plan_candidates = [
        state.get("anchor_landmark"),
        state.get("plan_anchor_landmark"),
        state.get("plan_landmark"),
    ]
    observed_match = any(_canonicalize_landmark_value(value) == landmark_id for value in observed_candidates)
    plan_match = any(_canonicalize_landmark_value(value) == landmark_id for value in plan_candidates)
    explicit_match = observed_match or plan_match
    if not explicit_match:
        return False

    current_airway = str(state.get("current_airway") or "").strip().upper()
    if (
        plan_match
        and not observed_match
        and bool(state.get("just_reached"))
        and not bool(state.get("first_time_landmark"))
        and current_airway.startswith(("RB", "LB"))
    ):
        return False

    return bool(state.get("first_time_landmark")) or bool(state.get("just_reached"))


def build_landmark_teaching_payload(*, state: State, plan: Plan) -> Dict[str, Any]:
    state = dict(state or {})
    if isinstance(plan, dict):
        state.setdefault("anchor_landmark", plan.get("anchor_landmark"))
        state.setdefault("plan_anchor_landmark", plan.get("anchor_landmark"))
        state.setdefault("plan_landmark", plan.get("landmark"))
    landmark_id = resolve_landmark_id(state=state, plan=plan)
    if not landmark_id:
        return {}
    card = get_landmark_card(landmark_id)
    if not card:
        return {}
    return {
        "landmark_id": card.id,
        "display_name": card.display_name,
        "when_to_teach": card.when_to_teach,
        "recognition_cues": list(card.recognition_cues),
        "common_confusions": list(card.common_confusions),
        "memory_hook_type": card.memory_hook_type,
        "memory_hook_core": card.memory_hook_core,
        "memory_hook_rhythm": card.memory_hook_rhythm,
        "action_anchor": card.action_anchor,
        "default_teaching_line": card.default_teaching_line,
        "reinforcement_line": card.reinforcement_line,
        "repair_line": card.repair_line,
        "quiz_line": card.quiz_line,
        "notes": card.notes,
        "first_arrival": should_fire_landmark_teaching(state=state, landmark_id=landmark_id),
        "followup_target": str((plan or {}).get("next_airway") or "").strip().upper(),
        "local_navigation_tip": local_navigation_cue(card.id, str((plan or {}).get("next_airway") or "").strip().upper()),
    }


def _build_landmark_teaching_utterance(*, landmark_id: str, state: State) -> str:
    card = get_landmark_card(landmark_id)
    if not card:
        return ""
    first_teach = should_fire_landmark_teaching(state=state, landmark_id=landmark_id)
    if not first_teach:
        return ""
    text = card.default_teaching_line
    if first_teach and card.memory_hook_rhythm:
        low_text = text.lower()
        low_hook = card.memory_hook_rhythm.lower()
        if low_hook not in low_text and "remember:" not in low_text:
            text = f"{text} {card.memory_hook_rhythm}"
    return compress_utterance(text, max_sentences=3, max_words_per_sentence=28)


def build_landmark_teaching_record(*, state: State, plan: Plan, guidance_text: str = "") -> Dict[str, Any]:
    state = dict(state or {})
    if isinstance(plan, dict):
        state.setdefault("anchor_landmark", plan.get("anchor_landmark"))
        state.setdefault("plan_anchor_landmark", plan.get("anchor_landmark"))
        state.setdefault("plan_landmark", plan.get("landmark"))
    landmark_id = resolve_landmark_id(state=state, plan=plan)
    if not landmark_id:
        return build_skill_record(
            skill="landmark_teaching_skill",
            active=False,
            priority=0.0,
            reason="no supported landmark available",
            data={},
            utterance="",
        )

    active = should_fire_landmark_teaching(state=state, landmark_id=landmark_id)
    payload = build_landmark_teaching_payload(state=state, plan=plan)
    utterance = _build_landmark_teaching_utterance(landmark_id=landmark_id, state=state) if active else ""
    local_tip = str(payload.get("local_navigation_tip") or "").strip()
    if active and local_tip and local_tip.lower() not in utterance.lower():
        utterance = light_cleanup_ui_text(f"{utterance} Local cue: {local_tip}")
    if active and guidance_text:
        utterance = merge_teaching_with_guidance(utterance, guidance_text)
    return build_skill_record(
        skill="landmark_teaching_skill",
        active=active,
        priority=0.83 if active else 0.0,
        reason="first arrival at validated landmark" if active else "landmark recognized but teaching suppressed",
        data=payload,
        utterance=utterance,
    )


def build_landmark_teaching_frame(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    payload = build_landmark_teaching_payload(state=state, plan=plan)
    landmark_id = str(payload.get("landmark_id") or "").strip()
    if not landmark_id:
        return {}

    target_airway = str((plan or {}).get("next_airway") or "").strip().upper()
    target_label = _airway_token(target_airway)
    cue = _recognition_cue(plan)
    teaching_line = _build_landmark_teaching_utterance(landmark_id=landmark_id, state=state)
    local_tip = str(payload.get("local_navigation_tip") or "").strip()
    next_step = local_tip or _fallback_guidance(plan=plan)
    next_step = light_cleanup_ui_text(next_step) if next_step else ""
    if next_step and teaching_line:
        base = _join_lines(teaching_line, next_step)
    else:
        base = teaching_line or next_step

    return _build_frame(
        mode="teaching",
        question_mode="none",
        safety_mode="normal",
        intent="landmark_teaching",
        target_airway=target_airway,
        target_label=target_label,
        recognition_cue=cue,
        prefix="",
        action_line=next_step,
        cue_line="",
        answer_core=str(payload.get("default_teaching_line") or "").strip(),
        next_step=next_step,
        question="",
        fallback_guidance=next_step or _fallback_guidance(plan=plan),
        base_utterance=base,
    )


class LandmarkTeachingSkill(BaseSkill):
    """Foreground first-arrival landmark teaching skill."""

    name = "landmark_teaching_skill"
    default_priority = 0.92

    def execute(
        self,
        *,
        state: Dict[str, Any],
        plan: Dict[str, Any],
        model: Any = None,
        current_situation: str = "",
        previous_msgs: str = "",
        policy_text: str = "",
        **kwargs: Any,
    ) -> SkillResult:
        state = dict(state or {})
        if isinstance(plan, dict):
            state.setdefault("anchor_landmark", plan.get("anchor_landmark"))
            state.setdefault("plan_anchor_landmark", plan.get("anchor_landmark"))
            state.setdefault("plan_landmark", plan.get("landmark"))

        payload = build_landmark_teaching_payload(state=state, plan=plan)
        landmark_id = str(payload.get("landmark_id") or "").strip()
        if not landmark_id:
            return SkillResult(
                skill=self.name,
                active=False,
                priority=0.0,
                reason="no supported landmark available",
                data={},
                utterance="",
            )

        active = should_fire_landmark_teaching(state=state, landmark_id=landmark_id)
        frame = build_landmark_teaching_frame(state=state, plan=plan) if active else {}
        deterministic = deterministic_frame_text(frame) if frame else ""
        wants_realization = bool(active and deterministic and model is not None and not getattr(model, "is_fallback_backend", False))
        return SkillResult(
            skill=self.name,
            active=active,
            priority=self.default_priority if active else 0.0,
            reason="first arrival at validated landmark" if active else "landmark recognized but teaching suppressed",
            data={
                **payload,
                "frame": frame,
                "frame_mode": str((frame or {}).get("mode") or "teaching"),
                "deterministic_utterance": deterministic,
                "realized": False,
            },
            utterance=deterministic,
            frame=frame,
            deterministic_text=deterministic,
            wants_realization=wants_realization,
            constraints={"max_sentences": 3, "mode": "teaching", "preserve_teaching": True},
            debug_reason="skill prepared teaching frame; manager must decide whether to realize it",
        )


def landmark_teaching_skill(**kwargs: Any) -> Dict[str, Any]:
    return LandmarkTeachingSkill().execute(**kwargs).to_dict()


def mark_landmark_as_taught(state: State, landmark_id: str) -> State:
    history = list(get_taught_landmarks(state))
    canonical = _canonicalize_landmark_value(landmark_id)
    if canonical and canonical not in history:
        history.append(canonical)
    new_state = dict(state)
    new_state["landmark_teaching_history"] = history
    return new_state
