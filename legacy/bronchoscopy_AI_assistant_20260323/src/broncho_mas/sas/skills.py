from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence, Set

from ..shared.curriculum import CurriculumEngine
from ..shared.directional_hint_builder import DirectionalHintBuilder
from ..shared.utterance_postprocess import (
    compress_utterance,
    normalize_landmark_language,
    sanitize_micro_step,
)


_AIRWAY_LABELS = {
    "CARINA": "the carina",
    "TRACHEA": "the trachea",
    "RMB": "the right main bronchus",
    "LMB": "the left main bronchus",
    "BI": "the bronchus intermedius",
    "RUL": "the right upper lobe opening",
    "RML": "the right middle lobe opening",
    "RLL": "the right lower lobe opening",
    "LUL": "the left upper lobe opening",
    "LLL": "the left lower lobe opening",
    "RB1": "the apical segment of the right upper lobe",
    "RB2": "the posterior segment of the right upper lobe",
    "RB3": "the anterior segment of the right upper lobe",
    "RB4": "the lateral segment of the right middle lobe",
    "RB5": "the medial segment of the right middle lobe",
    "RB6": "the superior segment of the right lower lobe",
    "RB7": "the medial basal segment of the right lower lobe",
    "RB8": "the anterior basal segment of the right lower lobe",
    "RB9": "the lateral basal segment of the right lower lobe",
    "RB10": "the posterior basal segment of the right lower lobe",
    "LB1+2": "the apicoposterior segment of the left upper lobe",
    "LB3": "the anterior segment of the left upper lobe",
    "LB4": "the superior lingular segment",
    "LB5": "the inferior lingular segment",
    "LB6": "the superior segment of the left lower lobe",
    "LB8": "the anteromedial basal segment of the left lower lobe",
    "LB9": "the lateral basal segment of the left lower lobe",
    "LB10": "the posterior basal segment of the left lower lobe",
}


# ---------------------------------------------------------------------------
# Core skills
# ---------------------------------------------------------------------------


def curriculum_skill(
    *,
    curriculum: CurriculumEngine,
    state: Dict[str, Any],
    reached_set: Set[str],
    current_airway: str,
    target_hint: str,
) -> Dict[str, Any]:
    return curriculum.make_plan(
        current_airway=current_airway,
        reached=reached_set,
        student_question=str(state.get("student_question") or "").strip(),
        requested_next_airway=target_hint,
        is_back=bool(state.get("backtracking", False)),
        stagnating=(not bool(state.get("is_target_visible", False))),
    )


def directional_skill(
    *,
    builder: DirectionalHintBuilder,
    m_jointsVelRel: Sequence[float],
    event_packet: Dict[str, Any],
) -> Dict[str, Any]:
    hint = builder.build(
        m_jointsVelRel,
        event_flag=event_packet.get("flag") if isinstance(event_packet, dict) else None,
    )
    return hint.to_dict()


def question_router_skill(*, state: Dict[str, Any], plan: Dict[str, Any]) -> str:
    q = str(state.get("student_question") or "").strip().lower()
    if not q:
        return "none"

    if any(token in q for token in ["picture", "diagram", "image", "show me", "draw", "point to", "visual"]):
        return "visual_relevant"

    if any(token in q for token in [
        "weather", "lunch", "coffee", "weekend", "music", "joke", "tired",
        "mad at me", "how are you", "are you okay", "can we talk about"
    ]):
        return "off_task_social"

    anatomy_tokens = [
        "what is", "where are we", "where am i", "is this", "why", "how",
        "which branch", "where do i go", "what branch", "carina", "mercedes",
        "rb", "lb", "rul", "rml", "rll", "lul", "lll", "bronchus", "lobe",
    ]
    if any(token in q for token in anatomy_tokens):
        return "teaching_relevant"

    next_airway = str((plan or {}).get("next_airway") or "").lower()
    if next_airway and next_airway.lower() in q:
        return "teaching_relevant"

    return "other"


def qa_skill(
    *,
    model: Any,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    current_situation: str,
    previous_msgs: str,
    question_mode: str,
    fallback_guidance: str,
) -> str:
    question = str(state.get("student_question") or "").strip()
    if not question or question_mode == "none":
        return ""

    if _safety_risk(state):
        return ""

    raw = _llm_qa_response(
        model=model,
        question=question,
        question_mode=question_mode,
        current_situation=current_situation,
        previous_msgs=previous_msgs,
        plan=plan,
        fallback_guidance=fallback_guidance,
    )
    if raw:
        return compress_utterance(raw, max_sentences=2, max_words_per_sentence=22)

    return _deterministic_qa_response(
        state=state,
        plan=plan,
        question_mode=question_mode,
        fallback_guidance=fallback_guidance,
    )


def guidance_skill(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    directional_hint: Dict[str, Any],
) -> str:
    """
    Produce short, spoken bronchoscopy coaching.

    Design goals:
    - safe and stepwise
    - curriculum-grounded
    - anatomically specific
    - friendly but still clinical
    - one concrete action per turn, with one recognition cue when helpful
    """
    intent = _choose_primary_intent(state=state, plan=plan, directional_hint=directional_hint)
    prefix = _choose_supportive_prefix(state=state, intent=intent)
    action_line = _realize_primary_action(intent=intent, state=state, plan=plan, directional_hint=directional_hint)
    cue_line = _maybe_add_cue(intent=intent, state=state, plan=plan)

    parts: List[str] = []
    if prefix:
        parts.append(prefix)
    if action_line:
        parts.append(action_line)
    if cue_line:
        parts.append(cue_line)

    text = " ".join(parts).strip()
    if text:
        return compress_utterance(text)

    fallback = _fallback_guidance(plan=plan)
    return compress_utterance(fallback)


def statistics_skill(
    *,
    current_situation: str,
    current_airway: str,
    next_airway: str,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    text = str(current_situation or "").lower()
    mode = str((plan or {}).get("mode", "")).lower()
    cue = _recognition_cue(plan)
    target_label = _airway_label(next_airway or str((plan or {}).get("next_airway") or ""))

    if mode == "backtrack":
        return {
            "trend": "stable",
            "likely_issue": "lost orientation or unsafe advance",
            "coach_focus_next": "withdraw to the carina and re-center",
            "teaching_point": "Reset orientation before advancing again.",
            "notes": f"current={current_airway or 'unknown'}; target={next_airway or 'unknown'}; mode=backtrack.",
        }
    if mode in {"reorient", "locate"} or "not visible" in text:
        return {
            "trend": "stable",
            "likely_issue": "target lumen not yet visualized",
            "coach_focus_next": f"keep centered and identify {target_label or 'the target opening'}",
            "teaching_point": (f"Use the landmark cue: {cue}." if cue else "Do not advance until the target lumen is clearly identified."),
            "notes": f"current={current_airway or 'unknown'}; target={next_airway or 'unknown'}; mode={mode or 'locate'}.",
        }
    return {
        "trend": "stable",
        "likely_issue": "normal navigation",
        "coach_focus_next": "advance with the lumen centered",
        "teaching_point": (f"Confirm {cue} before committing forward." if cue else "Advance only with a centered view."),
        "notes": f"current={current_airway or 'unknown'}; target={next_airway or 'unknown'}; mode={mode or 'advance'}.",
    }


# ---------------------------------------------------------------------------
# Q&A helpers
# ---------------------------------------------------------------------------


def _safety_risk(state: Dict[str, Any]) -> bool:
    return bool(state.get("wall_contact_risk", False)) or bool(state.get("need_recenter", False)) or bool(state.get("drift_detected", False))


def _llm_qa_response(
    *,
    model: Any,
    question: str,
    question_mode: str,
    current_situation: str,
    previous_msgs: str,
    plan: Dict[str, Any],
    fallback_guidance: str,
) -> str:
    if getattr(model, "is_fallback_backend", False):
        return ""

    style = {
        "off_task_social": "Respond lightly and a little humorously in the first sentence, then redirect back to bronchoscopy.",
        "visual_relevant": "Answer briefly and concretely, then give the next bronchoscopy step.",
        "teaching_relevant": "Answer briefly and educationally, then give the next bronchoscopy step.",
        "other": "Acknowledge briefly, then redirect back to bronchoscopy.",
    }.get(question_mode, "Answer briefly, then return to bronchoscopy.")

    plan_json = json.dumps(plan, ensure_ascii=True)
    prompt = (
        "ROLE: Bronchoscopy teaching coach with brief Q&A support\n"
        "TASK: Respond to the student question briefly, then return to the bronchoscopy task.\n\n"
        "RULES:\n"
        "1. Keep to one or two short spoken sentences.\n"
        "2. Stay consistent with PLAN_JSON for anatomy and the next step.\n"
        "3. Do not invent anatomy or unsafe steps.\n"
        "4. End with a concrete bronchoscopy instruction aligned with the plan.\n"
        "5. If the question is off-task, do not continue the off-task topic.\n"
        f"6. {style}\n\n"
        f"PLAN_JSON:\n{plan_json}\n\n"
        f"CURRENT_SITUATION:\n{current_situation.strip()}\n\n"
        f"PREVIOUS_MSGS:\n{previous_msgs.strip()}\n\n"
        f"STUDENT_QUESTION:\n{question.strip()}\n\n"
        f"FALLBACK_GUIDANCE:\n{fallback_guidance.strip()}\n\n"
        "ACTION:\nReturn the spoken answer now."
    )

    raw = _call_model_text(model, prompt)
    return raw.strip()


def _deterministic_qa_response(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    question_mode: str,
    fallback_guidance: str,
) -> str:
    question = str(state.get("student_question") or "").strip().lower()
    target = _airway_label(str((plan or {}).get("next_airway") or ""))
    cue = _recognition_cue(plan)
    fallback = fallback_guidance or _fallback_guidance(plan=plan)

    if question_mode == "off_task_social":
        return compress_utterance(f"Tempting, but let’s finish this airway first. {fallback}")

    if question_mode == "visual_relevant":
        if target and cue:
            return compress_utterance(f"Use {cue} to orient yourself to {target}. {fallback}")
        if target:
            return compress_utterance(f"Orient to {target}. {fallback}")
        return compress_utterance(f"Stay with the current airway view. {fallback}")

    if question_mode == "teaching_relevant":
        if "carina" in question:
            return compress_utterance(f"The carina is the main bifurcation where the trachea splits. {fallback}")
        if "mercedes" in question:
            return compress_utterance(f"The Mercedes sign helps confirm the right upper lobe trifurcation. {fallback}")
        if "why" in question and ("re-center" in question or "recenter" in question or "center" in question):
            return compress_utterance(f"Re-centering gives you a safer view and better control. {fallback}")
        if "where" in question or "which branch" in question or "what branch" in question or "where do i go" in question:
            if target:
                return compress_utterance(f"You are working toward {target}. {fallback}")
        if target:
            return compress_utterance(f"Keep the target as {target}. {fallback}")
        return compress_utterance(fallback)

    return compress_utterance(f"Let’s stay with the bronchoscopy. {fallback}")


def _call_model_text(model: Any, prompt: str) -> str:
    def _extract(resp: Any) -> str:
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
        for attr in ("text", "content"):
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
        return ""

    try:
        if hasattr(model, "generate"):
            text = _extract(model.generate(prompt))
            if text:
                return text
    except Exception:
        pass

    if callable(model):
        for messages in (
            [{"role": "user", "content": prompt}],
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            prompt,
        ):
            try:
                text = _extract(model(messages))
                if text:
                    return text
            except Exception:
                continue
    return ""


# ---------------------------------------------------------------------------
# Guidance realization helpers
# ---------------------------------------------------------------------------


def _choose_primary_intent(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    directional_hint: Dict[str, Any],
) -> str:
    mode = str((plan or {}).get("mode", "")).lower()
    primary_action = str((directional_hint or {}).get("primary_action", "")).lower()
    event_flag = str(state.get("event_flag") or (state.get("event_packet") or {}).get("flag") or "").lower()

    if bool(state.get("backtracking", False)) or mode == "backtrack":
        return "pull_back"
    if bool(state.get("need_recenter", False)):
        return "recenter"
    if "collision" in event_flag or "wall" in event_flag or bool(state.get("wall_contact_risk", False)):
        return "pull_back"
    if bool(state.get("drift_detected", False)):
        return "hold"
    if bool(state.get("just_reached", False)):
        return "confirm_view"

    direction = _extract_directional_side(plan=plan, directional_hint=directional_hint)
    if primary_action:
        mapped = _map_action_to_intent(primary_action)
        if mapped in {"turn_left", "turn_right", "advance", "pull_back", "hold", "recenter"}:
            return mapped
    if direction == "right":
        return "turn_right"
    if direction == "left":
        return "turn_left"

    if bool(state.get("is_target_visible", False)):
        return "confirm_view"
    if mode in {"reorient", "locate"}:
        return "identify_target"
    if mode == "advance":
        return "advance"
    if bool(state.get("is_centered", False)) or bool(state.get("is_stable", False)):
        return "advance"
    return "hold"


def _choose_supportive_prefix(*, state: Dict[str, Any], intent: str) -> str:
    if intent in {"pull_back", "recenter"}:
        return "Pause there."
    if intent == "hold" and bool(state.get("drift_detected", False)):
        return "Easy."
    if bool(state.get("just_reached", False)):
        return "Good."
    if bool(state.get("is_target_visible", False)):
        return "Good."
    if bool(state.get("is_centered", False)) and bool(state.get("is_stable", False)):
        return "Nice."
    return ""


def _realize_primary_action(
    *,
    intent: str,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    directional_hint: Dict[str, Any],
) -> str:
    target = _airway_label(str((plan or {}).get("next_airway") or ""))
    cue = _recognition_cue(plan)

    if intent == "recenter":
        return "Re-center before advancing."
    if intent == "hold":
        if cue:
            return f"Hold steady and keep {cue} in view."
        if target:
            return f"Hold steady and keep the view centered toward {target}."
        return "Hold steady and keep the lumen centered."
    if intent == "pull_back":
        if bool(state.get("backtracking", False)) or str((plan or {}).get("mode", "")).lower() == "backtrack":
            return "Pull back gently toward the carina."
        return "Pull back a little and re-center."
    if intent == "turn_right":
        return _build_navigation_action(
            directional_hint=directional_hint,
            fallback="Ease right a little",
            target=target,
            cue=cue,
        )
    if intent == "turn_left":
        return _build_navigation_action(
            directional_hint=directional_hint,
            fallback="Ease left a little",
            target=target,
            cue=cue,
        )
    if intent == "advance":
        return _build_navigation_action(
            directional_hint=directional_hint,
            fallback="Ease forward slowly",
            target=target,
            cue=cue,
        )
    if intent == "confirm_view":
        if target:
            return f"Keep {target} centered."
        if cue:
            return f"Keep {cue} centered in view."
        return "Keep that branch centered."
    if intent == "identify_target":
        if target and cue:
            return f"Stay centered and identify {target} by {cue}."
        if target:
            return f"Stay centered and identify {target}."
        if cue:
            return f"Stay centered and look for {cue}."
        return "Stay centered and identify the target lumen."

    first_step = _first_coachable_micro_step(plan.get("micro_steps") or [])
    if first_step:
        return _de_mechanize_step(first_step, target=target, cue=cue)

    return _fallback_action_from_plan(plan)


def _maybe_add_cue(*, intent: str, state: Dict[str, Any], plan: Dict[str, Any]) -> str:
    if intent in {"pull_back", "recenter"}:
        return ""

    cue = _recognition_cue(plan)
    target = _airway_label(str((plan or {}).get("next_airway") or ""))
    visible = bool(state.get("is_target_visible", False))

    if intent == "hold" and cue and not visible:
        return f"Do not advance until you see {cue}."
    if intent in {"turn_left", "turn_right", "advance", "identify_target"} and cue and not visible:
        if target:
            return f"Look for {cue} to confirm {target}."
        return f"Look for {cue} before advancing."
    if intent == "confirm_view" and target and cue:
        return f"This confirms {target}."
    return ""


# ---------------------------------------------------------------------------
# Low-level text helpers
# ---------------------------------------------------------------------------


def _map_action_to_intent(action: str) -> str:
    low = str(action or "").lower()
    if not low:
        return ""
    if "counter-clockwise" in low or "left" in low:
        return "turn_left"
    if "clockwise" in low or "right" in low:
        return "turn_right"
    if any(token in low for token in ["push", "advance", "forward"]):
        return "advance"
    if any(token in low for token in ["pull back", "withdraw", "back out", "back"]):
        return "pull_back"
    if any(token in low for token in ["center", "re-center", "recent"]):
        return "recenter"
    if any(token in low for token in ["hold", "steady", "pause"]):
        return "hold"
    return ""


def _extract_directional_side(*, plan: Dict[str, Any], directional_hint: Dict[str, Any]) -> str:
    for source in (directional_hint or {}, (plan or {}).get("directional_hint") or {}):
        for key in ("direction", "side", "turn"):
            value = str(source.get(key) or "").strip().lower()
            if value in {"left", "right"}:
                return value
        for key in ("primary_action", "rotate_text", "bend_text", "translate_text"):
            value = str(source.get(key) or "").strip().lower()
            if "left" in value or "counter-clockwise" in value:
                return "left"
            if "right" in value or "clockwise" in value:
                return "right"
    return ""


def _build_navigation_action(
    *,
    directional_hint: Dict[str, Any],
    fallback: str,
    target: str,
    cue: str,
) -> str:
    surface = _best_directional_surface(
        directional_hint,
        preferred_keys=("rotate_text", "bend_text", "translate_text", "primary_action", "secondary_action"),
        fallback=fallback,
    )
    base = _strip_terminal_period(surface)
    if target:
        return f"{base} toward {target}."
    if cue:
        return f"{base} and keep the lumen centered."
    return f"{base}."


def _best_directional_surface(
    directional_hint: Dict[str, Any],
    *,
    preferred_keys: Sequence[str],
    fallback: str,
) -> str:
    for key in preferred_keys:
        raw = str((directional_hint or {}).get(key) or "").strip()
        if not raw:
            continue
        sentence = _surface_directional_text(raw)
        if sentence:
            return sentence
    return fallback if fallback.endswith(".") else f"{fallback}."


def _surface_directional_text(text: str) -> str:
    cleaned = sanitize_micro_step(text)
    cleaned = _strip_check_clause(cleaned)
    cleaned = _normalize_step_style(cleaned)
    cleaned = _remove_multi_action_tail(cleaned)
    cleaned = _de_mechanize_step(cleaned)
    return cleaned


def _de_mechanize_step(text: str, *, target: str = "", cue: str = "") -> str:
    cleaned = normalize_landmark_language(str(text or "").strip())
    if not cleaned:
        return ""

    replacements = [
        (r"\bkeep the lumen centered\b", "keep the lumen centered"),
        (r"\bmove slowly\b", "ease forward slowly"),
        (r"\badvance into\b", "ease into"),
        (r"\badvance toward\b", "ease toward"),
        (r"\brotate clockwise slightly\b", "ease right a little"),
        (r"\brotate counter-clockwise slightly\b", "ease left a little"),
        (r"\brotate clockwise\b", "ease right a little"),
        (r"\brotate counter-clockwise\b", "ease left a little"),
        (r"\bpush forward the scope\b", "ease forward slowly"),
        (r"\bpull back the scope\b", "pull back a little"),
        (r"\bhold steady at the view\b", "hold steady"),
        (r"\bhold steady at\b", "hold steady at"),
    ]
    out = cleaned
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.I)

    if cue:
        out = re.sub(r"\bfind the mercedes sign at the right upper lobe\b", f"look for {cue}", out, flags=re.I)

    out = re.sub(r"\b(?:while|and then|then|before)\b.*$", "", out, flags=re.I)
    out = re.sub(r"\s+", " ", out).strip(" ,;:")
    if out and out[-1] not in ".!?":
        out += "."
    return out[:1].upper() + out[1:] if out else ""


def _remove_multi_action_tail(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    low = cleaned.lower()
    split_markers = [" and ", " while ", ";", ", then ", " then "]
    for marker in split_markers:
        idx = low.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx].strip(" ,;:")
            break
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _airway_label(airway: str) -> str:
    key = str(airway or "").strip().upper()
    if not key:
        return ""
    if key in _AIRWAY_LABELS:
        return _AIRWAY_LABELS[key]
    normalized = normalize_landmark_language(key)
    normalized = re.sub(r"[_\-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    if not normalized:
        return ""
    if normalized.startswith("the "):
        return normalized
    return f"the {normalized}"


def _strip_terminal_period(text: str) -> str:
    return str(text or "").strip().rstrip(".!?").strip()


def _fallback_action_from_plan(plan: Dict[str, Any]) -> str:
    target = _airway_label(str((plan or {}).get("next_airway") or ""))
    cue = _recognition_cue(plan)
    if target and cue:
        return f"Stay centered and ease toward {target}."
    if target:
        return f"Stay centered and ease toward {target}."
    if cue:
        return f"Stay centered and look for {cue}."
    return "Keep the lumen centered and advance slowly."


def _fallback_guidance(*, plan: Dict[str, Any]) -> str:
    target = _airway_label(str((plan or {}).get("next_airway") or ""))
    cue = _recognition_cue(plan)
    if target and cue:
        return f"Hold steady and ease toward {target}. Look for {cue}."
    if target:
        return f"Hold steady and ease toward {target}."
    if cue:
        return f"Hold steady. Look for {cue}."
    return "Hold steady and move slowly."


def _recognition_cue(plan: Dict[str, Any]) -> str:
    cue = normalize_landmark_language(str((plan or {}).get("recognition_cue", "")).strip())
    cue = re.sub(r"\b(?:at|toward|into)\s+(?:the\s+)?(?:right|left)\s+(?:upper|lower|middle)\s+lobe\b", "", cue, flags=re.I)
    cue = re.sub(r"\s+", " ", cue).strip(" ,;:")
    return cue


# ---------------------------------------------------------------------------
# Existing helpers retained for compatibility
# ---------------------------------------------------------------------------


def _first_coachable_micro_step(micro_steps: Sequence[Any]) -> str:
    for step in micro_steps:
        text = sanitize_micro_step(str(step))
        text = _strip_check_clause(text)
        text = _normalize_step_style(text)
        if text:
            return text
    return ""


def _strip_check_clause(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    marker = lowered.find(" check:")
    if marker != -1:
        cleaned = cleaned[:marker].strip(" .;:")
    return cleaned


def _normalize_step_style(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("Action:", "").replace("action:", "").strip()
    if cleaned and not cleaned.endswith("."):
        cleaned += "."
    return cleaned


def _coachify(action: str) -> str:
    replacements = {
        "tilt up the knob": "Tilt the knob up.",
        "tilt down the knob": "Tilt the knob down.",
        "rotate clockwise": "Rotate clockwise slightly.",
        "rotate counter-clockwise": "Rotate counter-clockwise slightly.",
        "push forward the scope": "Advance slowly.",
        "pull back the scope": "Pull back slowly.",
    }
    return replacements.get(action, action.strip().capitalize() + ".")



# ---------------------------------------------------------------------------
# Frame-based content + constrained realization
# ---------------------------------------------------------------------------


def guidance_frame_skill(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    directional_hint: Dict[str, Any],
) -> Dict[str, Any]:
    intent = _choose_primary_intent(state=state, plan=plan, directional_hint=directional_hint)
    prefix = _choose_supportive_prefix(state=state, intent=intent)
    action_line = _realize_primary_action(intent=intent, state=state, plan=plan, directional_hint=directional_hint)
    cue_line = _maybe_add_cue(intent=intent, state=state, plan=plan)
    target_airway = str((plan or {}).get("next_airway") or "").strip().upper()
    target_label = _airway_label(target_airway)
    cue = _recognition_cue(plan)
    safety_mode = "safety" if _safety_risk(state) or intent in {"pull_back", "recenter"} else "normal"
    base = _join_lines(prefix, action_line, cue_line)
    if not base:
        base = _fallback_guidance(plan=plan)
    return {
        "mode": "guidance",
        "question_mode": "none",
        "safety_mode": safety_mode,
        "intent": intent,
        "target_airway": target_airway,
        "target_label": target_label,
        "recognition_cue": cue,
        "prefix": prefix,
        "action_line": action_line,
        "cue_line": cue_line,
        "base_utterance": compress_utterance(base, max_sentences=2, max_words_per_sentence=22),
    }


def qa_frame_skill(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    question_mode: str,
    fallback_guidance: str,
) -> Dict[str, Any]:
    question = str(state.get("student_question") or "").strip()
    target_airway = str((plan or {}).get("next_airway") or "").strip().upper()
    target_label = _airway_label(target_airway)
    cue = _recognition_cue(plan)
    fallback = fallback_guidance or _fallback_guidance(plan=plan)
    answer_core = ""
    if question_mode == "off_task_social":
        answer_core = "Tempting, but let’s finish this airway first."
    elif question_mode == "visual_relevant":
        if target_label and cue:
            answer_core = f"Use {cue} to orient to {target_label}."
        elif target_label:
            answer_core = f"Orient to {target_label}."
        else:
            answer_core = "Stay with the current airway view."
    elif question_mode == "teaching_relevant":
        q = question.lower()
        if "carina" in q:
            answer_core = "The carina is the main bifurcation where the trachea splits."
        elif "mercedes" in q:
            answer_core = "The Mercedes sign helps confirm the right upper lobe trifurcation."
        elif "why" in q and ("re-center" in q or "recenter" in q or "center" in q):
            answer_core = "Re-centering gives you a safer view and better control."
        elif ("where" in q or "which branch" in q or "what branch" in q or "where do i go" in q) and target_label:
            answer_core = f"You are working toward {target_label}."
        elif target_label:
            answer_core = f"Keep the target as {target_label}."
        else:
            answer_core = "Let’s stay with the bronchoscopy anatomy."
    else:
        answer_core = "Let’s stay with the bronchoscopy."
    base = _join_lines(answer_core, fallback)
    return {
        "mode": "qa",
        "question_mode": question_mode,
        "safety_mode": "normal",
        "intent": "qa",
        "target_airway": target_airway,
        "target_label": target_label,
        "recognition_cue": cue,
        "question": question,
        "answer_core": answer_core,
        "next_step": fallback,
        "base_utterance": compress_utterance(base, max_sentences=2, max_words_per_sentence=22),
    }


def realize_response_skill(
    *,
    model: Any,
    frame: Dict[str, Any],
    current_situation: str,
    previous_msgs: str,
) -> str:
    base = str((frame or {}).get("base_utterance") or "").strip()
    if not base:
        return ""
    if getattr(model, "is_fallback_backend", False):
        return base
    if str((frame or {}).get("safety_mode") or "").lower() == "safety":
        return base
    prompt = _build_realization_prompt(frame=frame, current_situation=current_situation, previous_msgs=previous_msgs)
    raw = _call_model_text(model, prompt).strip()
    if not raw:
        return base
    return compress_utterance(raw, max_sentences=2, max_words_per_sentence=22)


def frame_to_text(frame: Dict[str, Any]) -> str:
    return compress_utterance(str((frame or {}).get("base_utterance") or "").strip(), max_sentences=2, max_words_per_sentence=22)


def _join_lines(*parts: str) -> str:
    clean = [str(p).strip() for p in parts if str(p).strip()]
    return " ".join(clean).strip()


def _build_realization_prompt(*, frame: Dict[str, Any], current_situation: str, previous_msgs: str) -> str:
    frame_json = json.dumps(frame, ensure_ascii=True)
    mode = str((frame or {}).get("mode") or "guidance")
    extra_style = (
        "You may be light and a little humorous in the first sentence, but do not continue the off-task topic."
        if str((frame or {}).get("question_mode") or "") == "off_task_social"
        else "Use calm, friendly bedside teaching language."
    )
    return (
        "ROLE: Bronchoscopy bedside coach\n"
        "TASK: Rewrite the deterministic coaching frame into natural spoken guidance.\n\n"
        "HARD RULES:\n"
        "1. Preserve the action, target airway, anatomy, and safety priority exactly as given.\n"
        "2. Do not add new anatomy, new targets, or extra procedural steps.\n"
        "3. Keep the response to one or two short spoken sentences.\n"
        "4. Keep it friendly, clear, and memorable, but still clinically grounded.\n"
        "5. If the frame includes a question response, answer briefly and then return to the bronchoscopy step.\n"
        f"6. {extra_style}\n\n"
        f"FRAME_JSON:\n{frame_json}\n\n"
        f"CURRENT_SITUATION:\n{current_situation.strip()}\n\n"
        f"PREVIOUS_MSGS:\n{previous_msgs.strip()}\n\n"
        f"BASE_UTTERANCE:\n{str((frame or {}).get('base_utterance') or '').strip()}\n\n"
        f"MODE: {mode}\n\n"
        "ACTION:\nReturn only the spoken response."
    )
