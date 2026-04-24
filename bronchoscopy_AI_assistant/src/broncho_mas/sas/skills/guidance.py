from __future__ import annotations

import re
from typing import Any, Dict, Sequence

from ...shared.utterance_postprocess import (
    normalize_landmark_language,
    sanitize_micro_step,
    sentence_split,
)
from .base import BaseSkill, SkillResult
from .realization import _build_frame, _join_lines, deterministic_frame_text, realize_frame_response
from .utterance_helpers import (
    _airway_token,
    _arrival_or_route_override,
    _from_carina_route_phrase,
    _is_curriculum_airway,
    _look_for_target_phrase,
    _route_phrase,
    _with_target_location_cue,
    safety_risk,
)

def _llm_trigger_active(state: Dict[str, Any]) -> bool:
    event_packet = state.get("event_packet") if isinstance(state.get("event_packet"), dict) else {}
    return bool(
        state.get("need_llm", False)
        or state.get("llm_trigger_flag", False)
        or event_packet.get("llm_trigger_flag", False)
        or str(state.get("soft_prompt") or "").strip()
        or str(state.get("llm_reason") or "").strip()
        or str(event_packet.get("soft_prompt") or "").strip()
        or str(event_packet.get("reason") or "").strip()
    )


# ---------------------------------------------------------------------------
# Guidance intent selection
# ---------------------------------------------------------------------------


def _choose_primary_intent(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    directional_hint: Dict[str, Any],
) -> str:
    mode = str((plan or {}).get("mode", "")).lower()
    primary_action = str((directional_hint or {}).get("primary_action", "")).lower()
    event_packet = state.get("event_packet") if isinstance(state.get("event_packet"), dict) else {}
    event_flag_value = state.get("event_flag", event_packet.get("flag"))
    event_reason = str(state.get("llm_reason") or event_packet.get("reason") or "").lower()

    if mode == "backtrack":
        return "pull_back"
    if bool(state.get("need_recenter", False)):
        return "recenter"
    if "collision" in event_reason or "wall" in event_reason or bool(state.get("wall_contact_risk", False)):
        return "pull_back"
    if isinstance(event_flag_value, (int, float)) and int(event_flag_value) > 0 and bool(state.get("wall_contact_risk", False)):
        return "pull_back"
    if bool(state.get("drift_detected", False)):
        return "hold"
    if bool(state.get("just_reached", False)):
        return "confirm_view"

    direction = _extract_directional_side(plan=plan, directional_hint=directional_hint)
    if primary_action:
        mapped = _map_action_to_intent(primary_action)
        if mapped in {"bend_up", "bend_down", "turn_left", "turn_right", "advance", "pull_back", "hold", "recenter"}:
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
    current_airway = str(state.get("current_airway") or "").strip().upper()
    if bool(state.get("just_reached", False)):
        return ""
    if bool(state.get("is_target_visible", False)) and not _is_curriculum_airway(current_airway):
        return "Good."
    if bool(state.get("is_centered", False)) and bool(state.get("is_stable", False)):
        return "Nice."
    return ""


def _merge_prefix_with_action(prefix: str, action_line: str) -> str:
    pre = str(prefix or "").strip()
    action = str(action_line or "").strip()
    if not pre:
        return action
    if not action:
        return pre
    token = pre.rstrip(".").strip()
    if token.lower() in {"good", "nice", "easy", "okay", "ok"}:
        if len(sentence_split(action)) <= 1:
            return f"{token}. {action}"
        first = action[:1].lower() + action[1:] if action[:1].isupper() else action
        return f"{token}, {first}"
    return f"{pre} {action}".strip()


def _realize_primary_action(
    *,
    intent: str,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    directional_hint: Dict[str, Any],
) -> str:
    current = _airway_token(str(state.get("current_airway") or ""))
    target = _airway_token(str((plan or {}).get("next_airway") or ""))
    cue = _recognition_cue(plan)

    override = _arrival_or_route_override(state=state, plan=plan)
    if override and intent in {"confirm_view", "identify_target", "advance", "pull_back", "hold", "recenter"}:
        return override

    if intent == "recenter":
        queued = _queued_directional_followup(directional_hint)
        if queued:
            return f"Re-center first, then {queued}."
        if current == "Carina" and target:
            return f"We’re back at the carina. Now move to {target}."
        return "Re-center before advancing."
    if intent == "hold":
        if cue:
            return f"Hold steady and keep {cue} in view."
        if target:
            return f"Hold steady and keep the view centered toward {target}."
        return "Hold steady and keep the lumen centered."
    if intent == "pull_back":
        if bool(state.get("backtracking", False)) or str((plan or {}).get("mode", "")).lower() == "backtrack":
            return "Pull back gently to the carina."
        return "Pull back a little and re-center."
    if intent == "bend_down":
        return _build_navigation_action(
            directional_hint=directional_hint,
            fallback="Tilt the knob down a little",
            target=target,
            cue=cue,
        )
    if intent == "bend_up":
        return _build_navigation_action(
            directional_hint=directional_hint,
            fallback="Tilt the knob up a little",
            target=target,
            cue=cue,
        )
    if intent == "turn_right":
        return _build_navigation_action(
            directional_hint=directional_hint,
            fallback="Rotate clockwise a little",
            target=target,
            cue=cue,
        )
    if intent == "turn_left":
        return _build_navigation_action(
            directional_hint=directional_hint,
            fallback="Rotate counter-clockwise a little",
            target=target,
            cue=cue,
        )
    if intent == "advance":
        route = _route_phrase(plan=plan, current_airway=current)
        if route:
            return route
        return _build_navigation_action(
            directional_hint=directional_hint,
            fallback="Ease forward slowly",
            target=target,
            cue=cue,
        )
    if intent == "confirm_view":
        if current and _is_curriculum_airway(current):
            return f"You’re at {current}. Let’s go back to the carina."
        if current == "Carina" and target:
            return f"We’re back at the carina. Now move to {target}."
        return "Hold that view steady."
    if intent == "identify_target":
        route = _route_phrase(plan=plan, current_airway=current)
        if route:
            return route
        if current == "Carina" and target:
            return f"We’re back at the carina. Now move to {target}."
        if target:
            return f"Stay centered and move toward {target}."
        if cue:
            short_cue = _shorten_recognition_cue(cue, level="short")
            return f"Stay centered and look for {short_cue}."
        return "Stay centered and find the target lumen."

    first_step = _first_coachable_micro_step(plan.get("micro_steps") or [])
    if first_step:
        return _de_mechanize_step(first_step, target=target, cue=cue)

    return _fallback_action_from_plan(plan)


def _maybe_add_cue(*, intent: str, state: Dict[str, Any], plan: Dict[str, Any]) -> str:
    if intent in {"pull_back", "recenter", "confirm_view"}:
        return ""

    cue = _recognition_cue(plan)
    target = _airway_token(str((plan or {}).get("next_airway") or ""))
    visible = bool(state.get("is_target_visible", False))

    short_cue = _shorten_recognition_cue(cue, level="short")
    medium_cue = _shorten_recognition_cue(cue, level="medium")

    if intent == "hold" and cue and not visible:
        return f"Do not advance until you see {short_cue}."
    if intent in {"turn_left", "turn_right", "advance", "identify_target"} and cue and not visible:
        if target:
            return f"Look for {medium_cue} as you move toward {target}."
        return f"Look for {short_cue} before advancing."
    return ""


def _maybe_add_technique_tip(
    *,
    intent: str,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    directional_hint: Dict[str, Any],
) -> str:
    transition_type = str((plan or {}).get("transition_type") or "").strip().lower()
    target_visible = bool(state.get("is_target_visible", False))
    rotate_text = str((directional_hint or {}).get("rotate_text") or "").strip().lower()
    primary_action = str((directional_hint or {}).get("primary_action") or "").strip().lower()
    event_packet = state.get("event_packet") if isinstance(state.get("event_packet"), dict) else {}
    event_flag = int(event_packet.get("flag") or 0)
    previous_msgs = str(state.get("previous_msgs") or "").lower()
    plane_mentions = previous_msgs.count("up-down plane")

    if plane_mentions >= 3:
        return ""

    if intent in {"turn_left", "turn_right", "identify_target"} and (
        not target_visible or transition_type in {"local_sibling", "regional_reanchor"} or event_flag in {3, 4}
    ):
        return "Rotate until the opening lines up with the tip's up-down plane, then use the knob to enter."

    if intent in {"bend_up", "bend_down"} and (rotate_text or "rotate" in primary_action):
        return "Line up the opening with rotation first; the tip then works in the up-down plane."

    if intent in {"turn_left", "turn_right"} and (rotate_text or "rotate" in primary_action):
        return "Keep a light grip on the insertion tube so the scope can rotate freely."

    return ""

# ---------------------------------------------------------------------------
# Guidance text shaping
# ---------------------------------------------------------------------------


def _queued_directional_followup(directional_hint: Dict[str, Any]) -> str:
    """Return a lower-case follow-up action suitable after a recenter command.

    Safety can stay primary, but the grounded directional action should still be
    verbalized when available.
    """
    hint = directional_hint or {}
    primary = str(hint.get("primary_action") or "").strip()
    secondary = str(hint.get("secondary_action") or "").strip()

    candidates: List[str] = []
    if primary:
        candidates.append(primary)
    if secondary and secondary.lower() != primary.lower():
        candidates.append(secondary)
    for key in ("translate_text", "rotate_text", "bend_text"):
        raw = str(hint.get(key) or "").strip()
        if raw and raw.lower() not in {c.lower() for c in candidates}:
            candidates.append(raw)

    for raw in candidates:
        mapped = _map_action_to_intent(raw)
        if mapped in {"", "recenter", "hold"}:
            continue
        coached = _coachify_directional_action(raw)
        if coached:
            clause = coached.strip()
            if clause.endswith('.'):
                clause = clause[:-1]
            return clause[:1].lower() + clause[1:]
    return ""


def _map_action_to_intent(action: str) -> str:
    low = str(action or "").lower()
    if not low:
        return ""
    if any(token in low for token in ["tilt up the knob", "tilt the knob up", "knob up", "bend up"]):
        return "bend_up"
    if any(token in low for token in ["tilt down the knob", "tilt the knob down", "knob down", "bend down"]):
        return "bend_down"
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
    paired = _compose_directional_actions(directional_hint)
    if paired:
        return paired

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


def _compose_directional_actions(directional_hint: Dict[str, Any]) -> str:
    primary = _coachify_directional_action(str((directional_hint or {}).get("primary_action") or ""))
    secondary = _coachify_directional_action(str((directional_hint or {}).get("secondary_action") or ""))

    if primary and secondary and secondary.lower() != primary.lower():
        return f"{primary} and {secondary.lower()}."
    if primary:
        return f"{primary}."
    if secondary:
        return f"{secondary}."
    return ""


def _coachify_directional_action(action: str) -> str:
    cleaned = str(action or "").strip().lower()
    mapping = {
        "tilt up the knob": "Tilt the knob up",
        "tilt down the knob": "Tilt the knob down",
        "rotate clockwise": "Rotate clockwise a little",
        "rotate counter-clockwise": "Rotate counter-clockwise a little",
        "push forward the scope": "Advance the scope slowly",
        "pull back the scope": "Pull back the scope a little",
    }
    if cleaned in mapping:
        return mapping[cleaned]
    cleaned = _strip_terminal_period(normalize_landmark_language(cleaned))
    if not cleaned:
        return ""
    return cleaned[:1].upper() + cleaned[1:]


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
        (r"\brotate clockwise slightly\b", "rotate clockwise a little"),
        (r"\brotate counter-clockwise slightly\b", "rotate counter-clockwise a little"),
        (r"\brotate clockwise\b", "rotate clockwise"),
        (r"\brotate counter-clockwise\b", "rotate counter-clockwise"),
        (r"\bpush forward the scope\b", "advance the scope slowly"),
        (r"\bpull back the scope\b", "pull back the scope a little"),
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


def _airway_token(airway: str) -> str:
    key = str(airway or "").strip().upper()
    if not key or key == "BACK":
        return ""
    return normalize_landmark_language(key)


def _strip_terminal_period(text: str) -> str:
    return str(text or "").strip().rstrip(".!?").strip()


def _is_curriculum_airway(value: str) -> bool:
    token = str(value or "").strip().upper()
    return bool(re.fullmatch(r"(?:RB\d+|LB\d(?:\+\d)?)", token))


def _is_intermediate_airway(value: str) -> bool:
    return str(value or "").strip().upper() in {"CARINA", "RMB", "LMB", "BI", "RUL", "RML", "RLL", "LUL", "LLL"}


def _session_turn_index(state: Dict[str, Any]) -> int:
    try:
        return int(state.get("session_turn_index") or 0)
    except Exception:
        return 0


def _is_destination_arrival(*, state: Dict[str, Any], current_upper: str, target_upper: str) -> bool:
    return bool(state.get("just_reached", False)) and _is_curriculum_airway(current_upper) and current_upper != ""


def _is_waypoint_arrival(*, state: Dict[str, Any], current_upper: str, target_upper: str) -> bool:
    waypoint_upper = str(state.get("waypoint_target") or "").strip().upper()
    training_upper = str(state.get("training_target") or target_upper or "").strip().upper()
    if not bool(state.get("just_reached", False)):
        return False
    if not current_upper or not _is_intermediate_airway(current_upper):
        return False
    if waypoint_upper:
        return waypoint_upper == current_upper and current_upper != training_upper
    return current_upper != training_upper


def _should_acknowledge_waypoint(*, state: Dict[str, Any], current_upper: str, cooldown_turns: int = 4) -> bool:
    if not current_upper:
        return False
    turn_index = _session_turn_index(state)
    last_waypoint = str(state.get("last_acknowledged_waypoint") or "").strip().upper()
    last_turn = int(state.get("last_acknowledged_waypoint_turn") or -10_000)
    if last_waypoint == current_upper and (turn_index - last_turn) <= cooldown_turns:
        return False
    low_prev = str(state.get("previous_msgs") or "").lower()
    if f"you're at the {current_upper.lower()}" in low_prev or f"you’re at the {current_upper.lower()}" in low_prev:
        return False
    if f"you're at {current_upper.lower()}" in low_prev or f"you’re at {current_upper.lower()}" in low_prev:
        return False
    return True


def _family_local_navigation_phrase(*, current_airway: str, target_airway: str) -> str:
    current = str(current_airway or "").strip().upper()
    target = str(target_airway or "").strip().upper()
    if not target:
        return ""

    pair_map = {
        ("RB1", "RB2"): "Back out just a touch. In the clockwise upper-lobe sweep, RB2 is next and should come up on your right.",
        ("RB2", "RB3"): "Back out just a touch. Keep the clockwise upper-lobe sweep: RB3 comes next on your left and forward.",
        ("RB4", "RB5"): "Back out just a touch. Stay in the middle-lobe pair: RB5 is the inner branch.",
        ("LB1+2", "LB3"): "Back out just a touch. In the left upper division, LB3 is the front, forward branch.",
        ("LB4", "LB5"): "Back out just a touch. In the lingula, LB5 is the lower partner.",
        ("RB6", "RB7"): "Back out just a touch. After six, the inner basal branch is RB7.",
        ("RB7", "RB8"): "Back out just a touch. After the inner basal branch, RB8 comes forward.",
        ("RB8", "RB9"): "Back out just a touch. Keep the basal sweep moving outward: RB9 is lateral.",
        ("RB9", "RB10"): "Back out just a touch. Finish the basal sweep with RB10 at the back.",
        ("LB6", "LB8"): "Back out just a touch. On the left lower side, skip framework LB7 and go forward to LB8.",
        ("LB8", "LB9"): "Back out just a touch. Keep the left basal sweep moving outward: LB9 is lateral.",
        ("LB9", "LB10"): "Back out just a touch. Finish the left basal sweep with LB10 at the back.",
    }
    if (current, target) in pair_map:
        return pair_map[(current, target)]

    target_map = {
        "RB1": "Stay local in the upper-lobe set. Start the clockwise sweep at RB1.",
        "RB2": "Back out just a touch. In the clockwise upper-lobe sweep, RB2 is next and should come up on your right.",
        "RB3": "Back out just a touch. Complete the upper-lobe clockwise sweep at RB3 on your left and forward.",
        "RB4": "Back out just a touch. Start the middle-lobe pair with RB4, the outer branch.",
        "RB5": "Back out just a touch. Finish the middle-lobe pair with RB5, the inner branch.",
        "RB6": "Back out just a touch. Six sits high before the basal sweep.",
        "RB7": "Back out just a touch. Start the right basal sweep at the inner branch, RB7.",
        "RB8": "Back out just a touch. RB8 is the front basal branch.",
        "RB9": "Back out just a touch. RB9 is the outer basal branch.",
        "RB10": "Back out just a touch. RB10 is the back basal branch.",
        "LB1+2": "Back out just a touch. In the left upper division, LB1+2 sits back and up.",
        "LB3": "Back out just a touch. In the left upper division, LB3 is the front, forward branch.",
        "LB4": "Back out just a touch. Start the lingular pair at LB4, the upper opening.",
        "LB5": "Back out just a touch. Finish the lingular pair at LB5, the lower opening.",
        "LB6": "Back out just a touch. Six sits high before the left basal sweep.",
        "LB8": "Back out just a touch. In this framework, go from six forward to LB8.",
        "LB9": "Back out just a touch. LB9 is the outer basal branch.",
        "LB10": "Back out just a touch. LB10 is the back basal branch.",
    }
    return target_map.get(target, "")


def _plan_route_steps(plan: Dict[str, Any]) -> List[str]:
    raw = (plan or {}).get("route") or []
    steps: List[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            token = _airway_token(str(item or ""))
            if token:
                steps.append(token)
    return steps


def _route_phrase(*, plan: Dict[str, Any], current_airway: str = "") -> str:
    from ...shared import get_airway_info

    steps = _plan_route_steps(plan)
    transition_type = str((plan or {}).get("transition_type") or "").strip().lower()
    target = _airway_token(str((plan or {}).get("next_airway") or ""))
    target_upper = str((plan or {}).get("next_airway") or "").strip().upper()
    reanchor_target = str((plan or {}).get("reanchor_target") or "").strip().upper()
    directional_hint = dict((plan or {}).get("directional_hint") or {})
    turn_side = _extract_directional_side(plan=plan, directional_hint=directional_hint)
    current_upper = str(current_airway or "").strip().upper()

    if transition_type == "local_sibling" and target:
        family_phrase = _family_local_navigation_phrase(current_airway=current_upper, target_airway=target_upper)
        if family_phrase:
            return family_phrase
        if turn_side in {"left", "right"}:
            return f"Withdraw a little to neutral, then turn {turn_side} into {target}."
        return f"Withdraw a little to neutral, then redirect into {target}."
    if transition_type == "regional_reanchor" and target:
        reanchor_info = get_airway_info(reanchor_target)
        look_for = _look_for_target_phrase(target)
        if reanchor_info and reanchor_target not in {"CARINA", "TRACHEA"}:
            return f"Come back slightly to the {reanchor_info.label}, then {look_for}."
        return _with_target_location_cue(f"Retrace slightly, then head toward {target}.", target)

    if not steps:
        return ""
    current = _airway_token(current_airway)
    current_upper = str(current or "").upper()
    if current and current in steps:
        idx = steps.index(current)
        if current_upper == "CARINA":
            return _from_carina_route_phrase(plan=plan, target=target)
        else:
            steps = steps[idx:]
    if not steps:
        return ""
    if len(steps) == 1:
        return f"Now move to {steps[0]}."
    if len(steps) == 2:
        return f"From here, go to {steps[-1]}."
    mids = ", then ".join(f"through the {step}" for step in steps[:-1])
    return f"From here, go {mids}, then into {steps[-1]}."


def _arrival_or_route_override(*, state: Dict[str, Any], plan: Dict[str, Any]) -> str:
    from ...shared import get_airway_info

    current = _airway_token(str(state.get("current_airway") or ""))
    current_upper = str(current or "").upper()
    target = _airway_token(str((plan or {}).get("next_airway") or ""))
    mode = str((plan or {}).get("mode") or "").lower()
    family_label = str((plan or {}).get("family_label") or "").strip()
    transition_type = str((plan or {}).get("transition_type") or "").strip().lower()
    reanchor_target = str((plan or {}).get("reanchor_target") or "").strip().upper()

    if mode == "backtrack":
        return "Pull back gently to the carina."

    if _is_destination_arrival(state=state, current_upper=current_upper, target_upper=str(target or "").upper()):
        if current_upper == "CARINA":
            if target:
                return _from_carina_route_phrase(plan=plan, target=target)
            return "We're back at the carina. Re-center here."
        if transition_type == "local_sibling" and target:
            route = _route_phrase(plan=plan, current_airway=current)
            if route:
                return f"Good, you've reached {current}. {route}"
            if family_label:
                return f"Good, you've reached {current}. Stay in the {family_label} and redirect to {target}."
            return f"Good, you've reached {current}. Redirect to {target}."
        if transition_type == "regional_reanchor" and target:
            reanchor_info = get_airway_info(reanchor_target)
            look_for = _look_for_target_phrase(target)
            if reanchor_info and reanchor_target not in {"CARINA", "TRACHEA"}:
                return f"Good, you've reached {current}. Now come back slightly to the {reanchor_info.label}, then {look_for}."
            return _with_target_location_cue(
                f"Good, you've reached {current}. Now withdraw slightly and look for {target}.",
                target,
            )
        return f"Good, you've reached {current}. Now pull back gently to the carina."

    if _is_waypoint_arrival(state=state, current_upper=current_upper, target_upper=str(target or "").upper()):
        route = _route_phrase(plan=plan, current_airway=current)
        if _should_acknowledge_waypoint(state=state, current_upper=current_upper):
            if route:
                return f"You're at the {current}. {route}"
            if target:
                return f"You're at the {current}. Continue toward {target}."
        return route or (f"Continue toward {target}." if target else "")

    if transition_type == "local_sibling" and target and _is_curriculum_airway(current_upper):
        if family_label:
            route = _route_phrase(plan=plan, current_airway=current)
            return route or f"Stay in the {family_label}. Redirect to {target}."
        return _route_phrase(plan=plan, current_airway=current) or f"Stay local and redirect to {target}."

    if current_upper == "CARINA" and target:
        return _from_carina_route_phrase(plan=plan, target=target)

    if current and _is_intermediate_airway(current_upper):
        route = _route_phrase(plan=plan, current_airway=current)
        if route:
            return _with_target_location_cue(f"You're at the {current}. {route}", target)

    if current and target and family_label and _is_curriculum_airway(current_upper) and current != target:
        if transition_type == "regional_reanchor":
            reanchor_info = get_airway_info(reanchor_target)
            if reanchor_info and reanchor_target not in {"CARINA", "TRACHEA"}:
                return _with_target_location_cue(
                    f"Retrace slightly to the {reanchor_info.label}. Then move toward {target}.",
                    target,
                )
        return _with_target_location_cue(f"Stay with the {family_label}. Next is {target}.", target)

    return ""


def _fallback_action_from_plan(plan: Dict[str, Any]) -> str:
    target = _airway_token(str((plan or {}).get("next_airway") or ""))
    cue = _recognition_cue(plan)
    if target and cue:
        return f"Stay centered and ease toward {target}."
    if target:
        return f"Stay centered and ease toward {target}."
    if cue:
        return f"Stay centered and look for {cue}."
    return "Keep the lumen centered and advance slowly."


def _fallback_guidance(*, plan: Dict[str, Any]) -> str:
    target = _airway_token(str((plan or {}).get("next_airway") or ""))
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


def _shorten_recognition_cue(cue: str, *, level: str = "short") -> str:
    text = str(cue or "").strip()
    if not text:
        return ""

    low = text.lower()

    if "carina" in low and "both main bronchi" in low:
        if level == "full":
            return "the carina with both main bronchi in view"
        if level == "medium":
            return "the carina bifurcation"
        return "the carina"

    if "carina bifurcation" in low:
        if level == "full":
            return "the carina bifurcation"
        return "the carina"

    if "mercedes sign" in low:
        if level == "full":
            return "the Mercedes sign at the right upper lobe trifurcation"
        return "the Mercedes sign"

    return text


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


# ---------------------------------------------------------------------------
# Frame construction and realization
# ---------------------------------------------------------------------------


def build_guidance_frame(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    directional_hint: Dict[str, Any],
) -> Dict[str, Any]:
    intent = _choose_primary_intent(state=state, plan=plan, directional_hint=directional_hint)
    prefix = _choose_supportive_prefix(state=state, intent=intent)
    action_line = _realize_primary_action(intent=intent, state=state, plan=plan, directional_hint=directional_hint)
    cue_line = _maybe_add_cue(intent=intent, state=state, plan=plan)
    technique_tip = _maybe_add_technique_tip(intent=intent, state=state, plan=plan, directional_hint=directional_hint)
    if technique_tip and prefix.strip().rstrip(".").lower() in {"good", "nice"}:
        prefix = ""
    target_airway = str((plan or {}).get("next_airway") or "").strip().upper()
    target_label = _airway_token(target_airway)
    cue = _recognition_cue(plan)
    safety_mode = "safety" if safety_risk(state) or intent in {"pull_back", "recenter"} else "normal"
    followup_line = technique_tip or cue_line
    base = _join_lines(_merge_prefix_with_action(prefix, action_line), followup_line)
    if not base:
        base = _fallback_guidance(plan=plan)
    return _build_frame(
        mode="guidance",
        question_mode="none",
        safety_mode=safety_mode,
        intent=intent,
        target_airway=target_airway,
        target_label=target_label,
        recognition_cue=cue,
        prefix=prefix,
        action_line=action_line,
        cue_line=followup_line,
        answer_core="",
        next_step=action_line,
        question="",
        fallback_guidance=_fallback_guidance(plan=plan),
        base_utterance=base,
    )

class GuidanceSkill(BaseSkill):
    """Main live-guidance skill."""

    name = "guidance_skill"
    default_priority = 0.8

    def should_activate(self, **kwargs: Any) -> bool:
        return True

    def _build_frame(
        self,
        *,
        state: Dict[str, Any],
        plan: Dict[str, Any],
        directional_hint: Dict[str, Any],
    ) -> Dict[str, Any]:
        return build_guidance_frame(
            state=state,
            plan=plan,
            directional_hint=directional_hint,
        )

    def _should_realize(self, *, state: Dict[str, Any], model: Any, frame: Dict[str, Any]) -> bool:
        base = str((frame or {}).get("base_utterance") or "").strip()
        if not base:
            return False
        if getattr(model, "is_fallback_backend", False):
            return False
        if str((frame or {}).get("safety_mode") or "").lower() == "safety":
            return False
        return _llm_trigger_active(state)

    def _realize(
        self,
        *,
        state: Dict[str, Any],
        model: Any,
        frame: Dict[str, Any],
        current_situation: str,
        previous_msgs: str,
        policy_text: str = "",
    ) -> str:
        if not self._should_realize(state=state, model=model, frame=frame):
            return ""
        return realize_frame_response(
            model=model,
            state=state,
            frame=frame,
            current_situation=current_situation,
            previous_msgs=previous_msgs,
            policy_text=policy_text,
        )

    def execute(
        self,
        *,
        state: Dict[str, Any],
        plan: Dict[str, Any],
        directional_hint: Dict[str, Any],
        model: Any = None,
        current_situation: str = "",
        previous_msgs: str = "",
        policy_text: str = "",
    ) -> SkillResult:
        frame = self._build_frame(
            state=state,
            plan=plan,
            directional_hint=directional_hint,
        )
        deterministic = deterministic_frame_text(frame)
        wants_realization = bool(model is not None and self._should_realize(state=state, model=model, frame=frame))
        priority = 1.0 if safety_risk(state) else self.default_priority
        return SkillResult(
            skill=self.name,
            active=True,
            priority=priority,
            reason="live guidance frame selected",
            data={
                "frame": frame,
                "frame_mode": str((frame or {}).get("mode") or "guidance"),
                "deterministic_utterance": deterministic,
                "realized": False,
            },
            utterance=deterministic,
            frame=frame,
            deterministic_text=deterministic,
            wants_realization=wants_realization,
            constraints={"max_sentences": 2, "mode": "guidance"},
            debug_reason="skill prepared guidance frame; manager must decide whether to realize it",
        )


def guidance_skill(**kwargs: Any) -> Dict[str, Any]:
    return GuidanceSkill().execute(**kwargs).to_dict()
