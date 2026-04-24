from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from ...shared.utterance_postprocess import (
    has_directional_action,
    light_cleanup_ui_text,
    normalize_landmark_language,
    sentence_split,
    trim_line_words,
)
from .base import Plan, State

def merge_teaching_with_guidance(teaching_text: str, guidance_text: str) -> str:
    teaching = str(teaching_text or "").strip()
    guidance = str(guidance_text or "").strip()

    def _guidance_priority(line: str) -> tuple[int, int]:
        low = str(line or "").strip().lower()
        if not low:
            return (0, 0)
        action_markers = (
            "keep ", "move ", "go ", "advance", "withdraw", "pull back", "redirect",
            "look for", "tilt ", "rotate ", "re-center", "recenter", "hold ", "stay ",
        )
        praise_only = {"good.", "nice.", "good", "nice"}
        if low in praise_only:
            return (0, -len(low))
        if has_directional_action(low) or any(marker in low for marker in action_markers):
            return (3, len(low))
        if any(token in low for token in ("rb", "lb", "carina", "rmb", "lmb", "bi")):
            return (2, len(low))
        return (1, len(low))

    def _lines(text: str, *, max_words: int) -> List[str]:
        out: List[str] = []
        for line in sentence_split(normalize_landmark_language(text)):
            compact = trim_line_words(line, max_words=max_words)
            if compact and compact not in out:
                out.append(compact)
        return out

    teaching_lines = _lines(teaching, max_words=30)
    guidance_lines = _lines(guidance, max_words=24)
    local_teaching_lines = [line for line in teaching_lines if line.lower().startswith("local cue:")]
    primary_teaching_lines = [line for line in teaching_lines if line not in local_teaching_lines]

    if teaching_lines and guidance_lines:
        merged: List[str] = []
        ordered_guidance = sorted(guidance_lines, key=_guidance_priority, reverse=True)
        for line in ordered_guidance[:2]:
            if line not in merged:
                merged.append(line)
        teaching_priority = primary_teaching_lines[:1] + local_teaching_lines[:1]
        for line in teaching_priority:
            if line not in merged:
                merged.append(line)
        if len(merged) < 3:
            for line in guidance_lines + primary_teaching_lines[1:] + local_teaching_lines[1:]:
                if line not in merged:
                    merged.append(line)
                    if len(merged) >= 3:
                        break
        return light_cleanup_ui_text(" ".join(merged[:4]))
    if guidance_lines:
        return light_cleanup_ui_text(" ".join(guidance_lines[:2]))
    if teaching_lines:
        return light_cleanup_ui_text(" ".join(teaching_lines[:2]))
    return ""

def safety_risk(state: State) -> bool:
    return bool(state.get("wall_contact_risk", False)) or bool(state.get("need_recenter", False)) or bool(state.get("drift_detected", False))

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


def _target_location_cue(target_airway: str) -> str:
    target = str(target_airway or "").strip().upper()
    target_map = {
        "RB1": "RB1 usually climbs apically from the right upper-lobe split.",
        "RB2": "RB2 usually sits posteriorly in the right upper-lobe split.",
        "RB3": "RB3 usually comes anteriorly from the right upper-lobe split.",
        "RB4": "RB4 is usually the lateral branch of the middle-lobe pair.",
        "RB5": "RB5 is usually the medial branch of the middle-lobe pair.",
        "RB6": "RB6 usually sits high before the basal fan.",
        "RB7": "RB7 is usually the medial basal branch; it can be variable.",
        "RB8": "RB8 is usually the anterior basal branch.",
        "RB9": "RB9 is usually the lateral basal branch.",
        "RB10": "RB10 is usually the posterior basal branch.",
        "LB1+2": "LB1+2 usually goes back and up in the left upper division.",
        "LB3": "LB3 usually comes forward in the left upper division.",
        "LB4": "LB4 is usually the superior branch of the lingular pair.",
        "LB5": "LB5 is usually the inferior branch of the lingular pair.",
        "LB6": "LB6 usually sits high before the left basal branches.",
        "LB8": "LB8 is usually the forward basal route in this framework.",
        "LB9": "LB9 is usually the lateral basal branch.",
        "LB10": "LB10 is usually the posterior basal branch.",
    }
    return target_map.get(target, "")


def _target_location_clause(target_airway: str) -> str:
    target = str(target_airway or "").strip().upper()
    target_map = {
        "RB1": "which usually climbs apically from the right upper-lobe split",
        "RB2": "which usually sits posteriorly in the right upper-lobe split",
        "RB3": "which usually comes anteriorly from the right upper-lobe split",
        "RB4": "which is usually the lateral branch of the middle-lobe pair",
        "RB5": "which is usually the medial branch of the middle-lobe pair",
        "RB6": "which usually sits high before the basal fan",
        "RB7": "which is usually the medial basal branch, though it can be variable",
        "RB8": "which is usually the anterior basal branch",
        "RB9": "which is usually the lateral basal branch",
        "RB10": "which is usually the posterior basal branch",
        "LB1+2": "which usually goes back and up in the left upper division",
        "LB3": "which usually comes forward in the left upper division",
        "LB4": "which is usually the superior branch of the lingular pair",
        "LB5": "which is usually the inferior branch of the lingular pair",
        "LB6": "which usually sits high before the left basal branches",
        "LB8": "which is usually the forward basal route in this framework",
        "LB9": "which is usually the lateral basal branch",
        "LB10": "which is usually the posterior basal branch",
    }
    return target_map.get(target, "")


def _look_for_target_phrase(target_airway: str) -> str:
    target = _airway_token(target_airway)
    if not target:
        return ""
    clause = _target_location_clause(target)
    if clause:
        return f"look for {target}, {clause}"
    return f"look for {target}"


def _with_target_location_cue(text: str, target_airway: str) -> str:
    base = str(text or "").strip()
    cue = _target_location_cue(target_airway)
    if not base or not cue:
        return base
    if cue.lower() in base.lower():
        return base
    return f"{base} {cue}"


def _plan_route_steps(plan: Dict[str, Any]) -> List[str]:
    raw = (plan or {}).get("route") or []
    steps: List[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            token = _airway_token(str(item or ""))
            if token:
                steps.append(token)
    return steps


def _route_step_label(step: str) -> str:
    from ...shared import get_airway_info

    token = _airway_token(step)
    info = get_airway_info(token)
    if info and info.label:
        label = str(info.label).strip()
        if _is_curriculum_airway(token) and label.lower() != token.lower():
            return f"{token} ({label})"
        return label
    return token


def _from_carina_route_phrase(*, plan: Dict[str, Any], target: str) -> str:
    steps = _plan_route_steps(plan)
    route = [step for step in steps if step.upper() not in {"CARINA", "TRACHEA"}]
    if not route:
        return f"We're at the carina. Re-center here."
    target_token = target or route[-1]
    route_labels = [_route_step_label(step) for step in route]
    if len(route) == 1:
        return _with_target_location_cue(f"We're at the carina. Move toward {route_labels[0]}.", target_token)
    if len(route) == 2:
        route_text = f"{route_labels[0]}, then {route_labels[1]}"
    else:
        route_text = ", then ".join(route_labels)
    return _with_target_location_cue(
        f"We're at the carina. Follow the route through {route_text}.",
        target_token,
    )


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
