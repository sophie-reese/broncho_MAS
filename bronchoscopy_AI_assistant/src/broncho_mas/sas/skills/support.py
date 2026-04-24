from __future__ import annotations

from typing import Any

from .base import BaseSkill, Plan, SkillResult, State
from .utterance_helpers import _airway_token, _is_destination_arrival, _is_waypoint_arrival

def _student_recovery_message(student_text: str) -> tuple[str, str]:
    low = str(student_text or "").lower()
    panic_markers = (
        "panic", "panicking", "scared", "afraid", "stress", "stressed",
        "overwhelmed", "freaking out", "can't do this", "cannot do this",
    )
    lost_markers = (
        "lost", "confused", "don't know where", "do not know where",
        "where am i", "where i am", "not sure where", "help",
    )
    if any(marker in low for marker in panic_markers):
        return (
            "distress_recovery",
            "Pause, hold still, and re-center the lumen. You are okay.",
        )
    if any(marker in low for marker in lost_markers):
        return (
            "lost_orientation",
            "Pause, hold still, and return to the carina to reset. Re-center the lumen first.",
        )
    return "", ""


def _recent_reset_count(previous_msgs: str) -> int:
    low = str(previous_msgs or "").lower()
    markers = ("pull back", "back out", "withdraw", "re-center", "recenter", "reset")
    return sum(low.count(marker) for marker in markers)


def _recent_support_count(previous_msgs: str) -> int:
    low = str(previous_msgs or "").lower().replace("’", "'")
    markers = (
        "good.",
        "nice and steady. keep going.",
        "easy. re-center first.",
    )
    return sum(low.count(marker) for marker in markers)


def _select_support_message(*, state: State, plan: Plan, previous_msgs: str) -> tuple[str, str, str, float]:
    target_label = _airway_token(str((plan or {}).get("next_airway") or ""))
    current_airway = _airway_token(str(state.get("current_airway") or ""))
    current_upper = str(current_airway or "").upper()
    event_reason = str(state.get("llm_reason") or "").lower()
    low_prev = str(previous_msgs or "").lower()
    student_text = str(state.get("student_question") or "").strip()

    recovery_mode, recovery_message = _student_recovery_message(student_text)
    if recovery_message:
        return (
            recovery_mode,
            recovery_message,
            "student expressed distress or lost orientation",
            1.08,
        )

    if bool(student_text):
        return "none", "", "support suppressed while handling student question", 0.0

    if bool(state.get("wall_contact_risk", False)) or bool(state.get("need_recenter", False)) or bool(state.get("drift_detected", False)):
        return (
            "stabilize",
            "Easy. Re-center first.",
            "runtime safety stabilization selected",
            1.0,
        )

    if _is_destination_arrival(state=state, current_upper=current_upper, target_upper=str(target_label or "").upper()):
        if current_airway:
            utterance = f"Good, you've reached {current_airway}."
        elif target_label:
            utterance = f"Good, you've reached {target_label}."
        else:
            utterance = "Good."
        return (
            "arrival_feedback",
            utterance,
            "destination arrival detected; brief acknowledgement selected",
            0.52,
        )

    if _is_waypoint_arrival(state=state, current_upper=current_upper, target_upper=str(target_label or "").upper()):
        return "none", "", "waypoint arrival handled by main guidance without praise", 0.0

    repeated_turn = sum(low_prev.count(marker) for marker in ("rotate_clockwise", "rotate_counter_clockwise", "clockwise", "counter-clockwise"))
    repeated_search = low_prev.count("toward") + low_prev.count("move to") + low_prev.count("go to")
    recent_support = _recent_support_count(previous_msgs)

    if bool(state.get("backtracking", False)) and _recent_reset_count(previous_msgs) >= 2 and recent_support == 0:
        return (
            "support_after_repeat_reset",
            "Re-center and try again from the carina.",
            "repeated backtracking reset detected",
            0.82,
        )

    if (
        recent_support == 0
        and ("spent more than usual time" in event_reason or "not visible" in event_reason or not bool(state.get("is_target_visible", False)))
        and (repeated_turn >= 2 or repeated_search >= 2)
    ):
        if target_label:
            return (
                "support_repeated_probing",
                f"Re-center first, then try again toward {target_label}.",
                "repeated probing detected without progress",
                0.78,
            )
        return (
            "support_repeated_probing",
            "Re-center first, then try again.",
            "repeated probing detected without progress",
            0.78,
        )

    if bool(state.get("is_centered", False)) and bool(state.get("is_stable", False)) and not bool(state.get("is_target_visible", False)):
        return (
            "encourage_progress",
            "Nice and steady. Keep going.",
            "stable but not yet at target; encourage continued progress",
            0.4,
        )

    return "none", "", "no support intervention selected", 0.0


class SupportSkill(BaseSkill):
    """Short emotional support and coaching-tone skill."""

    name = "support_skill"
    default_priority = 0.6

    def execute(
        self,
        *,
        state: State,
        plan: Plan,
        previous_msgs: str = "",
    ) -> SkillResult:
        mode, utterance, reason, priority = _select_support_message(
            state=state,
            plan=plan,
            previous_msgs=previous_msgs,
        )
        active = bool(utterance)
        return SkillResult(
            skill=self.name,
            active=active,
            priority=priority if active else 0.0,
            reason=reason,
            data={"support_mode": mode},
            utterance=utterance,
            frame={},
            deterministic_text=utterance,
            wants_realization=False,
            constraints={"max_sentences": 1, "mode": "support"},
            debug_reason="support is deterministic only",
        )


def support_skill(**kwargs: Any) -> Dict[str, Any]:
    return SupportSkill().execute(**kwargs).to_dict()
