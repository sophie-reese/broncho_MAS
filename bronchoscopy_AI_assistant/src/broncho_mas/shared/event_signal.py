from __future__ import annotations

from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from . import classify_anatomical_relationship, normalize_airway_code


# -----------------------------
# Constants
# -----------------------------

ANCHOR_POSITIONS = {
    "LMB", "RMB", "BI", "RUL", "RML", "RLL",
    "CARINA", "UDB", "LI", "LLL", "LUL"
}


# -----------------------------
# Small helpers
# -----------------------------

def _safe_str(x: Any) -> str:
    return str(x or "").strip()


def _safe_upper(x: Any) -> str:
    return _safe_str(x).upper()


def _bool(x: Any) -> bool:
    return bool(x)


def _plan_view(plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize curriculum/runtime plan into a lightweight stable view.
    """
    plan = plan or {}
    return {
        "mode": _safe_str(plan.get("mode")),
        "next_airway": _safe_str(plan.get("next_airway")),
        "anchor_landmark": _safe_str(plan.get("anchor_landmark")),
        "transition_type": _safe_str(plan.get("transition_type")),
        "reanchor_target": _safe_str(plan.get("reanchor_target")),
        "route": list(plan.get("route", []) or []),
        "recognition_cue": _safe_str(plan.get("recognition_cue")),
        "micro_steps": list(plan.get("micro_steps", []) or []),
        "why": _safe_str(plan.get("why")),
    }


def _route_inconsistency_flag(
    *,
    position: str,
    target: str,
    planned_next: str,
    plan_view: Dict[str, Any],
) -> bool:
    pos = normalize_airway_code(position)
    tgt = normalize_airway_code(target)
    nxt = normalize_airway_code(planned_next)
    transition_type = _safe_str(plan_view.get("transition_type")).lower()
    planned_mode = _safe_str(plan_view.get("mode")).lower()
    reanchor_target = normalize_airway_code(plan_view.get("reanchor_target"))
    route = [normalize_airway_code(x) for x in (plan_view.get("route") or []) if normalize_airway_code(x)]

    if not pos or not nxt:
        return False
    if planned_mode in {"reorient", "locate", "backtrack", "done"}:
        return False

    relationship = classify_anatomical_relationship(pos, nxt)

    if transition_type == "local_sibling":
        return relationship in {"cross_main_bronchus", "cross_side", "regional_branch_change"}

    if transition_type == "regional_reanchor":
        valid_positions = set(route + [nxt, reanchor_target])
        if pos in valid_positions:
            return False
        return relationship in {"cross_main_bronchus", "cross_side"}

    if transition_type == "global_reanchor":
        return False

    if route and pos not in set(route + [nxt]):
        return relationship in {"cross_main_bronchus", "cross_side"}

    if pos == nxt and tgt and normalize_airway_code(tgt) != pos:
        return True

    return False


def _estimate_bbox_center_norm(frame: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Best-effort estimate of target position in normalized image coordinates [-1, 1].

    Expected box format example:
      [x1, y1, x2, y2, conf, cls_id]
    Reads from:
      - frame["AnatomicalCoherence"]
      - frame["bounding_boxes"]

    Returns:
      (nx, ny) or None
    """
    boxes = frame.get("AnatomicalCoherence") or frame.get("bounding_boxes") or []
    if not isinstance(boxes, list) or not boxes:
        return None

    valid_boxes = [b for b in boxes if isinstance(b, (list, tuple)) and len(b) >= 4]
    if not valid_boxes:
        return None

    box = valid_boxes[0]
    x1, y1, x2, y2 = map(float, box[:4])
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    max_x = max(float(b[2]) for b in valid_boxes)
    max_y = max(float(b[3]) for b in valid_boxes)

    if max_x <= 0 or max_y <= 0:
        return None

    nx = (cx / max_x) * 2.0 - 1.0
    ny = (cy / max_y) * 2.0 - 1.0
    return (round(nx, 3), round(ny, 3))


def _position_sentence(target: str, bbox_center_norm: Optional[Tuple[float, float]]) -> str:
    if not target or bbox_center_norm is None:
        return ""

    x, y = bbox_center_norm
    horiz = "left" if x < -0.15 else "right" if x > 0.15 else "center"
    vert = "upper" if y < -0.15 else "lower" if y > 0.15 else "center"

    if horiz == "center" and vert == "center":
        pos_str = "center"
    elif horiz == "center":
        pos_str = vert
    elif vert == "center":
        pos_str = horiz
    else:
        pos_str = f"{horiz} {vert}"

    return f'the "{target}" is at the {pos_str} position.'


def _movement_sentence_from_velocity(velocity: Any, target: str = "") -> str:
    """
    Lightweight wording from m_jointsVelRel ~ [bend, rotation, insertion]
    """
    if not isinstance(velocity, (list, tuple)) or len(velocity) < 3:
        return ""

    try:
        knob = float(velocity[0] or 0.0)
        rot = float(velocity[1] or 0.0)
        ins = float(velocity[2] or 0.0)
    except Exception:
        return ""

    parts: List[str] = []

    if rot > 0.08:
        parts.append("rotate clockwise")
    elif rot < -0.08:
        parts.append("rotate counter-clockwise")

    if knob > 0.08:
        parts.append("tilt the tip up")
    elif knob < -0.08:
        parts.append("tilt the tip down")

    if ins > 0.08:
        parts.append("advance gently")
    elif ins < -0.08:
        parts.append("withdraw slightly")

    if not parts:
        return ""

    if len(parts) == 1:
        return f'the student should {parts[0]} to reach the "{target}".' if target else f"the student should {parts[0]}."
    return (
        f'the student should {", ".join(parts[:-1])}, and {parts[-1]} to reach the "{target}".'
        if target else
        f'the student should {", ".join(parts[:-1])}, and {parts[-1]}.'
    )


# -----------------------------
# Event engine
# -----------------------------

class EventSignalEngine:
    """
    Shared event detector for runtime + MAS.

    Flag semantics:
      0 = normal / no trigger
      1 = intermediate anchor / waypoint reached
      2 = destination / step completed
      3 = target not visible / target lost
      4 = unusual transition / path suspicion
      5 = drift / unstable recent motion
      6 = backtrack required

    Output schema:
    {
        "flag": int,
        "ema": float | None,
        "reason": str | None,
        "soft_prompt": str | None,
        "llm_trigger_flag": bool,
    }
    """

    def __init__(
        self,
        flow_mem: int = 5,
        ema_alpha: float = 0.35,
        drift_thresh: float = -0.5,
    ):
        self.flag: int = 0
        self.ema_score: Optional[float] = None
        self.ema_alpha = float(ema_alpha)
        self.drift_thresh = float(drift_thresh)

        self.motion_hist: Deque[float] = deque(maxlen=int(flow_mem))
        self.target_hist: Deque[str] = deque(maxlen=20)
        self._step_index: int = 0
        self._last_emitted_step: Dict[str, int] = {}
        self._last_anchor_signature: str = ""
        self._last_anchor_step: int = -10_000
        self._last_invisible_signature: str = ""
        self._last_invisible_step: int = -10_000

    # -------------------------
    # Emit suppression helpers
    # -------------------------

    def _emit_allowed(self, signature: str, *, cooldown_steps: int) -> bool:
        last_step = self._last_emitted_step.get(signature, -10_000)
        return (self._step_index - last_step) > int(cooldown_steps)

    def _mark_emitted(self, signature: str) -> None:
        self._last_emitted_step[signature] = self._step_index

    def _anchor_emit_allowed(self, position: str, target: str, planned_next: str) -> bool:
        signature = f"{_safe_upper(position)}|{_safe_upper(target)}|{_safe_upper(planned_next)}"
        if signature == self._last_anchor_signature and (self._step_index - self._last_anchor_step) <= 8:
            return False
        self._last_anchor_signature = signature
        self._last_anchor_step = self._step_index
        return True

    def _invisible_emit_allowed(self, position: str, target: str, planned_next: str) -> bool:
        signature = f"{_safe_upper(position)}|{_safe_upper(target)}|{_safe_upper(planned_next)}"
        if signature == self._last_invisible_signature and (self._step_index - self._last_invisible_step) <= 6:
            return False
        self._last_invisible_signature = signature
        self._last_invisible_step = self._step_index
        return True

    # -------------------------
    # Flag detection
    # -------------------------

    def _detect_flag(
        self,
        frame: Dict[str, Any],
        plan: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        self._step_index += 1
        plan_view = _plan_view(plan)

        target = _safe_str(frame.get("current_target"))
        position = _safe_str(frame.get("anatomical_position"))
        next_destination = _safe_str(frame.get("next_destination"))
        just_reached = _bool(frame.get("just_reached", False))

        planned_mode = plan_view.get("mode", "")
        planned_next = plan_view.get("next_airway", "") or next_destination

        need_event = False

        # 6 = explicit backtrack
        if target.lower() == "back" or planned_mode.lower() == "backtrack":
            self.flag = 6
            signature = f"flag6|{_safe_upper(position)}|{_safe_upper(planned_next)}|{_safe_upper(anchor_landmark:=plan_view.get('anchor_landmark',''))}"
            if self._emit_allowed(signature, cooldown_steps=8):
                need_event = True
                self._mark_emitted(signature)
            return need_event, {"flag": self.flag, "ema": self.ema_score}

        # 1 = at anchor/waypoint
        if target and _safe_upper(target) == _safe_upper(position) and _safe_upper(position) in ANCHOR_POSITIONS:
            if planned_next and _safe_upper(planned_next) != _safe_upper(position) and not just_reached:
                self.flag = 0
                return need_event, {"flag": self.flag, "ema": self.ema_score}
            self.flag = 1
            signature = f"flag1|{_safe_upper(position)}|{_safe_upper(target)}|{_safe_upper(planned_next)}"
            if self._emit_allowed(signature, cooldown_steps=10) and self._anchor_emit_allowed(position, target, planned_next):
                need_event = True
                self._mark_emitted(signature)
            return need_event, {"flag": self.flag, "ema": self.ema_score}

        # 2 = explicit completion event
        if just_reached:
            self.flag = 2
            signature = f"flag2|{_safe_upper(position)}|{_safe_upper(planned_next)}|{len(frame.get('reached_regions', []) or [])}"
            if self._emit_allowed(signature, cooldown_steps=2):
                need_event = True
                self._mark_emitted(signature)
            return need_event, {"flag": self.flag, "ema": self.ema_score}

        # 3 = target invisible / lost
        bbox_center = _estimate_bbox_center_norm(frame)
        explicit_visible = frame.get("is_target_visible", None)
        target_missing = (explicit_visible is False) or (explicit_visible is None and bbox_center is None)
        if target and target_missing:
            self.flag = 3
            signature = f"flag3|{_safe_upper(position)}|{_safe_upper(target)}|{_safe_upper(planned_next)}"
            if self._emit_allowed(signature, cooldown_steps=7) and self._invisible_emit_allowed(position, target, planned_next):
                need_event = True
                self._mark_emitted(signature)
            return need_event, {"flag": self.flag, "ema": self.ema_score}

        # 4 / 5 / 0 = transition stability logic
        state_key = f"{position}->{target}->{planned_next}->{planned_mode}"
        self.target_hist.append(state_key)

        most_common = Counter(self.target_hist).most_common(1)
        stability = (most_common[0][1] / len(self.target_hist)) if most_common else 1.0
        score = stability * 2.0 - 1.0  # [0,1] -> [-1,1]

        self.ema_score = score if self.ema_score is None else (
            self.ema_alpha * score + (1.0 - self.ema_alpha) * self.ema_score
        )

        # drift / unstable recent dynamics
        if self.ema_score < self.drift_thresh:
            self.flag = 5
            signature = f"flag5|{_safe_upper(position)}|{_safe_upper(target)}|{_safe_upper(planned_next)}"
            if self._emit_allowed(signature, cooldown_steps=6):
                need_event = True
                self._mark_emitted(signature)
            return need_event, {"flag": self.flag, "ema": self.ema_score}

        # route inconsistency heuristic
        if position and planned_next and target:
            if _route_inconsistency_flag(
                position=position,
                target=target,
                planned_next=planned_next,
                plan_view=plan_view,
            ):
                self.flag = 4
                signature = f"flag4|{_safe_upper(position)}|{_safe_upper(target)}|{_safe_upper(planned_next)}"
                if self._emit_allowed(signature, cooldown_steps=6):
                    need_event = True
                    self._mark_emitted(signature)
                return need_event, {"flag": self.flag, "ema": self.ema_score}

        # otherwise normal
        self.flag = 0
        return need_event, {"flag": self.flag, "ema": self.ema_score}

    # -------------------------
    # Reason text
    # -------------------------

    def _reason_from_flag(
        self,
        flag: int,
        frame: Dict[str, Any],
        plan: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        plan_view = _plan_view(plan)

        target = _safe_str(frame.get("current_target"))
        position = _safe_str(frame.get("anatomical_position"))
        next_destination = _safe_str(frame.get("next_destination"))

        planned_next = plan_view.get("next_airway", "") or next_destination
        planned_mode = plan_view.get("mode", "")
        anchor_landmark = plan_view.get("anchor_landmark", "")
        transition_type = plan_view.get("transition_type", "")
        reanchor_target = plan_view.get("reanchor_target", "")

        if flag == 1:
            if position:
                msg = (
                    f"so the student has just reached intermediate waypoint '{position}' "
                    "and needs feedback and guidance for the next step."
                )
                if planned_next:
                    msg += f" The next airway in the curriculum is '{planned_next}'."
                return msg
            return "The student has just reached the current target and needs feedback and next-step guidance."

        if flag == 2:
            if planned_next and position:
                return (
                    f"The student just completed the current step at '{position}'. "
                    f"The next airway in the curriculum is '{planned_next}', so the student now needs inspection guidance and the next navigation instruction."
                )
            return "The student has reached the planned destination and needs instructions for what to do next."

        if flag == 3:
            if target:
                msg = (
                    f"The desired target lumen '{target}' is not visible in the current camera view, "
                    "so the student needs guidance on how to re-locate it."
                )
                if anchor_landmark:
                    msg += f" Use '{anchor_landmark}' as the main landmark anchor."
                return msg
            return "The desired target lumen is not visible in the current view, so guidance on how to find it is needed."

        if flag == 4:
            if position and target:
                if transition_type == "local_sibling" and planned_next:
                    return (
                        f"The current airway '{position}' is inconsistent with a local move toward '{planned_next}', "
                        "so the student likely left the intended airway family and needs corrective guidance."
                    )
                if transition_type == "regional_reanchor" and reanchor_target:
                    return (
                        f"The current airway '{position}' is outside the expected route toward '{planned_next}'. "
                        f"Re-anchor at '{reanchor_target}' before continuing."
                    )
                return (
                    f"The current airway '{position}' is inconsistent with the planned route toward '{target}', "
                    "so the student needs corrective guidance."
                )
            return "The current airway is inconsistent with the planned route, so corrective guidance is needed."

        if flag == 5:
            if planned_mode:
                return (
                    f"The recent motion of the scope deviates from the ideal direction over multiple frames, "
                    f"so the student needs corrective steering instructions while staying in '{planned_mode}' mode."
                )
            return (
                "The recent motion of the scope deviates from the ideal direction over multiple frames, "
                "so the student needs corrective steering instructions."
            )

        if flag == 6:
            if anchor_landmark:
                return (
                    "The navigation target is 'back', meaning the scope should be guided back towards the carina; "
                    f"the student needs instructions on how to back out safely and then re-anchor using '{anchor_landmark}'."
                )
            return (
                "The navigation target is 'back', meaning the scope should be guided back towards the carina; "
                "the student needs instructions on how to back out safely."
            )

        return None

    # -------------------------
    # Soft prompt builder
    # -------------------------

    def _make_soft_prompt(
        self,
        current_data: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None,
        plan: Optional[Dict[str, Any]] = None,
    ) -> str:
        plan_view = _plan_view(plan)

        target = _safe_str(current_data.get("current_target"))
        training_target = _safe_str(current_data.get("training_target"))
        waypoint_target = _safe_str(current_data.get("waypoint_target"))
        position = _safe_str(current_data.get("anatomical_position"))
        next_dest = _safe_str(current_data.get("next_destination"))
        just_reached = _bool(current_data.get("just_reached", False))
        reached_regions = list(current_data.get("reached_regions", []) or [])
        velocity = current_data.get("m_jointsVelRel", None)

        planned_next = plan_view.get("next_airway", "") or next_dest
        anchor_landmark = plan_view.get("anchor_landmark", "")
        recognition_cue = plan_view.get("recognition_cue", "")
        micro_steps = plan_view.get("micro_steps", [])
        planned_mode = plan_view.get("mode", "")
        plan_why = plan_view.get("why", "")

        route_sentence = ""
        if position and planned_next:
            route_sentence = (
                f'the student is currently at {position}, and the next lumen to be explored is "{planned_next}".'
            )
            if waypoint_target and _safe_upper(waypoint_target) != _safe_upper(planned_next):
                route_sentence += f' use "{waypoint_target}" only as a waypoint or re-anchor step if needed.'
        elif position:
            route_sentence = f"the student is currently at {position}."

        plan_sentence = ""
        if planned_mode:
            plan_sentence = f"the current curriculum mode is {planned_mode}."
        if anchor_landmark:
            plan_sentence += f' use "{anchor_landmark}" as the anchor landmark.'
        if recognition_cue:
            plan_sentence += f" recognition cue: {recognition_cue}"
        if plan_why:
            plan_sentence += f" rationale: {plan_why}"

        micro_step_sentence = ""
        if micro_steps:
            micro_step_sentence = " recommended micro-steps: " + " | ".join(str(x) for x in micro_steps[:3])

        praise = ""
        if just_reached:
            position_upper = _safe_upper(position)
            if position_upper and position_upper.startswith(("RB", "LB")):
                praise = f"the student just reached {position}, so acknowledge the arrival first and then give the next instruction."
            elif position and waypoint_target and _safe_upper(waypoint_target) == position_upper:
                praise = f"the student is at waypoint {position}; acknowledge it briefly without praise and continue toward \"{planned_next or training_target or target}\"."

        encourage = ""
        if history and len(history) >= 15:
            recent_targets = [_safe_str(x.get("current_target")) for x in history[-15:]]
            if target and recent_targets.count(target) >= 10:
                encourage = "the student has spent more than usual time to reach the current target, they need to be encouraged."

        inspected = [_safe_upper(x) for x in reached_regions]
        remaining = []
        visit_order = current_data.get("airway_visit_order", []) or []
        if visit_order:
            inspected_set = set(inspected)
            remaining = [_safe_upper(x) for x in visit_order if _safe_upper(x) not in inspected_set]

        inspected_sentence = (
            f"the following areas have been inspected: {', '.join(x.lower() for x in inspected)}."
            if inspected else
            "No target region has been reached yet."
        )

        remaining_sentence = (
            f"the following areas still need to be examined: {', '.join(x.lower() for x in remaining)}."
            if remaining else
            "All target regions have been reached, and the student should be praised."
        )

        location_sentence = _position_sentence(target, _estimate_bbox_center_norm(current_data))
        movement_sentence = _movement_sentence_from_velocity(velocity, target)

        parts: List[str] = []
        for x in [
            route_sentence,
            plan_sentence,
            micro_step_sentence,
            location_sentence,
            movement_sentence,
            praise,
            encourage,
            inspected_sentence,
            remaining_sentence,
        ]:
            x = _safe_str(x)
            if x:
                parts.append(x)

        return " ".join(parts)

    # -------------------------
    # Public API
    # -------------------------

    def step(
        self,
        frame: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None,
        plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        need_event, base_meta = self._detect_flag(frame, plan=plan)

        reason = None
        soft_prompt = None
        if need_event:
            reason = self._reason_from_flag(base_meta["flag"], frame, plan=plan)
            soft_prompt = self._make_soft_prompt(frame, history=history, plan=plan)

        return {
            "flag": int(base_meta.get("flag", 0) or 0),
            "ema": base_meta.get("ema"),
            "reason": reason,
            "soft_prompt": soft_prompt,
            "llm_trigger_flag": bool(need_event),
        }
