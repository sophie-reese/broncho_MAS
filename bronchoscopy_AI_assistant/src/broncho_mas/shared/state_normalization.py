from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


DEFAULT_ANCHOR_AIRWAYS: Set[str] = {"CARINA", "LMB", "RMB", "TRACHEA"}


def extract_block(text: str, tag: str) -> str:
    if not text:
        return ""
    match = re.search(rf"{re.escape(tag)}\s*:(.*?)(?=\n\s*[A-Z_]+\s*:|\Z)", str(text), flags=re.S)
    return (match.group(1) if match else "").strip()


def coerce_m_joints_vel_rel(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return [0.0, 0.0, 0.0]
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return [0.0, 0.0, 0.0]


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def coerce_upper_token(value: Any) -> str:
    return str(value or "").strip().upper()


def extract_named_field(current_situation: str, *labels: str) -> str:
    text = str(current_situation or "")
    for label in labels:
        match = re.search(rf"{re.escape(label)}\s*[:=]\s*\"?([A-Za-z0-9+_\- ]+)\"?", text, flags=re.I)
        if match:
            return coerce_upper_token(match.group(1))
    return ""


def extract_reached_regions(current_situation: str, allowed: Optional[Iterable[str]] = None) -> List[str]:
    text = str(current_situation or "")
    allowed_set = {str(x).strip().upper() for x in (allowed or []) if str(x).strip()}
    out: List[str] = []
    patterns = [
        r"reached_regions\(last\)\s*[:=]\s*(\[[^\]]*\])",
        r"REACHED_REGIONS\s*[:=]\s*(\[[^\]]*\])",
        r"reached_regions\s*[:=]\s*(\[[^\]]*\])",
        r"timeline_reached\s*[:=]\s*(\[[^\]]*\])",
        r"regions_seen\s*[:=]\s*(\[[^\]]*\])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        chunk = match.group(1)
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(chunk)
                if isinstance(value, list):
                    for item in value:
                        region = str(item).strip().upper()
                        if not region:
                            continue
                        if allowed_set and region not in allowed_set:
                            continue
                        if region not in out:
                            out.append(region)
                    return out
            except Exception:
                continue
    return out


def extract_current_airway(
    current_situation: str,
    *,
    allowed_airways: Optional[Iterable[str]] = None,
    anchors: Optional[Iterable[str]] = None,
) -> str:
    text = str(current_situation or "")
    allowed = {str(x).strip().upper() for x in (allowed_airways or []) if str(x).strip()}
    allowed |= {str(x).strip().upper() for x in (anchors or DEFAULT_ANCHOR_AIRWAYS) if str(x).strip()}
    patterns = [
        r"anatomical_position\s*[:=]\s*\"?([A-Za-z0-9+\-]+)\"?",
        r"current airway\s*[:=]\s*([A-Za-z0-9+\-]+)",
        r"current region\s*[:=]\s*([A-Za-z0-9+\-]+)",
        r'currently at\s+\"?([A-Za-z0-9+\-]+)\"?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        value = match.group(1).strip().upper()
        if value in {"", "NONE", "NULL", "UNKNOWN"}:
            return ""
        if not allowed or value in allowed:
            return value
    return ""


def extract_target_hint(current_situation: str, allowed_airways: Optional[Iterable[str]] = None) -> str:
    text = str(current_situation or "")
    allowed = {str(x).strip().upper() for x in (allowed_airways or []) if str(x).strip()}
    patterns = [
        r"target region\s*[:=]\s*([A-Za-z0-9+\-]+)",
        r"target airway\s*[:=]\s*([A-Za-z0-9+\-]+)",
        r"navigation target is ['\"]?([A-Za-z0-9+\-]+)['\"]?",
        r"next lumen to be explored\s*(?:is)?\s*[:=]?\s*['\"]?([A-Za-z0-9+\-]+)['\"]?",
        r"requested next airway\s*[:=]\s*([A-Za-z0-9+\-]+)",
        r"current_target\s*[:=]\s*([A-Za-z0-9+\-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        value = match.group(1).strip().upper()
        if not allowed or value in allowed:
            return value
    return ""


def extract_m_joints_vel_rel(current_situation: str, raw_payload: Optional[Dict[str, Any]] = None) -> List[float]:
    payload = dict(raw_payload or {})
    for candidate in (payload.get("m_jointsVelRel"), payload.get("joints_vel_rel")):
        triplet = coerce_m_joints_vel_rel(candidate)
        if triplet != [0.0, 0.0, 0.0] or isinstance(candidate, (list, tuple)):
            return triplet

    text = str(current_situation or "")
    patterns = [
        r"m_jointsVelRel\s*[:=]\s*(\[[^\]]+\])",
        r"joints_vel_rel\s*[:=]\s*(\[[^\]]+\])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        chunk = match.group(1)
        for parser in (json.loads, ast.literal_eval):
            try:
                return coerce_m_joints_vel_rel(parser(chunk))
            except Exception:
                continue
    return [0.0, 0.0, 0.0]


def normalize_runtime_payload(raw: Dict[str, Any], airway_visit_order: Sequence[str]) -> Dict[str, Any]:
    raw_payload = dict(raw or {})
    event_packet = raw_payload.get("event_packet") if isinstance(raw_payload.get("event_packet"), dict) else {}
    raw_payload.setdefault("airway_visit_order", list(airway_visit_order))
    reached_regions = raw_payload.get("reached_regions") or raw_payload.get("regions_seen") or []
    if not isinstance(reached_regions, list):
        try:
            reached_regions = list(reached_regions)
        except Exception:
            reached_regions = []
    reached_regions = [str(x).strip().upper() for x in reached_regions if str(x).strip()]

    current_airway = (
        raw_payload.get("anatomical_position")
        or raw_payload.get("current_airway")
        or raw_payload.get("current_region")
        or ""
    )
    current_target_raw = str(raw_payload.get("current_target") or "").strip().upper()
    planned_target_raw = str(
        raw_payload.get("requested_next_airway")
        or raw_payload.get("next_destination")
        or raw_payload.get("target_airway")
        or raw_payload.get("target_region")
        or ""
    ).strip().upper()
    if current_target_raw == "BACK":
        target_airway = planned_target_raw
    else:
        target_airway = planned_target_raw or current_target_raw

    backtracking = coerce_bool(raw_payload.get("backtracking", False)) or current_target_raw == "BACK"

    validated_landmark = coerce_upper_token(
        raw_payload.get("validated_landmark")
        or raw_payload.get("landmark_id")
        or event_packet.get("validated_landmark")
        or event_packet.get("landmark_id")
        or event_packet.get("current_landmark")
        or ""
    )
    current_landmark = coerce_upper_token(raw_payload.get("current_landmark") or event_packet.get("current_landmark") or "")
    teaching_landmark = coerce_upper_token(raw_payload.get("teaching_landmark") or event_packet.get("teaching_landmark") or "")
    previous_landmark = coerce_upper_token(raw_payload.get("previous_landmark") or event_packet.get("previous_landmark") or "")

    return {
        "raw_payload": raw_payload,
        "prompt_text": str(raw_payload.get("llm_prompt_text") or "").strip(),
        "previous_msgs": str(raw_payload.get("previous_msgs") or raw_payload.get("history") or raw_payload.get("llm_history") or "").strip(),
        "student_question": str(raw_payload.get("student_question") or raw_payload.get("student_q") or raw_payload.get("question") or "").strip(),
        "soft_prompt": str(raw_payload.get("soft_prompt") or event_packet.get("soft_prompt") or "").strip(),
        "need_llm": bool(raw_payload.get("need_llm", raw_payload.get("llm_trigger_flag", event_packet.get("llm_trigger_flag", False)))),
        "llm_trigger_flag": bool(raw_payload.get("llm_trigger_flag", event_packet.get("llm_trigger_flag", False))),
        "llm_reason": str(raw_payload.get("llm_reason") or event_packet.get("reason") or "").strip(),
        "phase": str(raw_payload.get("phase") or "").strip(),
        "current_airway": str(current_airway).strip().upper(),
        "target_airway": target_airway,
        "requested_next_airway": str(raw_payload.get("requested_next_airway") or "").strip().upper(),
        "validated_landmark": validated_landmark,
        "landmark_id": coerce_upper_token(raw_payload.get("landmark_id") or event_packet.get("landmark_id") or ""),
        "current_landmark": current_landmark,
        "teaching_landmark": teaching_landmark,
        "previous_landmark": previous_landmark,
        "reached_regions": reached_regions,
        "just_reached": coerce_bool(raw_payload.get("just_reached", False)),
        "first_time_landmark": coerce_bool(raw_payload.get("first_time_landmark", event_packet.get("first_time_landmark", False))),
        "backtracking": backtracking,
        "drift_detected": coerce_bool(raw_payload.get("drift_detected", False)),
        "is_centered": coerce_bool(raw_payload.get("is_centered", False)),
        "is_stable": coerce_bool(raw_payload.get("is_stable", False)),
        "is_target_visible": coerce_bool(raw_payload.get("is_target_visible", False)),
        "wall_contact_risk": coerce_bool(raw_payload.get("wall_contact_risk", False)),
        "need_recenter": coerce_bool(raw_payload.get("need_recenter", False)),
        "m_jointsVelRel": extract_m_joints_vel_rel("", raw_payload),
        "robot_joints": raw_payload.get("robot_joints") or [0, 0, 0],
        "bounding_boxes": raw_payload.get("bounding_boxes") or [],
        "mode": str(raw_payload.get("mode") or "").strip(),
        "visualization_mode": str(raw_payload.get("visualization_mode") or "").strip(),
        "landmark_teaching_history": raw_payload.get("landmark_teaching_history") or raw_payload.get("teaching_history") or raw_payload.get("taught_landmarks") or [],
        "last_landmark_taught": str(raw_payload.get("last_landmark_taught") or "").strip(),
        "last_landmark_taught_turn": int(raw_payload.get("last_landmark_taught_turn") or 0),
        "last_acknowledged_waypoint": str(raw_payload.get("last_acknowledged_waypoint") or "").strip().upper(),
        "last_acknowledged_waypoint_turn": int(raw_payload.get("last_acknowledged_waypoint_turn") or 0),
        "last_acknowledged_destination": str(raw_payload.get("last_acknowledged_destination") or "").strip().upper(),
        "last_acknowledged_destination_turn": int(raw_payload.get("last_acknowledged_destination_turn") or 0),
        "session_turn_index": int(raw_payload.get("session_turn_index") or 0),
        "event_packet": event_packet,
    }


def normalize_legacy_prompt(
    prompt_text: str,
    airway_visit_order: Sequence[str],
    *,
    anchors: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    current_situation = extract_block(prompt_text, "CURRENT_SITUATION")
    previous_msgs = extract_block(prompt_text, "PREVIOUS_MSGS")
    student_question = extract_block(prompt_text, "STUDENT_QUESTION")
    return {
        "raw_payload": {},
        "prompt_text": prompt_text,
        "previous_msgs": previous_msgs,
        "student_question": student_question,
        "soft_prompt": "",
        "need_llm": False,
        "llm_trigger_flag": False,
        "llm_reason": "",
        "phase": "",
        "current_airway": extract_current_airway(current_situation, allowed_airways=airway_visit_order, anchors=anchors),
        "target_airway": extract_target_hint(current_situation, allowed_airways=airway_visit_order),
        "requested_next_airway": "",
        "validated_landmark": extract_named_field(current_situation, "validated_landmark", "validated landmark", "landmark_id", "landmark id"),
        "landmark_id": extract_named_field(current_situation, "landmark_id", "landmark id"),
        "current_landmark": extract_named_field(current_situation, "current_landmark", "current landmark"),
        "teaching_landmark": extract_named_field(current_situation, "teaching_landmark", "teaching landmark"),
        "previous_landmark": extract_named_field(current_situation, "previous_landmark", "previous landmark"),
        "reached_regions": extract_reached_regions(current_situation, allowed=airway_visit_order),
        "just_reached": False,
        "first_time_landmark": coerce_bool(extract_named_field(current_situation, "first_time_landmark", "first time landmark")),
        "backtracking": False,
        "drift_detected": False,
        "is_centered": False,
        "is_stable": False,
        "is_target_visible": False,
        "wall_contact_risk": False,
        "need_recenter": False,
        "m_jointsVelRel": extract_m_joints_vel_rel(current_situation),
        "robot_joints": [0, 0, 0],
        "bounding_boxes": [],
        "mode": "",
        "visualization_mode": "",
        "landmark_teaching_history": [],
        "last_landmark_taught": "",
        "last_landmark_taught_turn": 0,
        "last_acknowledged_waypoint": "",
        "last_acknowledged_waypoint_turn": 0,
        "last_acknowledged_destination": "",
        "last_acknowledged_destination_turn": 0,
        "session_turn_index": 0,
        "event_packet": {},
    }


def build_current_situation_from_state(state: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key, label in [
        ("phase", "Phase"),
        ("current_airway", "Current airway"),
        ("target_airway", "Target airway"),
        ("requested_next_airway", "Requested next airway"),
        ("soft_prompt", "Soft prompt"),
        ("llm_reason", "LLM reason"),
        ("validated_landmark", "Validated landmark"),
        ("current_landmark", "Current landmark"),
        ("previous_landmark", "Previous landmark"),
    ]:
        value = state.get(key)
        if value not in (None, ""):
            parts.append(f"{label}: {value}")
    for key, label in [
        ("need_llm", "Need LLM"),
        ("backtracking", "Backtracking"),
        ("drift_detected", "Drift detected"),
        ("is_centered", "Is centered"),
        ("is_stable", "Is stable"),
        ("is_target_visible", "Target visible"),
        ("wall_contact_risk", "Wall contact risk"),
        ("need_recenter", "Need recenter"),
        ("first_time_landmark", "First time landmark"),
    ]:
        parts.append(f"{label}: {str(bool(state.get(key, False))).lower()}")
    parts.append(f"reached_regions: {json.dumps(state.get('reached_regions', []))}")
    parts.append(f"m_jointsVelRel: {json.dumps(state.get('m_jointsVelRel', [0.0, 0.0, 0.0]))}")
    return "\n".join(parts).strip()
