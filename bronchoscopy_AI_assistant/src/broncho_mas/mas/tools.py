from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from smolagents import tool

from .util import compute_curriculum_progress

try:
    from ..shared.curriculum import CurriculumEngine as SharedCurriculumEngine  # type: ignore
except Exception:
    SharedCurriculumEngine = None  # type: ignore


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------

def _parse_json_like(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value

    s = str(value).strip()
    if not s:
        return default

    s = s.replace("\\]", "]").replace('\\"', '"').replace("\\'", "'")

    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        return ast.literal_eval(s)
    except Exception:
        return default


def _parse_list_str(s: Any) -> List[Any]:
    obj = _parse_json_like(s, [])
    if isinstance(obj, list):
        return obj
    if obj is None:
        return []
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, str) and obj.strip():
        return [obj.strip("'\"")]
    return []


def _normalize_upper_list(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in values or []:
        key = str(x).strip().upper()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


# ---------------------------------------------------------------------
# Shared curriculum adapter helpers
# ---------------------------------------------------------------------

def _make_shared_engine(curriculum_order: Sequence[Any]) -> Optional[Any]:
    if SharedCurriculumEngine is None:
        return None
    try:
        return SharedCurriculumEngine([str(x).upper() for x in curriculum_order])
    except Exception:
        return None


def _angles_to_text(angles: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for item in angles or []:
        angle = str(item.get("angle", "")).strip()
        purpose = str(item.get("purpose", "")).strip()
        if angle and purpose:
            parts.append(f"{angle} ({purpose})")
        elif angle:
            parts.append(angle)
    return " -> ".join(parts) if parts else ""


# ---------------------------------------------------------------------
# Timeline scoring helpers
# ---------------------------------------------------------------------

def _extract_region_from_row(row: Dict[str, Any]) -> str:
    candidates = [
        row.get("reached_region"),
        row.get("region"),
        row.get("airway"),
        row.get("current_airway"),
        row.get("current_region"),
        (row.get("statepacket") or {}).get("current_airway") if isinstance(row.get("statepacket"), dict) else None,
    ]

    cp = row.get("curriculum_progress")
    if isinstance(cp, dict):
        reached = cp.get("reached")
        if isinstance(reached, list) and reached:
            candidates.append(reached[-1])

    auth = row.get("auth_plan")
    if isinstance(auth, dict):
        candidates.append(auth.get("current_airway"))

    for value in candidates:
        key = str(value or "").strip().upper()
        if key and (key.startswith("RB") or key.startswith("LB")):
            return key
    return ""


def _extract_ordered_first_visits(timeline: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in timeline or []:
        if not isinstance(item, dict):
            continue
        region = _extract_region_from_row(item)
        if region and region not in seen:
            seen.add(region)
            out.append(region)
    return out


def _sequence_alignment(first_visits: Sequence[str], visit_order: Sequence[str]) -> float:
    target_index = {a: i for i, a in enumerate(visit_order)}
    filtered = [x for x in first_visits if x in target_index]
    if not filtered:
        return 0.0

    in_order = 1
    last_idx = target_index[filtered[0]]
    for x in filtered[1:]:
        idx = target_index[x]
        if idx >= last_idx:
            in_order += 1
            last_idx = idx
    return round(in_order / max(len(filtered), 1), 4)


def _pace_proxy(first_visits: Sequence[str], visit_order: Sequence[str], timeline_len: int) -> float:
    total_targets = max(len(visit_order), 1)
    unique_progress = len([x for x in first_visits if x in set(visit_order)])
    if timeline_len <= 0:
        return 0.0
    raw = unique_progress / float(timeline_len)
    scaled = min(raw * total_targets, 1.0)
    return round(scaled, 4)


# ---------------------------------------------------------------------
# Domain tools
# ---------------------------------------------------------------------

@tool
def curriculum_progress_tool(regions_seen_json: str, curriculum_order_json: str) -> str:
    """
    Compute curriculum progress deterministically.

    Args:
        regions_seen_json: JSON string of visited regions, e.g. '["RB1", "RB2"]'.
        curriculum_order_json: JSON string of the target visit order.
    """
    regions_seen = _normalize_upper_list(_parse_list_str(regions_seen_json))
    curriculum_order = _normalize_upper_list(_parse_list_str(curriculum_order_json))

    engine = _make_shared_engine(curriculum_order)
    if engine is not None and hasattr(engine, "progress_snapshot"):
        try:
            out = engine.progress_snapshot(regions_seen)
            out["authority"] = "shared_curriculum"
            return json.dumps(out, ensure_ascii=False)
        except Exception:
            pass

    out = compute_curriculum_progress(regions_seen, curriculum_order)
    if isinstance(out, dict):
        out["authority"] = "research_util_fallback"
    return json.dumps(out, ensure_ascii=False)


@tool
def landmark_lookup_tool(next_airway: str, curriculum_order_json: str = "[]") -> str:
    """
    Return teaching landmark information for a target airway.

    Args:
        next_airway: Target airway label, e.g. 'RB1'.
        curriculum_order_json: Optional curriculum order so the shared engine can be initialized.
    """
    airway = str(next_airway or "").strip().upper()
    curriculum_order = _normalize_upper_list(_parse_list_str(curriculum_order_json))
    engine = _make_shared_engine(curriculum_order or [airway] if airway else [])

    if engine is not None and airway and hasattr(engine, "landmark_for_airway"):
        try:
            info = engine.landmark_for_airway(airway)
            angles = list(getattr(info, "recommended_angles", []) or [])
            out = {
                "anchor_landmark": getattr(info, "landmark_id", ""),
                "recommended_angles": angles,
                "recommended_angle": _angles_to_text(angles),
                "recognition_cue": getattr(info, "recognition_cue", ""),
                "authority": "shared_curriculum",
            }
            if hasattr(engine, "inspect_rule_text"):
                out["inspect_rule_text"] = engine.inspect_rule_text()
            return json.dumps(out, ensure_ascii=False)
        except Exception:
            pass

    # conservative fallback
    if airway in {"RB1", "RB2", "RB3"}:
        out = {
            "anchor_landmark": "L2_RUL",
            "recommended_angles": [{"angle": "90° right", "purpose": "access right upper lobe"}],
            "recommended_angle": "90° right (access right upper lobe)",
            "recognition_cue": "the right upper-lobe trifurcation",
            "authority": "fallback_map",
        }
    elif airway.startswith("RB"):
        out = {
            "anchor_landmark": "L3_RML_RLL",
            "recommended_angles": [{"angle": "45° right", "purpose": "access bronchus intermedius / middle lobe pathway"}],
            "recommended_angle": "45° right (access bronchus intermedius / middle lobe pathway)",
            "recognition_cue": "Bronchus intermedius: longer airway with sequential segmental openings along the lumen",
            "authority": "fallback_map",
        }
    elif airway.startswith("LB"):
        out = {
            "anchor_landmark": "L4_LEFT",
            "recommended_angles": [{"angle": "90° left", "purpose": "access left upper lobe + lingula"}],
            "recommended_angle": "90° left (access left upper lobe + lingula)",
            "recognition_cue": "Left main bronchus takeoff: more horizontal course than the right; stable bifurcation landmarks",
            "authority": "fallback_map",
        }
    else:
        out = {
            "anchor_landmark": "L1_CARINA",
            "recommended_angles": [{"angle": "0°", "purpose": "neutral reference at carina"}],
            "recommended_angle": "0° (neutral reference at carina)",
            "recognition_cue": "Carina bifurcation; symmetric right/left main bronchi",
            "authority": "fallback_map",
        }
    return json.dumps(out, ensure_ascii=False)


@tool
def score_session_tool(timeline_json: str, visit_order_json: str) -> str:
    """
    Compute simple session scores from timeline rows.

    Args:
        timeline_json: JSON string representing the session timeline rows.
        visit_order_json: JSON string representing the target curriculum order.
    """
    timeline = _parse_json_like(timeline_json, [])
    visit_order = _normalize_upper_list(_parse_list_str(visit_order_json))

    if not isinstance(timeline, list):
        return json.dumps({}, ensure_ascii=False)

    first_visits = _extract_ordered_first_visits(timeline)
    visit_set = set(visit_order)
    covered = [x for x in first_visits if x in visit_set]

    total_targets = max(len(visit_order), 1)
    dc = round(len(set(covered)) / total_targets, 4)
    sp = _sequence_alignment(first_visits, visit_order)
    pt = _pace_proxy(first_visits, visit_order, len(timeline))

    out = {
        "DC": dc,
        "SP": sp,
        "PT": pt,
        "visited_unique": covered,
        "visited_count": len(covered),
        "timeline_rows": len(timeline),
        "authority": "tool_heuristic_v1",
    }
    return json.dumps(out, ensure_ascii=False)


# ---------------------------------------------------------------------
# Output tools
# ---------------------------------------------------------------------

@tool
def submit_statistics(trend: str, likely_issue: str, coach_focus_next: str, notes: str) -> str:
    """
    Submit the side-channel statistical analysis for the turn.

    Args:
        trend: One of 'improving', 'stable', or 'declining'.
        likely_issue: Short diagnosis of student struggle.
        coach_focus_next: One specific habit to enforce next.
        notes: Internal summary notes for later reporting.
    """
    return json.dumps(
        {
            "trend": str(trend or "").strip(),
            "likely_issue": str(likely_issue or "").strip(),
            "coach_focus_next": str(coach_focus_next or "").strip(),
            "notes": str(notes or "").strip(),
        },
        ensure_ascii=False,
    )


@tool
def submit_guidance(utterance: str, needs_visual_guidance: str) -> str:
    """
    Submit the instructor's final guidance to the student.

    Args:
        utterance: Spoken advice to be delivered to the student.
        needs_visual_guidance: 'true' if a visual aid is needed, else 'false'.
    """
    viz = str(needs_visual_guidance).lower().strip() in ("true", "yes", "1")
    return json.dumps(
        {
            "utterance": str(utterance or "").strip(),
            "needs_visual_guidance": viz,
        },
        ensure_ascii=False,
    )
