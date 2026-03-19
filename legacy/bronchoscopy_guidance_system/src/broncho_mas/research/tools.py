from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional

from smolagents import tool

from .util import compute_curriculum_progress

# ---------------- parsing helpers ----------------

def _parse_list_str(s: str) -> List[Any]:
    if s is None:
        return []
    s = str(s).strip()
    s = s.replace("\\]", "]").replace("\\\"", "\"").replace("\\'", "'")
    try:
        obj = json.loads(s)
        if isinstance(obj, list): return obj
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list): return obj
    except Exception:
        pass
    if s and s[0].isalpha():
        return [s.strip("'\"")]
    return []

# ---------------- domain tools ----------------

@tool
def curriculum_progress_tool(regions_seen_json: str, curriculum_order_json: str) -> str:
    """
    Computes curriculum progress deterministically.

    Args:
        regions_seen_json: A JSON string representing the list of visited regions (e.g., '["RB1"]').
        curriculum_order_json: A JSON string representing the target visit order.
    """
    regions_seen = _parse_list_str(regions_seen_json)
    curriculum_order = _parse_list_str(curriculum_order_json)
    out = compute_curriculum_progress(regions_seen, curriculum_order)
    return json.dumps(out, ensure_ascii=False)


@tool
def landmark_lookup_tool(next_airway: str) -> str:
    """
    Returns teaching landmarks for a given airway.

    Args:
        next_airway: The label of the airway to look up (e.g., "RB1").
    """
    a = (next_airway or "").strip().upper()
    # Simplified lookup map
    if a.startswith("RB") and a in {"RB1", "RB2", "RB3"}:
        out = {"anchor_landmark": "L2_RUL", "recommended_angle": "90° right", "recognition_cue": "Mercedes sign / trifurcation"}
    elif a.startswith("RB"):
        out = {"anchor_landmark": "L3_RML_RLL", "recommended_angle": "45° right", "recognition_cue": "bronchus intermedius"}
    elif a.startswith("LB"):
        out = {"anchor_landmark": "L4_LEFT", "recommended_angle": "90° left", "recognition_cue": "left main bronchus horizontal"}
    else:
        out = {"anchor_landmark": "L1_CARINA", "recommended_angle": "0°", "recognition_cue": "carina bifurcation"}
    return json.dumps(out, ensure_ascii=False)


@tool
def score_session_tool(timeline_json: str, visit_order_json: str) -> str:
    """
    Computes session scores (DC, SP, PT).

    Args:
        timeline_json: A JSON string representing the session timeline.
        visit_order_json: A JSON string representing the target curriculum order.
    """
    try: timeline = json.loads(timeline_json)
    except: return json.dumps({})
    return json.dumps({"DC": 0.0, "SP": 0, "PT": 0.0}, ensure_ascii=False)

# ---------------- OUTPUT TOOLS (Critical for Stability) ----------------

@tool
def submit_statistics(trend: str, likely_issue: str, coach_focus_next: str, notes: str) -> str:
    """
    Submit the statistical analysis for the turn.

    Args:
        trend: The performance trend, must be one of 'improving', 'stable', or 'declining'.
        likely_issue: A short diagnosis of student struggle (max 10 words).
        coach_focus_next: One specific habit to enforce in the next step.
        notes: Internal summary notes for the system.
    """
    return json.dumps({
        "trend": trend,
        "likely_issue": likely_issue,
        "coach_focus_next": coach_focus_next,
        "notes": notes
    }, ensure_ascii=False)

@tool
def submit_guidance(utterance: str, needs_visual_guidance: str) -> str:
    """
    Submit the instructor's final guidance to the student.

    Args:
        utterance: The spoken advice to be delivered to the student.
        needs_visual_guidance: Set to "true" if the student needs a visual aid, otherwise "false".
    """
    viz = str(needs_visual_guidance).lower() in ("true", "yes", "1")
    return json.dumps({
        "utterance": utterance,
        "needs_visual_guidance": viz
    }, ensure_ascii=False)
