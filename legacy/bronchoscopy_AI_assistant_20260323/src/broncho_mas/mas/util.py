from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional


def compute_curriculum_progress(regions_seen: List[str], curriculum_order: List[str]) -> Dict[str, Any]:
    """Compute curriculum progress deterministically (single source of truth)."""
    seen = {str(x).upper() for x in (regions_seen or [])}
    order = [str(x).upper() for x in (curriculum_order or [])]

    reached = [a for a in order if a in seen]
    missing = [a for a in order if a not in seen]
    next_airway = missing[0] if missing else ""
    coverage = (len(reached) / len(order)) if order else 0.0

    return {
        "reached": reached,
        "missing": missing,
        "reached_count": len(reached),
        "total": len(order),
        "coverage_ratio": round(coverage, 3),
        "next_airway": next_airway,
    }


def json_load_or_none(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON -> dict parser used for LLM outputs."""
    if not text:
        return None
    t = text.strip()

    # Strip fenced code blocks if the model returns ```json ... ```
    if t.startswith("```"):
        t = t.strip().strip("`")
        t = t.replace("json", "", 1).strip()

    # 1) Strict JSON
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2) Python-literal dict fallback (handles single quotes)
    try:
        obj = ast.literal_eval(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def validate_instructor_payload(x: Any) -> tuple[Dict[str, Any], List[str]]:
    """Lightweight schema validation for instructor payload.

    Expected minimal shape:
      - utterance: str
      - needs_visual_guidance: bool
    Returns (validated_payload, error_codes).
    """
    errs: List[str] = []
    if not isinstance(x, dict):
        return {"utterance": "", "needs_visual_guidance": False}, ["instructor_not_dict"]

    utt = x.get("utterance")
    nvg = x.get("needs_visual_guidance")

    if not isinstance(utt, str):
        errs.append("utterance_not_str")
        utt = ""
    if not isinstance(nvg, bool):
        errs.append("needs_visual_guidance_not_bool")
        nvg = False

    return {"utterance": utt, "needs_visual_guidance": nvg}, errs


def validate_statistics_payload(x: Any) -> tuple[Dict[str, Any], List[str]]:
    """Lightweight schema validation for statistics payload (must be dict)."""
    if isinstance(x, dict):
        return x, []
    return {}, ["statistics_not_dict"]
