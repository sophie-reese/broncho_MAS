from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# -------------------------------------------------------------------
# Output sanitization helpers
# -------------------------------------------------------------------

_RE_REACT_PREFIX = re.compile(r"^\s*(Thought|Action|Observation|Observations)\s*:\s*", flags=re.IGNORECASE)

def strip_react_traces(text: str) -> str:
    if not text:
        return ""
    out_lines: List[str] = []
    for ln in str(text).splitlines():
        if _RE_REACT_PREFIX.match(ln):
            continue
        if "Reached max steps" in ln:
            continue
        if "final answer tool call" in ln:
            continue
        out_lines.append(ln)
    return "\n".join(out_lines).strip()

# -------------------------------------------------------------------
# JoVE-aligned scoring / report composition (DC / SP / PT)
# -------------------------------------------------------------------

def build_core_report(
    *,
    allowed_reached: List[str],
    visit_order: List[str],
    curriculum_progress: Dict[str, Any],
    session_metrics: Dict[str, Any],
    sp_score: float,
    teach_line: Optional[str] = None,
) -> str:
    """
    Deterministic report skeleton with fixed headings.
    """
    coverage = float(curriculum_progress.get("coverage_ratio", 0.0))
    reached_count = int(curriculum_progress.get("reached_count", 0))
    total = int(curriculum_progress.get("total", max(len(visit_order), 1)))

    # PT (do not invent)
    duration = session_metrics.get("duration_seconds", None)
    if isinstance(duration, (int, float)) and duration >= 0:
        pt_line = f"- Procedure time (PT): {int(duration)} seconds."
    else:
        pt_line = "- Procedure time (PT): not recorded."

    dc_line = f"- Diagnostic completeness (DC): {reached_count}/{total} segments ({round(coverage*100)}%)."
    sp_line = f"- Structured progress (SP): {sp_score:.2f} (ordered progression ratio)."

    segs_line = f"- Segments visualized: {', '.join(allowed_reached) if allowed_reached else 'None'}."

    # Robust missing calculation
    allowed_upper = set(x.upper() for x in allowed_reached)
    missing = [a for a in visit_order if a.upper() not in allowed_upper]
    missing_preview = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
    missing_line = f"- Segments not yet visualized: {missing_preview if missing_preview else 'None'}."

    next_airway = curriculum_progress.get("next_airway") or "N/A"
    next_line = f"- Next target segment: {next_airway}."

    if not teach_line:
        teach_line = "- Focus next: stabilize view and keep the lumen centered before any forward movement."

    # backtrack & questions
    backtrack_ratio = session_metrics.get("backtrack_ratio", "N/A")
    # Handle potentially nested student questions if coming from complex dicts
    sq_raw = curriculum_progress.get("student_questions", session_metrics.get("student_questions", 0))
    sq = int(sq_raw) if isinstance(sq_raw, (int, float, str)) and str(sq_raw).isdigit() else 0

    return (
        "Clinical performance note\n"
        f"{dc_line}\n"
        f"{sp_line}\n"
        f"{pt_line}\n"
        "Teaching feedback note\n"
        f"{teach_line}\n"
        "Curriculum coverage\n"
        f"{segs_line}\n"
        f"{missing_line}\n"
        f"{next_line}\n"
        "Session metrics\n"
        f"- Backtrack ratio: {backtrack_ratio}.\n"
        f"- Student questions: {sq}.\n"
    )


def report_has_required_structure(report_text: str) -> bool:
    if not report_text:
        return False
    required = [
        "Clinical performance note",
        "Teaching feedback note",
        "Curriculum coverage",
        "Session metrics",
    ]
    return all(h in report_text for h in required)
