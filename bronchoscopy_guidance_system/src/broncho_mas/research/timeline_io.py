from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List


def load_session_metrics(recording_dir: str, airway_visit_order: List[str]) -> Dict[str, Any]:
    """Load session-level metrics from timeline.json (pure I/O + parsing)."""
    if not recording_dir:
        return {}
    timeline_path = Path(recording_dir) / "timeline.json"
    if not timeline_path.exists():
        return {}

    try:
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[mas] failed to read {timeline_path}: {exc}")
        return {}

    if not isinstance(timeline, list) or not timeline:
        return {}

    frames = [row for row in timeline if isinstance(row, dict) and "_meta" not in row]
    if not frames:
        return {}

    max_t = 0.0
    for row in frames:
        t_rel = row.get("t_rel")
        if isinstance(t_rel, (int, float)):
            max_t = max(max_t, float(t_rel))

    order = [x.upper() for x in (airway_visit_order or [])]
    order_set = set(order)
    reached: set[str] = set()

    for row in frames:
        rr = row.get("reached_regions")
        if isinstance(rr, list):
            for r in rr:
                r_up = str(r).upper()
                if r_up in order_set:
                    reached.add(r_up)

        if row.get("just_reached") and row.get("current_target"):
            ct = str(row.get("current_target")).upper()
            if ct in order_set:
                reached.add(ct)

    reached_in_order = [r for r in order if r in reached]
    coverage_ratio = round(len(reached_in_order) / max(len(order), 1), 3)

    # anatomical_position fallback (if logs only have anatomical positions)
    for row in frames:
        pos = row.get("anatomical_position")
        if pos:
            pos_up = str(pos).upper()
            if str(row.get("current_target") or "").lower() != "back":
                if pos_up in order_set:
                    reached.add(pos_up)

    next_airway = ""
    for r in order:
        if r not in reached:
            next_airway = r
            break

    back_frames = sum(1 for row in frames if str(row.get("current_target") or "").lower() == "back")
    backtrack_ratio = round(back_frames / max(len(frames), 1), 3)

    return {
        "recording_dir": os.fspath(recording_dir),
        "duration_seconds": round(max_t, 2),
        "timeline_frames": len(frames),
        "coverage_ratio": coverage_ratio,
        "reached_in_curriculum": reached_in_order,
        "next_airway": next_airway,
        "backtrack_ratio": backtrack_ratio,
    }
