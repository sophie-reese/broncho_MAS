from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# Timeline write side
# ---------------------------------------------------------------------

class TimelineLogger:
    def __init__(self, save_path: str, auto_flush_every: int = 1):
        self.save_path = Path(save_path)
        self.auto_flush_every = max(1, int(auto_flush_every))
        self.rows: List[Dict[str, Any]] = []

    def append(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) % self.auto_flush_every == 0:
            self.flush()

    def flush(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_path.write_text(
            json.dumps(self.rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self.rows = []

    def last(self) -> Optional[Dict[str, Any]]:
        return self.rows[-1] if self.rows else None


def build_timeline_entry(
    *,
    iteration: int,
    frame_state: Dict[str, Any],
    event_packet: Optional[Dict[str, Any]] = None,
    runtime_result: Optional[Dict[str, Any]] = None,
    llm_prompt: Optional[str] = None,
    llm_called: bool = False,
) -> Dict[str, Any]:
    event_packet = event_packet or {}
    runtime_result = runtime_result or {}
    raw = runtime_result.get("raw", {}) if isinstance(runtime_result, dict) else {}

    return {
        "iteration": iteration,
        "t_rel": frame_state.get("t_rel"),
        "time": frame_state.get("time"),
        "time_raw": frame_state.get("time_raw"),
        "mode": frame_state.get("mode"),
        "visualization_mode": frame_state.get("visualization_mode"),
        "robot_joints": frame_state.get("robot_joints"),
        "m_jointsVelRel": frame_state.get("m_jointsVelRel"),
        "bounding_boxes": frame_state.get("bounding_boxes"),
        "AnatomicalCoherence": frame_state.get("AnatomicalCoherence"),
        "anatomical_position": frame_state.get("anatomical_position"),
        "next_destination": frame_state.get("next_destination"),
        "current_target": frame_state.get("current_target"),
        "reached_regions": frame_state.get("reached_regions"),
        "just_reached": frame_state.get("just_reached", False),
        "llm_trigger_flag": bool(event_packet.get("llm_trigger_flag", False)),
        "event_flag": event_packet.get("flag"),
        "event_ema": event_packet.get("ema"),
        "llm_reason": event_packet.get("reason"),
        "soft_prompt": event_packet.get("soft_prompt"),
        "llm_call_started": bool(llm_called),
        "llm_prompt": llm_prompt,
        "student_question_current": frame_state.get("student_question_current"),
        "visual_guidance_required_flag": frame_state.get("visual_guidance_required_flag", False),
        "pure_voice_guidance_flag": frame_state.get("pure_voice_guidance_flag", False),
        "deterministic_ui": raw.get("deterministic_ui"),
        "llm_ui": raw.get("llm_ui"),
        "final_ui_text": runtime_result.get("ui_text") if isinstance(runtime_result, dict) else None,
        "utterance_full": runtime_result.get("utterance_full") if isinstance(runtime_result, dict) else None,
        "plan_json": runtime_result.get("plan_json") if isinstance(runtime_result, dict) else None,
        "statistics": runtime_result.get("statistics") if isinstance(runtime_result, dict) else None,
    }


# ---------------------------------------------------------------------
# Timeline read side
# ---------------------------------------------------------------------


def _safe_load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[mas] failed to read {path}: {exc}")
        return None



def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    except Exception as exc:
        print(f"[mas] failed to read {path}: {exc}")
    return rows



def _frames_from_legacy_timeline(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if isinstance(row, dict) and "_meta" not in row]



def _frames_from_shared_timeline(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("event_type") != "step":
            continue

        state = event.get("state") or {}
        if not isinstance(state, dict):
            state = {}

        extra = event.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {}

        frame = {
            "anatomical_position": state.get("anatomical_position") or state.get("current_airway") or "",
            "current_target": state.get("current_target") or state.get("next_destination") or "",
            "reached_regions": state.get("reached_regions") or [],
            "just_reached": state.get("just_reached", False),
        }

        latency_ms = extra.get("latency_ms")
        if isinstance(latency_ms, (int, float)):
            frame["t_rel"] = round(float(latency_ms) / 1000.0, 3)
        else:
            ts = event.get("ts")
            if isinstance(ts, (int, float)):
                frame["t_rel"] = float(ts)

        frames.append(frame)
    return frames



def load_session_metrics(recording_dir: str, airway_visit_order: List[str]) -> Dict[str, Any]:
    """Load session metrics from either legacy timeline.json or shared timeline.jsonl."""
    if not recording_dir:
        return {}

    recording_path = Path(recording_dir)
    legacy_path = recording_path / "timeline.json"
    shared_path = recording_path / "timeline.jsonl"

    frames: List[Dict[str, Any]] = []

    if legacy_path.exists():
        timeline = _safe_load_json(legacy_path)
        if isinstance(timeline, list):
            frames = _frames_from_legacy_timeline(timeline)
    elif shared_path.exists():
        events = _load_jsonl(shared_path)
        frames = _frames_from_shared_timeline(events)
    else:
        return {}

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
        "timeline_format": "legacy_json" if legacy_path.exists() else "shared_jsonl",
    }
