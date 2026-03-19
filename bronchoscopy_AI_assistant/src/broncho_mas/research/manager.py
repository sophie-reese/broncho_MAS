"""
Research manager updated to use shared logging and shared event_signal.

Design goals:
- Keep the original research flow intact.
- Add shared RunLogger support (meta.json + timeline.jsonl + errors.jsonl).
- Add EventSignalEngine support so research and runtime share the same event layer.
- Preserve legacy-safe output behavior for online.py / notebook integration.
"""

from __future__ import annotations

import ast
import json
import os
import platform
import re
import time
from typing import Any, Dict, List, Optional, Union

from smolagents import CodeAgent, ToolCallingAgent

# --- import compatibility layer ---
try:
    from ..shared.model_selector import create_model
except Exception:
    from .model_selector import create_model  # type: ignore

try:
    from ..shared.logging_utils import RunLogger
except Exception:
    from .logging_utils import RunLogger  # type: ignore

try:
    from ..shared.event_signal import EventSignalEngine
except Exception:
    EventSignalEngine = None  # type: ignore

try:
    from ..shared.directional_hint_builder import DirectionalHintBuilder
except Exception:
    from directional_hint_builder import DirectionalHintBuilder  # type: ignore

from .util import validate_instructor_payload, validate_statistics_payload

# Prefer shared curriculum only if it supports make_plan; otherwise use research one.
try:
    from ..shared.curriculum import CurriculumEngine as _SharedCurriculumEngine  # type: ignore
except Exception:
    _SharedCurriculumEngine = None  # type: ignore

try:
    from .curriculum_engine import CurriculumEngine as _ResearchCurriculumEngine  # type: ignore
except Exception:
    _ResearchCurriculumEngine = None  # type: ignore

from .tools import (
    curriculum_progress_tool,
    landmark_lookup_tool,
    score_session_tool,
    submit_guidance,
    submit_statistics,
)
try:
    from ..shared.timeline import load_session_metrics  # type: ignore
except Exception:
    try:
        from .timeline import load_session_metrics  # type: ignore
    except Exception:
        from .timeline_io import load_session_metrics  # type: ignore
from .reporting import build_core_report

# Prompt builders moved in the new layout; keep fallback to old local prompts.
try:
    from ..shared.prompting import (  # type: ignore
        build_instructor_prompt,
        build_report_prompt,
        build_statistics_prompt,
    )
except Exception:
    from .prompts import build_instructor_prompt, build_report_prompt, build_statistics_prompt  # type: ignore


ANCHOR_AIRWAYS = {"CARINA", "LMB", "RMB", "TRACHEA"}


def _extract_json_object(text: Union[str, Dict[str, Any], Any]) -> Dict[str, Any]:
    """Extract a JSON object robustly from dicts, pure JSON strings, or text containing JSON."""
    if isinstance(text, dict):
        return text
    if not text:
        return {}
    text = str(text).strip()

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
    return {}


def _coerce_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return bool(val)


def _unwrap_tool_arguments(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Accept both direct business payloads and tool-wrapper payloads."""
    if not isinstance(obj, dict):
        return {}

    if "arguments" in obj and isinstance(obj["arguments"], dict):
        return obj["arguments"]

    if obj.get("name") in {"submit_guidance", "submit_statistics"} and isinstance(obj.get("arguments"), dict):
        return obj["arguments"]

    fn = obj.get("function")
    if isinstance(fn, dict) and fn.get("name") in {"submit_guidance", "submit_statistics"}:
        args = fn.get("arguments", {})
        if isinstance(args, str):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(args)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            return {}
        return args if isinstance(args, dict) else {}

    return obj


def _ensure_dict(val: Any) -> Dict[str, Any]:
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            obj = json.loads(val)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
    return {}


def _coerce_instructor_result(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        raw = _unwrap_tool_arguments(raw)
        return {
            "utterance": str(raw.get("utterance", "")).strip(),
            "needs_visual_guidance": _coerce_bool(raw.get("needs_visual_guidance", False)),
        }

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            obj = _unwrap_tool_arguments(item)
            utterance = str(obj.get("utterance", "")).strip()
            if utterance:
                return {
                    "utterance": utterance,
                    "needs_visual_guidance": _coerce_bool(obj.get("needs_visual_guidance", False)),
                }

    text = str(raw or "").strip()
    if not text:
        return {"utterance": "", "needs_visual_guidance": False}

    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(text)
            return _coerce_instructor_result(obj)
        except Exception:
            pass

    embedded = _extract_json_object(text)
    embedded = _unwrap_tool_arguments(embedded)
    if embedded:
        return {
            "utterance": str(embedded.get("utterance", "")).strip(),
            "needs_visual_guidance": _coerce_bool(embedded.get("needs_visual_guidance", False)),
        }

    return {
        "utterance": text,
        "needs_visual_guidance": False,
    }


class MultiAgentManager:
    """Bronchoscopy MAS manager with shared logging + shared event signal."""

    AIRWAY_VISIT_ORDER: List[str] = [
        "RB1", "RB2", "RB3", "RB4", "RB5", "RB6", "RB7", "RB8", "RB9", "RB10",
        "LB1+2", "LB3", "LB4", "LB5", "LB6", "LB8", "LB9", "LB10",
    ]

    def __init__(
        self,
        model_name: Optional[str] = None,
        log_root: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        if model_name:
            os.environ["BRONCHO_MODEL"] = model_name
        os.environ.setdefault("BRONCHO_PROVIDER", "hf")
        self.model = create_model(None)

        self.engine = self._make_engine()
        self.event_engine = EventSignalEngine() if EventSignalEngine is not None else None
        self.directional_builder = DirectionalHintBuilder()

        self.statistics_agent = ToolCallingAgent(
            tools=[submit_statistics],
            model=self.model,
            name="statistics",
            description="Analyzes performance and calls submit_statistics.",
            max_steps=2,
        )

        self.instructor_agent = ToolCallingAgent(
            tools=[submit_guidance],
            model=self.model,
            name="instructor",
            description="Provides coaching and calls submit_guidance.",
            max_steps=2,
        )

        self.report_agent = ToolCallingAgent(
            tools=[],
            model=self.model,
            name="report_writer",
            description="Writes reports.",
            max_steps=2,
        )

        self.orchestrator = CodeAgent(
            tools=[curriculum_progress_tool, landmark_lookup_tool],
            model=self.model,
            additional_authorized_imports=["json"],
            max_steps=4,
        )

        self._turn_idx = 0
        self._last_ui_text = ""

        self.logger = RunLogger(
            log_root=log_root,
            session_id=session_id,
            pipeline="research",
        )
        self._write_run_meta()

        print(
            f"[broncho_mas manager] version=0.0.4-research "
            f"research_mode=agentic provider={os.environ.get('BRONCHO_PROVIDER', 'hf')} "
            f"model={os.environ.get('BRONCHO_MODEL', '')}"
        )

        self._ORCH_RULES = (
            "ORCHESTRATOR RULES:\n"
            "- Call curriculum_progress_tool first.\n"
            "- Then call landmark_lookup_tool.\n"
            "- Finally output ONE JSON object via final_answer.\n"
            "- Output keys: curriculum_progress, landmark_hint.\n"
        )

    def _write_run_meta(self) -> None:
        try:
            self.logger.write_meta(
                {
                    "schema": "broncho.meta.v2",
                    "system_family": "broncho_mas",
                    "pipeline": "research",
                    "manager": "MultiAgentManager",
                    "broncho_mas_version": "0.0.4-research",
                    "provider": os.environ.get("BRONCHO_PROVIDER", "hf"),
                    "model": os.environ.get("BRONCHO_MODEL", ""),
                    "tool_choice": os.environ.get("BRONCHO_TOOL_CHOICE", ""),
                    "event_signal_enabled": bool(self.event_engine is not None),
                    "curriculum_version": os.environ.get("BRONCHO_CURRICULUM_VERSION", "curriculum.v1"),
                    "prompt_version": os.environ.get("BRONCHO_PROMPT_VERSION", "research_prompt.v1"),
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                }
            )
        except Exception as exc:
            print(f"[research] failed to write meta: {exc}")

    def _log_error(self, where: str, exc: Exception, extra: Optional[Dict[str, Any]] = None) -> None:
        try:
            payload: Dict[str, Any] = {
                "where": where,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            if extra:
                payload["extra"] = extra
            self.logger.append_error(payload)
        except Exception as log_exc:
            print(f"[research] failed to log error: {log_exc}")

    def _log_note(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        try:
            payload: Dict[str, Any] = {"event_type": "note", "message": message}
            if extra:
                payload["extra"] = extra
            self.logger.append_timeline(payload)
        except Exception as exc:
            print(f"[research] failed to log note: {exc}")

    def _make_engine(self):
        if _SharedCurriculumEngine is not None:
            try:
                eng = _SharedCurriculumEngine(self.AIRWAY_VISIT_ORDER)
                if hasattr(eng, "make_plan") and hasattr(eng, "next_airway"):
                    return eng
            except Exception:
                pass

        if _ResearchCurriculumEngine is not None:
            return _ResearchCurriculumEngine(self.AIRWAY_VISIT_ORDER)

        raise RuntimeError("No usable CurriculumEngine found.")


    def _landmark_hint_from_target(self, target_airway: str) -> Dict[str, Any]:
        target = str(target_airway or "").strip().upper()
        if not target or not hasattr(self.engine, "landmark_for_airway"):
            return {}

        try:
            landmark = self.engine.landmark_for_airway(target)
        except Exception:
            return {}

        angles = list(getattr(landmark, "recommended_angles", []) or [])
        return {
            "landmark_id": str(getattr(landmark, "landmark_id", "") or "").strip(),
            "recommended_angles": angles,
            "recommended_angle": angles[0] if angles else {},
            "recognition_cue": str(getattr(landmark, "recognition_cue", "") or "").strip(),
            "target_airway": target,
            "authority": "shared_curriculum",
        }

    def _curriculum_progress_from_truth(self, reached_list: List[str]) -> Dict[str, Any]:
        reached_u = [str(x).upper() for x in (reached_list or [])]
        if hasattr(self.engine, "progress_snapshot"):
            try:
                snap = self.engine.progress_snapshot(reached_u)
                if isinstance(snap, dict):
                    out = dict(snap)
                    out["authority"] = "shared_curriculum"
                    return out
            except Exception:
                pass

        next_airway = self.engine.next_airway(set(reached_u)) if hasattr(self.engine, "next_airway") else ""
        total = len(self.AIRWAY_VISIT_ORDER)
        reached_in_order = [a for a in self.AIRWAY_VISIT_ORDER if a in set(reached_u)]
        return {
            "reached": reached_u,
            "next_airway": next_airway,
            "reached_count": len(reached_in_order),
            "total": total,
            "coverage_ratio": round(len(reached_in_order) / max(total, 1), 4),
            "authority": "shared_curriculum",
        }

    def build_statepacket(
        self,
        current_situation: str,
        previous_msgs: str,
        student_question: str,
        reached_list: List[str],
        curriculum_progress: Dict[str, Any],
        landmark_hint: Dict[str, Any],
        auth_plan: Dict[str, Any],
        raw_payload: Optional[Dict[str, Any]] = None,
        event_packet: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "schema": "statepacket.research.v2",
            "current_situation": current_situation,
            "previous_msgs": previous_msgs,
            "student_question": student_question,
            "reached_regions": [str(x).upper() for x in (reached_list or [])],
            "curriculum_progress": curriculum_progress or {},
            "landmark_hint": landmark_hint or {},
            "auth_plan": auth_plan or {},
            "signals": {
                "eeg": None,
                "event_flag": (event_packet or {}).get("flag"),
                "event_ema": (event_packet or {}).get("ema"),
                "llm_trigger_flag": (event_packet or {}).get("llm_trigger_flag"),
            },
            "event_packet": event_packet or {},
            "raw_payload": raw_payload or {},
            "meta": {
                "broncho_mas_version": "0.0.4-research",
                "provider": os.environ.get("BRONCHO_PROVIDER", "hf"),
                "model": os.environ.get("BRONCHO_MODEL", ""),
            },
        }

    def _extract_visual_context(self, text: str) -> Dict[str, Any]:
        s = str(text or "")

        def _grab(label: str) -> str:
            m = re.search(rf"{label}\s*:\s*([A-Za-z0-9+_-]+)", s, flags=re.I)
            return m.group(1).upper() if m else ""

        return {
            "current_region": _grab("Current region"),
            "target_region": _grab("Target region"),
            "current_airway": _grab("Current airway"),
            "target_airway": _grab("Target airway"),
            "drift": bool(re.search(r"\bdrift\b", s, flags=re.I)),
            "not_centered": bool(re.search(r"not yet centered|not centered|off[- ]center", s, flags=re.I)),
            "wall_facing": bool(re.search(r"wall-facing|red/pink blur|facing the wall", s, flags=re.I)),
        }

    @staticmethod
    def _airway_family(region: str) -> str:
        r = str(region or "").upper()
        if r in {"RB1", "RB2", "RB3"}:
            return "RUL"
        if r in {"RB4", "RB5"}:
            return "RML"
        if r.startswith("RB"):
            return "RLL"
        if r in {"LB1+2", "LB3"}:
            return "LUL"
        if r in {"LB4", "LB5"}:
            return "LINGULA"
        if r.startswith("LB"):
            return "LLL"
        return ""

    def _extract_best_utterance(self, raw: Any, parsed: Dict[str, Any]) -> str:
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                obj = _unwrap_tool_arguments(item)
                utt = str(obj.get("utterance", "")).strip()
                if utt:
                    return utt
        if isinstance(raw, dict):
            obj = _unwrap_tool_arguments(raw)
            utt = str(obj.get("utterance", "")).strip()
            if utt:
                return utt
        utt = str((parsed or {}).get("utterance", "")).strip()
        if utt and not utt.lower().startswith(("guidance delivered:", "student instructed to", "guidance provided")):
            return utt
        return ""

    def _deterministic_guidance_fallback(
        self,
        curriculum_progress: Dict[str, Any],
        landmark_hint: Dict[str, Any],
        auth_plan: Dict[str, Any],
    ) -> str:
        directional = (auth_plan or {}).get("directional_hint", {})
        primary = str(directional.get("primary_action") or "").strip()
        secondary = str(directional.get("secondary_action") or "").strip()
        if primary:
            spoken: List[str] = ["Good.", self.directional_builder._coachify(primary)]
            if secondary:
                spoken.append(self.directional_builder._coachify(secondary))
            return " ".join(spoken[:3]).strip()

        next_airway = str((curriculum_progress or {}).get("next_airway", "")).strip().upper()
        cue = str((landmark_hint or {}).get("recognition_cue", "")).strip()
        angle = (landmark_hint or {}).get("recommended_angle", "")
        if isinstance(angle, dict):
            angle_text = str(angle.get("angle") or angle.get("name") or "").strip()
        else:
            angle_text = str(angle or "").strip()

        micro_steps = (auth_plan or {}).get("micro_steps") or []
        if isinstance(micro_steps, list):
            cleaned = []
            for step in micro_steps:
                s = str(step or "").strip()
                if not s:
                    continue
                if "Check:" in s:
                    s = s.split("Check:", 1)[0].strip()
                if "Action:" in s:
                    s = s.split("Action:", 1)[1].strip()
                cleaned.append(s)
            if cleaned:
                return ". ".join(s.rstrip(".") for s in cleaned[:3]) + "."

        parts = []
        if angle_text:
            parts.append(f"Rotate to {angle_text}.")
        parts.append("Keep the lumen centered.")
        if cue:
            parts.append(f"Look for {cue}.")
        if next_airway:
            parts.append(f"Advance toward {next_airway} in small steps.")
        else:
            parts.append("Advance in small steps.")
        return " ".join(parts).strip()

    def _normalize_ui_text(
        self,
        text: str,
        curriculum_progress: Dict[str, Any],
        landmark_hint: Dict[str, Any],
        auth_plan: Dict[str, Any],
    ) -> str:
        raw = str(text or "").strip()
        if not raw:
            return self._deterministic_guidance_fallback(curriculum_progress, landmark_hint, auth_plan)

        x = raw
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(x)
                coerced = _coerce_instructor_result(parsed)
                utt = str(coerced.get("utterance", "")).strip()
                if utt:
                    x = utt
                    break
            except Exception:
                pass

        x = re.sub(r"(?is)^guidance\s+provided\s+to\s+(student|students)\s*:\s*", "", x)
        x = re.sub(r"(?is)^guidance\s+delivered\s*:\s*", "", x)
        x = re.sub(r"(?is)^student\s+instructed\s+to\s*", "", x)
        x = re.sub(r"(?is)^spoken\s+guidance\s*:\s*", "", x)
        x = re.sub(r"(?is)^instruction(al)?\s+guidance\s*:\s*", "", x)
        x = x.replace("**", "").replace("•", " ")
        x = x.replace("Mercedes sign / trifurcation", "Mercedes sign")
        x = x.replace("Y-shaped trifurcation", "Mercedes sign")
        x = re.sub(r"\s+", " ", x).strip()

        if not x:
            return self._deterministic_guidance_fallback(curriculum_progress, landmark_hint, auth_plan)

        chunks = re.split(r"(?<=[.!?])\s+", x)
        cleaned: List[str] = []
        for chunk in chunks:
            s = str(chunk or "").strip(" -:;")
            if not s:
                continue
            lower = s.lower()
            if lower.startswith((
                "guidance delivered", "student instructed to", "guidance provided",
                "submit_guidance", "final_answer", "calling tool", "tool call"
            )):
                continue
            if s not in cleaned:
                cleaned.append(s)

        out = " ".join(cleaned).strip()
        out = re.sub(r"\s+", " ", out).strip()
        bad = ["submit_guidance", "guidance delivered", "student instructed to", "'function'", '"function"', "'arguments'", '"arguments"']
        if not out or any(b in out.lower() for b in bad):
            return self._deterministic_guidance_fallback(curriculum_progress, landmark_hint, auth_plan)
        return out

    def _content_conflicts(self, text: str, visual_ctx: Dict[str, Any]) -> bool:
        s = str(text or "").lower()
        if not s:
            return True
        target_family = self._airway_family(visual_ctx.get("target_region", "") or visual_ctx.get("target_airway", ""))
        if s.startswith(("guidance delivered:", "student instructed to", "guidance provided")):
            return True
        if target_family == "RUL" and any(k in s for k in ["right lower lobe", "lower lobe branch", "rb6", "rb7", "rb8", "rb9", "rb10"]):
            return True
        if target_family == "RML" and any(k in s for k in ["right lower lobe", "rb6", "rb7", "rb8", "rb9", "rb10"]):
            return True
        if target_family == "RLL" and any(k in s for k in ["right upper lobe", "mercedes sign"]):
            return True
        return False

    def _bedside_safe_guidance(
        self,
        visual_ctx: Dict[str, Any],
        curriculum_progress: Dict[str, Any],
        landmark_hint: Dict[str, Any],
        auth_plan: Dict[str, Any],
    ) -> str:
        current_region = str((visual_ctx or {}).get("current_region") or (visual_ctx or {}).get("current_airway") or "").upper()
        target_region = str((visual_ctx or {}).get("target_region") or (visual_ctx or {}).get("target_airway") or "").upper()
        drift = bool((visual_ctx or {}).get("drift"))
        not_centered = bool((visual_ctx or {}).get("not_centered"))
        wall_facing = bool((visual_ctx or {}).get("wall_facing"))

        if wall_facing:
            return "Pull back now. Re-center the lumen before advancing."

        parts: List[str] = []
        if drift or not_centered:
            parts.append("Pause and re-center the lumen")
        elif current_region:
            parts.append("Hold the view steady")

        if target_region:
            fam = self._airway_family(target_region)
            if current_region and current_region != target_region:
                if fam == "RUL":
                    parts.append(f"Stay in the right upper lobe view and identify {target_region} before advancing")
                elif fam == "RML":
                    parts.append(f"Identify the right middle lobe opening for {target_region} before advancing")
                elif fam == "RLL":
                    parts.append(f"Identify the {target_region} opening before advancing")
                elif fam == "LUL":
                    parts.append(f"Stay in the left upper lobe view and identify {target_region} before advancing")
                elif fam == "LINGULA":
                    parts.append(f"Identify the lingular opening for {target_region} before advancing")
                elif fam == "LLL":
                    parts.append(f"Identify the {target_region} opening before advancing")
                else:
                    parts.append(f"Identify the {target_region} opening before advancing")
            else:
                parts.append(f"Confirm the {target_region} opening before advancing")
        else:
            return self._deterministic_guidance_fallback(curriculum_progress, landmark_hint, auth_plan)

        parts.append("Use small controlled movements")
        return ". ".join(p.strip().rstrip(".") for p in parts if p.strip()) + "."

    def _build_event_frame(
        self,
        current_situation: str,
        student_question: str,
        visual_ctx: Dict[str, Any],
        reached_list: List[str],
        curriculum_progress: Dict[str, Any],
    ) -> Dict[str, Any]:
        current_airway = (
            visual_ctx.get("current_region")
            or visual_ctx.get("current_airway")
            or (reached_list[-1] if reached_list else "")
        )
        current_target = (
            visual_ctx.get("target_region")
            or visual_ctx.get("target_airway")
            or curriculum_progress.get("next_airway", "")
        )
        return {
            "anatomical_position": current_airway,
            "current_target": current_target,
            "next_destination": curriculum_progress.get("next_airway", ""),
            "reached_regions": [str(x).upper() for x in reached_list],
            "just_reached": False,
            "student_question": student_question,
            "drift_detected": bool(visual_ctx.get("drift")),
            "need_recenter": bool(visual_ctx.get("not_centered") or visual_ctx.get("wall_facing")),
            "raw_current_situation": current_situation,
        }

    def _extract_control_triplet(self, text: str) -> Optional[List[float]]:
        s = str(text or "")
        patterns = [
            r"m_jointsVelRel\s*[:=]\s*(\[[^\]]+\])",
            r"joints_vel_rel\s*[:=]\s*(\[[^\]]+\])",
            r"control_triplet\s*[:=]\s*(\[[^\]]+\])",
        ]
        for pat in patterns:
            m = re.search(pat, s, flags=re.I)
            if not m:
                continue
            chunk = m.group(1)
            for parser in (json.loads, ast.literal_eval):
                try:
                    val = parser(chunk)
                    if isinstance(val, (list, tuple)) and len(val) >= 3:
                        return [float(val[0]), float(val[1]), float(val[2])]
                except Exception:
                    pass
        return None

    def _extract_block(self, prompt: str, header: str) -> str:
        if not prompt:
            return ""
        marker = f"{header}:"
        idx = prompt.find(marker)
        if idx == -1:
            return ""
        tail = prompt[idx + len(marker) :]
        m = re.search(r"\n[A-Z_]+:\s*\n", tail)
        if m:
            return tail[: m.start()].strip()
        return tail.strip()

    def _extract_reached_regions(self, current_situation: str) -> List[str]:
        text = current_situation or ""
        patterns = [
            r"reached_regions\(last\)\s*=\s*(\[[^\]]*\])",
            r"REACHED_REGIONS\s*:\s*(\[[^\]]*\])",
            r"reached_regions\s*:\s*(\[[^\]]*\])",
            r"timeline_reached\s*:\s*(\[[^\]]*\])",
            r"regions_seen\s*:\s*(\[[^\]]*\])",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if not m:
                continue
            chunk = m.group(1)
            for parser in (json.loads, ast.literal_eval):
                try:
                    val = parser(chunk)
                    if isinstance(val, list):
                        return [str(x) for x in val]
                except Exception:
                    pass
        return []

    def run(self, prompt: str) -> Dict[str, Any]:
        t0 = time.time()
        self._turn_idx += 1

        try:
            current_situation = self._extract_block(prompt, "CURRENT_SITUATION")
            previous_msgs = self._extract_block(prompt, "PREVIOUS_MSGS")
            student_question = self._extract_block(prompt, "STUDENT_QUESTION")

            if not current_situation and not student_question:
                current_situation = prompt.strip()

            visual_ctx = self._extract_visual_context(current_situation + "\n" + student_question)
            reached_list = self._extract_reached_regions(current_situation)

            reached_json = json.dumps([str(x).upper() for x in reached_list])
            order_json = json.dumps(self.AIRWAY_VISIT_ORDER)

            orch_task = (
                f"{self._ORCH_RULES}\n\n"
                "INPUTS:\n"
                f"- regions_seen_json: {reached_json}\n"
                f"- curriculum_order_json: {order_json}\n"
            )

            t_orch = time.perf_counter()
            plan_raw = self.orchestrator.run(orch_task)
            orch_latency_ms = round((time.perf_counter() - t_orch) * 1000.0, 2)
            plan = _extract_json_object(plan_raw)

            cp_agent = _ensure_dict(plan.get("curriculum_progress", {}))
            cp_agent = _unwrap_tool_arguments(cp_agent)
            lh_agent = _ensure_dict(plan.get("landmark_hint", {}))
            lh_agent = _unwrap_tool_arguments(lh_agent)

            cp = self._curriculum_progress_from_truth(reached_list)
            cp["agent_curriculum_progress"] = cp_agent

            next_airway = str(cp.get("next_airway", "") or "").strip().upper()
            lh = self._landmark_hint_from_target(next_airway)
            if lh_agent:
                lh["agent_landmark_hint"] = lh_agent

            latest_event = {
                "current_situation": current_situation,
                "student_question": student_question,
                "timeline_reached": reached_list,
            }

            stats_prompt = build_statistics_prompt(
                curriculum_progress=cp,
                latest_event=latest_event,
                landmark_hint=lh,
            )
            t_stats = time.perf_counter()
            stats_raw = self.statistics_agent.run(stats_prompt)
            stats_latency_ms = round((time.perf_counter() - t_stats) * 1000.0, 2)
            stats_obj = _extract_json_object(stats_raw)
            stats_obj = _unwrap_tool_arguments(stats_obj)

            if stats_obj:
                if "key_habit" in stats_obj and "coach_focus_next" not in stats_obj:
                    stats_obj["coach_focus_next"] = stats_obj.get("key_habit", "")
                if "analysis" in stats_obj and "notes" not in stats_obj:
                    stats_obj["notes"] = stats_obj.get("analysis", "")
                if "likely_issue" not in stats_obj:
                    stats_obj["likely_issue"] = stats_obj.get("analysis", "") or "navigation needs improvement"

            # Side-channel analytics only.
            # Keep this agent in the multi-agent architecture for later expansion,
            # but do not use it to condition real-time instructor utterance.
            stats_valid, stats_errs = validate_statistics_payload(stats_obj)

            current_airway = (
                visual_ctx.get("current_region")
                or visual_ctx.get("current_airway")
                or (cp.get("reached") or [reached_list[-1] if reached_list else ""])[-1]
            )
            auth_plan_json = self.engine.make_plan(
                current_airway=str(current_airway).upper(),
                reached={str(x).upper() for x in reached_list},
                student_question=student_question,
                requested_next_airway=str(cp.get("next_airway", "") or "").strip().upper(),
            )
            auth_plan_json["curriculum_authority"] = "shared_curriculum"

            event_frame = self._build_event_frame(
                current_situation=current_situation,
                student_question=student_question,
                visual_ctx=visual_ctx,
                reached_list=reached_list,
                curriculum_progress=cp,
            )

            event_packet: Dict[str, Any] = {}
            if self.event_engine is not None:
                try:
                    event_packet = self.event_engine.step(event_frame, history=None, plan=auth_plan_json)
                except Exception as exc:
                    self._log_error("event_engine.step", exc)
                    event_packet = {}

            control_triplet = self._extract_control_triplet(current_situation + "\n" + student_question)
            if control_triplet is not None:
                try:
                    hint = self.directional_builder.build(
                        control_triplet,
                        event_flag=event_packet.get("flag") if isinstance(event_packet, dict) else None,
                    )
                    auth_plan_json["directional_hint"] = hint.to_dict()
                except Exception as exc:
                    self._log_error("directional_hint", exc)

            instructor_prompt = build_instructor_prompt(
                plan=auth_plan_json,
                current=current_situation,
                previous_msgs=previous_msgs,
                student_q=student_question,
            )
            directional = auth_plan_json.get("directional_hint")
            if directional:
                instructor_prompt += "\n\nDIRECTIONAL_HINT:\n" + json.dumps(directional, ensure_ascii=False)
            event_soft_prompt = str(event_packet.get("soft_prompt") or "").strip() if isinstance(event_packet, dict) else ""
            if event_soft_prompt:
                instructor_prompt += "\n\nEVENT_CONTEXT:\n" + event_soft_prompt

            t_instr = time.perf_counter()
            instr_raw = self.instructor_agent.run(instructor_prompt)
            instr_latency_ms = round((time.perf_counter() - t_instr) * 1000.0, 2)
            instr_candidate = _coerce_instructor_result(instr_raw)
            instr_json, instr_errs = validate_instructor_payload(instr_candidate)

            best_utterance = self._extract_best_utterance(instr_raw, instr_json)
            if not best_utterance:
                best_utterance = str(instr_json.get("utterance", "")).strip()

            has_visual_context = bool(visual_ctx.get("target_region") or visual_ctx.get("current_region") or visual_ctx.get("target_airway") or visual_ctx.get("current_airway"))
            has_conflict = self._content_conflicts(best_utterance, visual_ctx)

            if has_visual_context and has_conflict:
                ui_text = self._bedside_safe_guidance(
                    visual_ctx=visual_ctx,
                    curriculum_progress=cp,
                    landmark_hint=lh,
                    auth_plan=auth_plan_json,
                )
            else:
                if has_conflict:
                    best_utterance = ""
                if not best_utterance:
                    best_utterance = self._deterministic_guidance_fallback(
                        curriculum_progress=cp,
                        landmark_hint=lh,
                        auth_plan=auth_plan_json,
                    )
                ui_text = self._normalize_ui_text(
                    best_utterance,
                    curriculum_progress=cp,
                    landmark_hint=lh,
                    auth_plan=auth_plan_json,
                )

            if not ui_text:
                ui_text = self._deterministic_guidance_fallback(
                    curriculum_progress=cp,
                    landmark_hint=lh,
                    auth_plan=auth_plan_json,
                )
            instr_json["utterance"] = ui_text

            self._last_ui_text = ui_text

            statepacket = self.build_statepacket(
                current_situation=current_situation,
                previous_msgs=previous_msgs,
                student_question=student_question,
                reached_list=reached_list,
                curriculum_progress=cp,
                landmark_hint=lh,
                auth_plan=auth_plan_json,
                raw_payload={"prompt": prompt},
                event_packet=event_packet,
            )

            stats_valid["curriculum"] = cp
            stats_valid["landmark_hint"] = lh

            total_latency_ms = int((time.time() - t0) * 1000)

            # Shared step event
            try:
                self.logger.log_step(
                    step_index=self._turn_idx,
                    state={
                        "anatomical_position": event_frame.get("anatomical_position"),
                        "current_target": event_frame.get("current_target"),
                        "next_destination": event_frame.get("next_destination"),
                        "reached_regions": event_frame.get("reached_regions"),
                        "just_reached": event_frame.get("just_reached"),
                        "current_airway": current_airway,
                    },
                    control={
                        "event_flag": event_packet.get("flag"),
                        "event_ema": event_packet.get("ema"),
                        "llm_trigger_flag": event_packet.get("llm_trigger_flag"),
                        "llm_call_started": event_packet.get("llm_call_started"),
                        "visual_guidance_required_flag": event_packet.get("visual_guidance_required_flag"),
                        "pure_voice_guidance_flag": event_packet.get("pure_voice_guidance_flag"),
                    },
                    guidance={
                        "final_ui_text": ui_text,
                        "utterance_full": ui_text,
                    },
                    plan=auth_plan_json,
                    statistics=stats_valid,
                    extra={
                        "curriculum_progress": cp,
                        "landmark_hint": lh,
                        "statepacket": statepacket,
                        "event_packet": event_packet,
                        "latency_ms": total_latency_ms,
                    },
                )
            except Exception as exc:
                self._log_error("logger.log_step", exc)

            # Agent-turn events
            try:
                self.logger.log_agent_turn(
                    step_index=self._turn_idx,
                    agent_name="orchestrator",
                    role="curriculum_and_landmark_lookup",
                    input_data={
                        "regions_seen_json": reached_json,
                        "curriculum_order_json": order_json,
                        "task": orch_task,
                    },
                    output_data={
                        "raw": plan_raw,
                        "curriculum_progress": cp,
                        "landmark_hint": lh,
                    },
                    extra={"latency_ms": orch_latency_ms},
                )
                self.logger.log_agent_turn(
                    step_index=self._turn_idx,
                    agent_name="statistics",
                    role="performance_analysis",
                    input_data={
                        "prompt": stats_prompt,
                        "curriculum_progress": cp,
                        "latest_event": latest_event,
                        "landmark_hint": lh,
                    },
                    output_data={
                        "raw": stats_raw,
                        "validated": stats_valid,
                        "errors": stats_errs,
                    },
                    extra={"latency_ms": stats_latency_ms},
                )
                self.logger.log_agent_turn(
                    step_index=self._turn_idx,
                    agent_name="instructor",
                    role="guidance_generation",
                    input_data={
                        "prompt": instructor_prompt,
                        "plan": auth_plan_json,
                        "current": current_situation,
                        "previous_msgs": previous_msgs,
                        "student_q": student_question,
                    },
                    output_data={
                        "raw": instr_raw,
                        "coerced": instr_candidate,
                        "validated": instr_json,
                        "errors": instr_errs,
                    },
                    extra={"latency_ms": instr_latency_ms, "accepted_ui_text": ui_text},
                )
            except Exception as exc:
                self._log_error("logger.log_agent_turn", exc)

            # LLM-call style timeline events for analysis symmetry with runtime.
            try:
                self.logger.log_llm_call(
                    step_index=self._turn_idx,
                    provider=os.environ.get("BRONCHO_PROVIDER", "hf"),
                    model=os.environ.get("BRONCHO_MODEL", ""),
                    prompt=stats_prompt,
                    response=str(stats_raw),
                    success=bool(stats_raw),
                    latency_ms=stats_latency_ms,
                    extra={"agent_name": "statistics"},
                )
                self.logger.log_llm_call(
                    step_index=self._turn_idx,
                    provider=os.environ.get("BRONCHO_PROVIDER", "hf"),
                    model=os.environ.get("BRONCHO_MODEL", ""),
                    prompt=instructor_prompt,
                    response=str(instr_raw),
                    success=bool(instr_raw),
                    latency_ms=instr_latency_ms,
                    extra={"agent_name": "instructor", "accepted_ui_text": ui_text},
                )
            except Exception as exc:
                self._log_error("logger.log_llm_call", exc)

            print("[research] final ui_text =", repr(ui_text))

            return {
                "ui_text": ui_text,
                "needs_visual_guidance": bool(instr_json.get("needs_visual_guidance", False)),
                "statistics": stats_valid,
                "curriculum_progress": cp,
                "landmark_hint": lh,
                "plan_json": auth_plan_json,
                "statepacket": statepacket,
                "event_packet": event_packet,
                "raw": {
                    "stats_raw": stats_raw,
                    "instr_raw": instr_raw,
                    "orchestrator_raw": plan_raw,
                },
            }
        except Exception as exc:
            self._log_error("MultiAgentManager.run", exc)
            raise

    def get_report(self, recording_dir: str) -> str:
        if not recording_dir or not os.path.exists(recording_dir):
            return "Error: Recording directory not found."

        metrics = load_session_metrics(recording_dir, self.AIRWAY_VISIT_ORDER)

        timeline_path = os.path.join(recording_dir, "timeline.json")
        timeline_jsonl_path = os.path.join(recording_dir, "timeline.jsonl")
        timeline_txt = "[]"
        try:
            if os.path.exists(timeline_path):
                with open(timeline_path, "r", encoding="utf-8") as f:
                    timeline_txt = f.read()
            elif os.path.exists(timeline_jsonl_path):
                rows = []
                with open(timeline_jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            continue
                timeline_txt = json.dumps(rows, ensure_ascii=False)
        except Exception:
            timeline_txt = "[]"

        scores_json = score_session_tool(timeline_txt, json.dumps(self.AIRWAY_VISIT_ORDER))
        scores = json.loads(scores_json)

        core_report = build_core_report(
            allowed_reached=metrics.get("reached_in_curriculum", []),
            visit_order=self.AIRWAY_VISIT_ORDER,
            curriculum_progress={
                "coverage_ratio": metrics.get("coverage_ratio", 0),
                "reached_count": len(metrics.get("reached_in_curriculum", [])),
                "total": len(self.AIRWAY_VISIT_ORDER),
                "next_airway": metrics.get("next_airway", ""),
            },
            session_metrics=metrics,
            sp_score=0.0,
        )

        prompt = build_report_prompt(
            core_report=core_report,
            session_scores=scores,
            curriculum_progress=metrics,
            session_metrics=metrics,
        )

        return str(self.report_agent.run(prompt))
