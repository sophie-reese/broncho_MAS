from __future__ import annotations

import json
import os
import platform
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


PathLike = Union[str, Path]


def _session_id_now() -> str:
    """Return a high-resolution session id to avoid collisions."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _looks_like_session_dir(path: Path) -> bool:
    name = str(path.name or "")
    if re.fullmatch(r"\d{8}_\d{6}", name):
        return True
    if re.fullmatch(r"\d{8}_\d{6}_\d{6}", name):
        return True
    markers = ("meta.json", "timeline.jsonl", "errors.jsonl")
    return any((path / marker).exists() for marker in markers)


def _resolve_existing_run_dir(
    log_root: Optional[PathLike] = None,
    session_id: Optional[str] = None,
) -> Optional[Path]:
    if session_id is not None and str(session_id).strip():
        return None

    env_run_dir = os.environ.get("BRONCHO_RUN_DIR") or os.environ.get("BRONCHO_RECORDING_DIR")
    if env_run_dir and str(env_run_dir).strip():
        candidate = Path(env_run_dir).expanduser().resolve()
        if candidate.exists():
            return candidate

    if log_root is not None and str(log_root).strip():
        candidate = Path(log_root).expanduser().resolve()
        if candidate.exists() and _looks_like_session_dir(candidate):
            return candidate
    return None



def _resolve_log_root(log_root: Optional[PathLike] = None) -> Path:
    """
    Resolve the root log directory.

    New rule:
    - If log_root is explicitly given, use it.
    - Otherwise always create and use ./mas_log under the current run directory.
    """
    if log_root is not None and str(log_root).strip():
        root = Path(log_root).expanduser().resolve()
    else:
        root = (Path.cwd() / "mas_log").resolve()

    root.mkdir(parents=True, exist_ok=True)
    os.environ["BRONCHO_LOG_ROOT"] = str(root)
    return root


class RunLogger:
    """Shared append-only JSONL logger for runtime and research pipelines."""

    def __init__(
        self,
        log_root: Optional[PathLike] = None,
        session_id: Optional[str] = None,
        pipeline: str = "runtime",
    ):
        existing_run_dir = _resolve_existing_run_dir(log_root=log_root, session_id=session_id)
        if existing_run_dir is not None:
            self.run_dir = existing_run_dir
            self.log_root = self.run_dir.parent
            self.session_id = self.run_dir.name
        else:
            self.log_root = _resolve_log_root(log_root)
            self.session_id = session_id or os.environ.get("BRONCHO_SESSION_ID") or _session_id_now()
            self.run_dir = self.log_root / self.session_id
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = str(pipeline or "runtime")

        os.environ["BRONCHO_LOG_ROOT"] = str(self.log_root)
        os.environ["BRONCHO_RUN_DIR"] = str(self.run_dir)
        os.environ["BRONCHO_SESSION_ID"] = str(self.session_id)
        os.environ.setdefault("BRONCHO_RECORDING_DIR", str(self.run_dir))

        self.meta_path = self.run_dir / "meta.json"
        self.timeline_path = self.run_dir / "timeline.jsonl"
        self.errors_path = self.run_dir / "errors.jsonl"
        self.artifacts_dir = self.run_dir / "artifacts"

        # Useful during debugging: immediately show where logs are going.
        print(f"[broncho_mas] log dir = {self.run_dir}")

    def write_meta(self, meta: Optional[Dict[str, Any]] = None) -> None:
        payload = dict(meta or {})
        payload.setdefault("schema", "broncho.meta.v2")
        payload.setdefault("session_id", self.session_id)
        payload.setdefault("pipeline", self.pipeline)
        payload.setdefault("created_ts", time.time())
        payload.setdefault("platform", platform.platform())
        payload.setdefault("python", platform.python_version())
        payload.setdefault("cwd", str(Path.cwd().resolve()))
        payload.setdefault("log_root", str(self.log_root))
        payload.setdefault("run_dir", str(self.run_dir))

        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def append_timeline(self, obj: Dict[str, Any]) -> None:
        payload = dict(obj or {})
        payload.setdefault("schema", "broncho.timeline.v2")
        payload.setdefault("ts", time.time())
        payload.setdefault("session_id", self.session_id)
        payload.setdefault("pipeline", self.pipeline)
        payload.setdefault("event_id", str(uuid.uuid4()))

        with self.timeline_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def append_error(self, obj: Dict[str, Any]) -> None:
        payload = dict(obj or {})
        payload.setdefault("schema", "broncho.error.v1")
        payload.setdefault("ts", time.time())
        payload.setdefault("session_id", self.session_id)
        payload.setdefault("pipeline", self.pipeline)

        with self.errors_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_step(
        self,
        step_index: int,
        state: Optional[Dict[str, Any]] = None,
        control: Optional[Dict[str, Any]] = None,
        guidance: Optional[Dict[str, Any]] = None,
        plan: Optional[Dict[str, Any]] = None,
        statistics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "event_type": "step",
            "step_index": step_index,
            "state": state or {},
            "control": control or {},
            "guidance": guidance or {},
            "plan": plan or {},
            "statistics": statistics or {},
        }
        if extra:
            payload["extra"] = extra
        self.append_timeline(payload)

    def log_llm_call(
        self,
        step_index: Optional[int],
        provider: str,
        model: str,
        prompt: str,
        response: Optional[str] = None,
        success: bool = True,
        latency_ms: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "event_type": "llm_call",
            "step_index": step_index,
            "llm": {
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "response": response,
                "success": success,
                "latency_ms": latency_ms,
            },
        }
        if extra:
            payload["extra"] = extra
        self.append_timeline(payload)

    def log_agent_turn(
        self,
        step_index: Optional[int],
        agent_name: str,
        role: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "event_type": "agent_turn",
            "step_index": step_index,
            "agent": {
                "name": agent_name,
                "role": role,
            },
            "input": input_data or {},
            "output": output_data or {},
        }
        if extra:
            payload["extra"] = extra
        self.append_timeline(payload)


    def ensure_artifacts_dir(self) -> Path:
        """Create and return the artifacts directory only when it is actually needed."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        return self.artifacts_dir


MASLogger = RunLogger
