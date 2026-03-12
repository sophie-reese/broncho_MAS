from __future__ import annotations

import json
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


def _session_id_now() -> str:
    # e.g., 20260302_151234
    return time.strftime("%Y%m%d_%H%M%S")


class MASLogger:
    """Append-only JSONL logger for MAS-side reproducible research logs."""

    def __init__(self, log_root: Optional[str] = None, session_id: Optional[str] = None):
        self.log_root = Path(log_root or os.environ.get("BRONCHO_MAS_LOG_ROOT", "./log_mas"))
        self.session_id = session_id or _session_id_now()
        self.run_dir = self.log_root / self.session_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.turns_path = self.run_dir / "turns.jsonl"
        self.errors_path = self.run_dir / "errors.jsonl"
        self.meta_path = self.run_dir / "meta.json"

    def write_meta(self, meta: Dict[str, Any]) -> None:
        meta = dict(meta or {})
        meta.setdefault("schema", "mas.meta.v1")
        meta.setdefault("session_id", self.session_id)
        meta.setdefault("created_ts", time.time())
        meta.setdefault("platform", platform.platform())
        meta.setdefault("python", platform.python_version())
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def append_turn(self, obj: Dict[str, Any]) -> None:
        obj = dict(obj or {})
        obj.setdefault("ts", time.time())
        obj.setdefault("turn_id", str(uuid.uuid4()))
        with self.turns_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def append_error(self, obj: Dict[str, Any]) -> None:
        obj = dict(obj or {})
        obj.setdefault("ts", time.time())
        with self.errors_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
