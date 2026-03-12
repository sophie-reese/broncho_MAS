from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _select_pipeline() -> str:
    return (os.environ.get("BRONCHO_PIPELINE") or "runtime").strip().lower()


def _build_manager(model_name: str):
    """
    Manager selector.

    BRONCHO_PIPELINE=runtime   -> RuntimeManager (default)
    BRONCHO_PIPELINE=research  -> MultiAgentManager
    BRONCHO_PIPELINE=mas       -> MultiAgentManager

    Any unknown value falls back to runtime first, then research.
    """
    pipeline = _select_pipeline()

    if pipeline in {"research", "mas"}:
        from .research.manager import MultiAgentManager  # type: ignore
        return MultiAgentManager(model_name=model_name)

    if pipeline == "runtime":
        try:
            from .runtime.runtime_manager import RuntimeManager  # type: ignore
            return RuntimeManager(model_name=model_name)
        except Exception:
            from .research.manager import MultiAgentManager  # type: ignore
            return MultiAgentManager(model_name=model_name)

    try:
        from .runtime.runtime_manager import RuntimeManager  # type: ignore
        return RuntimeManager(model_name=model_name)
    except Exception:
        from .research.manager import MultiAgentManager  # type: ignore
        return MultiAgentManager(model_name=model_name)


class SmolAgentsLLM:
    def __init__(self, model_name: str = "Qwen/Qwen3.5-27B"):
        self.manager = _build_manager(model_name=model_name)

    def ask(self, prompt: str) -> str:
        result = self.manager.run(prompt)
        if isinstance(result, dict):
            return str(result.get("ui_text", "")).strip()
        return str(result).strip()

    def ask_structured(self, prompt: str) -> Dict[str, Any]:
        result = self.manager.run(prompt)
        if isinstance(result, dict):
            return result
        return {"ui_text": str(result).strip()}

    def get_report(self, recording_dir: Optional[str] = None) -> str:
        if hasattr(self.manager, "get_report"):
            return self.manager.get_report(recording_dir=recording_dir)
        raise NotImplementedError(
            "The active runtime manager does not implement get_report(). "
            "Set BRONCHO_PIPELINE=research for report generation."
        )
