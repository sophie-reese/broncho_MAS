from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _select_pipeline() -> str:
    return (os.environ.get("BRONCHO_PIPELINE") or "runtime").strip().lower()


def _build_manager(model_name: str):
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
    """
    Thin compatibility adapter.

    Rules:
    - If a payload has been injected into _payload_for_next_call, use it.
    - Otherwise fall back to the legacy prompt-only path.
    - Do not do state inference or payload reshaping here; that belongs in the manager.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-27B",
        temperature: float = 0.2,
        max_tokens: int = 256,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.manager = _build_manager(model_name=model_name)
        self._payload_for_next_call: Optional[Dict[str, Any]] = None

    def _pop_next_payload(self) -> Optional[Dict[str, Any]]:
        payload = getattr(self, "_payload_for_next_call", None)
        self._payload_for_next_call = None
        return payload if isinstance(payload, dict) else None

    def _run_payload(self, payload: Dict[str, Any]) -> Any:
        if hasattr(self.manager, "step"):
            return self.manager.step(payload)
        if hasattr(self.manager, "run"):
            return self.manager.run(payload)
        raise AttributeError(
            f"Active manager {type(self.manager).__name__} exposes neither "
            f"step(payload) nor run(payload)."
        )

    def _run_prompt(self, prompt: str) -> Any:
        if hasattr(self.manager, "run"):
            return self.manager.run(prompt)
        if hasattr(self.manager, "step"):
            return self.manager.step(prompt)
        raise AttributeError(
            f"Active manager {type(self.manager).__name__} exposes neither "
            f"run(prompt) nor step(prompt)."
        )

    @staticmethod
    def _normalize_result(result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            out = dict(result)
            out["ui_text"] = str(out.get("ui_text", "")).strip()
            return out
        return {"ui_text": str(result).strip()}

    def ask(self, prompt: str) -> str:
        payload = self._pop_next_payload()
        result = self._run_payload(payload) if payload is not None else self._run_prompt(prompt)
        return self._normalize_result(result)["ui_text"]

    def ask_structured(self, prompt: str) -> Dict[str, Any]:
        payload = self._pop_next_payload()
        result = self._run_payload(payload) if payload is not None else self._run_prompt(prompt)
        return self._normalize_result(result)

    def generate(self, prompt: str) -> Dict[str, Any]:
        return self.ask_structured(prompt)

    def __call__(self, prompt: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.ask_structured(prompt)

    def get_report(self, recording_dir: Optional[str] = None) -> str:
        if hasattr(self.manager, "get_report"):
            return self.manager.get_report(recording_dir=recording_dir)
        raise NotImplementedError(
            "The active runtime manager does not implement get_report(). "
            "Set BRONCHO_PIPELINE=research for report generation."
        )
