from __future__ import annotations

__version__ = "0.0.3-runtime-research"

from .adapter import SmolAgentsLLM
from .runtime.runtime_manager import RuntimeManager

try:
    from .research.manager import MultiAgentManager
except Exception:  # pragma: no cover
    MultiAgentManager = None

__all__ = ["SmolAgentsLLM", "RuntimeManager", "MultiAgentManager"]
