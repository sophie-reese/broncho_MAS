from __future__ import annotations

__version__ = "0.0.5-runtime-sas-mas"

from .adapter import SmolAgentsLLM
from .runtime.manager import RuntimeManager
from .sas.manager import SASManager, SingleAgentManager

try:
    from .mas.manager import MASManager, MultiAgentManager
except Exception:  # pragma: no cover
    MASManager = None
    MultiAgentManager = None

__all__ = [
    "SmolAgentsLLM",
    "RuntimeManager",
    "SASManager",
    "SingleAgentManager",
    "MASManager",
    "MultiAgentManager",
]
