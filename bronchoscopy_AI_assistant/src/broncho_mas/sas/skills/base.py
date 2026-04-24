from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from ...shared.curriculum import DEFAULT_AIRWAY_VISIT_ORDER

State = Dict[str, Any]
Plan = Dict[str, Any]
Frame = Dict[str, Any]
DirectionalHint = Dict[str, Any]

DEFAULT_VISIT_ORDER: tuple[str, ...] = DEFAULT_AIRWAY_VISIT_ORDER


@dataclass
class SkillResult:
    skill: str
    active: bool
    priority: float
    reason: str
    data: Dict[str, Any] = field(default_factory=dict)
    utterance: str = ""
    frame: Dict[str, Any] = field(default_factory=dict)
    deterministic_text: str = ""
    wants_realization: bool = False
    constraints: Dict[str, Any] = field(default_factory=dict)
    debug_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseSkill:
    name: str = "base_skill"
    default_priority: float = 0.5

    def should_activate(self, **kwargs: Any) -> bool:
        return True

    def priority(self, **kwargs: Any) -> float:
        return float(self.default_priority)

    def reason(self, **kwargs: Any) -> str:
        return "skill activated"

    def execute(self, **kwargs: Any) -> SkillResult:
        raise NotImplementedError


def build_skill_record(
    *,
    skill: str,
    active: bool,
    priority: float,
    reason: str,
    data: Dict[str, Any] | None = None,
    utterance: str = "",
    frame: Dict[str, Any] | None = None,
    deterministic_text: str = "",
    wants_realization: bool = False,
    constraints: Dict[str, Any] | None = None,
    debug_reason: str = "",
) -> Dict[str, Any]:
    return SkillResult(
        skill=skill,
        active=bool(active),
        priority=float(priority),
        reason=str(reason or ""),
        data=dict(data or {}),
        utterance=str(utterance or "").strip(),
        frame=dict(frame or {}),
        deterministic_text=str(deterministic_text or "").strip(),
        wants_realization=bool(wants_realization),
        constraints=dict(constraints or {}),
        debug_reason=str(debug_reason or ""),
    ).to_dict()
