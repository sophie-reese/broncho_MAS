from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from .skills import (
    guidance_skill,
    landmark_teaching_skill,
    qa_skill,
    reporting_skill,
    statistics_skill,
    support_skill,
)


@dataclass(frozen=True)
class SkillSpec:
    name: str
    descriptor: str
    category: str
    backend: Callable[..., Any]
    produces_utterance: bool = False


SKILL_REGISTRY: Dict[str, SkillSpec] = {
    "guidance_skill": SkillSpec(
        name="guidance_skill",
        descriptor="live coaching realization from curriculum plan, directional hint, and current runtime context",
        category="guidance",
        backend=guidance_skill,
        produces_utterance=True,
    ),
    "support_skill": SkillSpec(
        name="support_skill",
        descriptor="brief grounding, encouragement, and emotional stabilization during live bronchoscopy coaching",
        category="support",
        backend=support_skill,
        produces_utterance=True,
    ),
    "landmark_teaching_skill": SkillSpec(
        name="landmark_teaching_skill",
        descriptor="first-arrival teaching overlay for key bronchoscopy landmarks using short recognition and memory cues",
        category="teaching",
        backend=landmark_teaching_skill,
        produces_utterance=True,
    ),
    "qa_skill": SkillSpec(
        name="qa_skill",
        descriptor="short in-procedure answering, observation handling, and redirection during bronchoscopy training",
        category="interaction",
        backend=qa_skill,
        produces_utterance=True,
    ),
    "statistics_skill": SkillSpec(
        name="statistics_skill",
        descriptor="compact analytic navigation summary for logging, monitoring, and downstream reporting",
        category="analytics",
        backend=statistics_skill,
        produces_utterance=False,
    ),
    "reporting_skill": SkillSpec(
        name="reporting_skill",
        descriptor="end-of-session educator-facing bronchoscopy report generation from grounded session facts",
        category="reporting",
        backend=reporting_skill,
        produces_utterance=True,
    ),
}
