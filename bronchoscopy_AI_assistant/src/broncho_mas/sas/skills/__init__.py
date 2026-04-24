from __future__ import annotations

from ...shared import LANDMARK_ALIASES, LANDMARK_CARDS, LandmarkCard
from .base import BaseSkill, SkillResult, build_skill_record
from .guidance import GuidanceSkill, build_guidance_frame, guidance_skill
from .qa import QASkill, _QuestionRouter, build_qa_frame, qa_skill, route_question_mode
from .reporting import build_report_facts, generate_report_text, render_report_template, reporting_skill
from .realization import deterministic_frame_text, realize_frame_response
from .statistics import build_statistics_payload, statistics_skill
from .support import SupportSkill, support_skill
from .teaching import (
    LandmarkTeachingSkill,
    build_landmark_teaching_frame,
    build_landmark_teaching_payload,
    landmark_teaching_skill,
    mark_landmark_as_taught,
    resolve_landmark_id,
    should_fire_landmark_teaching,
)
from .utterance_helpers import (
    _airway_token,
    _arrival_or_route_override,
    _family_local_navigation_phrase,
    _route_phrase,
    merge_teaching_with_guidance,
    safety_risk,
)

__all__ = [
    "SkillResult",
    "LandmarkCard",
    "LANDMARK_CARDS",
    "LANDMARK_ALIASES",
    "build_skill_record",
    "landmark_teaching_skill",
    "mark_landmark_as_taught",
    "guidance_skill",
    "qa_skill",
    "support_skill",
    "reporting_skill",
    "statistics_skill",
]
