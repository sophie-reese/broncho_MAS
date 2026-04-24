from __future__ import annotations

from typing import Any, Dict

from .base import BaseSkill, SkillResult
from .guidance import _recognition_cue
from .utterance_helpers import _airway_token

def build_statistics_payload(
    *,
    current_situation: str,
    current_airway: str,
    next_airway: str,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    text = str(current_situation or "").lower()
    mode = str((plan or {}).get("mode", "")).lower()
    cue = _recognition_cue(plan)
    target_label = _airway_token(next_airway or str((plan or {}).get("next_airway") or ""))

    if mode == "backtrack":
        return {
            "trend": "stable",
            "likely_issue": "lost orientation or unsafe advance",
            "coach_focus_next": "withdraw to the carina and re-center",
            "teaching_point": "Reset orientation before advancing again.",
            "notes": f"current={current_airway or 'unknown'}; target={next_airway or 'unknown'}; mode=backtrack.",
        }
    if mode in {"reorient", "locate"} or "not visible" in text:
        return {
            "trend": "stable",
            "likely_issue": "target lumen not yet visualized",
            "coach_focus_next": f"keep centered and identify {target_label or 'the target opening'}",
            "teaching_point": (f"Use the landmark cue: {cue}." if cue else "Do not advance until the target lumen is clearly identified."),
            "notes": f"current={current_airway or 'unknown'}; target={next_airway or 'unknown'}; mode={mode or 'locate'}.",
        }
    return {
        "trend": "stable",
        "likely_issue": "normal navigation",
        "coach_focus_next": "advance with the lumen centered",
        "teaching_point": (f"Confirm {cue} before committing forward." if cue else "Advance only with a centered view."),
        "notes": f"current={current_airway or 'unknown'}; target={next_airway or 'unknown'}; mode={mode or 'advance'}.",
    }

class StatisticsSkill(BaseSkill):
    name = "statistics_skill"
    default_priority = 0.4

    def execute(
        self,
        *,
        current_situation: str,
        current_airway: str,
        next_airway: str,
        plan: Dict[str, Any],
    ) -> SkillResult:
        stats = build_statistics_payload(
            current_situation=current_situation,
            current_airway=current_airway,
            next_airway=next_airway,
            plan=plan,
        )
        return SkillResult(
            skill=self.name,
            active=True,
            priority=self.default_priority,
            reason="session statistics summary generated",
            data=stats,
            utterance="",
        )

def statistics_skill(**kwargs: Any) -> Dict[str, Any]:
    return StatisticsSkill().execute(**kwargs).data
