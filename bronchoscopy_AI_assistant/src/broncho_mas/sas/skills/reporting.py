from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .base import BaseSkill, SkillResult

REPORT_SYSTEM_PROMPT = (
    "You are an experienced bronchoscopy trainer writing an end-of-session training report. "
    "Write like a concise clinical educator. Use only the provided facts. "
    "Do not invent missing findings, metrics, or airway segments. "
    "Keep exactly these four section headers and spell them exactly as given: "
    "Clinical performance note, Teaching feedback note, Curriculum coverage, Session metrics. "
    "Under each header, write short bullet lines. Keep the report professional, specific, and actionable. "
    "In the Teaching feedback note section, focus on the single most important improvement target and make it practical for the next session. "
    "If a metric is missing, explicitly say it was not recorded."
)


class ReportLLM:
    """Small local wrapper for report writing."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        system_prompt: str = REPORT_SYSTEM_PROMPT,
        temperature: float = 0.2,
        max_tokens: int = 350,
    ) -> None:
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.environ.get("BRONCHO_REPORT_MODEL") or os.environ.get("BRONCHO_LLM_MODEL", "gpt-4.1-mini")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    def ask(self, facts: Dict[str, Any]) -> str:
        user_content = json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


class ReportingSkill(BaseSkill):
    name = "reporting_skill"
    default_priority = 0.3

    def execute(
        self,
        *,
        allowed_reached: List[str],
        visit_order: List[str],
        curriculum_progress: Dict[str, Any],
        session_metrics: Dict[str, Any],
        sp_score: float,
        teach_line: Optional[str] = None,
        report_llm: Optional[Any] = None,
        use_llm: Optional[bool] = None,
    ) -> SkillResult:
        facts = build_report_facts(
            allowed_reached=allowed_reached,
            visit_order=visit_order,
            curriculum_progress=curriculum_progress,
            session_metrics=session_metrics,
            sp_score=sp_score,
            teach_line=teach_line,
        )
        report_text, report_mode = generate_report_text(
            facts=facts,
            report_llm=report_llm,
            use_llm=use_llm,
        )
        return SkillResult(
            skill=self.name,
            active=True,
            priority=self.default_priority,
            reason=f"end-of-session report generated ({report_mode})",
            data={
                "report_text": report_text,
                "required_structure_ok": report_has_required_structure(report_text),
                "report_mode": report_mode,
                "report_facts": facts,
            },
            utterance=report_text,
        )


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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_report_facts(
    *,
    allowed_reached: List[str],
    visit_order: List[str],
    curriculum_progress: Dict[str, Any],
    session_metrics: Dict[str, Any],
    sp_score: float,
    teach_line: Optional[str] = None,
) -> Dict[str, Any]:
    coverage = _safe_float(curriculum_progress.get("coverage_ratio", 0.0), 0.0)
    reached_count = _safe_int(curriculum_progress.get("reached_count", len(allowed_reached)), len(allowed_reached))
    total = _safe_int(curriculum_progress.get("total", max(len(visit_order), 1)), max(len(visit_order), 1))

    duration = session_metrics.get("duration_seconds", None)
    duration_seconds = int(duration) if isinstance(duration, (int, float)) and duration >= 0 else None

    allowed_upper = {x.upper() for x in allowed_reached}
    missing = [a for a in visit_order if a.upper() not in allowed_upper]
    next_airway = curriculum_progress.get("next_airway") or "N/A"
    backtrack_ratio = session_metrics.get("backtrack_ratio", "N/A")
    sq_raw = curriculum_progress.get("student_questions", session_metrics.get("student_questions", 0))
    student_questions = _safe_int(sq_raw, 0)

    if not teach_line:
        teach_line = "- Overall teaching focus: continue building a systematic segment-by-segment bronchoscopy technique, with attention to airway orientation, centered scope handling, and complete, atraumatic examination."

    facts = {
        "report_constraints": {
            "required_headers": [
                "Clinical performance note",
                "Teaching feedback note",
                "Curriculum coverage",
                "Session metrics",
            ],
            "style": "concise clinical educator",
            "use_only_provided_facts": True,
            "bullets_required": True,
        },
        "performance": {
            "diagnostic_completeness": {
                "reached_count": reached_count,
                "total": total,
                "coverage_ratio": coverage,
                "coverage_percent_rounded": round(coverage * 100),
            },
            "structured_progress_score": round(float(sp_score), 2),
            "procedure_time_seconds": duration_seconds,
        },
        "curriculum": {
            "segments_visualized": list(allowed_reached),
            "segments_not_yet_visualized": list(missing),
            "segments_not_yet_visualized_preview": list(missing[:10]),
            "next_target_segment": next_airway,
        },
        "session_metrics": {
            "backtrack_ratio": backtrack_ratio,
            "student_questions": student_questions,
        },
        "teaching": {
            "focus_line": teach_line,
        },
    }
    return facts


def _should_use_llm(use_llm: Optional[bool]) -> bool:
    if use_llm is not None:
        return bool(use_llm)
    env_value = os.environ.get("BRONCHO_REPORT_USE_LLM", "1").strip().lower()
    if env_value in {"0", "false", "no", "off"}:
        return False
    return bool(os.environ.get("OPENAI_API_KEY"))


def _get_report_llm(report_llm: Optional[Any] = None) -> Any:
    return report_llm if report_llm is not None else ReportLLM()


def generate_report_text(
    *,
    facts: Dict[str, Any],
    report_llm: Optional[Any] = None,
    use_llm: Optional[bool] = None,
) -> tuple[str, str]:
    if _should_use_llm(use_llm):
        try:
            llm = _get_report_llm(report_llm)
            report_text = llm.ask(facts)
            report_text = normalize_report_text(report_text)
            if report_has_required_structure(report_text):
                return report_text, "llm"
        except Exception as exc:
            print(f"[reporting_skill] LLM report generation failed: {exc}")
    return render_report_template(facts), "template_fallback"


def render_report_template(facts: Dict[str, Any]) -> str:
    perf = facts.get("performance", {})
    curriculum = facts.get("curriculum", {})
    metrics = facts.get("session_metrics", {})
    teaching = facts.get("teaching", {})

    dc = perf.get("diagnostic_completeness", {})
    reached_count = _safe_int(dc.get("reached_count"), 0)
    total = _safe_int(dc.get("total"), 1)
    coverage_percent = _safe_int(dc.get("coverage_percent_rounded"), 0)

    duration_seconds = perf.get("procedure_time_seconds")
    if isinstance(duration_seconds, int) and duration_seconds >= 0:
        pt_line = f"- Procedure time (PT): {duration_seconds} seconds."
    else:
        pt_line = "- Procedure time (PT): not recorded."

    dc_line = f"- Diagnostic completeness (DC): {reached_count}/{total} segments ({coverage_percent}%)."
    sp_line = f"- Structured progress (SP): {_safe_float(perf.get('structured_progress_score'), 0.0):.2f} (ordered progression ratio)."

    visualized = curriculum.get("segments_visualized") or []
    segs_line = f"- Segments visualized: {', '.join(visualized) if visualized else 'None'}."

    missing_preview = curriculum.get("segments_not_yet_visualized_preview") or []
    missing_all = curriculum.get("segments_not_yet_visualized") or []
    missing_line = f"- Segments not yet visualized: {', '.join(missing_preview) if missing_preview else 'None'}{' ...' if len(missing_all) > len(missing_preview) and missing_preview else ''}."

    next_line = f"- Next target segment: {curriculum.get('next_target_segment') or 'N/A'}."
    teach_line = teaching.get("focus_line") or "- Overall teaching focus: continue building a systematic segment-by-segment bronchoscopy technique, with attention to airway orientation, centered scope handling, and complete, atraumatic examination."

    return (
        "Clinical performance note\n"
        f"{dc_line}\n"
        f"{sp_line}\n"
        f"{pt_line}\n"
        "Teaching feedback note\n"
        f"{teach_line}\n"
        "Curriculum coverage\n"
        f"{segs_line}\n"
        f"{missing_line}\n"
        f"{next_line}\n"
        "Session metrics\n"
        f"- Backtrack ratio: {metrics.get('backtrack_ratio', 'N/A')}.\n"
        f"- Student questions: {_safe_int(metrics.get('student_questions'), 0)}.\n"
    )


def normalize_report_text(report_text: str) -> str:
    text = (report_text or "").replace("\r\n", "\n").strip()
    if not text:
        return text
    lines = [line.rstrip() for line in text.split("\n")]
    normalized = "\n".join(lines).strip()
    return normalized


def report_has_required_structure(report_text: str) -> bool:
    if not report_text:
        return False
    required = [
        "Clinical performance note",
        "Teaching feedback note",
        "Curriculum coverage",
        "Session metrics",
    ]
    return all(h in report_text for h in required)


def reporting_skill(**kwargs: Any) -> Dict[str, Any]:
    result = ReportingSkill().execute(**kwargs)
    payload = dict(result.data)
    payload.setdefault("report_text", result.utterance)
    return payload
