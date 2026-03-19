from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple


ANCHOR_AIRWAYS: Set[str] = {"CARINA", "LMB", "RMB", "TRACHEA"}


def _safe_coverage_ratio(reached_count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(reached_count / total, 4)


class _SimpleLLM:
    """Silent fallback backend."""

    is_fallback_backend = True

    def generate(self, prompt: str) -> str:
        return ""

    def __call__(self, messages: Any) -> str:
        return ""


class RuntimeManager:
    """
    Real-time runtime manager for bronchoscopy tutoring.

    Design goals:
    - accept either legacy full-prompt text or structured payload dict
    - keep navigation decisions deterministic
    - keep utterances short enough for TTS / online use
    - preserve educational value with one concrete cue per turn
    - degrade safely when LLM/provider is unavailable
    """

    AIRWAY_VISIT_ORDER: List[str] = [
        "RB1", "RB2", "RB3", "RB4", "RB5", "RB6", "RB7", "RB8", "RB9", "RB10",
        "LB1+2", "LB3", "LB4", "LB5", "LB6", "LB8", "LB9", "LB10",
    ]

    def __init__(self, model_name: str = "Qwen/Qwen3.5-27B", llm: Optional[Any] = None):
        self.model_name = model_name
        self._last_ui_text = ""

        from ..shared.curriculum import CurriculumEngine
        from .realtime_engine import RealtimeInstructorEngine

        self.curriculum = CurriculumEngine(self.AIRWAY_VISIT_ORDER)
        self.rt_engine = RealtimeInstructorEngine(self.curriculum)
        self.model = llm or self._build_model()

    # ------------------------------------------------------------------
    # model loading
    # ------------------------------------------------------------------
    def _build_model(self) -> Any:
        try:
            os.environ.setdefault("BRONCHO_MODEL", self.model_name)
            os.environ.setdefault("BRONCHO_PROVIDER", os.environ.get("BRONCHO_PROVIDER", "hf"))
            from ..shared.model_selector import create_model  # type: ignore

            model = create_model(None)
            print(
                f"[runtime] provider={os.environ.get('BRONCHO_PROVIDER', '')} "
                f"model={os.environ.get('BRONCHO_MODEL', '')}"
            )
            return model
        except Exception as exc:
            print(f"[runtime] model init failed: {exc}")
            return _SimpleLLM()

    # ------------------------------------------------------------------
    # input normalization
    # ------------------------------------------------------------------
    def _extract_block(self, text: str, tag: str) -> str:
        if not text:
            return ""
        m = re.search(rf"{re.escape(tag)}\s*:(.*?)(?=\n\s*[A-Z_]+\s*:|\Z)", str(text), flags=re.S)
        return (m.group(1) if m else "").strip()

    def _parse_prompt_or_payload(self, payload: Any) -> Tuple[str, str, str, str, Dict[str, Any]]:
        """
        Returns:
            prompt_text, current_situation, previous_msgs, student_question, raw_payload
        """
        if isinstance(payload, dict):
            raw_payload = dict(payload)
            current_situation = self._payload_to_current_situation(raw_payload)
            previous_msgs = str(
                raw_payload.get("previous_msgs")
                or raw_payload.get("history")
                or raw_payload.get("llm_history")
                or ""
            ).strip()
            student_question = str(
                raw_payload.get("student_question")
                or raw_payload.get("student_q")
                or raw_payload.get("question")
                or ""
            ).strip()
            prompt_text = self._compose_prompt(current_situation, previous_msgs, student_question)
            return prompt_text, current_situation, previous_msgs, student_question, raw_payload

        prompt_text = str(payload or "")
        current_situation = self._extract_block(prompt_text, "CURRENT_SITUATION")
        previous_msgs = self._extract_block(prompt_text, "PREVIOUS_MSGS")
        student_question = self._extract_block(prompt_text, "STUDENT_QUESTION")
        return prompt_text, current_situation, previous_msgs, student_question, {}

    def _payload_to_current_situation(self, payload: Dict[str, Any]) -> str:
        parts: List[str] = []

        mapping = [
            ("phase", "Phase"),
            ("current_region", "Current region"),
            ("current_airway", "Current airway"),
            ("target_region", "Target region"),
            ("target_airway", "Target airway"),
            ("requested_next_airway", "Requested next airway"),
            ("soft_prompt", "Soft prompt"),
            ("need_llm", "Need LLM"),
        ]
        for key, label in mapping:
            val = payload.get(key)
            if val not in (None, ""):
                parts.append(f"{label}: {val}")

        bool_keys = [
            ("backtracking", "Backtracking"),
            ("drift_detected", "Drift detected"),
            ("is_centered", "Is centered"),
            ("is_stable", "Is stable"),
            ("is_target_visible", "Target visible"),
            ("wall_contact_risk", "Wall contact risk"),
            ("need_recenter", "Need recenter"),
        ]
        for key, label in bool_keys:
            if key in payload:
                parts.append(f"{label}: {str(bool(payload.get(key))).lower()}")

        for key, label in (("regions_seen", "regions_seen"), ("reached_regions", "reached_regions"), ("missing_regions", "missing_regions")):
            val = payload.get(key)
            if isinstance(val, list):
                parts.append(f"{label}: {json.dumps(val)}")

        return "\n".join(parts).strip()

    def _compose_prompt(self, current_situation: str, previous_msgs: str, student_question: str) -> str:
        prompt = f"CURRENT_SITUATION: {current_situation}".strip()
        if previous_msgs:
            prompt += f"\n\nPREVIOUS_MSGS: {previous_msgs}"
        if student_question:
            prompt += f"\n\nSTUDENT_QUESTION: {student_question}"
        return prompt

    # ------------------------------------------------------------------
    # extraction helpers
    # ------------------------------------------------------------------
    def _extract_reached_regions(self, current_situation: str) -> List[str]:
        text = str(current_situation or "")
        allowed = set(self.AIRWAY_VISIT_ORDER)
        out: List[str] = []
        patterns = [
            r"reached_regions\(last\)\s*[:=]\s*(\[[^\]]*\])",
            r"REACHED_REGIONS\s*[:=]\s*(\[[^\]]*\])",
            r"reached_regions\s*[:=]\s*(\[[^\]]*\])",
            r"timeline_reached\s*[:=]\s*(\[[^\]]*\])",
            r"regions_seen\s*[:=]\s*(\[[^\]]*\])",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.I)
            if not m:
                continue
            chunk = m.group(1)
            for parser in (json.loads, ast.literal_eval):
                try:
                    val = parser(chunk)
                    if isinstance(val, list):
                        for item in val:
                            x = str(item).strip().upper()
                            if x in allowed and x not in out:
                                out.append(x)
                        return out
                except Exception:
                    pass
        return out

    def _extract_current_airway(self, current_situation: str) -> str:
        text = str(current_situation or "")
        allowed = set(self.AIRWAY_VISIT_ORDER) | ANCHOR_AIRWAYS
        patterns = [
            r"anatomical_position\s*[:=]\s*\"?([A-Za-z0-9+\-]+)\"?",
            r"current airway\s*[:=]\s*([A-Za-z0-9+\-]+)",
            r"current region\s*[:=]\s*([A-Za-z0-9+\-]+)",
            r'currently at\s+\"?([A-Za-z0-9+\-]+)\"?',
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.I)
            if not m:
                continue
            x = m.group(1).strip().upper()
            if x in {"", "NONE", "NULL", "UNKNOWN"}:
                return ""
            if x in allowed:
                return x
        return ""

    def _extract_target_hint(self, current_situation: str) -> str:
        text = str(current_situation or "")
        allowed = set(self.AIRWAY_VISIT_ORDER)
        patterns = [
            r"target region\s*[:=]\s*([A-Za-z0-9+\-]+)",
            r"target airway\s*[:=]\s*([A-Za-z0-9+\-]+)",
            r"navigation target is ['\"]?([A-Za-z0-9+\-]+)['\"]?",
            r"next lumen to be explored\s*(?:is)?\s*[:=]?\s*['\"]?([A-Za-z0-9+\-]+)['\"]?",
            r"requested next airway\s*[:=]\s*([A-Za-z0-9+\-]+)",
            r"current_target\s*[:=]\s*([A-Za-z0-9+\-]+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.I)
            if not m:
                continue
            x = m.group(1).strip().upper()
            if x in allowed:
                return x
        return ""

    def _needs_visual_guidance(self, student_question: str) -> bool:
        q = str(student_question or "").lower()
        return any(k in q for k in ["picture", "diagram", "image", "show me", "draw", "where is", "point", "visual"])

    # ------------------------------------------------------------------
    # text cleanup
    # ------------------------------------------------------------------
    def _sentence_split(self, text: str) -> List[str]:
        raw = re.split(r"(?<=[.!?])\s+|\n+", str(text or "").strip())
        out: List[str] = []
        for s in raw:
            x = re.sub(r"\s+", " ", s).strip()
            if x:
                out.append(x)
        return out

    def _normalize_landmark_language(self, text: str) -> str:
        x = str(text or "")
        x = x.replace("Mercedes sign / trifurcation", "Mercedes sign")
        x = x.replace("Mercedes sign/trifurcation", "Mercedes sign")
        x = x.replace("trifurcation / Mercedes sign", "Mercedes sign")
        x = re.sub(r"\bMercedes sign and trifurcation\b", "Mercedes sign", x, flags=re.I)
        x = re.sub(r"\bMercedes sign or trifurcation\b", "Mercedes sign", x, flags=re.I)
        x = re.sub(r"\bthe trifurcation\b", "the Mercedes sign", x, flags=re.I)
        return x

    def _trim_line_words(self, text: str, max_words: int = 18) -> str:
        x = re.sub(r"\s+", " ", str(text or "").strip())
        if not x:
            return ""
        words = x.split()
        if len(words) <= max_words:
            return x.rstrip(" ,;:")
        cut = " ".join(words[:max_words]).rstrip(" ,;:")
        if cut and cut[-1] not in ".!?":
            cut += "."
        return cut

    def _compress_utterance(self, text: str) -> str:
        x = self._normalize_landmark_language(text)
        lines = self._sentence_split(x)
        if not lines:
            x = re.sub(r"\s+", " ", x).strip()
            return self._trim_line_words(x, max_words=18) if x else ""

        clean: List[str] = []
        for line in lines:
            y = re.sub(r"\s+", " ", line).strip(" ,;")
            if not y:
                continue
            clean.append(self._trim_line_words(y, max_words=18))
            if len(clean) == 2:
                break
        return " ".join(clean).strip()

    def _short_ui_text(self, text: str) -> str:
        x = self._normalize_landmark_language(text)
        x = re.sub(r"\s+", " ", str(x or "").strip())
        replacements = [
            ("Hold neutral at the anchor", "Hold center"),
            ("Advance toward", "Advance to"),
            ("in small increments", "slowly"),
            ("while keeping the lumen centered", ""),
            ("recognition cue", "landmark"),
            ("reacquire", "find again"),
            ("identify", "find"),
        ]
        for old, new in replacements:
            x = x.replace(old, new)
        x = re.sub(r"\s+", " ", x).strip(" ,;")

        sentences = self._sentence_split(x)
        if sentences:
            clipped = []
            for s in sentences[:2]:
                clipped.append(self._trim_line_words(s, max_words=18))
            x = " ".join(clipped).strip()
        else:
            x = self._trim_line_words(x, max_words=18)

        if x and x[-1] not in ".!?":
            x += "."
        return x

    def _too_similar(self, a: str, b: str) -> bool:
        na = re.sub(r"[^a-z0-9+]+", " ", str(a or "").lower()).strip()
        nb = re.sub(r"[^a-z0-9+]+", " ", str(b or "").lower()).strip()
        return bool(na and na == nb)

    def _extract_text_response(self, resp: Any) -> str:
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp.strip()
        if isinstance(resp, dict):
            for key in ("utterance", "text", "content", "response", "output"):
                val = resp.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        content = getattr(resp, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if txt:
                        parts.append(str(txt))
                else:
                    txt = getattr(item, "text", None)
                    if txt:
                        parts.append(str(txt))
            return "\n".join(parts).strip()
        text = getattr(resp, "text", None)
        if isinstance(text, str):
            return text.strip()
        return str(resp).strip()

    # ------------------------------------------------------------------
    # planning and guidance
    # ------------------------------------------------------------------
    def _lookup_landmark(self, airway: str) -> Dict[str, Any]:
        info = self.curriculum.landmark_for_airway(airway or "RB1")
        return {
            "landmark_id": getattr(info, "landmark_id", ""),
            "recommended_angles": getattr(info, "recommended_angles", []),
            "recognition_cue": getattr(info, "recognition_cue", ""),
        }

    def _build_authoritative_plan(
        self,
        current_situation: str,
        previous_msgs: str,
        student_question: str,
        current_airway: str,
        reached_set: Set[str],
        target_hint: str,
        landmark_hint: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            ctx = self.rt_engine.parse_context(
                current=current_situation,
                previous_msgs=previous_msgs,
                student_q=student_question,
            )
            plan = self.rt_engine.build_plan(ctx)
            if isinstance(plan, dict):
                if target_hint and not plan.get("next_airway"):
                    plan["next_airway"] = target_hint
                if current_airway and not plan.get("current_airway"):
                    plan["current_airway"] = current_airway
                if landmark_hint.get("recognition_cue") and not plan.get("recognition_cue"):
                    plan["recognition_cue"] = landmark_hint["recognition_cue"]
                return plan
        except Exception:
            pass

        plan = self.curriculum.make_plan(
            current_airway=str(current_airway or "").upper(),
            reached={str(x).upper() for x in reached_set},
            student_question=student_question,
        )
        if not isinstance(plan, dict):
            plan = {}
        if target_hint:
            plan["next_airway"] = target_hint
        if current_airway and not plan.get("current_airway"):
            plan["current_airway"] = current_airway
        if landmark_hint.get("recognition_cue") and not plan.get("recognition_cue"):
            plan["recognition_cue"] = landmark_hint["recognition_cue"]
        return plan

    def _build_deterministic_guidance(
        self,
        current_situation: str,
        previous_msgs: str,
        student_question: str,
        auth_plan_json: Dict[str, Any],
    ) -> str:
        try:
            ctx = self.rt_engine.parse_context(
                current=current_situation,
                previous_msgs=previous_msgs,
                student_q=student_question,
            )
            guidance = self.rt_engine.build_guidance(ctx, auth_plan_json)
            if isinstance(guidance, dict):
                return self._compress_utterance(guidance.get("utterance", ""))
        except Exception:
            pass

        steps = (auth_plan_json or {}).get("micro_steps") or []
        if isinstance(steps, list) and steps:
            compact = []
            for step in steps[:2]:
                t = re.sub(r"\s+", " ", str(step or "")).strip().rstrip(".")
                if t:
                    compact.append(t)
            if compact:
                return ". ".join(compact) + "."

        target = str((auth_plan_json or {}).get("next_airway", "")).strip()
        cue = str((auth_plan_json or {}).get("recognition_cue", "")).strip()
        if target and cue:
            return f"Advance to {target}. Look for {cue}."
        if target:
            return f"Advance to {target} slowly."
        return "Keep the lumen centered. Move slowly."

    def _llm_verbalize(
        self,
        current_situation: str,
        previous_msgs: str,
        student_question: str,
        auth_plan_json: Dict[str, Any],
    ) -> str:
        try:
            from . import build_runtime_instructor_prompt  # type: ignore

            prompt = build_runtime_instructor_prompt(
                plan=auth_plan_json,
                current=current_situation,
                previous_msgs=previous_msgs,
                student_q=student_question,
            )
        except Exception:
            prompt = (
                "ROLE: Real-time bronchoscopy instructor\n"
                "TASK: Rewrite the deterministic plan into 1 or 2 short teaching sentences.\n"
                "RULES:\n"
                "- Follow PLAN_JSON exactly.\n"
                "- Do not change the target airway.\n"
                "- Do not add anatomy not present in the plan.\n"
                "- Keep it under 22 words per sentence.\n"
                "- Include at most one landmark cue.\n\n"
                f"PLAN_JSON:\n{json.dumps(auth_plan_json, ensure_ascii=True)}\n\n"
                f"CURRENT_SITUATION:\n{current_situation}\n\n"
                f"PREVIOUS_MSGS:\n{previous_msgs}\n\n"
                f"STUDENT_QUESTION:\n{student_question}\n"
            )

        raw = ""
        try:
            if hasattr(self.model, "generate"):
                raw = self._extract_text_response(self.model.generate(prompt))
        except Exception:
            raw = ""

        if not raw and callable(self.model):
            candidate_messages = [
                [{"role": "user", "content": prompt}],
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                prompt,
            ]
            for messages in candidate_messages:
                try:
                    raw = self._extract_text_response(self.model(messages))
                    if raw:
                        break
                except Exception:
                    continue

        return self._compress_utterance(raw)

    def _guidance_is_safe(self, llm_text: str, plan: Dict[str, Any], deterministic_text: str) -> bool:
        llm_norm = str(llm_text or "").upper()
        target = str((plan or {}).get("next_airway", "")).upper()
        if not llm_text.strip():
            return False
        if len(self._sentence_split(llm_text)) > 2:
            return False
        if len(llm_text.split()) > 32:
            return False
        airway_mentions = set(re.findall(r"\b(?:RB\d+|LB\d(?:\+\d)?|LMB|RMB|CARINA|TRACHEA)\b", llm_norm))
        if target and any(x not in {target, "LMB", "RMB", "CARINA", "TRACHEA"} for x in airway_mentions):
            return False
        return True

    def _compact_statistics(
        self,
        current_situation: str,
        current_airway: str,
        next_airway: str,
        auth_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        s = str(current_situation or "").lower()
        mode = str((auth_plan or {}).get("mode", "")).lower()
        cue = str((auth_plan or {}).get("recognition_cue", "")).strip()

        if mode == "backtrack":
            return {
                "trend": "stable",
                "likely_issue": "lost orientation or unsafe advance",
                "coach_focus_next": "withdraw to the carina and re-center",
                "teaching_point": "Reset orientation before advancing again.",
                "notes": f"current={current_airway or 'unknown'}; target={next_airway or 'unknown'}; mode=backtrack.",
            }
        if mode == "locate" or "not visible" in s:
            return {
                "trend": "stable",
                "likely_issue": "target lumen not yet visualized",
                "coach_focus_next": "hold center and identify the next opening",
                "teaching_point": (f"Use the landmark cue: {cue}." if cue else "Do not advance until the lumen is identified."),
                "notes": f"current={current_airway or 'unknown'}; target={next_airway or 'unknown'}; mode=locate.",
            }
        return {
            "trend": "stable",
            "likely_issue": "normal navigation",
            "coach_focus_next": "advance with the lumen centered",
            "teaching_point": (f"Confirm {cue}." if cue else "Advance only with a centered view."),
            "notes": f"current={current_airway or 'unknown'}; target={next_airway or 'unknown'}; mode={mode or 'advance'}.",
        }

    def _build_statepacket(
        self,
        prompt: str,
        current_situation: str,
        previous_msgs: str,
        student_question: str,
        reached_list: List[str],
        curriculum_progress: Dict[str, Any],
        landmark_hint: Dict[str, Any],
        auth_plan_json: Dict[str, Any],
        raw_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "schema": "statepacket.runtime.v2",
            "prompt": prompt,
            "current_situation": current_situation,
            "previous_msgs": previous_msgs,
            "student_question": student_question,
            "reached_regions": [str(x).upper() for x in reached_list],
            "curriculum_progress": curriculum_progress,
            "landmark_hint": landmark_hint,
            "auth_plan": auth_plan_json,
            "raw_payload": raw_payload,
            "meta": {
                "manager": "RuntimeManager",
                "provider": os.environ.get("BRONCHO_PROVIDER", ""),
                "model": os.environ.get("BRONCHO_MODEL", self.model_name),
                "fallback_backend": bool(getattr(self.model, "is_fallback_backend", False)),
            },
        }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(self, payload: Any) -> Dict[str, Any]:
        prompt, current_situation, previous_msgs, student_question, raw_payload = self._parse_prompt_or_payload(payload)

        reached_list = self._extract_reached_regions(current_situation)
        reached_set: Set[str] = {str(x).upper() for x in reached_list}

        current_airway = self._extract_current_airway(current_situation)
        target_hint = self._extract_target_hint(current_situation)

        deterministic_next = self.curriculum.next_airway(reached_set)
        next_airway = target_hint or deterministic_next or ""
        landmark_hint = self._lookup_landmark(next_airway or current_airway or "RB1")

        auth_plan_json = self._build_authoritative_plan(
            current_situation=current_situation,
            previous_msgs=previous_msgs,
            student_question=student_question,
            current_airway=current_airway,
            reached_set=reached_set,
            target_hint=next_airway,
            landmark_hint=landmark_hint,
        )

        auth_next = str(auth_plan_json.get("next_airway", "")).strip().upper() or next_airway
        if auth_next:
            next_airway = auth_next

        curriculum_progress = {
            "reached": [str(x).upper() for x in reached_list],
            "next_airway": next_airway,
            "coverage_ratio": _safe_coverage_ratio(len(reached_list), len(self.AIRWAY_VISIT_ORDER)),
            "reached_count": len(reached_list),
            "total": len(self.AIRWAY_VISIT_ORDER),
        }

        deterministic_ui = self._build_deterministic_guidance(
            current_situation=current_situation,
            previous_msgs=previous_msgs,
            student_question=student_question,
            auth_plan_json=auth_plan_json,
        )

        llm_ui = ""
        if not getattr(self.model, "is_fallback_backend", False):
            llm_ui = self._llm_verbalize(
                current_situation=current_situation,
                previous_msgs=previous_msgs,
                student_question=student_question,
                auth_plan_json=auth_plan_json,
            )

        ui_text = deterministic_ui
        if self._guidance_is_safe(llm_ui, auth_plan_json, deterministic_ui):
            ui_text = llm_ui

        if not ui_text:
            ui_text = "Keep the lumen centered. Advance slowly."

        if self._too_similar(ui_text, self._last_ui_text):
            first_line = self._sentence_split(ui_text)[:1]
            if first_line:
                ui_text = first_line[0]

        ui_text = self._short_ui_text(ui_text)
        self._last_ui_text = ui_text

        statistics = self._compact_statistics(
            current_situation=current_situation,
            current_airway=current_airway,
            next_airway=next_airway,
            auth_plan=auth_plan_json,
        )

        statepacket = self._build_statepacket(
            prompt=prompt,
            current_situation=current_situation,
            previous_msgs=previous_msgs,
            student_question=student_question,
            reached_list=reached_list,
            curriculum_progress=curriculum_progress,
            landmark_hint=landmark_hint,
            auth_plan_json=auth_plan_json,
            raw_payload=raw_payload,
        )

        return {
            "ui_text": ui_text,
            "instructor": ui_text,
            "needs_visual_guidance": self._needs_visual_guidance(student_question),
            "curriculum_progress": curriculum_progress,
            "landmark_hint": landmark_hint,
            "plan_json": auth_plan_json,
            "statistics": statistics,
            "statepacket": statepacket,
            "raw": {
                "prompt": prompt,
                "current_situation": current_situation,
                "used_fallback_backend": bool(getattr(self.model, "is_fallback_backend", False)),
                "deterministic_ui": deterministic_ui,
                "llm_ui": llm_ui,
            },
        }

    def step(self, payload: Any) -> Dict[str, Any]:
        return self.run(payload)

    def __call__(self, payload: Any) -> Dict[str, Any]:
        return self.run(payload)
