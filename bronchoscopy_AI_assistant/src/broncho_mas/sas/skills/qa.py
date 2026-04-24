from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from ...shared import get_airway_info, normalize_airway_code, teaching_fact_for
from ...shared.curriculum import CurriculumEngine, DEFAULT_AIRWAY_VISIT_ORDER
from .base import BaseSkill, Plan, SkillResult, State, build_skill_record
from .guidance import _fallback_guidance, _recognition_cue
from .realization import _build_frame, _join_lines, deterministic_frame_text, realize_frame_response
from .teaching import build_landmark_teaching_record, get_taught_landmarks
from .utterance_helpers import _airway_token, safety_risk

_qa_curriculum_cache: CurriculumEngine | None = None


def _qa_curriculum(visit_order: tuple[str, ...] = DEFAULT_AIRWAY_VISIT_ORDER) -> CurriculumEngine:
    global _qa_curriculum_cache
    if _qa_curriculum_cache is None or tuple(_qa_curriculum_cache.visit_order) != tuple(str(x).upper() for x in visit_order):
        _qa_curriculum_cache = CurriculumEngine(visit_order)
    return _qa_curriculum_cache

_QA_COVER_MODES = {
    "teaching_relevant",
    "observation_relevant",
    "visual_relevant",
    "off_task_social",
}


def _contains_any(text: str, patterns: Sequence[str]) -> bool:
    low = str(text or "").lower()
    return any(p in low for p in patterns)


def _contains_anatomy_or_navigation_signal(text: str, next_airway: str = "") -> bool:
    low = str(text or "").lower()
    anatomy_tokens = [
        "carina", "mercedes", "bifurcation", "main bronchi", "bronchi", "bronchus",
        "airway", "opening", "branch", "lumen", "scope", "wall", "center", "re-center",
        "recenter", "advance", "pull back", "back up", "withdraw", "rotate", "turn",
        "tilt", "up", "down", "rb", "lb", "rul", "rml", "rll", "lul", "lll", "trachea",
        "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10",
    ]
    if next_airway and next_airway.lower() in low:
        return True
    return any(token in low for token in anatomy_tokens)


def _looks_like_fragment_unclear(text: str, next_airway: str = "") -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    if _contains_anatomy_or_navigation_signal(raw, next_airway=next_airway):
        return False
    if any(ch in raw for ch in "?!."):
        return False
    words = re.findall(r"[a-z0-9+]+", raw)
    if len(words) <= 2:
        return True
    weak_phrases = {
        "this one", "that one", "over here", "right here", "maybe", "not sure",
        "i think", "hmm", "uh", "um", "wait", "hold on",
    }
    return raw in weak_phrases or (len(words) <= 4 and raw.endswith(("maybe", "i think", "not sure")))


def _is_visual_request(text: str) -> bool:
    return _contains_any(text, ["picture", "diagram", "image", "show me", "draw", "point to", "visual"])


def _is_off_task_social(text: str) -> bool:
    return _contains_any(
        text,
        [
            "weather", "lunch", "coffee", "weekend", "music", "joke", "tired",
            "mad at me", "how are you", "are you okay", "can we talk about",
            "talk about", "chat about", "tell me about", "hockey", "football",
            "soccer", "basketball", "baseball", "sports", "movie", "game",
        ],
    )


def _is_teaching_question(text: str, next_airway: str = "") -> bool:
    low = str(text or "").lower()
    question_cues = [
        "?", "what", "where", "why", "how", "which", "is this", "are we", "am i",
        "do i", "should i", "can i", "what branch", "which branch", "where do i go",
    ]
    return _contains_anatomy_or_navigation_signal(low, next_airway=next_airway) and any(cue in low for cue in question_cues)


def _is_route_navigation_question(text: str, next_airway: str = "") -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False

    direct_phrases = [
        "where do i go",
        "where should i go",
        "which way",
        "where next",
        "what next",
        "how do i find",
        "how can i find",
        "how do i reach",
        "how can i reach",
        "how do i get to",
        "how can i get to",
        "how do i enter",
        "how can i enter",
        "how do i go to",
        "how can i go to",
    ]
    if any(phrase in low for phrase in direct_phrases):
        return True

    route_verbs = ["find", "reach", "get to", "enter", "go to"]
    if any(v in low for v in route_verbs):
        airway_match = re.search(r"\b(?:rb\d{1,2}|lb\d{1,2}|rul|rml|rll|lul|lll|carina|rmb|lmb|bi)\b", low)
        if airway_match:
            return True
        if next_airway and next_airway.lower() in low:
            return True

    if ("how" in low or "where" in low) and _contains_anatomy_or_navigation_signal(low, next_airway=next_airway):
        if any(tok in low for tok in ["opening", "branch", "airway", "segment", "find", "reach", "enter"]):
            return True

    return False


def _is_observation_relevant(text: str, next_airway: str = "") -> bool:
    low = str(text or "").lower()
    if not _contains_anatomy_or_navigation_signal(low, next_airway=next_airway):
        return False
    observation_cues = [
        "i can see", "i see", "i think", "looks like", "it looks like",
        "this is", "that is", "we are in", "we're in", "i am in", "i'm in",
        "i'm at", "we're at", "the next opening", "next opening", "opening here",
        "i found", "i found it", "there is", "there's",
    ]
    return any(cue in low for cue in observation_cues) and not _is_teaching_question(low, next_airway=next_airway)


def can_question_mode_cover_turn(mode: str) -> bool:
    return str(mode or "").strip().lower() in _QA_COVER_MODES


class _QuestionRouter(BaseSkill):
    name = "question_router_internal"
    default_priority = 0.85

    def execute(self, *, state: Dict[str, Any], plan: Dict[str, Any]) -> SkillResult:
        q_raw = str(state.get("student_question") or "").strip()
        q = q_raw.lower()
        next_airway = str((plan or {}).get("next_airway") or "").strip().lower()

        if not q:
            mode = "none"
            reason = "no student question present"
        elif _looks_like_fragment_unclear(q, next_airway=next_airway):
            mode = "fragment_unclear"
            reason = "student input is too fragmentary or ambiguous to answer safely"
        elif _is_visual_request(q):
            mode = "visual_relevant"
            reason = "student asks for visual grounding"
        elif _is_off_task_social(q):
            mode = "off_task_social"
            reason = "student input is off-task social chatter"
        elif _is_teaching_question(q, next_airway=next_airway):
            mode = "teaching_relevant"
            reason = "student asks an anatomically or procedurally relevant question"
        elif _is_observation_relevant(q, next_airway=next_airway):
            mode = "observation_relevant"
            reason = "student makes a clinically relevant observation that should be checked"
        else:
            mode = "other"
            reason = "student input is present but not worth taking the turn"

        return SkillResult(
            skill=self.name,
            active=can_question_mode_cover_turn(mode),
            priority=self.default_priority if can_question_mode_cover_turn(mode) else 0.0,
            reason=reason,
            data={"question_mode": mode},
            utterance="",
        )


def route_question_mode(*, state: State, plan: Plan) -> str:
    return str(_QuestionRouter().execute(state=state, plan=plan).data.get("question_mode") or "none")

def _airway_teaching_fact(value: str) -> str:
    return str(teaching_fact_for(value) or "").strip()


def _airway_label(value: str) -> str:
    info = get_airway_info(value)
    if info and info.label:
        return str(info.label).strip()
    return _airway_token(value)


def _location_phrase(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        return ""
    low = text.lower()
    if low.startswith(("the ", "a ", "an ")):
        return text
    if re.fullmatch(r"[RL]B\d(?:\+\d)?|BI|RMB|LMB|RUL|RML|RLL|LUL|LLL", text.upper()):
        return text
    return f"the {text}"


def _qa_local_navigation_step(*, current_airway: str, target_airway: str, visit_order: tuple[str, ...] = DEFAULT_AIRWAY_VISIT_ORDER) -> str:
    route = _qa_curriculum(visit_order).route_to_airway(target_airway, current_airway=current_airway)
    if not route:
        return ""
    current = normalize_airway_code(current_airway)
    target = normalize_airway_code(target_airway)
    right_upper_targets = {
        "RB1": "the apical, superior branch of the right upper-lobe trifurcation",
        "RB2": "the posterior branch of the right upper-lobe trifurcation",
        "RB3": "the anterior branch of the right upper-lobe trifurcation",
    }
    if current == "CARINA" and target in right_upper_targets:
        target_detail = right_upper_targets[target]
        return (
            "From the carina, enter the right main bronchus and identify the right upper-lobe bronchus "
            f"before you pass down into bronchus intermedius. At the right upper-lobe trifurcation, {target} is {target_detail}; "
            "keep the lumen centered as you align with that branch."
        )
    if current in {"RMB", "RUL"} and target in right_upper_targets:
        target_detail = right_upper_targets[target]
        return (
            f"Find the right upper-lobe trifurcation. {target} is {target_detail}; "
            "center that opening before advancing."
        )
    if len(route) == 1:
        nxt = _airway_token(route[0])
        return f"Work locally toward {nxt}." if nxt else ""
    route_labels = [_airway_label(node) for node in route]
    route_labels = [label for label in route_labels if label]
    target_label = route_labels[-1] if route_labels else _airway_label(target)
    intermediate = route_labels[:-1]
    current_label = _airway_label(current) if current else ""
    target_fact = _airway_teaching_fact(target)
    if intermediate and current_label:
        route_text = " -> ".join(intermediate + [target_label])
        base = f"From {_location_phrase(current_label)}, follow the route through {route_text}."
    elif intermediate:
        route_text = " -> ".join(intermediate + [target_label])
        base = f"Follow the route through {route_text}."
    elif target_label:
        base = f"Center the opening for {target_label} before advancing."
    else:
        base = ""
    if target_fact:
        return f"{base} {target_fact}".strip()
    if base:
        return base
    first_hop = _airway_token(route[0])
    final = _airway_token(route[-1])
    if first_hop and final and first_hop != final:
        return f"Re-anchor at {first_hop}, then continue toward {final}."
    if final:
        return f"Continue toward {final}."
    return ""


def _qa_grounded_answer(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    question_mode: str,
    visit_order: tuple[str, ...] = DEFAULT_AIRWAY_VISIT_ORDER,
) -> str:
    question = str(state.get("student_question") or "").strip()
    q = question.lower()
    current_airway = normalize_airway_code(state.get("current_airway"))
    target_airway = normalize_airway_code((plan or {}).get("next_airway"))
    current_label = _airway_label(current_airway)
    target_label = _airway_label(target_airway)
    target_fact = _airway_teaching_fact(target_airway)
    current_fact = _airway_teaching_fact(current_airway)
    cue = _recognition_cue(plan)
    relation = _qa_curriculum(visit_order).transition_context(current_airway, target_airway).get("relationship", "") if target_airway else ""

    if question_mode == "off_task_social":
        if target_airway:
            return f"Let's save that for later and stay with the bronchoscopy for now. Keep following the current route toward {target_airway}."
        return "Let's save that for later and stay with the bronchoscopy for now. Keep the lumen centered and continue the task."

    if question_mode == "visual_relevant":
        if cue and target_label:
            return f"Use {cue} to orient, then bring {target_label} into view."
        if cue:
            return f"Use {cue} to orient yourself before you advance."
        step = _qa_local_navigation_step(current_airway=current_airway, target_airway=target_airway, visit_order=visit_order)
        return step or (f"Work toward {target_label}." if target_label else "Stay with the current airway view.")

    if question_mode == "observation_relevant":
        if current_label and cue:
            return f"Do not commit yet. Check for {cue} and confirm whether this view matches {current_label}."
        if current_fact:
            return f"Do not commit yet. Use the anatomy: {current_fact}"
        return "Do not commit yet. Confirm the landmark view before you advance."

    if question_mode != "teaching_relevant":
        return ""

    if _is_route_navigation_question(q, next_airway=target_airway):
        step = _qa_local_navigation_step(current_airway=current_airway, target_airway=target_airway, visit_order=visit_order)
        if step:
            return step
        if target_label:
            return f"Your working target is {target_label}."
        return "Stay systematic and advance one branch at a time."

    if any(phrase in q for phrase in ["where am i", "what is this", "which branch is this", "is this", "are we in", "am i in"]):
        if current_label and current_fact:
            return f"Right now you are most likely at {current_label}. {current_fact}"
        if current_label:
            return f"Right now treat this as {current_label}, but confirm the view before you commit."
        return "Confirm the landmark view before naming the branch."

    if "carina" in q:
        return _airway_teaching_fact("CARINA") or "The carina is where the trachea splits into the two main bronchi."

    if "mercedes" in q or ("right upper" in q and ("trifurcation" in q or "three" in q or "sign" in q)):
        fact = _airway_teaching_fact("RUL")
        return fact or "The Mercedes sign marks the right upper-lobe trifurcation into RB1, RB2, and RB3."

    if "why" in q and any(tok in q for tok in ["re-center", "recenter", "center", "reset"]):
        return "Re-centering gives you a stable landmark view before the next branch choice."

    if any(tok in q for tok in ["rb", "lb", "segment", "branch", "opening"]):
        if target_fact:
            return target_fact
        if current_fact:
            return current_fact

    if relation == "same_family" and target_label:
        return f"You are still in the same family. Keep the sequence organized and work toward {target_label}."

    if target_fact:
        return target_fact
    if target_label:
        return f"Keep the working target as {target_label}."
    return "Stay with the bronchoscopy anatomy."


def build_qa_frame(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    question_mode: str,
    fallback_guidance: str,
    visit_order: tuple[str, ...] = DEFAULT_AIRWAY_VISIT_ORDER,
) -> Dict[str, Any]:
    question = str(state.get("student_question") or "").strip()
    target_airway = str((plan or {}).get("next_airway") or "").strip().upper()
    target_label = _airway_token(target_airway)
    cue = _recognition_cue(plan)
    fallback = fallback_guidance or _fallback_guidance(plan=plan)
    answer_core = _qa_grounded_answer(
        state=state,
        plan=plan,
        question_mode=question_mode,
        visit_order=visit_order,
    )

    base = _join_lines(answer_core, fallback) if answer_core else ""
    return _build_frame(
        mode="qa",
        question_mode=question_mode,
        safety_mode="normal",
        intent="qa",
        target_airway=target_airway,
        target_label=target_label,
        recognition_cue=cue,
        prefix="",
        action_line="",
        cue_line="",
        answer_core=answer_core,
        next_step=fallback,
        question=question,
        fallback_guidance=fallback,
        base_utterance=base,
    )


def build_response_candidates(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    directional_hint: Dict[str, Any],
    question_mode: str,
    qa_allowed: bool,
) -> Dict[str, Any]:
    guidance_frame = build_guidance_frame(
        state=state,
        plan=plan,
        directional_hint=directional_hint,
    )
    deterministic_guidance = deterministic_frame_text(guidance_frame)
    landmark_teaching = build_landmark_teaching_record(
        state=state,
        plan=plan,
        guidance_text=deterministic_guidance,
    )

    has_question = bool(str(state.get("student_question") or "").strip()) and question_mode != "none"
    qa_frame: Dict[str, Any] | None = None
    if has_question and qa_allowed and can_question_mode_cover_turn(question_mode):
        qa_frame = build_qa_frame(
            state=state,
            plan=plan,
            question_mode=question_mode,
            fallback_guidance=deterministic_guidance,
        )

    skill_records: List[Dict[str, Any]] = [
        build_skill_record(
                skill="guidance_frame_internal",
            active=True,
            priority=0.8 if not safety_risk(state) else 1.0,
            reason="primary real-time coaching path",
            data=guidance_frame,
            utterance=deterministic_guidance,
        )
    ]
    if landmark_teaching.get("active"):
        skill_records.append(landmark_teaching)
    if has_question:
        skill_records.append(
            build_skill_record(
                skill="qa_frame_internal",
                active=bool(qa_allowed and qa_frame and can_question_mode_cover_turn(question_mode)),
                priority=0.95 if qa_allowed and qa_frame and can_question_mode_cover_turn(question_mode) else 0.2,
                reason=(
                    f"student question routed as {question_mode}; selected"
                    if qa_allowed and qa_frame
                    else f"student question routed as {question_mode}; suppressed for safety or relevance"
                ),
                data=qa_frame or {"question_mode": question_mode},
                utterance=deterministic_frame_text(qa_frame or {}),
            )
        )

    selected_skill = max(
        [record for record in skill_records if record.get("active")],
        key=lambda record: float(record.get("priority") or 0.0),
    )
    if selected_skill.get("skill") == "landmark_teaching_skill" and selected_skill.get("active"):
        taught = list(get_taught_landmarks(state))
        landmark_id = str(selected_skill.get("data", {}).get("landmark_id") or "").strip()
        if landmark_id and landmark_id not in taught:
            taught.append(landmark_id)
        state["landmark_teaching_history"] = taught

    selected_frame = qa_frame if selected_skill.get("skill") == "qa_frame_internal" and qa_frame else guidance_frame
    return {
        "guidance_frame": guidance_frame,
        "qa_frame": qa_frame,
        "selected_frame": selected_frame,
        "selected_frame_mode": str((selected_frame or {}).get("mode") or "guidance"),
        "selected_skill": selected_skill,
        "skill_records": skill_records,
        "deterministic_guidance": deterministic_guidance,
        "landmark_teaching": landmark_teaching,
        "qa_allowed": bool(qa_allowed),
        "has_question": has_question,
    }

class QASkill(BaseSkill):
    """Main question-answering and explanation skill."""

    name = "qa_skill"
    default_priority = 0.95

    def _route_question(self, *, state: Dict[str, Any], plan: Dict[str, Any]) -> str:
        routed = _QuestionRouter().execute(state=state, plan=plan)
        return str((routed.data or {}).get("question_mode") or "none")

    def should_activate(
        self,
        *,
        state: Dict[str, Any],
        plan: Dict[str, Any],
        qa_allowed: bool = True,
        **kwargs: Any,
    ) -> bool:
        has_question = bool(str((state or {}).get("student_question") or "").strip())
        if not has_question or not bool(qa_allowed):
            return False
        return can_question_mode_cover_turn(self._route_question(state=state, plan=plan))

    def _build_frame(
        self,
        *,
        state: Dict[str, Any],
        plan: Dict[str, Any],
        question_mode: str,
        fallback_guidance: str,
        visit_order: tuple[str, ...],
    ) -> Dict[str, Any]:
        return build_qa_frame(
            state=state,
            plan=plan,
            question_mode=question_mode,
            fallback_guidance=fallback_guidance,
            visit_order=visit_order,
        )

    def _should_realize(self, *, state: Dict[str, Any], model: Any, frame: Dict[str, Any]) -> bool:
        # Keep QA grounded and deterministic by default.
        # This avoids adding unverified anatomy through free-form paraphrase.
        if not bool(state.get("allow_llm_qa", False)):
            return False
        base = str((frame or {}).get("base_utterance") or "").strip()
        if not base:
            return False
        if getattr(model, "is_fallback_backend", False):
            return False
        if str((frame or {}).get("safety_mode") or "").lower() == "safety":
            return False
        return True

    def _realize(
        self,
        *,
        state: Dict[str, Any],
        model: Any,
        frame: Dict[str, Any],
        current_situation: str,
        previous_msgs: str,
        policy_text: str = "",
    ) -> str:
        if not self._should_realize(state=state, model=model, frame=frame):
            return ""
        return realize_frame_response(
            model=model,
            state=state,
            frame=frame,
            current_situation=current_situation,
            previous_msgs=previous_msgs,
            policy_text=policy_text,
        )

    def execute(
        self,
        *,
        state: Dict[str, Any],
        plan: Dict[str, Any],
        fallback_guidance: str,
        qa_allowed: bool = True,
        visit_order: tuple[str, ...] = DEFAULT_AIRWAY_VISIT_ORDER,
        model: Any = None,
        current_situation: str = "",
        previous_msgs: str = "",
        policy_text: str = "",
    ) -> SkillResult:
        question_mode = self._route_question(state=state, plan=plan)
        active = (
            bool(str((state or {}).get("student_question") or "").strip())
            and can_question_mode_cover_turn(question_mode)
            and bool(qa_allowed)
        )

        frame = (
            self._build_frame(
                state=state,
                plan=plan,
                question_mode=question_mode,
                fallback_guidance=fallback_guidance,
                visit_order=visit_order,
            )
            if active else
            {"question_mode": question_mode, "mode": "guidance"}
        )

        deterministic = deterministic_frame_text(frame) if active else ""
        wants_realization = bool(active and model is not None and self._should_realize(state=state, model=model, frame=frame))
        reason = (
            f"student question routed as {question_mode}; qa selected"
            if active else
            f"student question routed as {question_mode}; suppressed for safety or relevance"
        )
        return SkillResult(
            skill=self.name,
            active=active,
            priority=self.default_priority if active else 0.0,
            reason=reason,
            data={
                "question_mode": question_mode,
                "frame": frame,
                "frame_mode": str((frame or {}).get("mode") or "qa"),
                "deterministic_utterance": deterministic,
                "realized": False,
                "qa_allowed": bool(qa_allowed),
            },
            utterance=deterministic,
            frame=dict(frame or {}),
            deterministic_text=deterministic,
            wants_realization=wants_realization,
            constraints={"max_sentences": 2, "mode": "qa", "answer_then_return": True},
            debug_reason="skill prepared qa frame; manager must decide whether to realize it",
        )


def qa_skill(**kwargs: Any) -> Dict[str, Any]:
    return QASkill().execute(**kwargs).to_dict()
