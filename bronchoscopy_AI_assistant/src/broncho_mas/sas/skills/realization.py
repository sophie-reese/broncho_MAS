from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from ...shared.utterance_postprocess import compress_utterance

def _extract_text_chunks(value: Any) -> List[str]:
    chunks: List[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                chunks.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
    return chunks


def _extract_model_text(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp.strip()
    if isinstance(resp, dict):
        for key in ("text", "content"):
            value = resp.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            chunks = _extract_text_chunks(value)
            if chunks:
                return " ".join(chunks).strip()
    for attr in ("text", "content"):
        value = getattr(resp, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        chunks = _extract_text_chunks(value)
        if chunks:
            return " ".join(chunks).strip()
    return ""


def _model_call_variants(prompt: str) -> Sequence[Any]:
    return (
        [{"role": "user", "content": prompt}],
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        prompt,
    )


def _call_model_text(model: Any, prompt: str) -> str:
    """Best-effort text extraction across the supported local model adapters."""

    try:
        if hasattr(model, "generate"):
            text = _extract_model_text(model.generate(prompt))
            if text:
                return text
    except Exception:
        pass

    if callable(model):
        for messages in _model_call_variants(prompt):
            try:
                text = _extract_model_text(model(messages))
                if text:
                    return text
            except Exception:
                continue
    return ""

def realize_frame_response(
    *,
    model: Any,
    state: Dict[str, Any],
    frame: Dict[str, Any],
    current_situation: str,
    previous_msgs: str,
    policy_text: str = "",
) -> str:
    base = str((frame or {}).get("base_utterance") or "").strip()
    if not base:
        return ""
    if getattr(model, "is_fallback_backend", False):
        return base
    if str((frame or {}).get("safety_mode") or "").lower() == "safety":
        return base
    prompt = _build_realization_prompt(
        frame=frame,
        current_situation=current_situation,
        previous_msgs=previous_msgs,
        policy_text=policy_text,
        soft_prompt=str(state.get("soft_prompt") or ""),
        llm_reason=str(state.get("llm_reason") or ""),
    )
    raw = _call_model_text(model, prompt).strip()
    if not raw:
        return base
    return compress_utterance(raw, max_sentences=2, max_words_per_sentence=24)


def deterministic_frame_text(frame: Dict[str, Any]) -> str:
    max_sentences = 3 if str((frame or {}).get("mode") or "").strip().lower() == "teaching" else 2
    return compress_utterance(
        str((frame or {}).get("base_utterance") or "").strip(),
        max_sentences=max_sentences,
        max_words_per_sentence=24,
    )


def _build_frame(
    *,
    mode: str,
    question_mode: str,
    safety_mode: str,
    intent: str,
    target_airway: str,
    target_label: str,
    recognition_cue: str,
    prefix: str,
    action_line: str,
    cue_line: str,
    answer_core: str,
    next_step: str,
    question: str,
    fallback_guidance: str,
    base_utterance: str,
) -> Dict[str, Any]:
    max_sentences = 3 if str(mode or "").strip().lower() == "teaching" else 2
    compressed = compress_utterance(base_utterance, max_sentences=max_sentences, max_words_per_sentence=24)
    return {
        "mode": mode,
        "question_mode": question_mode,
        "safety_mode": safety_mode,
        "intent": intent,
        "target_airway": target_airway,
        "target_label": target_label,
        "recognition_cue": recognition_cue,
        "prefix": prefix,
        "action_line": action_line,
        "cue_line": cue_line,
        "answer_core": answer_core,
        "next_step": next_step,
        "question": question,
        "fallback_guidance": fallback_guidance,
        "base_utterance": compressed,
    }


def _join_lines(*parts: str) -> str:
    clean = [str(p).strip() for p in parts if str(p).strip()]
    return " ".join(clean).strip()


def _build_realization_prompt(
    *,
    frame: Dict[str, Any],
    current_situation: str,
    previous_msgs: str,
    policy_text: str = "",
    soft_prompt: str = "",
    llm_reason: str = "",
) -> str:
    frame_json = json.dumps(frame, ensure_ascii=True)
    mode = str((frame or {}).get("mode") or "guidance")
    extra_style = (
        "You may be light and a little humorous in the first sentence, but do not continue the off-task topic."
        if str((frame or {}).get("question_mode") or "") == "off_task_social"
        else "Use calm, friendly bedside teaching language."
    )
    policy_block = (
        "SYSTEM_POLICY_SOURCE: These policy notes were compiled from SAS skill markdown files.\n"
        "Apply them as runtime constraints for this response.\n\n"
        f"{policy_text.strip()}\n\n"
    ) if policy_text.strip() else ""
    hint_block = ""
    if llm_reason.strip() or soft_prompt.strip():
        hint_block = (
            f"LLM_REASON:\n{llm_reason.strip()}\n\n"
            f"SOFT_PROMPT:\n{soft_prompt.strip()}\n\n"
        )
    return (
        policy_block
        + "ROLE: Bronchoscopy bedside coach\n"
        "TASK: Rewrite the deterministic coaching frame into natural spoken guidance.\n\n"
        "HARD RULES:\n"
        "1. Preserve the action, target airway, anatomy, and safety priority exactly as given.\n"
        "2. Do not add new anatomy, new targets, or extra procedural steps.\n"
        "3. Keep the response to one or two short spoken sentences.\n"
        "3b. Do not repeat the exact same teaching phrase from PREVIOUS_STEP_SUMMARY unless it is required for safety.\n"
        "3c. If you mention a memory cue or teaching tip, keep the sentence complete rather than cutting it off.\n"
        "4. Keep it friendly, clear, and memorable, but still clinically grounded.\n"
        "5. If the frame includes a question response, answer briefly and then return to the bronchoscopy step.\n"
        f"6. {extra_style}\n"
        "7. Never mention JSON, internal modes, or hidden system logic.\n"
        "8. The final wording should be natural enough to display directly in the UI.\n\n"
        + f"{hint_block}"
        f"FRAME_JSON:\n{frame_json}\n\n"
        f"CURRENT_SITUATION:\n{current_situation.strip()}\n\n"
        f"PREVIOUS_MSGS:\n{previous_msgs.strip()}\n\n"
        f"BASE_UTTERANCE:\n{str((frame or {}).get('base_utterance') or '').strip()}\n\n"
        f"MODE: {mode}\n\n"
        "ACTION:\nReturn only the spoken response."
    )
