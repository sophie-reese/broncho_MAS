from __future__ import annotations

import re
from typing import List


def normalize_mojibake(text: str) -> str:
    out = str(text or "")
    for bad in ("\u00c3\u201a\u00c2\u00b0", "\u00c2\u00b0"):
        out = out.replace(bad, "°")
    return out


def sentence_split(text: str) -> List[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n+", normalize_mojibake(text).strip())
    return [re.sub(r"\s+", " ", part).strip() for part in raw if re.sub(r"\s+", " ", part).strip()]


def normalize_landmark_language(text: str) -> str:
    out = normalize_mojibake(text)
    replacements = [
        (r"\bL\d+_[A-Z0-9+]+\b", ""),
        (r"\bto\s*90°\s*right\b", "to the right"),
        (r"\bto\s*90°\s*left\b", "to the left"),
        (r"\b90°\s*right\b", "right"),
        (r"\b90°\s*left\b", "left"),
        (r"\b\d+\s*degrees?\b", ""),
        (r"\btrifurcation\b", "Mercedes sign"),
        (r"\banchor landmark\b", "view"),
        (r"\bHold neutral at\b", "Hold steady at"),
        (r"\bRotate clockwise to the right\b", "Turn right"),
        (r"\bRotate counter-clockwise to the left\b", "Turn left"),
        (r"\bRotate clockwise\b", "Turn right"),
        (r"\bRotate counter-clockwise\b", "Turn left"),
        (r"\bReacquire\b", "Find"),
        (r"\bLocate\b", "Find"),
        (r"\bidentify\b", "find"),
    ]
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out, flags=re.I)
    return re.sub(r"\s+", " ", out).strip(" ,;")


def sanitize_micro_step(text: str) -> str:
    out = normalize_landmark_language(text)
    for src, dst in [
        ("Hold neutral", "Hold steady"),
        ("Advance toward", "Advance into"),
        ("in small increments", "slowly"),
        ("while keeping the lumen centered", ""),
        ("recognition cue", "landmark"),
    ]:
        out = out.replace(src, dst)
    out = re.sub(r"\bAction:\s*", "", out, flags=re.I)
    out = re.sub(r"\bCheck:\s*", "", out, flags=re.I)
    out = re.sub(r"\bexact\b", "", out, flags=re.I)
    out = re.sub(r"\bprecisely\b", "", out, flags=re.I)
    return re.sub(r"\s+", " ", out).strip(" ,;")


def trim_line_words(text: str, max_words: int = 18) -> str:
    cleaned = re.sub(r"\s+", " ", normalize_mojibake(text).strip())
    if not cleaned:
        return ""
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned.rstrip(" ,;:")
    cut = " ".join(words[:max_words]).rstrip(" ,;:")
    if cut and cut[-1] not in ".!?":
        cut += "."
    return cut


def compress_utterance(text: str, *, max_sentences: int = 2, max_words_per_sentence: int = 18) -> str:
    normalized = normalize_landmark_language(text)
    lines = sentence_split(normalized)
    if not lines:
        return trim_line_words(normalized, max_words=max_words_per_sentence) if normalized.strip() else ""
    clean: List[str] = []
    for line in lines:
        compact = trim_line_words(line, max_words=max_words_per_sentence)
        if compact and compact not in clean:
            clean.append(compact)
        if len(clean) == max_sentences:
            break
    return " ".join(clean).strip()


def strip_instruction_prefixes(text: str) -> str:
    out = normalize_mojibake(text)
    prefixes = [
        r"(?is)^guidance\s+provided\s+to\s+(student|students)\s*:\s*",
        r"(?is)^guidance\s+delivered\s*:\s*",
        r"(?is)^student\s+instructed\s+to\s*",
        r"(?is)^spoken\s+guidance\s*:\s*",
        r"(?is)^instruction(al)?\s+guidance\s*:\s*",
    ]
    for pattern in prefixes:
        out = re.sub(pattern, "", out)
    out = out.replace("**", "").replace("•", " ")
    out = out.replace("Mercedes sign / trifurcation", "Mercedes sign")
    out = out.replace("Y-shaped trifurcation", "Mercedes sign")
    return re.sub(r"\s+", " ", out).strip()


def dedupe_sentences(text: str) -> str:
    cleaned: List[str] = []
    for chunk in re.split(r"(?<=[.!?])\s+", strip_instruction_prefixes(text)):
        sentence = str(chunk or "").strip(" -:;")
        if not sentence:
            continue
        lower = sentence.lower()
        if lower.startswith(("guidance delivered", "student instructed to", "guidance provided", "submit_guidance", "final_answer", "calling tool", "tool call")):
            continue
        if sentence not in cleaned:
            cleaned.append(sentence)
    return re.sub(r"\s+", " ", " ".join(cleaned)).strip()


def clinicalize_ui_text(text: str) -> str:
    out = normalize_landmark_language(text)
    for pattern in [r"\bL\d+_[A-Z0-9+]+\b", r"\b90°\b", r"\banchor landmark\b"]:
        out = re.sub(pattern, "", out, flags=re.I)
    return re.sub(r"\s+", " ", out).strip(" ,;")


def short_ui_text(text: str, *, max_words: int = 28) -> str:
    cleaned = re.sub(r"\s+", " ", normalize_mojibake(text).strip())
    if not cleaned:
        return ""
    parts = sentence_split(cleaned)
    candidate = " ".join(parts[:2]).strip() if parts else cleaned
    words = candidate.split()
    if len(words) <= max_words:
        return candidate
    candidate = " ".join(words[:max_words]).rstrip(",;: ")
    while candidate.endswith(("and", "or", "to", "the", "a", "an", "with", "toward", "towards", "into")):
        candidate = " ".join(candidate.split()[:-1]).rstrip(",;: ")
    if candidate and candidate[-1] not in ".!?":
        candidate += "."
    return candidate



def relaxed_ui_text(text: str, *, max_words: int = 42) -> str:
    cleaned = re.sub(r"\s+", " ", normalize_mojibake(text).strip())
    if not cleaned:
        return ""
    parts = sentence_split(cleaned)
    candidate = " ".join(parts[:2]).strip() if parts else cleaned
    words = candidate.split()
    if len(words) <= max_words:
        return candidate
    candidate = " ".join(words[:max_words]).rstrip(",;: ")
    while candidate.endswith(("and", "or", "to", "the", "a", "an", "with", "toward", "towards", "into")):
        candidate = " ".join(candidate.split()[:-1]).rstrip(",;: ")
    if candidate and candidate[-1] not in ".!?":
        candidate += "."
    return candidate


def normalize_agent_utterance(text: str) -> str:
    return dedupe_sentences(normalize_landmark_language(text))



def _strip_empty_parentheses(text: str) -> str:
    out = str(text or "")
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"\[\s*\]", "", out)
    return out


def light_cleanup_ui_text(text: str) -> str:
    out = normalize_mojibake(str(text or ""))
    out = out.replace("**", "")
    out = out.replace("__", "")
    out = _strip_empty_parentheses(out)
    out = out.replace("—", ". ")
    out = out.replace("–", "-")
    out = re.sub(r"\s*;\s*", ". ", out)
    out = re.sub(r"\s*,\s*", ", ", out)
    out = re.sub(r"\s*\.\s*\.\s*\.", ".", out)
    out = re.sub(r"\s+", " ", out).strip(" ,;:-")
    out = re.sub(r"\.\s*\.", ".", out)
    if out and out[-1] not in ".!?":
        out += "."
    return out
