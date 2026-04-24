from __future__ import annotations

import re
from typing import List


def normalize_mojibake(text: str) -> str:
    out = str(text or "")
    for bad in ("\u00c3\u201a\u00c2\u00b0", "\u00c2\u00b0"):
        out = out.replace(bad, "°")
    out = out.replace("â€™", "'")
    out = out.replace("â€œ", '"')
    out = out.replace("â€\x9d", '"')
    out = out.replace("â€“", "-")
    out = out.replace("â€”", "-")
    out = out.replace("â€¢", " ")
    out = out.replace("**", "")
    out = out.replace("__", "")
    return out


def normalize_display_text(text: str) -> str:
    out = normalize_mojibake(text)
    replacements = {
        "Â°": "°",
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€“": "-",
        "â€”": "-",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def sentence_split(text: str) -> List[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n+", normalize_display_text(text).strip())
    return [re.sub(r"\s+", " ", part).strip() for part in raw if re.sub(r"\s+", " ", part).strip()]



_BLACKLIST_PATTERNS = (
    (r"\bconfirm it\b[.!]?", ""),
    (r"\buse [^.?!]{0,120}? to confirm [^.?!]{0,120}?[.!]?", ""),
    (r"\bfind [^.?!]{0,120}? by [^.?!]{0,120}?\.\s*use [^.?!]{0,120}? to confirm [^.?!]{0,120}?[.!]?", ""),
    (r"\b(?:great work getting to|nice work reaching|good job reaching)\s+(?:the\s+)?(?:rmb|rul|bi|rml|rll|lmb|carina)\b[.!]?", ""),
)


def sanitize_blacklisted_phrases(text: str) -> str:
    out = normalize_display_text(str(text or ""))
    for pattern, replacement in _BLACKLIST_PATTERNS:
        out = re.sub(pattern, replacement, out, flags=re.I)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([.,!?])", r"\1", out)
    out = re.sub(r"([.!?])\1+", r"\1", out)
    return out.strip()


def remove_repeated_sentences(text: str) -> str:
    sentences = sentence_split(text)
    if not sentences:
        return normalize_display_text(str(text or "")).strip()

    cleaned: List[str] = []
    previous_norm = ""
    seen = set()
    for sentence in sentences:
        norm = re.sub(r"\s+", " ", sentence.strip().lower())
        if not norm:
            continue
        if norm == previous_norm:
            continue
        if norm in seen and len(sentences) <= 3:
            continue
        cleaned.append(sentence.strip())
        seen.add(norm)
        previous_norm = norm
    return " ".join(cleaned).strip()


def sanitize_ui_utterance(text: str) -> str:
    out = sanitize_blacklisted_phrases(text)
    out = remove_repeated_sentences(out)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\.\s*\.", ".", out)
    out = re.sub(r"\s*;\s*", ". ", out)
    return out.strip(" ,;:-")


_DIRECTIONAL_ACTION_PATTERNS = (
    "tilt the knob up",
    "tilt the knob down",
    "tilt up the knob",
    "tilt down the knob",
    "knob up",
    "knob down",
    "rotate clockwise",
    "rotate counter-clockwise",
    "counter-clockwise",
    "clockwise",
    "advance the scope",
    "advance slowly",
    "push forward the scope",
    "pull back the scope",
    "pull back",
    "withdraw",
)


def has_directional_action(text: str) -> bool:
    low = normalize_display_text(str(text or "")).lower()
    return any(pattern in low for pattern in _DIRECTIONAL_ACTION_PATTERNS)


def select_priority_sentences(text: str, *, max_sentences: int = 2) -> List[str]:
    lines = sentence_split(text)
    if not lines:
        return []

    chosen: List[str] = []
    action_lines = [line for line in lines if has_directional_action(line)]
    if action_lines:
        chosen.append(action_lines[0])
    local_cue_lines = [line for line in lines if line.lower().startswith("local cue:")]
    if local_cue_lines and local_cue_lines[0] not in chosen:
        chosen.append(local_cue_lines[0])

    for line in lines:
        if line not in chosen:
            chosen.append(line)
        if len(chosen) >= max_sentences:
            break
    return chosen[:max_sentences]

def normalize_landmark_language(text: str) -> str:
    out = normalize_mojibake(text)
    replacements = [
        (r"\bL\d+_[A-Z0-9+]+\b", ""),
        (r"\bto\s*90°\s*right\b", "to the right"),
        (r"\bto\s*90°\s*left\b", "to the left"),
        (r"\b90°\s*right\b", "right"),
        (r"\b90°\s*left\b", "left"),
        (r"\b\d+\s*degrees?\b", ""),
        (r"\banchor landmark\b", "view"),
        (r"\bHold neutral at\b", "Hold steady at"),
        (r"\bRotate clockwise to the right\b", "Rotate clockwise"),
        (r"\bRotate counter-clockwise to the left\b", "Rotate counter-clockwise"),
        (r"\bRotate clockwise\b", "Rotate clockwise"),
        (r"\bRotate counter-clockwise\b", "Rotate counter-clockwise"),
        (r"\bReacquire\b", "Find"),
        (r"\bLocate\b", "Find"),
        (r"\bidentify\b", "find"),
        (r"\bcarina bifurcation;\s*symmetric right/left main bronchi\b", "the carina with both main bronchi in view"),
        (r"\bcarina bifurcation;\s*symmetric main bronchi\b", "the carina with both main bronchi in view"),
        (r"\bsymmetric right/left main bronchi\b", "both main bronchi in view"),
        (r"\bsymmetric main bronchi\b", "both main bronchi in view"),
        (r"\bthat carina bifurcation and symmetry\b", "that carina view"),
        (r"\bcarina bifurcation and symmetry\b", "the carina with both main bronchi in view"),
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
    cleaned = re.sub(r"\s+", " ", normalize_display_text(text).strip())
    if not cleaned:
        return ""
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned.rstrip(" ,;:")

    overflow_limit = max_words + max(2, int(max_words * 0.3))
    if cleaned[-1:] in ".!?" and len(words) <= overflow_limit:
        return cleaned.rstrip(" ,;:")

    prefix_words = words[:max_words]
    prefix = " ".join(prefix_words).rstrip(" ,;:")
    boundary_positions = [prefix.rfind(token) for token in (". ", "! ", "? ", "; ", ", ")]
    best_boundary = max(boundary_positions) if boundary_positions else -1
    if best_boundary >= max(12, int(len(prefix) * 0.55)):
        bounded = prefix[: best_boundary + 1].rstrip(" ,;:")
        if bounded:
            return bounded

    cut = prefix.rstrip(" ,;:")
    if cut and cleaned[-1:] in ".!?":
        return f"{cut}..."
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
    out = normalize_display_text(text)
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
    out = out.replace("Mercedes sign / trifurcation", "right upper-lobe trifurcation")
    out = out.replace("Y-shaped trifurcation", "trifurcation")
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
    out = out.replace("\u00c2\u00b0", "\u00b0")
    out = out.replace("\u00e2\u20ac\u2122", "'")
    for pattern in [r"\bL\d+_[A-Z0-9+]+\b", r"\b90°\b", r"\banchor landmark\b"]:
        out = re.sub(pattern, "", out, flags=re.I)
    return re.sub(r"\s+", " ", out).strip(" ,;")


def short_ui_text(text: str, *, max_words: int = 28, max_sentences: int = 2) -> str:
    cleaned = re.sub(r"\s+", " ", normalize_display_text(text).strip())
    if not cleaned:
        return ""
    parts = select_priority_sentences(cleaned, max_sentences=max_sentences)
    candidate = " ".join(parts).strip() if parts else cleaned
    words = candidate.split()
    if len(words) <= max_words:
        return candidate
    candidate = trim_line_words(candidate, max_words=max_words).rstrip(",;: ")
    while candidate.endswith(("and", "or", "to", "the", "a", "an", "with", "toward", "towards", "into")):
        candidate = " ".join(candidate.split()[:-1]).rstrip(",;: ")
    if candidate and candidate[-1] not in ".!?":
        candidate += "."
    return candidate



def relaxed_ui_text(text: str, *, max_words: int = 42, max_sentences: int = 2) -> str:
    cleaned = re.sub(r"\s+", " ", normalize_display_text(text).strip())
    if not cleaned:
        return ""
    parts = select_priority_sentences(cleaned, max_sentences=max_sentences)
    candidate = " ".join(parts).strip() if parts else cleaned
    words = candidate.split()
    if len(words) <= max_words:
        return candidate
    candidate = trim_line_words(candidate, max_words=max_words).rstrip(",;: ")
    while candidate.endswith(("and", "or", "to", "the", "a", "an", "with", "toward", "towards", "into")):
        candidate = " ".join(candidate.split()[:-1]).rstrip(",;: ")
    if candidate and candidate[-1] not in ".!?":
        candidate += "."
    return candidate


def normalize_agent_utterance(text: str) -> str:
    return dedupe_sentences(sanitize_ui_utterance(normalize_landmark_language(text)))



def _strip_empty_parentheses(text: str) -> str:
    out = str(text or "")
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"\[\s*\]", "", out)
    return out


def light_cleanup_ui_text(text: str) -> str:
    out = sanitize_ui_utterance(text)
    out = normalize_display_text(out)
    out = out.replace("\u00c2\u00b0", "\u00b0")
    out = out.replace("\u00e2\u20ac\u2122", "'")
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
