"""Directional hint builder for bronchoscopy runtime guidance.

The goal is to restore grounded instrument-control language such as:
- tilt up the knob
- tilt down the knob
- rotate clockwise
- rotate counter-clockwise
- push forward the scope
- pull back the scope

while keeping the result modular, inspectable, and easy to integrate into a
new runtime manager.

Design principles
-----------------
1. Event signal decides *when* to speak.
2. DirectionalHintBuilder decides *which direction* the instrument suggests.
3. Your LLM / formatter decides *how to phrase it naturally*.

This file does NOT claim medical ground truth. It converts relative control
signals into explicit language hints. Treat its output as a grounded control
hint layer, not as final expert truth.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence


# Axis convention copied conceptually from the old control path,
# but reimplemented cleanly here.
BENDING_IDX = 0      # lever / tip flexion
ROTATION_IDX = 1     # shaft rotation
TRANSLATION_IDX = 2  # advance / withdraw


@dataclass
class DeadzoneConfig:
    """Thresholds for suppressing tiny noisy movements."""

    bend_eps: float = 0.08
    rotate_eps: float = 0.08
    translate_eps: float = 0.02


@dataclass
class DirectionalHint:
    """Structured output from the directional hint layer."""

    bend_label: str = "neutral"
    rotate_label: str = "neutral"
    translate_label: str = "neutral"

    bend_text: Optional[str] = None
    rotate_text: Optional[str] = None
    translate_text: Optional[str] = None

    primary_action: Optional[str] = None
    secondary_action: Optional[str] = None

    confidence: str = "heuristic"
    source: str = "m_jointsVelRel"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DirectionalHintBuilder:
    """Build explicit directional hints from relative control signals.

    Parameters
    ----------
    deadzone:
        Small values inside these thresholds are treated as neutral.
    prefer_order:
        Priority order used when choosing a primary action in non-backtrack
        cases. Default: rotation -> bend -> translation.
    """

    def __init__(
        self,
        deadzone: Optional[DeadzoneConfig] = None,
        prefer_order: Optional[Sequence[str]] = None,
    ) -> None:
        self.deadzone = deadzone or DeadzoneConfig()
        self.prefer_order: List[str] = list(prefer_order or ("rotate", "bend", "translate"))
        self._last_primary_action: Optional[str] = None
        self._repeat_primary_count: int = 0

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def build(
        self,
        m_jointsVelRel: Optional[Sequence[float]],
        event_flag: Optional[int] = None,
        *,
        force_backtrack_primary: bool = True,
        include_secondary: bool = True,
    ) -> DirectionalHint:
        """Convert relative control values into explicit directional hints.

        Parameters
        ----------
        m_jointsVelRel:
            Expected shape [bend, rotate, translate]. If missing or invalid,
            a neutral hint is returned.
        event_flag:
            Optional event signal. If event_flag == 6 (backtrack), withdraw is
            prioritized when available.
        force_backtrack_primary:
            If True, backtrack prefers translation first.
        include_secondary:
            If True, choose a secondary action as well.
        """
        if not self._valid_triplet(m_jointsVelRel):
            return DirectionalHint(notes="No valid control triplet provided.")

        bend_value = float(m_jointsVelRel[BENDING_IDX])
        rotate_value = float(m_jointsVelRel[ROTATION_IDX])
        translate_value = float(m_jointsVelRel[TRANSLATION_IDX])

        bend_label = self._signed_label(bend_value, self.deadzone.bend_eps)
        rotate_label = self._signed_label(rotate_value, self.deadzone.rotate_eps)
        translate_label = self._signed_label(translate_value, self.deadzone.translate_eps)

        bend_text = self._bend_text(bend_label)
        rotate_text = self._rotate_text(rotate_label)
        translate_text = self._translate_text(translate_label)

        primary_action, secondary_action = self._select_actions(
            bend_text=bend_text,
            rotate_text=rotate_text,
            translate_text=translate_text,
            event_flag=event_flag,
            force_backtrack_primary=force_backtrack_primary,
            include_secondary=include_secondary,
        )
        primary_action, secondary_action, stale_note = self._suppress_stale_primary(
            primary_action=primary_action,
            secondary_action=secondary_action,
            bend_text=bend_text,
            rotate_text=rotate_text,
            translate_text=translate_text,
            event_flag=event_flag,
        )

        return DirectionalHint(
            bend_label=bend_label,
            rotate_label=rotate_label,
            translate_label=translate_label,
            bend_text=bend_text,
            rotate_text=rotate_text,
            translate_text=translate_text,
            primary_action=primary_action,
            secondary_action=secondary_action,
            confidence="heuristic",
            source="m_jointsVelRel",
            notes=self._notes_for(event_flag, bend_label, rotate_label, translate_label, stale_note=stale_note),
        )

    def build_prompt_block(
        self,
        hint: DirectionalHint,
        *,
        include_secondary: bool = True,
        include_axis_summary: bool = True,
    ) -> str:
        """Create a compact prompt block that can be appended to your LLM prompt."""
        lines: List[str] = ["Directional control hint:"]

        if hint.primary_action:
            lines.append(f"- primary action: {hint.primary_action}")
        if include_secondary and hint.secondary_action:
            lines.append(f"- secondary action: {hint.secondary_action}")

        if include_axis_summary:
            axis_bits = []
            if hint.bend_text:
                axis_bits.append(f"bend={hint.bend_text}")
            if hint.rotate_text:
                axis_bits.append(f"rotate={hint.rotate_text}")
            if hint.translate_text:
                axis_bits.append(f"translate={hint.translate_text}")
            if axis_bits:
                lines.append("- axis summary: " + "; ".join(axis_bits))

        lines.append("- Treat these as grounded control hints, not full explanations.")
        return "\n".join(lines)

    def build_coach_lines(
        self,
        hint: DirectionalHint,
        *,
        opener: Optional[str] = None,
        include_secondary: bool = False,
    ) -> List[str]:
        """Create short coach-style lines for UI/TTS.

        Example:
            ["Good.", "Rotate clockwise slightly.", "Tilt the knob up."]
        """
        lines: List[str] = []
        if opener:
            lines.append(self._sentence_case(opener.strip()))

        if hint.primary_action:
            lines.append(self._coachify(hint.primary_action))
        if include_secondary and hint.secondary_action:
            lines.append(self._coachify(hint.secondary_action))
        return lines

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    @staticmethod
    def _valid_triplet(values: Optional[Sequence[float]]) -> bool:
        if values is None:
            return False
        try:
            return len(values) >= 3
        except TypeError:
            return False

    @staticmethod
    def _signed_label(value: float, eps: float) -> str:
        if value > eps:
            return "positive"
        if value < -eps:
            return "negative"
        return "neutral"

    @staticmethod
    def _bend_text(label: str) -> Optional[str]:
        mapping = {
            "positive": "tilt up the knob",
            "negative": "tilt down the knob",
            "neutral": None,
        }
        return mapping[label]

    @staticmethod
    def _rotate_text(label: str) -> Optional[str]:
        mapping = {
            "positive": "rotate clockwise",
            "negative": "rotate counter-clockwise",
            "neutral": None,
        }
        return mapping[label]

    @staticmethod
    def _translate_text(label: str) -> Optional[str]:
        mapping = {
            "positive": "push forward the scope",
            "negative": "pull back the scope",
            "neutral": None,
        }
        return mapping[label]

    def _select_actions(
        self,
        *,
        bend_text: Optional[str],
        rotate_text: Optional[str],
        translate_text: Optional[str],
        event_flag: Optional[int],
        force_backtrack_primary: bool,
        include_secondary: bool,
    ) -> tuple[Optional[str], Optional[str]]:
        action_map = {
            "bend": bend_text,
            "rotate": rotate_text,
            "translate": translate_text,
        }

        ordered_keys: List[str] = []
        if force_backtrack_primary and event_flag == 6:
            ordered_keys = ["translate", "bend", "rotate"]
        else:
            ordered_keys = list(self.prefer_order)

        primary: Optional[str] = None
        secondary: Optional[str] = None

        for key in ordered_keys:
            candidate = action_map.get(key)
            if candidate:
                primary = candidate
                break

        if include_secondary and primary is not None:
            for key in ordered_keys:
                candidate = action_map.get(key)
                if candidate and candidate != primary:
                    secondary = candidate
                    break

        return primary, secondary

    def _suppress_stale_primary(
        self,
        *,
        primary_action: Optional[str],
        secondary_action: Optional[str],
        bend_text: Optional[str],
        rotate_text: Optional[str],
        translate_text: Optional[str],
        event_flag: Optional[int],
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        stale_note: Optional[str] = None

        if primary_action and primary_action == self._last_primary_action:
            self._repeat_primary_count += 1
        else:
            self._repeat_primary_count = 1 if primary_action else 0
            self._last_primary_action = primary_action

        if not primary_action:
            self._last_primary_action = None
            self._repeat_primary_count = 0
            return primary_action, secondary_action, stale_note

        if event_flag == 6:
            return primary_action, secondary_action, stale_note

        rotate_repeat = primary_action in {"rotate clockwise", "rotate counter-clockwise"} and self._repeat_primary_count >= 3
        if not rotate_repeat:
            return primary_action, secondary_action, stale_note

        stale_note = f"stale_primary={primary_action};repeat={self._repeat_primary_count}"
        fallback_primary = None
        fallback_secondary = None

        if translate_text == "pull back the scope":
            fallback_primary = translate_text
            fallback_secondary = "re-center the view"
        elif bend_text and bend_text != primary_action:
            fallback_primary = "re-center the view"
            fallback_secondary = bend_text
        else:
            fallback_primary = "re-center the view"
            if translate_text and translate_text != primary_action:
                fallback_secondary = translate_text

        self._last_primary_action = fallback_primary
        self._repeat_primary_count = 1
        return fallback_primary, fallback_secondary or secondary_action, stale_note

    @staticmethod
    def _notes_for(
        event_flag: Optional[int],
        bend_label: str,
        rotate_label: str,
        translate_label: str,
        *,
        stale_note: Optional[str] = None,
    ) -> str:
        bits = [
            f"event_flag={event_flag}",
            f"bend={bend_label}",
            f"rotate={rotate_label}",
            f"translate={translate_label}",
        ]
        if stale_note:
            bits.append(stale_note)
        return "; ".join(bits)

    @staticmethod
    def _sentence_case(text: str) -> str:
        if not text:
            return text
        text = text.strip()
        text = text[0].upper() + text[1:]
        if text[-1] not in ".!?":
            text += "."
        return text

    @staticmethod
    def _coachify(action: str) -> str:
        replacements = {
            "tilt up the knob": "Tilt the knob up.",
            "tilt down the knob": "Tilt the knob down.",
            "rotate clockwise": "Rotate clockwise slightly.",
            "rotate counter-clockwise": "Rotate counter-clockwise slightly.",
            "push forward the scope": "Advance slowly.",
            "pull back the scope": "Pull back slowly.",
            "re-center the view": "Re-center first.",
        }
        return replacements.get(action, DirectionalHintBuilder._sentence_case(action))


def attach_directional_hint(
    plan_json: Optional[Dict[str, Any]],
    m_jointsVelRel: Optional[Sequence[float]],
    *,
    event_flag: Optional[int] = None,
    builder: Optional[DirectionalHintBuilder] = None,
) -> Dict[str, Any]:
    """Attach a structured directional hint into an existing plan_json dict.

    This is useful when your runtime already produces plan_json and you want
    to enrich it with explicit control-direction metadata.
    """
    builder = builder or DirectionalHintBuilder()
    hint = builder.build(m_jointsVelRel, event_flag=event_flag)

    out: Dict[str, Any] = dict(plan_json or {})
    out["directional_hint"] = hint.to_dict()
    return out


def prompt_with_directional_hint(
    base_prompt: str,
    m_jointsVelRel: Optional[Sequence[float]],
    *,
    event_flag: Optional[int] = None,
    builder: Optional[DirectionalHintBuilder] = None,
) -> str:
    """Append a compact directional-hint block to an existing prompt."""
    builder = builder or DirectionalHintBuilder()
    hint = builder.build(m_jointsVelRel, event_flag=event_flag)
    block = builder.build_prompt_block(hint)

    base = (base_prompt or "").rstrip()
    if not base:
        return block
    return f"{base}\n\n{block}"


def _demo() -> None:
    builder = DirectionalHintBuilder()

    samples = {
        "locate_example": [0.22, -0.31, 0.0],
        "advance_example": [0.0, 0.0, 0.15],
        "backtrack_example": [-0.11, 0.04, -0.30],
        "neutral_example": [0.01, -0.02, 0.0],
    }

    for name, triplet in samples.items():
        hint = builder.build(triplet, event_flag=6 if name == "backtrack_example" else 3)
        print(f"\n=== {name} ===")
        print(hint.to_dict())
        print(builder.build_prompt_block(hint))
        print(builder.build_coach_lines(hint, opener="good", include_secondary=True))


if __name__ == "__main__":
    _demo()
