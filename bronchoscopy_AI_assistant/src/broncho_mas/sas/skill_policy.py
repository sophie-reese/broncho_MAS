from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class SkillPolicyCard:
    name: str
    source_path: str = ""
    purpose: str = ""
    activation: str = ""
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    required_headers: List[str] = field(default_factory=list)
    raw_text: str = ""

    def to_prompt_block(self) -> str:
        lines: List[str] = [f"[{self.name}]"]
        if self.purpose:
            lines.append(f"Purpose: {self.purpose}")
        if self.activation:
            lines.append(f"Activation: {self.activation}")
        if self.inputs:
            lines.append("Inputs: " + ", ".join(self.inputs))
        if self.outputs:
            lines.append("Outputs: " + ", ".join(self.outputs))
        if self.constraints:
            lines.append("Constraints:")
            for item in self.constraints:
                lines.append(f"- {item}")
        if self.required_headers:
            lines.append("Required headers:")
            for item in self.required_headers:
                lines.append(f"- {item}")
        return "\n".join(lines).strip()


class SkillPolicyLoader:
    """
    Minimal MD loader for SAS skill-policy files.

    It extracts a compact policy card from a markdown file so the manager can
    compile short policy context for the current call.
    """

    DEFAULT_FILE_MAP = {
        "guidance_skill": "guidance_skill.md",
        "qa_skill": "qa_skill.md",
        "support_skill": "support_skill.md",
        "statistics_skill": "statistics_skill.md",
        "reporting_skill": "reporting_skill.md",
        "landmark_teaching_skill": "landmark_teaching_skill.md",
    }

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent / "policies"

    def load_file(self, path: str) -> SkillPolicyCard:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.base_dir / file_path
        text = file_path.read_text(encoding="utf-8")
        name = file_path.stem
        return self._parse_markdown(name=name, source_path=str(file_path), text=text)

    def resolve_skill_path(self, skill_name: str) -> Path:
        filename = self.DEFAULT_FILE_MAP.get(skill_name, f"{skill_name}.md")
        return self.base_dir / filename

    def load_many(self, paths: Iterable[str]) -> Dict[str, SkillPolicyCard]:
        cards: Dict[str, SkillPolicyCard] = {}
        for path in paths:
            card = self.load_file(path)
            cards[card.name] = card
        return cards

    def load_cards(self, skill_names: Iterable[str]) -> List[SkillPolicyCard]:
        cards: List[SkillPolicyCard] = []
        for skill_name in skill_names:
            path = self.resolve_skill_path(skill_name)
            card = self.load_file(path)
            card.name = skill_name
            cards.append(card)
        return cards

    def _parse_markdown(self, *, name: str, source_path: str, text: str) -> SkillPolicyCard:
        card = SkillPolicyCard(name=name, source_path=source_path, raw_text=text)

        lines = [ln.rstrip() for ln in text.splitlines()]

        bullet_re = re.compile(r"^\s*-\s+")
        key_value_re = re.compile(r"^\s*-\s*([^:]+):\s*(.+?)\s*$")
        header_re = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")

        section: Optional[str] = None
        purpose_lines: List[str] = []
        activation_lines: List[str] = []
        constraints: List[str] = []
        inputs: List[str] = []
        outputs: List[str] = []
        required_headers: List[str] = []

        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            header_match = header_re.match(raw)
            if header_match:
                section = header_match.group(1).strip().lower()
                continue

            kv = key_value_re.match(raw)
            if kv:
                key = kv.group(1).strip().lower()
                value = kv.group(2).strip()
                if key == "inputs":
                    inputs.extend(self._split_csv_like(value))
                    continue
                if key == "outputs":
                    outputs.extend(self._split_csv_like(value))
                    continue
                if key == "activation":
                    activation_lines.append(value)
                    continue
                if key in {"purpose", "goal", "role"}:
                    purpose_lines.append(value)
                    continue

            if section:
                if "purpose" in section or "goal" in section or "role" in section:
                    if bullet_re.match(raw):
                        purpose_lines.append(bullet_re.sub("", raw).strip())
                    else:
                        purpose_lines.append(line)
                    continue
                if "activation" in section:
                    if bullet_re.match(raw):
                        activation_lines.append(bullet_re.sub("", raw).strip())
                    else:
                        activation_lines.append(line)
                    continue
                if "constraint" in section or "instruction" in section or "policy" in section:
                    if bullet_re.match(raw):
                        constraints.append(bullet_re.sub("", raw).strip())
                    else:
                        constraints.append(line)
                    continue
                if "input" in section:
                    if bullet_re.match(raw):
                        inputs.append(bullet_re.sub("", raw).strip())
                    else:
                        inputs.extend(self._split_csv_like(line))
                    continue
                if "output" in section:
                    if bullet_re.match(raw):
                        outputs.append(bullet_re.sub("", raw).strip())
                    else:
                        outputs.extend(self._split_csv_like(line))
                    continue
                if "required header" in section:
                    if bullet_re.match(raw):
                        required_headers.append(bullet_re.sub("", raw).strip())
                    else:
                        required_headers.append(line)
                    continue

            low = line.lower()
            if "must" in low or "do not" in low or "should not" in low or "only" in low:
                constraints.append(line)

        card.purpose = self._dedupe_join(purpose_lines)
        card.activation = self._dedupe_join(activation_lines)
        card.inputs = self._dedupe_list(inputs)
        card.outputs = self._dedupe_list(outputs)
        card.constraints = self._dedupe_list(constraints)
        card.required_headers = self._dedupe_list(required_headers)
        return card

    @staticmethod
    def _split_csv_like(text: str) -> List[str]:
        parts = re.split(r"[;,]", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _dedupe_list(items: Iterable[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for item in items:
            norm = re.sub(r"\s+", " ", item).strip()
            if not norm:
                continue
            key = norm.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(norm)
        return result

    @staticmethod
    def _dedupe_join(items: Iterable[str]) -> str:
        deduped = SkillPolicyLoader._dedupe_list(items)
        return " ".join(deduped).strip()


from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_STAGE_SKILLS: Dict[str, List[str]] = {
    "runtime_guidance": ["guidance_skill", "support_skill", "landmark_teaching_skill", "qa_skill"],
    "reporting": ["statistics_skill", "reporting_skill"],
}


class SkillPolicyCompiler:
    def __init__(self, *, loader: Optional[SkillPolicyLoader] = None) -> None:
        self.loader = loader or SkillPolicyLoader(base_dir=Path(__file__).resolve().parent / "policies")

    def compile_for_skills(self, skill_names: Iterable[str], *, title: str = "SKILL_POLICY") -> str:
        cards = self.loader.load_cards(skill_names)
        blocks = [card.to_prompt_block() for card in cards if card.to_prompt_block().strip()]
        return f"{title}:\n" + ("\n\n".join(blocks) if blocks else "[no skill policy loaded]")

    def compile_for_stage(self, stage: str) -> str:
        skill_names = DEFAULT_STAGE_SKILLS.get(stage, [])
        return self.compile_for_skills(skill_names, title=f"SKILL_POLICY::{stage}")

    def compile_runtime_guidance_policy(self) -> str:
        return self.compile_for_stage("runtime_guidance")

    def compile_reporting_policy(self) -> str:
        return self.compile_for_stage("reporting")


def build_runtime_guidance_policy(*, base_dir: Optional[Path] = None) -> str:
    compiler = SkillPolicyCompiler(loader=SkillPolicyLoader(base_dir=base_dir or Path(__file__).resolve().parent / "policies"))
    return compiler.compile_runtime_guidance_policy()


def build_reporting_policy(*, base_dir: Optional[Path] = None) -> str:
    compiler = SkillPolicyCompiler(loader=SkillPolicyLoader(base_dir=base_dir or Path(__file__).resolve().parent / "policies"))
    return compiler.compile_reporting_policy()
