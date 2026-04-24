from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class AirwayInfo:
    code: str
    label: str
    kind: str
    side: str = ""
    parent: str = ""
    children: List[str] = field(default_factory=list)
    family: str = ""
    aliases: List[str] = field(default_factory=list)
    teaching_fact: str = ""
    recognition_cues: List[str] = field(default_factory=list)
    pitfalls: List[str] = field(default_factory=list)
    reanchor_point: str = ""
    repair_hint: str = ""
    quiz_prompt: str = ""
    references: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class LandmarkCard:
    id: str
    display_name: str
    aliases: List[str] = field(default_factory=list)
    when_to_teach: str = "first_arrival_only"
    recognition_cues: List[str] = field(default_factory=list)
    common_confusions: List[str] = field(default_factory=list)
    memory_hook_type: str = "position_hook"
    memory_hook_core: str = ""
    memory_hook_rhythm: str = ""
    action_anchor: str = ""
    default_teaching_line: str = ""
    reinforcement_line: str = ""
    repair_line: str = ""
    quiz_line: str = ""
    local_navigation_cues: Dict[str, str] = field(default_factory=dict)
    notes: str = ""
    references: List[str] = field(default_factory=list)


_KNOWLEDGE_PATH = Path(__file__).with_name("bronchoscopy_knowledge.yaml")


@lru_cache(maxsize=1)
def _load_knowledge() -> Dict[str, Any]:
    with _KNOWLEDGE_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("bronchoscopy_knowledge.yaml must contain a mapping at top level")
    return data


@lru_cache(maxsize=1)
def _build_airway_tree() -> Dict[str, AirwayInfo]:
    raw = _load_knowledge().get("airways") or []
    tree: Dict[str, AirwayInfo] = {}
    for item in raw:
        code = str(item.get("id") or "").strip().upper()
        if not code:
            continue
        tree[code] = AirwayInfo(
            code=code,
            label=str(item.get("definition") or code).strip(),
            kind=str(item.get("kind") or "airway").strip(),
            side=str(item.get("side") or "").strip(),
            parent=str(item.get("parent") or "").strip().upper(),
            children=[str(x).strip().upper() for x in (item.get("children") or []) if str(x).strip()],
            family=str(item.get("family") or "").strip(),
            aliases=[str(x).strip() for x in (item.get("aliases") or []) if str(x).strip()],
            teaching_fact=str(item.get("teaching_fact") or "").strip(),
            recognition_cues=[str(x).strip() for x in (item.get("recognition_cues") or []) if str(x).strip()],
            pitfalls=[str(x).strip() for x in (item.get("pitfalls") or []) if str(x).strip()],
            reanchor_point=str(item.get("reanchor_point") or "").strip().upper(),
            repair_hint=str(item.get("repair_hint") or "").strip(),
            quiz_prompt=str(item.get("quiz_prompt") or "").strip(),
            references=[str(x).strip() for x in (item.get("references") or []) if str(x).strip()],
        )
    return tree


@lru_cache(maxsize=1)
def _build_airway_aliases() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for code, info in AIRWAY_TREE.items():
        aliases.setdefault(code, code)
        aliases.setdefault(code.replace("+", "/"), code)
        aliases.setdefault(code.replace("_", " "), code)
        for alias in info.aliases:
            token = str(alias).strip().upper().replace("-", "_")
            token = " ".join(token.split())
            if token:
                aliases.setdefault(token, code)
    aliases.setdefault("RIGHT_MAIN_BRONCHUS", "RMB")
    aliases.setdefault("RIGHT MAIN BRONCHUS", "RMB")
    aliases.setdefault("LEFT_MAIN_BRONCHUS", "LMB")
    aliases.setdefault("LEFT MAIN BRONCHUS", "LMB")
    aliases.setdefault("BRONCHUS_INTERMEDIUS", "BI")
    aliases.setdefault("BRONCHUS INTERMEDIUS", "BI")
    aliases.setdefault("LB1/2", "LB1+2")
    return aliases


@lru_cache(maxsize=1)
def _load_landmark_rows() -> List[dict]:
    return list(_load_knowledge().get("landmarks") or [])


@lru_cache(maxsize=1)
def _build_landmark_cards() -> Dict[str, LandmarkCard]:
    cards: Dict[str, LandmarkCard] = {}
    for item in _load_landmark_rows():
        lid = str(item.get("id") or "").strip()
        if not lid:
            continue
        cards[lid] = LandmarkCard(
            id=lid,
            display_name=str(item.get("display_name") or item.get("definition") or lid).strip(),
            aliases=[str(x).strip() for x in (item.get("aliases") or []) if str(x).strip()],
            when_to_teach=str(item.get("when_to_teach") or "first_arrival_only").strip(),
            recognition_cues=[str(x).strip() for x in (item.get("recognition_cues") or []) if str(x).strip()],
            common_confusions=[str(x).strip() for x in (item.get("pitfalls") or []) if str(x).strip()],
            memory_hook_type=str(item.get("memory_hook_type") or "position_hook").strip(),
            memory_hook_core=str(item.get("memory_hook_core") or "").strip(),
            memory_hook_rhythm=str(item.get("memory_hook_rhythm") or "").strip(),
            action_anchor=str(item.get("action_anchor") or "").strip(),
            default_teaching_line=str(item.get("default_teaching_line") or item.get("teaching_fact") or "").strip(),
            reinforcement_line=str(item.get("reinforcement_line") or "").strip(),
            repair_line=str(item.get("repair_line") or item.get("repair_hint") or "").strip(),
            quiz_line=str(item.get("quiz_line") or item.get("quiz_prompt") or "").strip(),
            local_navigation_cues={str(k).strip().upper(): str(v).strip() for k, v in dict(item.get("local_navigation_cues") or {}).items() if str(k).strip() and str(v).strip()},
            notes=str(item.get("notes") or "").strip(),
            references=[str(x).strip() for x in (item.get("references") or []) if str(x).strip()],
        )
    return cards


@lru_cache(maxsize=1)
def _build_landmark_aliases() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for lid, card in LANDMARK_CARDS.items():
        aliases[lid] = lid
        aliases[lid.upper()] = lid
        for alias in card.aliases:
            token = str(alias).strip()
            if token:
                aliases[token] = lid
                aliases[token.upper()] = lid
    return aliases


AIRWAY_TREE: Dict[str, AirwayInfo] = _build_airway_tree()
AIRWAY_ALIASES: Dict[str, str] = _build_airway_aliases()
LANDMARK_CARDS: Dict[str, LandmarkCard] = _build_landmark_cards()
LANDMARK_ALIASES: Dict[str, str] = _build_landmark_aliases()


def normalize_airway_code(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    token = raw.upper().replace("-", "_")
    token = " ".join(token.split())
    return AIRWAY_ALIASES.get(token, token)


def get_airway_info(value: Optional[str]) -> Optional[AirwayInfo]:
    code = normalize_airway_code(value)
    if not code:
        return None
    return AIRWAY_TREE.get(code)


def normalize_landmark_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    return LANDMARK_ALIASES.get(stripped, LANDMARK_ALIASES.get(stripped.upper(), stripped))


def get_landmark_card(value: Optional[str]) -> Optional[LandmarkCard]:
    canonical = normalize_landmark_id(value)
    if canonical is None:
        return None
    return LANDMARK_CARDS.get(canonical)


def has_landmark_card(value: Optional[str]) -> bool:
    return get_landmark_card(value) is not None


def list_landmark_cards() -> List[LandmarkCard]:
    return list(LANDMARK_CARDS.values())


def local_navigation_cue(landmark_id: Optional[str], airway: Optional[str]) -> str:
    card = get_landmark_card(landmark_id)
    if not card:
        return ""
    key = str(airway or "").strip().upper()
    if not key:
        return ""
    return str(card.local_navigation_cues.get(key) or "").strip()


def airway_path_to_root(value: Optional[str]) -> List[str]:
    info = get_airway_info(value)
    if not info:
        return []
    path = [info.code]
    seen = {info.code}
    parent = info.parent
    while parent and parent not in seen:
        path.append(parent)
        seen.add(parent)
        parent_info = AIRWAY_TREE.get(parent)
        parent = parent_info.parent if parent_info else ""
    return path


def nearest_shared_ancestor(current: Optional[str], target: Optional[str]) -> str:
    current_path = airway_path_to_root(current)
    target_path = airway_path_to_root(target)
    target_set = set(target_path)
    for code in current_path:
        if code in target_set:
            return code
    return ""


def siblings_for(value: Optional[str]) -> List[str]:
    info = get_airway_info(value)
    if not info or not info.parent:
        return []
    parent = AIRWAY_TREE.get(info.parent)
    if not parent:
        return []
    return [child for child in parent.children if child != info.code]


def classify_anatomical_relationship(current: Optional[str], target: Optional[str]) -> str:
    current_info = get_airway_info(current)
    target_info = get_airway_info(target)
    if not current_info or not target_info:
        return "unknown"
    shared = nearest_shared_ancestor(current_info.code, target_info.code)
    if current_info.code == target_info.code:
        return "same_airway"
    if target_info.code in airway_path_to_root(current_info.code):
        return "ancestor_target"
    if current_info.code in airway_path_to_root(target_info.code):
        return "descendant_target"
    if current_info.parent and current_info.parent == target_info.parent:
        return "sibling"
    if current_info.family and current_info.family == target_info.family:
        return "same_family"
    if shared == "CARINA":
        return "cross_main_bronchus"
    if current_info.side and target_info.side and current_info.side != target_info.side:
        return "cross_side"
    if shared in {"RMB", "BI", "LMB", "LEFT_UPPER_TRUNK", "RUL", "RML", "RLL", "LUL", "LINGULA", "LLL"}:
        return "regional_branch_change"
    return "related"


def teaching_fact_for(value: Optional[str]) -> str:
    info = get_airway_info(value)
    return info.teaching_fact if info else ""


def recognition_cues_for(value: Optional[str]) -> List[str]:
    info = get_airway_info(value)
    return list(info.recognition_cues) if info else []


def repair_hint_for(value: Optional[str]) -> str:
    info = get_airway_info(value)
    return info.repair_hint if info else ""


def quiz_prompt_for(value: Optional[str]) -> str:
    info = get_airway_info(value)
    return info.quiz_prompt if info else ""


def references_for(value: Optional[str]) -> List[str]:
    info = get_airway_info(value)
    return list(info.references) if info else []


__all__ = [
    "AirwayInfo",
    "LandmarkCard",
    "AIRWAY_TREE",
    "AIRWAY_ALIASES",
    "LANDMARK_CARDS",
    "LANDMARK_ALIASES",
    "normalize_airway_code",
    "get_airway_info",
    "normalize_landmark_id",
    "get_landmark_card",
    "has_landmark_card",
    "list_landmark_cards",
    "local_navigation_cue",
    "airway_path_to_root",
    "nearest_shared_ancestor",
    "siblings_for",
    "classify_anatomical_relationship",
    "teaching_fact_for",
    "recognition_cues_for",
    "repair_hint_for",
    "quiz_prompt_for",
    "references_for",
]
