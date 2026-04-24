# SAS policies

This folder keeps the written skill specifications for the SAS line.

## Current skill specs
- guidance_skill.md
- landmark_teaching_skill.md
- qa_skill.md
- support_skill.md
- statistics_skill.md
- reporting_skill.md

Notes:
- `landmark_teaching_skill.md` defines the first-arrival landmark teaching overlay used during runtime guidance.
- `statistics_skill.md` is kept separate so runtime policy compilation can load reporting and statistics constraints independently.
- `curriculum_skill.md` and `directional_skill.md` are retained as design references for shared helper logic, not active runtime skills.
- These markdown files are policy/spec sources.
- The executable logic now lives in the split SAS skills package at `src/broncho_mas/sas/skills/`, not in a single `skills.py` file.
- `src/broncho_mas/sas/manager.py` is responsible for turn orchestration, including `SkillDispatcher`, `TurnLog`, `SessionMilestones`, and `TurnCounters`.
- The manager can compile selected policy text into LLM calls, but the Python package structure is the source of executable behavior.
