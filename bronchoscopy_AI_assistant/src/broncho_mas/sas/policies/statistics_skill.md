# Skill: Statistics

## Metadata
- Name: `statistics_skill`
- Purpose: produce a compact grounded analytics payload for logging, monitoring, and downstream report generation
- Inputs: `current_situation`, `current_airway`, `next_airway`, `plan`
- Outputs: `skill`, `active`, `priority`, `reason`, `data.trend`, `data.likely_issue`, `data.coach_focus_next`, `data.teaching_point`, `data.notes`, `utterance`
- Activation:
  - This skill is the default analytics path for each SAS step.

## Instructions
1. Read `plan.mode`, `current_airway`, `next_airway`, and `current_situation`.
2. If `plan.mode == "backtrack"`, return a backtrack-oriented payload:
   - `likely_issue = "lost orientation or unsafe advance"`
   - `coach_focus_next = "withdraw to the carina and re-center"`
3. If `plan.mode` is `reorient` or `locate`, or if `current_situation` indicates the target is not visible, return a reorientation payload:
   - `likely_issue = "target lumen not yet visualized"`
   - `coach_focus_next = "keep centered and identify <target>"`
4. Otherwise, return a normal-navigation payload:
   - `likely_issue = "normal navigation"`
   - `coach_focus_next = "advance with the lumen centered"`
5. Include:
   - `trend`
   - `likely_issue`
   - `coach_focus_next`
   - `teaching_point`
   - `notes`
6. Keep `utterance` empty. This skill returns structured analytics, not UI wording.

## Constraints
- Use only grounded facts from the supplied state text and plan.
- Do not invent measurements, scores, or unseen anatomy.
- Keep the payload compact and structured.
- Do not turn this skill into a prose report writer.
- Do not produce the final educator-facing report here.

## What this skill does not do
- It does not arbitrate between guidance and QA.
- It does not write the end-of-session report text.
- It does not emit spoken coaching as its primary product.

## Example 1: Backtrack case
### Scenario
- `current_airway = "RB2"`
- `next_airway = "RB3"`
- `plan.mode = "backtrack"`

### Illustrative result
```json
{
  "skill": "statistics_skill",
  "active": true,
  "priority": 0.4,
  "reason": "session statistics summary generated",
  "data": {
    "trend": "stable",
    "likely_issue": "lost orientation or unsafe advance",
    "coach_focus_next": "withdraw to the carina and re-center",
    "teaching_point": "Reset orientation before advancing again.",
    "notes": "current=RB2; target=RB3; mode=backtrack."
  },
  "utterance": ""
}
```

## Example 2: Reorientation case
### Scenario
- `current_airway = "CARINA"`
- `next_airway = "RB1"`
- `plan.mode = "reorient"`

### Illustrative result
```json
{
  "skill": "statistics_skill",
  "active": true,
  "priority": 0.4,
  "reason": "session statistics summary generated",
  "data": {
    "trend": "stable",
    "likely_issue": "target lumen not yet visualized",
    "coach_focus_next": "keep centered and identify RB1",
    "teaching_point": "Use the landmark cue: the carina with both main bronchi in view.",
    "notes": "current=CARINA; target=RB1; mode=reorient."
  },
  "utterance": ""
}
```

## Example 3: Ordinary progression
### Scenario
- `current_airway = "RB2"`
- `next_airway = "RB3"`
- `plan.mode = "advance"`

### Illustrative result
```json
{
  "skill": "statistics_skill",
  "active": true,
  "priority": 0.4,
  "reason": "session statistics summary generated",
  "data": {
    "trend": "stable",
    "likely_issue": "normal navigation",
    "coach_focus_next": "advance with the lumen centered",
    "teaching_point": "Confirm the landmark cue before committing forward.",
    "notes": "current=RB2; target=RB3; mode=advance."
  },
  "utterance": ""
}
```
