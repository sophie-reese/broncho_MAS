# Skill: Support

## Metadata
- Name: `support_skill`
- Purpose: add one short supportive or recovery layer during live bronchoscopy coaching without inventing navigation content
- Inputs: `state`, `plan`, optional `previous_msgs`
- Outputs: `skill`, `active`, `priority`, `reason`, `data.support_mode`, `utterance`
- Activation principle:
  - Activate only when there is a grounded support need in the current SAS state.
  - Do not activate just to make the system sound warmer.
  - Student panic/lost-orientation recovery is different from ordinary QA and should be handled here.
  - Arrival acknowledgment and repair support are not the same thing and must not be mixed.

## Core Policy
This skill has seven meaningful modes:
- `distress_recovery`
- `lost_orientation`
- `stabilize`
- `arrival_feedback`
- `support_after_repeat_reset`
- `support_repeated_probing`
- `encourage_progress`

If none applies, return `support_mode="none"` and an empty `utterance`.

## Instructions
1. Read the current normalized `state`, `plan`, and recent `previous_msgs`.
2. Check `state.student_question` first for emotional/orientation recovery language.
   - If the trainee says they are panicking, scared, overwhelmed, or cannot continue, use `distress_recovery`.
   - If the trainee says they are lost, confused, unsure where they are, or asks for help because orientation is lost, use `lost_orientation`.
3. If the trainee is actively asking a normal question and it is not distress/orientation recovery language, suppress support so QA can handle the turn.
4. If there is immediate stabilization risk, prefer `stabilize`.
   Grounded cues include:
   - `wall_contact_risk`
   - `need_recenter`
   - `drift_detected`
5. If there is a fresh arrival event at a true curriculum destination, `arrival_feedback` may be used.
   Grounded cues include:
   - `just_reached=True`
   - the current airway is a curriculum airway
   - the arrival is not just an intermediate waypoint
6. In a destination-arrival event, use only a light acknowledgment such as:
   - `"Good."`
7. Do not praise intermediate waypoints such as `RMB`, `LMB`, `BI`, `RUL`, `RML`, or `RLL`.
   - if those are acknowledged at all, that belongs to the main guidance path, not support
8. Only consider repair-style support when there is no fresh arrival event.
9. If repeated reset / backtracking behavior is grounded, use `support_after_repeat_reset`.
   Grounded cues include:
   - `backtracking=True`
   - recent reset language already occurred multiple times in `previous_msgs`
   - no recent support utterance has just fired
10. If repeated probing without progress is grounded, use `support_repeated_probing`.
    Grounded cues include:
    - the target is still not visible or the runtime reason explicitly says the target is not visible / more time than usual was spent
    - recent navigation history shows repeated turning or repeated "toward / move to / go to" behavior
    - no recent support utterance has just fired
11. If the trainee is centered and stable but the target is not yet visible, use `encourage_progress`.
12. Otherwise return no support.

## Current Mode Meanings
### `distress_recovery`
Use when the trainee explicitly expresses panic, fear, stress, or overwhelm.
Preferred utterance:
- `Pause, hold still, and re-center the lumen. You are okay.`

This mode may activate even though `student_question` is non-empty, because emotional recovery should not be treated like ordinary QA.

### `lost_orientation`
Use when the trainee explicitly says they are lost, confused, or unsure where they are.
Preferred utterance:
- `Pause, hold still, and return to the carina to reset. Re-center the lumen first.`

### `stabilize`
Use when the system needs immediate recentering or stabilization.
Preferred utterance:
- `Easy. Re-center first.`

### `arrival_feedback`
Use only when a true destination has been reached.
Preferred utterance:
- `Good.`

This is intentionally minimal. The main arrival confirmation and next-step wording should come from the main guidance path, not from repair support.

### `support_after_repeat_reset`
Use only when the trainee appears to be repeating a reset / backtrack cycle without progress.
Preferred utterance:
- `Re-center and try again from the carina.`

### `support_repeated_probing`
Use only when the trainee appears to be repeatedly searching for the same target without progress.
Preferred utterance:
- `Re-center first, then try again toward <target>.`
- or `Re-center first, then try again.` when no grounded target label is available

### `encourage_progress`
Use when the trainee is stable and centered but not yet at the target.
Preferred utterance:
- `Nice and steady. Keep going.`

## Constraints
- Keep the utterance extremely short.
- Do not invent anatomy, landmarks, targets, or procedural steps.
- Do not replace the main guidance.
- Do not suppress `distress_recovery` or `lost_orientation` just because `student_question` is non-empty.
- Do not use repair support during a fresh arrival event.
- Do not stack multiple support utterances in the same turn.
- Do not fire repeated support if recent support has already been given.
- Use only grounded cues already present in `state`, `plan`, or `previous_msgs`.

## Integration Rules
- This skill is a support layer, not the main instructor.
- On true destination arrival, `arrival_feedback` may coexist with main guidance.
- On waypoint arrival, support should usually stay off.
- On `just_reached`, repair support should stay off.
- If landmark teaching is active in the same turn, repair support should not take foreground.
- If guidance wins the turn, `encourage_progress` may be suppressed by the manager to avoid generic support text replacing anatomical guidance.
- Urgent safety correction still outranks landmark teaching.

## What this skill does not do
- It does not choose the next navigation target.
- It does not answer ordinary student questions.
- It does not provide the landmark teaching line.
- It does not generate directional hints.
- It does not decide the full arrival wording such as `You're at RB2...`; that belongs to the main guidance path.
- It does not praise waypoints.

## Example 1: Fresh destination arrival
### Scenario
- `state.just_reached = true`
- `state.current_airway = "RB2"`
- `plan.next_airway = "RB3"`

### Expected behavior
- `arrival_feedback` activates.
- The utterance is only a light acknowledgment.
- Repair support does not fire.

### Illustrative result
```json
{
  "skill": "support_skill",
  "active": true,
  "priority": 0.52,
  "reason": "destination arrival detected; brief acknowledgement selected",
  "data": {
    "support_mode": "arrival_feedback"
  },
  "utterance": "Good, you've reached RB2."
}
```

## Example 2: Waypoint arrival
### Scenario
- `state.just_reached = true`
- `state.current_airway = "RMB"`
- `state.training_target = "RB7"`

### Expected behavior
- Support stays inactive.
- Waypoint acknowledgment is left to the main guidance path.

### Illustrative result
```json
{
  "skill": "support_skill",
  "active": false,
  "priority": 0.0,
  "reason": "waypoint arrival handled by main guidance without praise",
  "data": {
    "support_mode": "none"
  },
  "utterance": ""
}
```

## Example 3: Immediate stabilization
### Scenario
- `state.wall_contact_risk = true`
- `plan.next_airway = "RB1"`

### Expected behavior
- `stabilize` activates.
- Safety takes priority over praise.

### Illustrative result
```json
{
  "skill": "support_skill",
  "active": true,
  "priority": 1.0,
  "reason": "runtime safety stabilization selected",
  "data": {
    "support_mode": "stabilize"
  },
  "utterance": "Easy. Re-center first."
}
```

## Example 4: Student panic
### Scenario
- `state.student_question = "I'm panicking"`
- current view is otherwise stable

### Expected behavior
- `distress_recovery` activates.
- The utterance gives one immediate recovery action before any navigation detail.

### Illustrative result
```json
{
  "skill": "support_skill",
  "active": true,
  "priority": 1.08,
  "reason": "student expressed distress or lost orientation",
  "data": {
    "support_mode": "distress_recovery"
  },
  "utterance": "Pause, hold still, and re-center the lumen. You are okay."
}
```

## Example 5: Student lost
### Scenario
- `state.student_question = "I'm lost"`
- current target is `RB6`

### Expected behavior
- `lost_orientation` activates.
- The utterance uses the carina as the reset anchor.

### Illustrative result
```json
{
  "skill": "support_skill",
  "active": true,
  "priority": 1.08,
  "reason": "student expressed distress or lost orientation",
  "data": {
    "support_mode": "lost_orientation"
  },
  "utterance": "Pause, hold still, and return to the carina to reset. Re-center the lumen first."
}
```
