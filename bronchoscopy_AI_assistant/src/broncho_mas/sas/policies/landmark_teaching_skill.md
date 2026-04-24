# Skill: Landmark Teaching

## Metadata
- Name: `landmark_teaching_skill`
- Purpose: deliver one short landmark teaching overlay when SAS reaches a supported bronchoscopy landmark for the first meaningful time in the session
- Inputs: `state`, `plan`, optional `guidance_text`
- Outputs: `skill`, `active`, `priority`, `reason`, `data`, `utterance`
- Scope: landmark recognition, sequence memory, and short local finding cues only

## Core Policy
This skill is a first-arrival teaching cue.
It is not a dwell-based mini-lecture.

The correct mental model is:
- resolve the landmark
- confirm it is supported
- check that it is safe and appropriate to teach now
- fire once on first meaningful arrival
- give one short landmark line
- optionally add one short local finding cue for the immediate target
- hand the floor back to the main guidance

## Landmark Resolution
Resolve a canonical landmark id from grounded state / plan fields in this order:
1. `state.validated_landmark`
2. `state.landmark_id`
3. `state.current_landmark`
4. `state.teaching_landmark`
5. `plan.anchor_landmark`
6. `plan.landmark`
7. `state.current_airway`

Plan-grounded landmark resolution is valid and should be accepted.

## Supported Landmarks
- `L1_CARINA`: carina / tracheal bifurcation reset anchor
- `L2_RUL`: right upper lobe landmark
- `L3_RIGHT_MIDDLE_LOWER`: right middle and lower lobes landmark
- `L4_LEFT_MAIN`: left lung landmark

## Activation
Activate only if all of the following are true:
1. A canonical landmark id is resolved.
2. The resolved landmark is one of the supported landmark cards.
3. No safety-risk condition is active.
4. `state.student_question` is empty.
5. The landmark has not already been taught in the current session history.
6. There is a first-arrival cue.
7. The same landmark was not just taught again within the recent cooldown window.

## First-Arrival Cues
Fire when one of these is true:
- `state.first_time_landmark`
- `state.just_reached`

This skill should be treated as arrival-triggered, not dwell-triggered.

## Suppression Conditions
Suppress activation if any of these apply:
- safety-critical correction is active
- recentering / wall-risk / drift handling is active
- the trainee is actively asking a question
- the same landmark has already been recorded in teaching history
- the same landmark was just taught again within cooldown
- landmark resolution fails or is unsupported

## Data Contract
When active, `data` should include:
- `landmark_id`
- `display_name`
- `recognition_cues`
- `memory_hook_type`
- `memory_hook_core`
- `memory_hook_rhythm`
- `action_anchor`
- `common_confusions`
- `first_arrival`
- `followup_target`
- `local_navigation_tip`

## Utterance Policy
- Keep the utterance to one short landmark line, or at most two short spoken sentences after merge.
- Prefer the landmark card's `default_teaching_line` on first teaching.
- If a `local_navigation_tip` exists for the immediate target, it may be added as a second short line.
- Prefer visual-first local cues such as left, right, upper, lower, medial, lateral, forward, or back.
- Keep the local cue actionable first, then add a brief memory cue if useful.
- Do not turn this into a long anatomy lesson.
- Do not enumerate full segment inventories unless the landmark card explicitly compresses them into a short memory aid.
- Keep wording speakable in live coaching.

## Integration Rules
- This skill complements `guidance_skill`; it does not replace the main navigation plan.
- If `guidance_text` is provided and the skill is active, merge in this order:
  1. arrival confirmation if present
  2. next-step guidance
  3. teaching line
  4. local cue, especially on first appearance
- When this skill fires, the caller should record the taught landmark so it is not re-taught repeatedly.
- If landmark teaching is active in the same turn, repair support should not take foreground.
- Urgent safety correction still outranks landmark teaching.

## Constraints
- Supported landmarks in v1 are only the carina, right upper lobe landmark, right middle/lower landmark, and left main landmark.
- Do not invent unsupported landmark ids.
- Do not activate from vague anatomy mentions that fail canonical resolution.
- Do not require a dwell frame after arrival.
- Do not repeat the same teaching card once the landmark has been recorded in session teaching history.
- Do not allow the first local finding cue to be dropped by later compression or cleanup.
- Use the landmark card as the source of truth for recognition cues, memory hooks, and local cues.

## What this skill does not do
- It does not choose the next navigation action.
- It does not answer the trainee's question.
- It does not replace the main guidance utterance.
- It does not perform general knowledge QA.
- It does not fire repeatedly just because the landmark remains in view.

## Example 1: Carina first arrival from plan anchor
### Scenario
- `state.validated_landmark` is empty
- `plan.anchor_landmark = "L1_CARINA"`
- `state.just_reached = true`
- carina has not been taught yet

### Expected behavior
- Landmark resolution succeeds from `plan.anchor_landmark`.
- The skill activates.
- A short reset-anchor teaching cue is produced.

### Illustrative result
```json
{
  "skill": "landmark_teaching_skill",
  "active": true,
  "priority": 0.62,
  "reason": "first arrival at supported landmark",
  "data": {
    "landmark_id": "L1_CARINA",
    "display_name": "carina",
    "first_arrival": true
  },
  "utterance": "Use the carina as your reset point when the training sequence needs a fresh start."
}
```

## Example 2: RUL landmark with local cue
### Scenario
- `state.current_landmark = "RUL"`
- `state.just_reached = true`
- `plan.next_airway = "RB2"`
- landmark not yet taught

### Expected behavior
- The skill activates.
- The teaching line highlights the training order.
- A short local cue may be added for the immediate next branch.

### Illustrative result
```json
{
  "skill": "landmark_teaching_skill",
  "active": true,
  "priority": 0.62,
  "reason": "first arrival at supported landmark",
  "data": {
    "landmark_id": "L2_RUL",
    "display_name": "right upper lobe landmark",
    "first_arrival": true,
    "followup_target": "RB2",
    "local_navigation_tip": "In this upper-lobe view, RB2 usually comes up on your right. Memory cue: the back branch."
  },
  "utterance": "Right upper lobe stays local: inspect RB1, then RB2, then RB3. Local cue: RB2 usually comes up on your right."
}
```

## Example 3: Teaching suppressed because already taught
### Scenario
- `state.current_landmark = "L1_CARINA"`
- carina already exists in `taught_landmarks`

### Expected behavior
- The skill stays inactive.
- It returns an empty teaching utterance instead of repeating the lesson.

### Illustrative result
```json
{
  "skill": "landmark_teaching_skill",
  "active": false,
  "priority": 0.0,
  "reason": "landmark already taught in current session",
  "data": {
    "landmark_id": "L1_CARINA"
  },
  "utterance": ""
}
```

## Example 4: Teaching suppressed during safety correction
### Scenario
- `state.just_reached = true`
- `state.wall_contact_risk = true`
- a supported landmark is resolved

### Expected behavior
- The skill does not fire.
- Safety correction takes precedence.

### Illustrative result
```json
{
  "skill": "landmark_teaching_skill",
  "active": false,
  "priority": 0.0,
  "reason": "suppressed by safety-risk condition",
  "utterance": ""
}
```
