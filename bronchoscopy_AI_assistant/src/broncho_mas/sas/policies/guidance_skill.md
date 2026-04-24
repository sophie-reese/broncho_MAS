# Skill: Guidance

## Metadata
- Name: guidance_skill
- Purpose: Produce the main real-time bronchoscopy guidance utterance for the current step.
- Inputs: `state`, `plan`, `directional_hint`, optional `model`, optional `current_situation`, optional `previous_msgs`
- Outputs: `skill`, `active`, `priority`, `reason`, `data.frame`, `data.frame_mode`, `data.deterministic_utterance`, `data.realized`, `utterance`
- Activation:
  - This skill is the default live-guidance path.
  - `should_activate()` always returns `True`.

## Instructions
1. Build a deterministic guidance frame from:
   - current `state`
   - current `plan`
   - current `directional_hint`
2. Render that frame into a deterministic utterance.
3. Treat `plan.next_airway` as the main training target.
4. Treat waypoint anchors such as `RMB`, `LMB`, `BI`, `RUL`, `RML`, and `RLL` as route checkpoints, not final success states.
5. On true destination arrival, acknowledge the reached airway first and then give the next move.
6. On waypoint arrival, allow only a brief acknowledgment such as `You're at RMB.` and continue toward the training target without praise.
7. For local sibling moves, stay local and use short visual finding cues such as left, right, upper, lower, medial, lateral, forward, or back.
8. If a short local memory cue is available, it should stay secondary to the finding cue.
9. For route moves starting from the carina, name the clinically meaningful route checkpoints rather than saying only `Now move to <target>`.
   - Example for `RB6`: right main bronchus, bronchus intermedius, right lower lobe bronchus, then RB6.
   - Include the anatomical target label when available, such as `RB6 (right lower lobe superior segment)`.
10. For superior lower-lobe segments, preserve the location cue when it is part of the guidance frame.
   - `RB6` usually sits high before the basal fan.
   - `LB6` usually sits high before the left basal branches.
11. If the current airway is already a reached destination, do not tell the trainee to complete that same destination or its sibling pair again.
   - Example: after reaching `RB5`, do not say `complete RB4 and RB5 first`; guide onward toward `RB6`.
12. Only attempt model-based realization if all of the following are true:
   - a `model` is provided
   - the frame contains a non-empty `base_utterance`
   - the model is **not** marked as a fallback backend
   - the frame is **not** in `safety` mode
   - runtime LLM triggering is active through state / event context such as `need_llm`, `llm_trigger_flag`, `soft_prompt`, `llm_reason`, or matching event-packet fields
13. If realization succeeds, use the realized text as the final `utterance`.
14. Otherwise, keep the deterministic utterance.
15. Raise priority to `1.0` if safety risk is detected from the current state.
16. Return a structured `SkillResult` that records both the frame and whether realization happened.

## Constraints
- Do not invent a new plan. This skill speaks from the provided `plan` and `directional_hint`.
- Do not force realization. Deterministic output is valid and expected when realization is blocked.
- Do not override safety mode with model wording.
- This skill is always active; selection against QA should happen at the orchestration layer, now mediated by `SkillDispatcher`.
- Do not praise `RMB`, `LMB`, `BI`, `RUL`, `RML`, or `RLL` as if they were completed segment targets.
- Do not let a pull-back instruction bury a true destination arrival acknowledgment.
- Do not collapse local sibling moves into generic `back to the carina` language.
- Do not collapse a carina-to-target route into generic `Now move to <target>` wording when the route checkpoints are known.
- Do not repeat already completed middle-lobe targets when the current target has advanced to `RB6`.

## What this skill does not do
- It does not decide whether the student question should go to QA.
- It does not classify question type.
- It does not compute curriculum progress.
- It does not generate the directional hint itself.
- It does not own cross-skill arbitration; the manager/dispatcher decides whether QA or teaching outranks it on a given turn.

## Example 1: Normal live guidance without realization
### Scenario
- `state` indicates ordinary live navigation
- `plan` says the student is moving locally from `RB1` to `RB2`
- `directional_hint` supports that move
- `model=None`

### Expected behavior
- The skill builds a guidance frame.
- It renders the frame into deterministic text.
- Because no model is provided, it does **not** realize the frame.
- Final output uses the deterministic utterance.

### Illustrative result
```json
{
  "skill": "guidance_skill",
  "active": true,
  "priority": 0.8,
  "reason": "live guidance frame selected",
  "data": {
    "frame_mode": "guidance",
    "deterministic_utterance": "Back out just a touch. RB2 should come up on your right.",
    "realized": false
  },
  "utterance": "Back out just a touch. RB2 should come up on your right."
}
```

## Example 2: Guidance with realization
### Scenario
- Same as Example 1
- `model` is available
- frame contains a non-empty `base_utterance`
- frame is not in `safety` mode
- model is not a fallback backend
- runtime LLM triggering is active

### Expected behavior
- The skill first builds the deterministic frame.
- It then asks the realization layer to rewrite the frame into smoother language.
- If realization returns non-empty text, that realized text becomes the final `utterance`.

### Illustrative result
```json
{
  "skill": "guidance_skill",
  "active": true,
  "priority": 0.8,
  "reason": "live guidance realized over deterministic frame",
  "data": {
    "frame_mode": "guidance",
    "deterministic_utterance": "Back out just a touch. RB2 should come up on your right.",
    "realized": true
  },
  "utterance": "Back out a touch. RB2 should appear on your right."
}
```

## Example 3: Safety-risk case
### Scenario
- `skills.utterance_helpers.safety_risk(state)` returns `True`
- frame may still be built normally
- model-based realization may or may not happen depending on the frame and model

### Expected behavior
- The skill stays active.
- Priority is raised to `1.0`.
- If the frame is marked `safety`, realization is blocked and deterministic wording is kept.

### Illustrative result
```json
{
  "skill": "guidance_skill",
  "active": true,
  "priority": 1.0,
  "reason": "live guidance frame selected",
  "data": {
    "frame_mode": "guidance",
    "deterministic_utterance": "Stop advancing. Re-center before moving again.",
    "realized": false
  },
  "utterance": "Stop advancing. Re-center before moving again."
}
```

## Example 4: Carina-to-RB6 route
### Scenario
- `state.current_airway = "CARINA"`
- `plan.next_airway = "RB6"`
- `plan.route` includes right-sided lower-lobe checkpoints

### Expected behavior
- The skill names the route through the right main bronchus and bronchus intermedius.
- It includes the right lower-lobe superior-segment label when available.
- It does not only say `Now move to RB6`.

### Illustrative result
```json
{
  "skill": "guidance_skill",
  "active": true,
  "priority": 0.8,
  "reason": "live guidance frame selected",
  "data": {
    "frame_mode": "guidance",
    "deterministic_utterance": "Nice, we're at the carina. Follow the route through right main bronchus, then bronchus intermedius, then right lower lobe bronchus, then RB6 (right lower lobe superior segment).",
    "realized": false
  },
  "utterance": "Nice, we're at the carina. Follow the route through right main bronchus, then bronchus intermedius, then right lower lobe bronchus, then RB6 (right lower lobe superior segment)."
}
```

## Example 5: RB5 to RB6 without repeating middle-lobe pair
### Scenario
- `state.current_airway = "RB5"`
- `plan.next_airway = "RB6"`
- the trainee has already reached `RB4` and `RB5`

### Expected behavior
- The skill guides onward toward `RB6`.
- The guidance frame keeps the RB6 location cue.
- It does not tell the trainee to complete `RB4` and `RB5` again.

### Illustrative result
```json
{
  "skill": "guidance_skill",
  "active": true,
  "priority": 0.8,
  "reason": "live guidance frame selected",
  "data": {
    "frame_mode": "guidance",
    "deterministic_utterance": "Nice, retrace slightly to the bronchus intermedius. Then move toward RB6.",
    "frame": {
      "next_step": "Retrace slightly to the bronchus intermedius. Then move toward RB6. RB6 usually sits high before the basal fan."
    },
    "realized": false
  },
  "utterance": "Nice, retrace slightly to the bronchus intermedius. Then move toward RB6."
}
```
