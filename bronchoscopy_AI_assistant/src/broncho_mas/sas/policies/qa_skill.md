# Skill: QA

## Metadata
- Name: qa_skill
- Purpose: Handle student question turns by routing the question, deciding whether QA is allowed, building a QA frame, and optionally realizing the final answer.
- Inputs: `state`, `plan`, `fallback_guidance`, optional `qa_allowed`, optional `visit_order`, optional `model`, optional `current_situation`, optional `previous_msgs`
- Outputs: `skill`, `active`, `priority`, `reason`, `data.question_mode`, `data.frame`, `data.frame_mode`, `data.deterministic_utterance`, `data.realized`, `data.qa_allowed`, `utterance`
- Activation:
  - QA requires a non-empty `student_question`.
  - QA requires `qa_allowed=True`.
  - QA requires the routed `question_mode` to be one of the coverable modes:
    - `teaching_relevant`
    - `observation_relevant`
    - `visual_relevant`
    - `off_task_social`

## Instructions
1. Read `state.student_question`.
2. Route the question into a `question_mode` using the internal routing logic.
3. Activate QA only if all three conditions hold:
   - `student_question` is non-empty
   - `qa_allowed` is `True`
   - `question_mode` is one of the coverable modes
4. If QA is active:
   - build a QA frame using `state`, `plan`, `question_mode`, `fallback_guidance`, and the active `visit_order`
   - render a deterministic utterance from that frame
5. If `question_mode="off_task_social"`, answer with a brief redirect and immediately return to the bronchoscopy task.
   - Do not silently ignore the student.
   - Do not answer the off-task topic.
   - Preserve the current navigation target when available, for example: `Keep following the current route toward RB6.`
6. Panic/lost-orientation language should be handled by `support_skill`, not by ordinary QA.
7. Only attempt model-based realization if all of the following are true:
   - QA is active
   - a `model` is provided
   - the frame contains a non-empty `base_utterance`
   - the model is **not** marked as a fallback backend
   - the frame is **not** in `safety` mode
8. If realization succeeds, use the realized text as the final `utterance`.
9. If QA is inactive, return:
   - `active=False`
   - `priority=0.0`
   - empty `utterance`
   - a reason explaining suppression

## Constraints
- Do not activate QA when there is no actual student question.
- Do not activate QA when `qa_allowed=False`, even if the question is present.
- Do not fabricate QA content when the question routes to `none`.
- Do not force realization. Deterministic QA text is valid when realization is blocked.
- For off-task social questions, redirect briefly and resume the active bronchoscopy target instead of staying silent.
- Do not answer off-task subject matter such as sports, weather, movies, or casual chat topics.
- Do not treat panic or lost-orientation statements as QA; those should be recovered by support.
- When inactive, return an empty `utterance` rather than pretending QA happened.
- Do not bind QA to a stale import-time curriculum; it should use the active visit order supplied by the manager.

## What this skill does not do
- It does not serve as the default live-guidance path.
- It does not compute directional hints.
- It does not compute curriculum progress.
- It does not decide global skill arbitration by itself; `SkillDispatcher` decides whether QA overrides guidance.

## Example 1: Student question activates QA
### Scenario
- `state.student_question = "How do I tell the carina from the main bronchi?"`
- `qa_allowed = true`
- routing returns `question_mode = "teaching_relevant"`
- `model=None`

### Expected behavior
- QA activates.
- The skill builds a QA frame.
- It renders deterministic QA text.
- Because no model is provided, realization does not happen.
- Final output uses deterministic QA wording.

### Illustrative result
```json
{
  "skill": "qa_skill",
  "active": true,
  "priority": 0.95,
  "reason": "student question routed as teaching_relevant; qa selected",
  "data": {
    "question_mode": "teaching_relevant",
    "frame_mode": "qa",
    "deterministic_utterance": "The carina is the bifurcation point where the airway divides into the two main bronchi.",
    "realized": false,
    "qa_allowed": true
  },
  "utterance": "The carina is the bifurcation point where the airway divides into the two main bronchi."
}
```

## Example 2: QA activates and is realized
### Scenario
- Same as Example 1
- a usable `model` is provided
- frame contains a non-empty `base_utterance`
- frame is not in `safety` mode

### Expected behavior
- QA activates.
- The skill builds the QA frame.
- It uses the realization layer to produce smoother final wording.
- Final `utterance` is the realized text.

### Illustrative result
```json
{
  "skill": "qa_skill",
  "active": true,
  "priority": 0.95,
  "reason": "student question routed as teaching_relevant; qa realized",
  "data": {
    "question_mode": "teaching_relevant",
    "frame_mode": "qa",
    "deterministic_utterance": "The carina is the bifurcation point where the airway divides into the two main bronchi.",
    "realized": true,
    "qa_allowed": true
  },
  "utterance": "Look for the point where the airway splits into the right and left main bronchi; that split is the carina."
}
```

## Example 3: Question present but QA suppressed
### Scenario
- `state.student_question = "What should I do now?"`
- either:
  - `qa_allowed = false`, or
  - routing returns `question_mode = "other"` or `question_mode = "fragment_unclear"`

### Expected behavior
- QA does not activate.
- The skill returns an empty `utterance`.
- The reason explicitly says QA was suppressed.

### Illustrative result
```json
{
  "skill": "qa_skill",
  "active": false,
  "priority": 0.0,
  "reason": "student question routed as other; suppressed for safety or relevance",
  "data": {
    "question_mode": "other",
    "frame_mode": "guidance",
    "deterministic_utterance": "",
    "realized": false,
    "qa_allowed": true
  },
  "utterance": ""
}
```

## Example 4: Off-task social question
### Scenario
- `state.student_question = "Can you talk about hockey?"`
- `plan.next_airway = "RB6"`
- `qa_allowed = true`

### Expected behavior
- QA activates with `question_mode = "off_task_social"`.
- The response briefly redirects away from the off-task topic.
- The response preserves the active bronchoscopy target.

### Illustrative result
```json
{
  "skill": "qa_skill",
  "active": true,
  "priority": 0.95,
  "reason": "student question routed as off_task_social; qa selected",
  "data": {
    "question_mode": "off_task_social",
    "frame_mode": "qa",
    "deterministic_utterance": "Let's save that for later and stay with the bronchoscopy for now. Keep following the current route toward RB6.",
    "realized": false,
    "qa_allowed": true
  },
  "utterance": "Let's save that for later and stay with the bronchoscopy for now. Keep following the current route toward RB6."
}
```
