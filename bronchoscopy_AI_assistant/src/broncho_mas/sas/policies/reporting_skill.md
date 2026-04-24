# Skill: Reporting

## Metadata
- Name: `reporting_skill`
- Purpose: generate a grounded end-of-session bronchoscopy training report from curriculum and session facts
- Inputs: `allowed_reached`, `visit_order`, `curriculum_progress`, `session_metrics`, `sp_score`, optional `teach_line`, optional `report_llm`, optional `use_llm`
- Outputs: `skill`, `active`, `priority`, `reason`, `data.report_text`, `data.required_structure_ok`, `data.report_mode`, `data.report_facts`, `utterance`
- Activation:
  - Use only for end-of-session reporting or an explicit reporting stage.

## Instructions
1. Build a grounded facts packet from:
   - reached segments
   - curriculum progress
   - session metrics
   - structured progress score
   - optional teaching line
2. If LLM report writing is enabled and available, ask the report writer to render the report.
3. If LLM report writing is disabled, fails, or returns a report missing required headers, fall back to the deterministic template renderer.
4. Return the report in the standard `SkillResult` envelope.
5. Keep `utterance` identical to `report_text`.

## Constraints
- Use only provided facts.
- Do not invent missing findings, metrics, or airway segments.
- The final report must contain all four headers exactly:
  - `Clinical performance note`
  - `Teaching feedback note`
  - `Curriculum coverage`
  - `Session metrics`
- If `duration_seconds` is missing or invalid, the report must say procedure time was not recorded.
- If `teach_line` is absent, use the built-in default teaching focus line.

## LLM behavior
- `use_llm=True` forces an LLM attempt.
- `use_llm=False` forces deterministic template rendering.
- If `use_llm` is omitted, the code uses:
  - `BRONCHO_REPORT_USE_LLM`
  - and the presence of `OPENAI_API_KEY`
- If the LLM path fails for any reason, the skill falls back to `template_fallback`.

## What this skill does not do
- It does not produce live spoken coaching.
- It does not arbitrate between QA and guidance.
- It does not replace `statistics_skill`; reporting consumes grounded facts but produces the human-readable end-of-session report.

## Example 1: Template fallback
### Scenario
- `use_llm = false`
- `allowed_reached = ["RB1", "RB2", "RB3"]`
- `visit_order` includes 18 curriculum targets
- `curriculum_progress.next_airway = "RB4"`
- `session_metrics.duration_seconds = 420`
- `session_metrics.backtrack_ratio = 0.18`
- `sp_score = 0.82`

### Illustrative result
```json
{
  "skill": "reporting_skill",
  "active": true,
  "priority": 0.3,
  "reason": "end-of-session report generated (template_fallback)",
  "data": {
    "report_text": "Clinical performance note\n- Diagnostic completeness (DC): 3/18 segments (17%).\n- Structured progress (SP): 0.82 (ordered progression ratio).\n- Procedure time (PT): 420 seconds.\nTeaching feedback note\n- Overall teaching focus: continue building a systematic segment-by-segment bronchoscopy technique, with attention to airway orientation, centered scope handling, and complete, atraumatic examination.\nCurriculum coverage\n- Segments visualized: RB1, RB2, RB3.\n- Segments not yet visualized: RB4, RB5, RB6, RB7, RB8, RB9, RB10, LB1+2, LB3, LB4 ...\n- Next target segment: RB4.\nSession metrics\n- Backtrack ratio: 0.18.\n- Student questions: 0.\n",
    "required_structure_ok": true,
    "report_mode": "template_fallback",
    "report_facts": {"...": "..."}
  },
  "utterance": "Clinical performance note\n..."
}
```

## Example 2: LLM success
### Scenario
- `use_llm = true`
- a working `report_llm` is available
- the returned report contains all required headers

### Expected behavior
- The skill uses the LLM output.
- `report_mode = "llm"`.
- `required_structure_ok = true`.

## Helper wrapper
The convenience wrapper

```python
reporting_skill(**kwargs) -> Dict[str, Any]
```

returns only the payload layer:

```json
{
  "report_text": "...",
  "required_structure_ok": true,
  "report_mode": "llm",
  "report_facts": {"...": "..."}
}
```
