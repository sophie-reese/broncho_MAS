# BronchoMAS

A modular bronchoscopy guidance framework with two complementary paths:

- **Runtime path** for short, real-time instructional coaching
- **Research path** for richer multi-agent experimentation, replay, and reporting

## What this project is

BronchoMAS is a Python project for bronchoscopy training guidance.

It was built to support a practical need: generate short, useful, medically guided coaching from structured bronchoscopy state, while also preserving a richer research line for experimentation and future academic work.

Instead of forcing one single architecture to do everything, this project keeps two parallel tracks:

- a **runtime track** for faster and more stable live guidance
- a **research track** for heavier orchestration, logging, and analysis

## Where to look first


- `bronchoscopy_guidance_system/src` 
  This is the main code area of the repository.

- `bronchoscopy_guidance_system/broncho_mas_demo.ipynb 
  This notebook is for quick testing, debugging, and demonstration.  
  After downloading the repository, these notebooks can be used as a simple entry point for running and inspecting the current prototype     without going through the full runtime integration path.

- `bronchoscopy_guidance_system/src/broncho_mas/runtime/`  
  Look here for the real-time guidance path. This is currently the more practical direction for live instructional support.

- `bronchoscopy_guidance_system/src/broncho_mas/sas/`
  This is the single-agent skill runtime path. SAS skills are now split into the `src/broncho_mas/sas/skills/` package by responsibility instead of living in one large module, and `src/broncho_mas/sas/manager.py` owns turn state, structured turn logging, and skill dispatch/arbitration.

- `bronchoscopy_guidance_system/src/broncho_mas/research/`  
  Look here for the richer research-oriented path, including multi-agent experimentation and reporting-related logic.

- `documentation/`  
  Contains project background, design rationale, and supporting notes.

---

## Why there are multiple tracks

Earlier experiments focused more heavily on a multi-agent pipeline. The content quality could be good, but the full chain was often too slow and too heavy for strict real-time use.

For bronchoscopy training, this matters. Guidance needs to be:

- short
- timely
- easy to speak through TTS
- stable under latency pressure
- grounded in structured upstream state

Because of that, the project evolved into a three-line design:

### 1. Runtime path
The runtime path is the practical live path.

It is designed to:

- consume structured state from upstream runtime/perception modules
- apply deterministic curriculum and navigation logic first
- generate short, human-readable coaching
- stay usable when latency matters
- degrade gracefully if the LLM is slow or unavailable

In this path, deterministic logic handles the parts that must stay stable, and the LLM is mainly used to verbalize the guidance more naturally.

### 2. MAS path
The MAS or research path is the richer experimental path.

It is designed to:

- support more agentic orchestration
- test prompting strategies
- generate logs, traces, and replayable outputs
- support reporting and debriefing
- preserve the broader multi-agent direction for academic exploration

This path is useful for experimentation and evaluation, but it is not the best choice for strict real-time constraints.

### 3. SAS path

To balance controllability and flexibility, the SAS path uses:

- one manager
- a small set of specialized skills
- deterministic helpers
- structured outputs
- optional LLM realization

This allows the system to remain modular and instructional, while still being stable enough for real-time bronchoscopy training.

---

## SAS design overview

The SAS path is the current single-agent skills design. It is selected with:

`BRONCHO_PIPELINE=sas`

SAS is organized around a manager that receives either a structured runtime payload or a legacy prompt, normalizes the state, builds an authoritative curriculum/navigation plan, dispatches skills, selects the most appropriate skill surface, applies shared post-processing, and logs a structured turn record.

Conceptually, the SAS design has these layers:

- **Upstream runtime environment**  
  Robot-connected simulation or runtime code provides perception, navigation, voice/question, and session context.

- **Payload construction and state normalization**  
  Runtime fields such as `current_airway`, `target_airway`, `reached_regions`, `event_packet`, `m_jointsVelRel`, safety/visibility flags, landmark state, and question text are normalized into a SAS state packet.

- **Knowledge and policy layer**  
  This layer contains the grounded material SAS should reason from: curriculum order, airway transition knowledge, bronchoscopy anatomy knowledge, landmark cards and aliases, skill policy markdown files, reporting constraints, and session memory such as taught landmarks or acknowledged waypoints.

- **SAS core**  
  `SASManager` orchestrates deterministic support logic, curriculum planning, event signals, directional hints, skill dispatch, arbitration, optional LLM realization, post-processing, and turn logging.

- **Skill layer**  
  Active SAS skills include `guidance_skill`, `support_skill`, `landmark_teaching_skill`, `qa_skill`, `statistics_skill`, and `reporting_skill`. Guidance, support, teaching, and QA can produce live utterances; statistics supports structured analytics; reporting produces end-of-session educator-facing reports.

- **Model/generation layer**  
  The model backend can be Hugging Face, OpenAI-compatible, LiteLLM, Transformers, Azure, or Bedrock depending on environment configuration. SAS keeps deterministic fallbacks so the system can still produce guidance when no model backend is available.

- **Output layer**  
  SAS returns a structured result dictionary containing `ui_text`, `utterance_full`, skill records, selected skill/frame information, `plan_json`, `statistics`, `statepacket`, `event_packet`, and logging/report artifacts when available. Actual TTS playback is handled outside SAS.

The knowledge and policy layer is intentionally described as its own layer because SAS is not only an LLM wrapper. The live behavior depends on grounded curriculum, anatomy, landmark, safety, reporting, and session-state constraints that feed the manager and skills.

---

## Current project status

This repository is a **working prototype in active development**.

At the current stage:

- the **runtime path** is the main direction for real-time integration
- the **research path** is preserved for richer experimentation and analysis
- the codebase reflects an ongoing effort to balance:
  - medical correctness
  - concise instructional output
  - latency constraints
  - future multi-agent research potential

This is not presented as a finished clinical system. It is a training-oriented guidance prototype.
