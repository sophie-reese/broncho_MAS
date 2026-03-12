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

- `bronchoscopy_guidance_system/src/broncho_mas/research/`  
  Look here for the richer research-oriented path, including multi-agent experimentation and reporting-related logic.

- `documentation/`  
  Contains project background, design rationale, and supporting notes.

---

## Why there are two tracks

Earlier experiments focused more heavily on a multi-agent pipeline. The content quality could be good, but the full chain was often too slow and too heavy for strict real-time use.

For bronchoscopy training, this matters. Guidance needs to be:

- short
- timely
- easy to speak through TTS
- stable under latency pressure
- grounded in structured upstream state

Because of that, the project evolved into a two-line design:

### 1. Runtime path
The runtime path is the practical live path.

It is designed to:

- consume structured state from upstream runtime/perception modules
- apply deterministic curriculum and navigation logic first
- generate short, human-readable coaching
- stay usable when latency matters
- degrade gracefully if the LLM is slow or unavailable

In this path, deterministic logic handles the parts that must stay stable, and the LLM is mainly used to verbalize the guidance more naturally.

### 2. Research path
The research path is the richer experimental path.

It is designed to:

- support more agentic orchestration
- test prompting strategies
- generate logs, traces, and replayable outputs
- support reporting and debriefing
- preserve the broader multi-agent direction for academic exploration

This path is useful for experimentation and evaluation, but it is not the best choice for strict real-time constraints.

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
