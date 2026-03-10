# BronchoMAS
A real-time and research-oriented multi-agent guidance framework for bronchoscopy training

## Overview

BronchoMAS is a modular Python framework for **conversational bronchoscopy guidance**. It is designed to support two complementary goals:

1. **Real-time coaching for bronchoscopy training**
   - low-latency, safety-aware, short instructional guidance
   - suitable for simulator or live training integration
   - deterministic curriculum logic with optional LLM verbalization

2. **Research-oriented multi-agent experimentation**
   - richer prompting and orchestration
   - statistics, trace logging, and session reporting
   - useful for replay analysis, debriefing, and future academic extension

The project was built around a practical constraint:  
**support real-time guidance without requiring invasive edits to co-worker runtime code**, while still preserving a more ambitious multi-agent research path for experimentation and evaluation. This follows the earlier design direction of adding a conversational coaching layer on top of an existing perception/runtime pipeline rather than replacing it. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

---

## Why this project exists

Bronchoscopy training benefits from guidance that is:

- **context-aware**: based on current landmark/procedure state
- **safe**: able to prioritize warnings and navigation recovery
- **concise**: 1–2 action bedside-style coaching
- **responsive**: fast enough to fit real-time training flow
- **traceable**: capable of producing logs, metrics, and debriefs

Earlier design work for this project defined a real-time conversational coach with:
- a central orchestrator,
- procedure-state reasoning,
- safety guard logic,
- tutoring/pedagogy guidance,
- event-driven message flow,
- and post-session evaluation goals. :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}

This repository turns that direction into a cleaner implementation path.

---

## Core design idea

The system is intentionally split into **two tracks**.

### 1. Runtime track
The runtime track is the **fast path**.

Its job is to:
- consume structured state
- apply deterministic curriculum/navigation logic first
- produce short, safe, human-readable coaching
- stay robust under latency pressure
- degrade gracefully when the LLM is slow or unavailable

This design follows the earlier real-time implementation blueprint in which curriculum is generated deterministically first, and the LLM is used mainly to verbalize or polish the plan rather than decide navigation itself. That is important for safety, stability, and latency. :contentReference[oaicite:7]{index=7}

### 2. Research track
The research track is the **rich path**.

Its job is to:
- support heavier multi-agent workflows
- produce richer outputs for analysis/reporting
- test prompt variants and orchestration strategies
- enable replay, metrics, and debrief generation

This track is not the one to trust for strict real-time guarantees. It is there because real educational systems often need both:
- a reliable live path
- and a richer offline/research path

---

## Architecture

At a high level, BronchoMAS follows a **coordinator + specialist agents** pattern.

### Shared conceptual roles

- **Session/Orchestration layer**  
  Maintains session state, routes events, and decides what should speak and when.

- **Procedure-state reasoning**  
  Converts perception/runtime signals into meaningful training state.

- **Curriculum planning**  
  Tracks visited regions and determines the next intended airway/landmark path.

- **Safety guard**  
  Prioritizes urgent interventions and prevents unsafe or inappropriate output.

- **Tutoring / instruction layer**  
  Turns state into concise learner-facing guidance.

- **Debrief / reporting layer**  
  Summarizes session performance after the interaction. :contentReference[oaicite:8]{index=8}

### Real-time policy principles

The real-time guidance design follows a few strict rules:

- safety first
- max 1–2 actions per utterance
- include a verification cue when possible
- interrupt only when necessary
- keep conversation aligned with procedural flow
- separate deterministic navigation/planning from natural-language rendering

These constraints come directly from the earlier architecture notes and are still central to the current project direction. :contentReference[oaicite:9]{index=9}

---

## Repository philosophy

This repository is not trying to pretend that one single agent does everything perfectly.

Instead, it reflects a more realistic engineering position:

- **deterministic logic** handles what must be stable
- **LLMs** handle language rendering, explanation, and richer reflection
- **real-time runtime** handles responsiveness
- **research pipeline** handles experimentation and reporting

That separation is deliberate.

---

## Main components

The exact filenames may evolve, but the repository is organized around three functional areas:

### `runtime/`
Fast, narrow, real-time instructional path.

Typical responsibilities:
- runtime orchestration
- concise tutoring output
- deterministic curriculum progression
- prompt fast-path handling
- integration adapters

### `research/`
Heavier, richer, multi-agent or reporting-oriented path.

Typical responsibilities:
- extended orchestration
- statistics/report generation
- logging and replay support
- prompt experimentation
- debrief generation

### `shared/`
Reusable logic across both paths.

Typical responsibilities:
- curriculum utilities
- model selection
- shared prompting helpers
- common contracts / state structures

---

## Earlier real-time implementation lineage

This repository also builds on an earlier, simpler runtime design, which introduced the following components:

- `contracts.py`: typed event/action contracts
- `agents.py`: procedure-state, safety, tutor, and curriculum agents
- `curriculum.py`: deterministic curriculum planner
- `prompting.py`: physician-style prompts for fast and legacy modes
- `orchestrator.py`: queue-based real-time orchestrator
- `adapter_example.py`: example integration layer

That earlier version already established several good principles that still matter here:
- **do not modify co-worker code unless necessary**
- use a wrapper/integration layer
- use a **fast prompt mode** for live guidance
- enforce **LLM timeout guards**
- keep deterministic planning ahead of language generation
- route urgent actions through interruption-aware output handling :contentReference[oaicite:10]{index=10}

---

## Integration strategy

A major design goal of this project is to make integration possible **without directly rewriting the existing bronchoscopy GUI/runtime**.

The intended integration pattern is:

1. keep the existing simulator/GUI runtime unchanged
2. read structured runtime/perception outputs in a wrapper layer
3. publish those outputs into the guidance orchestrator
4. include signals such as:
   - reached regions
   - drift/backtracking
   - stagnation
   - phase hints
5. poll guidance actions from the orchestrator
6. send utterances to the existing TTS layer
7. if an action is urgent, interrupt current speech before emitting the new one

This mirrors the earlier implementation plan and remains the preferred engineering strategy because it reduces coupling and makes the system easier to test in isolation. :contentReference[oaicite:11]{index=11}

---

## Safety and scope

This project is aimed at **training guidance**, not autonomous diagnosis or autonomous clinical decision making.

Safety design priorities include:
- prioritizing safety checks before tutoring output
- maintaining bounded utterance length
- avoiding uncontrolled navigation decisions by the LLM
- enabling urgent override behavior
- keeping logs for replay and inspection

The original architecture notes explicitly framed the system as a training copilot with a safety guard and non-diagnostic training scope, and that remains the intended scope here. :contentReference[oaicite:12]{index=12} :contentReference[oaicite:13]{index=13}

---

## Latency target

The real-time design target is conversational guidance that returns quickly enough to remain usable during bronchoscopy training flow.

Earlier design notes used an end-to-end target of roughly **< 1500 ms** from input update to spoken guidance, with bounded latency budgets across state update, safety checks, generation, and TTS startup. That target strongly shaped the runtime-first architecture in this project. :contentReference[oaicite:14]{index=14}

In practice, this means:
- the runtime path must stay narrow
- prompts must be compact
- timeouts and fallbacks matter
- verbose agentic behavior belongs in the research path, not the live path

---

## Current status

This repository is currently at the stage of a **nearly working integrated prototype** with a much cleaner split between:

- a **real-time runtime path** for live coaching
- a **research path** for richer multi-agent experimentation and reporting

The implementation is intended to support:
- synthetic prompt testing
- replay-based inspection
- structured handoff to supervisors/professors
- incremental integration with the existing bronchoscopy runtime

---

## What this project contributes

Compared with a pure legacy prompt chain or a pure end-to-end agentic design, this project contributes a more practical middle path:

- real-time-first engineering
- deterministic curriculum progression
- LLM-based language rendering where useful
- safety-aware orchestration
- a cleaner story for future research expansion

In short:  
**not “LLM decides everything,” and not “no intelligence at all,” but a controlled hybrid suitable for training systems.**

---

## Intended use cases

- bronchoscopy simulator coaching
- curriculum-guided navigation support
- offline replay analysis
- session debrief generation
- prompt/orchestration experiments for academic work
- future expansion into richer multi-agent educational support

---

## Documentation

Earlier and supporting design documents include:

- real-time conversational guidance architecture
- co-worker runtime integration planning
- real-time runtime implementation notes
- testing guide

These documents shaped the current structure and are useful for understanding the original design goals, especially around safety, latency, contracts, and integration boundaries. :contentReference[oaicite:15]{index=15} :contentReference[oaicite:16]{index=16}

---

## Suggested quick start

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
