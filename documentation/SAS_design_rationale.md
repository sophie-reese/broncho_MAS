# SAS Architecture Note

## Purpose

This note gives a short explanation of the **SAS (Single-Agent with Skills)** line in BronchoMAS: what it is, why the project uses it, and what each core skill is responsible for.

---

## 1. Why SAS exists

Earlier iterations explored a heavier **multi-agent system (MAS)** line. That was useful for experimentation and richer analysis, but it also increased latency, token usage, coordination overhead, and debugging complexity. For bronchoscopy guidance, those costs matter because the system needs to be short, stable, timely, and grounded in structured upstream state. Recent work on **single-agent with skills** supports this direction: one orchestrated agent can preserve modularity while reducing the communication overhead of explicit multi-agent coordination, especially when the skill library remains small and well organized. fileciteturn4file2

So in this project, **SAS** means:

> one orchestrating manager, a small set of explicit skills, deterministic helpers, and structured outputs grounded in the upstream payload.

This also matches the distinction between **skills** and **tools**: skills define bounded behavioral capabilities, while tools or helpers compute things deterministically. fileciteturn4file4

---

## 2. High-level structure

```text
upstream structured payload
        ↓
    SAS Manager
        ↓
  skill evaluation / routing
        ↓
 selected skill result
        ↓
 shared realization / postprocessing
        ↓
 final utterance + structured trace
```

The manager receives structured upstream state, evaluates available skills, selects the most appropriate active skill, and returns both a short utterance and a structured trace of why that skill was selected.

---

## 3. Skill overview and rationale

### Guidance skill
The **guidance skill** is the main real-time coaching path. It turns the current plan, directional hint, and safety state into short bedside guidance. This is the default live instructional skill and should stay short, direct, and grounded in structured upstream signals rather than free-form inference.

### QA skill
The **QA skill** handles trainee questions that are relevant to bronchoscopy teaching, navigation, or interpretation of what the trainee is seeing. It can take the turn when a question is worth answering, but it should still return quickly to the procedural task instead of opening long side conversations.

### Landmark teaching skill
The **landmark teaching skill** is a short teaching overlay, not a full replacement for guidance. Its job is to fire when the trainee reaches an important landmark for the first time and give a brief, memorable teaching line about recognition and recall. The purpose is not to dump anatomy facts, but to strengthen orientation and retention at key moments. To keep the skill library small and avoid many near-duplicate skills, the system uses **one landmark teaching skill** plus a separate **landmark card library** that stores the actual content for each landmark. This is consistent with the SAS design principle that skill boundaries should stay few and distinct, since large flat libraries with semantically similar skills can hurt routing quality. fileciteturn4file2

### Support skill
The **support skill** adds very light emotional or interactional support when useful, for example when the trainee is repeatedly resetting or when a small acknowledgment helps maintain flow. It should remain secondary and must never override safety-critical or anatomically important guidance.

### Statistics skill
The **statistics skill** produces compact structured summaries of the current procedural situation, such as likely issue, next coaching focus, and teaching point. It is not mainly for speaking to the trainee; it is there to support logging, analysis, and downstream reporting.

### Reporting skill
The **reporting skill** produces the end-of-session report. Its role is to summarize curriculum coverage, structured progress, and the main teaching takeaway in a concise educator-facing format. This skill belongs to session-end reflection, not live turn-by-turn navigation.

---

## 4. Core implementation principles

### Keep the skill set small
SAS works best when the skill library is small and clearly separated. Too many similar skills increase semantic overlap and make routing less reliable. fileciteturn4file2

### Preserve deterministic authority where needed
Curriculum logic, navigation authority, and safety-sensitive constraints should not be left entirely to free-form LLM behavior. The language model mainly helps realize the final response, while deterministic logic keeps the system grounded.

### Return structured outputs
A selected skill should return not only text, but also a structured record such as selected skill, priority, reason, and payload. This makes the system easier to debug, test, and extend.

---

## 5. Bottom line

The SAS line is the current primary architecture of BronchoMAS because it is a practical middle path:

- more modular than one monolithic prompt
- more efficient than explicit multi-agent coordination
- better suited to short, structured, real-time bronchoscopy guidance

The goal is not to copy a generic agent-skills framework for its own sake. The goal is to build a **disciplined, grounded, and explainable single-agent guidance architecture** that fits the actual needs of bronchoscopy training.
