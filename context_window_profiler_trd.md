# Context Window Profiler — Technical Requirements Document

**Version:** 1.0.0  
**Status:** Draft  
**Date:** May 31, 2026  

---

## 1. Overview

### Problem

Long prompts are mostly dead weight. Engineers stuff system prompts, examples, retrieved context, and instructions into the context window — but have no idea which parts the model actually uses to produce its output. The result:

- Prompts grow indefinitely because no one knows what's safe to remove
- Token costs balloon on content the model ignores
- Latency suffers from context the model didn't need
- Engineers cargo-cult instructions ("always include X") without evidence they help

There is no tool today that tells an engineer, with evidence, *which parts of a prompt actually matter*.

### Goal

A profiler that takes a prompt and a representative set of inputs, then automatically tests which sections of the prompt are load-bearing versus dead weight. It outputs a ranked heatmap showing each section's measured impact on output quality.

The engineer learns: "remove these 400 tokens and quality stays identical" or "this one instruction is doing 80% of the work — never touch it."

### Non-Goals

- Not a prompt optimizer — it diagnoses, it doesn't rewrite
- Not a token counter or cost estimator (those exist)
- Not a general eval harness — it answers one specific question
- Not real-time — profiling is an offline, deliberate process

---

## 2. Core Concept

The profiler works by **systematic ablation**: it removes one section of the prompt at a time, re-runs the same inputs, and measures how much the output quality changes. Sections whose removal causes large quality drops are load-bearing. Sections whose removal changes nothing are dead weight.

This is the same technique used in ML interpretability research, applied to prompts.

---

## 3. Functional Requirements

### 3.1 Prompt Sectioning

- Accept a prompt with engineer-defined section markers
- Each section gets a name and can be independently ablated
- Sections can be nested (a system prompt might have sub-sections)
- Support common structures: instructions, examples, context, format specs, persona

### 3.2 Input Set

- Engineer provides a set of representative inputs (10–50 is typical)
- Inputs should cover the real distribution of cases the prompt handles
- The same input set is used for every ablation run so results are comparable

### 3.3 Ablation Strategies

The profiler supports three modes:

**Leave-one-out** — remove one section at a time, baseline against the full prompt. Best for understanding individual section impact.

**Cumulative** — remove sections in ranked order of importance until quality breaks. Shows the minimum viable prompt.

**Pairwise** — test pairs of sections together to detect interaction effects (two sections that only matter when combined).

### 3.4 Quality Measurement

Output quality is measured by comparing each ablated run against the baseline (full prompt) run:

- An LLM judge rates whether the ablated output is equivalent, degraded, or improved versus baseline
- Optional deterministic checks (format validity, required fields, length bounds)
- Scores are averaged across the input set for each ablation

### 3.5 Output: The Heatmap

The primary deliverable is a ranked report:

| Section | Tokens | Impact Score | Verdict |
|---|---|---|---|
| Format spec | 120 | 0.94 | Critical — never remove |
| Example 1 | 340 | 0.71 | Important |
| Persona | 80 | 0.12 | Low value |
| Example 2 | 290 | 0.04 | Dead weight — safe to remove |
| Tone guidance | 60 | 0.01 | Dead weight — safe to remove |

Impact score = average quality drop when this section is removed.

### 3.6 Recommendations

Based on the heatmap, the profiler produces actionable suggestions:

- "Removing these 3 sections saves 540 tokens with no measured quality loss"
- "Section X has high variance — it helps some inputs but hurts others. Investigate."
- "These two sections seem redundant — keeping only one preserves quality."

---

## 4. CLI Interface

```
profile    Run ablation analysis on a prompt
report     Show the heatmap from a previous profiling run
trim       Generate a slimmed-down prompt based on profiling results
compare    Compare profiles of two prompt versions
```

---

## 5. Constraints

- Must work with any prompt structure, not just a fixed template
- API key configured via environment variable
- Results stored locally — no cloud dependency
- Profiling runs are expensive (many API calls); must be explicitly invoked, not automatic
- Should warn the user about expected cost before starting a run

---

## 6. Build Phases

### Phase 1 — Core Ablation
Section parsing, leave-one-out ablation, basic quality scoring, terminal heatmap output.

### Phase 2 — Smart Analysis
Cumulative and pairwise modes, variance detection, recommendations engine.

### Phase 3 — Workflow Integration
Trim command (generates a recommended slimmer prompt), compare across versions, configurable section markers.

### Stretch
- Visual HTML heatmap export
- Integration with prompt eval harnesses
- Multi-model profiling (same prompt, different models, compare which sections each one relies on)

---

## 7. Open Questions

| Question | Notes |
|---|---|
| Section granularity | Should sections be defined by the user (explicit markers) or auto-detected (paragraphs, headings)? Explicit is more accurate but more work. |
| Cost ceiling | A 10-section prompt with 30 inputs = 300+ API calls per profile run. Should there be a default cap and confirmation step? |
| Judge reliability | LLM judges can be inconsistent on subtle quality differences. How many judge calls per ablation are needed for confidence? |
| Order effects | Some prompts are sensitive to section order. Should the profiler also test reordering, not just removal? |
| Caching | If the same input + same prompt is run twice, results should be cached. How long is a cache entry valid? |

---

## 8. Success Metrics

- A profile of a 1,500-token prompt with 20 inputs completes in under 10 minutes
- At least 20% token reduction achieved on typical real-world prompts with no measurable quality loss
- Engineers report increased confidence in trimming prompts after using the tool
- The recommendations are actionable enough that users apply them more often than they discard them
