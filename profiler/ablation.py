import re
from dataclasses import dataclass, field
from itertools import combinations
from together import Together
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from .parser import Section
from .client import complete
from .judge import score_pair


# ── data classes ────────────────────────────────────────────────────────────

@dataclass
class SectionResult:
    section: Section
    impact_score: float
    token_estimate: int
    scores_per_input: list[float] = field(default_factory=list)
    sample_reasonings: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.impact_score >= 0.7:
            return "Critical — never remove"
        if self.impact_score >= 0.4:
            return "Important"
        if self.impact_score >= 0.1:
            return "Low value"
        return "Dead weight — safe to remove"


@dataclass
class CumulativeStep:
    removed_section_names: list[str]
    tokens_removed: int
    avg_quality: float  # 0–1 (higher = better quality remaining)


@dataclass
class CumulativeResult:
    steps: list[CumulativeStep]
    threshold: float

    @property
    def safe_steps(self) -> int:
        """Number of removal steps where quality stays at or above threshold."""
        return sum(1 for s in self.steps if s.avg_quality >= self.threshold)


@dataclass
class PairResult:
    section_a: str
    section_b: str
    combined_impact: float   # quality drop when both are removed
    expected_impact: float   # sum of individual LOO impacts
    delta: float             # combined - expected; positive = synergy, negative = redundant


@dataclass
class PairwiseResult:
    pairs: list[PairResult]


# ── ablation helpers ─────────────────────────────────────────────────────────

def _ablate(prompt: str, section: Section) -> str:
    ablated = prompt[: section.start] + prompt[section.end :]
    return re.sub(r'\n{3,}', '\n\n', ablated).strip()


def _ablate_multiple(prompt: str, sections: list[Section]) -> str:
    """Remove all given sections from prompt in a single pass."""
    if not sections:
        return prompt
    sorted_secs = sorted(sections, key=lambda s: s.start)
    parts: list[str] = []
    prev_end = 0
    for s in sorted_secs:
        parts.append(prompt[prev_end : s.start])
        prev_end = s.end
    parts.append(prompt[prev_end:])
    ablated = "".join(parts)
    return re.sub(r'\n{3,}', '\n\n', ablated).strip()


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    )


# ── runners ──────────────────────────────────────────────────────────────────

def run_leave_one_out(
    client: Together,
    model: str,
    judge_model: str,
    prompt: str,
    sections: list[Section],
    inputs: list[str],
    task_desc: str = "",
) -> tuple[list[SectionResult], list[str]]:
    """Returns (sorted results, baseline outputs) — baselines are reused by Phase 2 modes."""
    total_steps = len(inputs) + len(sections) * len(inputs) * 2

    with _make_progress() as progress:
        task = progress.add_task("Running baseline…", total=total_steps)

        baselines: list[str] = []
        for inp in inputs:
            baselines.append(complete(client, model, prompt, inp))
            progress.advance(task)

        results: list[SectionResult] = []
        for section in sections:
            ablated_prompt = _ablate(prompt, section)
            scores: list[float] = []
            reasonings: list[str] = []

            for i, inp in enumerate(inputs):
                progress.update(task, description=f"LOO '{section.name}' ({i + 1}/{len(inputs)})")
                ablated_out = complete(client, model, ablated_prompt, inp)
                progress.advance(task)

                j = score_pair(client, judge_model, baselines[i], ablated_out, section.name, task_desc)
                scores.append(j["score"])
                reasonings.append(j.get("reasoning", ""))
                progress.advance(task)

            avg_quality = sum(scores) / len(scores)
            results.append(SectionResult(
                section=section,
                impact_score=round(1.0 - avg_quality, 3),
                token_estimate=max(1, len(section.content) // 4),
                scores_per_input=[round(1.0 - s, 3) for s in scores],
                sample_reasonings=reasonings[:2],
            ))

    return sorted(results, key=lambda r: r.impact_score, reverse=True), baselines


def run_cumulative(
    client: Together,
    model: str,
    judge_model: str,
    prompt: str,
    sections: list[Section],
    inputs: list[str],
    loo_results: list[SectionResult],
    baselines: list[str],
    threshold: float = 0.8,
    task_desc: str = "",
) -> CumulativeResult:
    """Remove sections in ascending impact order (least → most important), measuring
    quality after each removal. Uses baselines already computed by LOO."""
    # Order sections from least to most impactful
    section_map = {s.name: s for s in sections}
    ordered = sorted(loo_results, key=lambda r: r.impact_score)  # ascending

    total_steps = len(ordered) * len(inputs) * 2
    steps: list[CumulativeStep] = []
    removed: list[Section] = []
    tokens_removed = 0

    with _make_progress() as progress:
        task = progress.add_task("Cumulative ablation…", total=total_steps)

        for result in ordered:
            sec = section_map[result.section.name]
            removed.append(sec)
            tokens_removed += result.token_estimate

            ablated_prompt = _ablate_multiple(prompt, removed)
            quality_scores: list[float] = []

            for i, inp in enumerate(inputs):
                label = f"Cumulative step {len(removed)}/{len(ordered)}"
                progress.update(task, description=label)
                ablated_out = complete(client, model, ablated_prompt, inp)
                progress.advance(task)

                j = score_pair(
                    client, judge_model, baselines[i], ablated_out,
                    f"{len(removed)} sections removed", task_desc,
                )
                quality_scores.append(j["score"])
                progress.advance(task)

            avg_quality = round(sum(quality_scores) / len(quality_scores), 3)
            steps.append(CumulativeStep(
                removed_section_names=[s.name for s in removed],
                tokens_removed=tokens_removed,
                avg_quality=avg_quality,
            ))

    return CumulativeResult(steps=steps, threshold=threshold)


def run_pairwise(
    client: Together,
    model: str,
    judge_model: str,
    prompt: str,
    sections: list[Section],
    inputs: list[str],
    loo_results: list[SectionResult],
    baselines: list[str],
    task_desc: str = "",
) -> PairwiseResult:
    """Test all section pairs together to detect interaction effects."""
    loo_map = {r.section.name: r.impact_score for r in loo_results}
    n = len(sections)
    pairs_list = list(combinations(sections, 2))
    total_steps = len(pairs_list) * len(inputs) * 2

    pairs: list[PairResult] = []

    with _make_progress() as progress:
        task = progress.add_task("Pairwise ablation…", total=total_steps)

        for idx, (sec_a, sec_b) in enumerate(pairs_list):
            ablated_prompt = _ablate_multiple(prompt, [sec_a, sec_b])
            quality_scores: list[float] = []

            for i, inp in enumerate(inputs):
                progress.update(task, description=f"Pair {idx + 1}/{len(pairs_list)}: '{sec_a.name}' + '{sec_b.name}'")
                ablated_out = complete(client, model, ablated_prompt, inp)
                progress.advance(task)

                j = score_pair(
                    client, judge_model, baselines[i], ablated_out,
                    f"{sec_a.name} + {sec_b.name}", task_desc,
                )
                quality_scores.append(j["score"])
                progress.advance(task)

            avg_quality = sum(quality_scores) / len(quality_scores)
            combined_impact = round(1.0 - avg_quality, 3)
            expected = round(loo_map.get(sec_a.name, 0) + loo_map.get(sec_b.name, 0), 3)
            delta = round(combined_impact - expected, 3)

            pairs.append(PairResult(
                section_a=sec_a.name,
                section_b=sec_b.name,
                combined_impact=combined_impact,
                expected_impact=expected,
                delta=delta,
            ))

    return PairwiseResult(pairs=sorted(pairs, key=lambda p: abs(p.delta), reverse=True))
