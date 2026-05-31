import re
from dataclasses import dataclass, field
from together import Together
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from .parser import Section
from .client import complete
from .judge import score_pair


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


def _ablate(prompt: str, section: Section) -> str:
    ablated = prompt[: section.start] + prompt[section.end :]
    return re.sub(r'\n{3,}', '\n\n', ablated).strip()


def run_leave_one_out(
    client: Together,
    model: str,
    judge_model: str,
    prompt: str,
    sections: list[Section],
    inputs: list[str],
    task_desc: str = "",
) -> list[SectionResult]:
    # baseline + (ablated output + judge call) per section per input
    total_steps = len(inputs) + len(sections) * len(inputs) * 2

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Running baseline…", total=total_steps)

        baselines = []
        for inp in inputs:
            baselines.append(complete(client, model, prompt, inp))
            progress.advance(task)

        results = []
        for section in sections:
            ablated_prompt = _ablate(prompt, section)
            scores: list[float] = []
            reasonings: list[str] = []

            for i, inp in enumerate(inputs):
                progress.update(task, description=f"Ablating '{section.name}' ({i + 1}/{len(inputs)})")
                ablated_out = complete(client, model, ablated_prompt, inp)
                progress.advance(task)

                j = score_pair(client, judge_model, baselines[i], ablated_out, section.name, task_desc)
                scores.append(j["score"])
                reasonings.append(j.get("reasoning", ""))
                progress.advance(task)

            avg_quality = sum(scores) / len(scores)
            impact = round(1.0 - avg_quality, 3)
            results.append(SectionResult(
                section=section,
                impact_score=impact,
                token_estimate=max(1, len(section.content) // 4),
                scores_per_input=[round(1.0 - s, 3) for s in scores],
                sample_reasonings=reasonings[:2],
            ))

    return sorted(results, key=lambda r: r.impact_score, reverse=True)
