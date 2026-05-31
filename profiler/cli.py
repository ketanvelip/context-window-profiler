import json
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

from .parser import parse_sections, Section
from .client import get_client
from .ablation import (
    run_leave_one_out,
    run_cumulative,
    run_pairwise,
    SectionResult,
    CumulativeResult,
    PairwiseResult,
)
from .storage import save_run, list_runs, load_run
from .report import render_heatmap, render_recommendations, render_cumulative, render_pairwise

console = Console()

_MODES = click.Choice(["leave-one-out", "cumulative", "pairwise", "all"], case_sensitive=False)


def _load_inputs(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    inputs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            inputs.append(obj.get("user", str(obj)))
        except json.JSONDecodeError:
            inputs.append(line)
    return inputs


def _estimate_calls(n: int, m: int, mode: str) -> int:
    loo = m + n * m * 2
    cumulative = n * m * 2
    pairwise = (n * (n - 1) // 2) * m * 2
    if mode == "leave-one-out":
        return loo
    if mode == "cumulative":
        return loo + cumulative
    if mode == "pairwise":
        return loo + pairwise
    return loo + cumulative + pairwise  # all


def _result_from_dict(d: dict) -> SectionResult:
    section = Section(name=d["section_name"], content="", start=0, end=0)
    return SectionResult(
        section=section,
        impact_score=d["impact_score"],
        token_estimate=d["token_estimate"],
        scores_per_input=d.get("scores_per_input", []),
        sample_reasonings=d.get("sample_reasonings", []),
    )


@click.group()
def cli():
    """Context Window Profiler — find which parts of your prompt actually matter."""


@cli.command()
@click.argument("prompt_file", type=click.Path(exists=True))
@click.argument("inputs_file", type=click.Path(exists=True))
@click.option("--model", envvar="TOGETHER_MODEL", default="meta-llama/Llama-3.3-70B-Instruct-Turbo", show_default=True)
@click.option("--judge-model", envvar="TOGETHER_JUDGE_MODEL", default=None, help="Defaults to --model")
@click.option("--task", default="", help="One-line task description (helps the judge evaluate quality)")
@click.option("--mode", type=_MODES, default="leave-one-out", show_default=True,
              help="leave-one-out, cumulative, pairwise, or all")
@click.option("--threshold", default=0.8, show_default=True, type=float,
              help="Quality threshold for cumulative mode (0–1)")
def profile(prompt_file, inputs_file, model, judge_model, task, mode, threshold):
    """Run ablation analysis on PROMPT_FILE with inputs from INPUTS_FILE."""
    judge_model = judge_model or model
    prompt = Path(prompt_file).read_text(encoding="utf-8")
    inputs = _load_inputs(inputs_file)

    if not inputs:
        console.print("[red]No inputs found.[/red]")
        sys.exit(1)

    sections = parse_sections(prompt)
    if len(sections) < 2:
        console.print(
            "[red]Fewer than 2 sections detected.[/red]\n"
            "Add section markers (## Heading, [SECTION: name]…[/SECTION], or "
            "<!-- [SECTION: name] -->…<!-- [/SECTION] -->) or ensure the prompt "
            "has multiple blank-line-separated paragraphs."
        )
        sys.exit(1)

    n, m = len(sections), len(inputs)
    est = _estimate_calls(n, m, mode)

    console.print(f"\n[bold]Sections detected:[/bold] {n}")
    for s in sections:
        console.print(f"  · {s.name}  (~{max(1, len(s.content) // 4)} tokens)")
    console.print(f"\n[bold]Inputs:[/bold]          {m}")
    console.print(f"[bold]Mode:[/bold]            {mode}")
    console.print(f"[bold]Estimated calls:[/bold] ~{est}")
    console.print(f"[bold]Model:[/bold]           {model}")
    console.print(f"[bold]Judge:[/bold]           {judge_model}\n")

    if not click.confirm("Proceed?"):
        raise click.Abort()

    client = get_client()

    # LOO is always required — it determines ordering for cumulative and
    # provides individual impact scores for pairwise delta calculation.
    loo_results, baselines = run_leave_one_out(
        client, model, judge_model, prompt, sections, inputs, task_desc=task
    )

    render_heatmap(loo_results, console)
    render_recommendations(loo_results, console)

    cumulative_result: CumulativeResult | None = None
    pairwise_result: PairwiseResult | None = None

    if mode in ("cumulative", "all"):
        cumulative_result = run_cumulative(
            client, model, judge_model, prompt, sections, inputs,
            loo_results, baselines, threshold=threshold, task_desc=task,
        )
        render_cumulative(cumulative_result, console)

    if mode in ("pairwise", "all"):
        pairwise_result = run_pairwise(
            client, model, judge_model, prompt, sections, inputs,
            loo_results, baselines, task_desc=task,
        )
        render_pairwise(pairwise_result, console)

    run_data: dict = {
        "model": model,
        "judge_model": judge_model,
        "mode": mode,
        "prompt": prompt,
        "inputs": inputs,
        "results": [
            {
                "section_name": r.section.name,
                "token_estimate": r.token_estimate,
                "impact_score": r.impact_score,
                "scores_per_input": r.scores_per_input,
                "sample_reasonings": r.sample_reasonings,
                "verdict": r.verdict,
            }
            for r in loo_results
        ],
    }

    if cumulative_result is not None:
        run_data["cumulative"] = {
            "threshold": cumulative_result.threshold,
            "steps": [
                {
                    "removed_section_names": s.removed_section_names,
                    "tokens_removed": s.tokens_removed,
                    "avg_quality": s.avg_quality,
                }
                for s in cumulative_result.steps
            ],
        }

    if pairwise_result is not None:
        run_data["pairwise"] = [
            {
                "section_a": p.section_a,
                "section_b": p.section_b,
                "combined_impact": p.combined_impact,
                "expected_impact": p.expected_impact,
                "delta": p.delta,
            }
            for p in pairwise_result.pairs
        ]

    path = save_run(run_data)
    console.print(f"\n[dim]Results saved → {path}[/dim]")


@cli.command()
@click.argument("run_file", type=click.Path(), required=False)
def report(run_file):
    """Show heatmap (and cumulative/pairwise if available) from a previous run."""
    if run_file:
        path = Path(run_file)
        if not path.exists():
            console.print(f"[red]File not found: {run_file}[/red]")
            sys.exit(1)
    else:
        runs = list_runs()
        if not runs:
            console.print("[red]No runs found in ~/.context-profiler/runs/[/red]")
            sys.exit(1)
        path = runs[0]
        console.print(f"[dim]Loading: {path}[/dim]")

    data = load_run(path)

    loo_results = [_result_from_dict(r) for r in data["results"]]
    render_heatmap(loo_results, console)
    render_recommendations(loo_results, console)

    if "cumulative" in data:
        from .ablation import CumulativeStep
        c = data["cumulative"]
        steps = [
            CumulativeStep(
                removed_section_names=s["removed_section_names"],
                tokens_removed=s["tokens_removed"],
                avg_quality=s["avg_quality"],
            )
            for s in c["steps"]
        ]
        render_cumulative(CumulativeResult(steps=steps, threshold=c["threshold"]), console)

    if "pairwise" in data:
        from .ablation import PairResult
        pairs = [
            PairResult(
                section_a=p["section_a"],
                section_b=p["section_b"],
                combined_impact=p["combined_impact"],
                expected_impact=p["expected_impact"],
                delta=p["delta"],
            )
            for p in data["pairwise"]
        ]
        render_pairwise(PairwiseResult(pairs=pairs), console)
