import json
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

from .parser import parse_sections, Section
from .client import get_client
from .ablation import run_leave_one_out, SectionResult
from .storage import save_run, list_runs, load_run
from .report import render_heatmap, render_recommendations

console = Console()


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
def profile(prompt_file, inputs_file, model, judge_model, task):
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
            "Add section markers (## Heading, [SECTION: name]…[/SECTION], or <!-- [SECTION: name] -->…<!-- [/SECTION] -->)"
            " or ensure the prompt has multiple blank-line-separated paragraphs."
        )
        sys.exit(1)

    n, m = len(sections), len(inputs)
    est_calls = m + n * m * 2
    console.print(f"\n[bold]Sections detected:[/bold] {n}")
    for s in sections:
        console.print(f"  · {s.name}  (~{max(1, len(s.content) // 4)} tokens)")
    console.print(f"\n[bold]Inputs:[/bold]          {m}")
    console.print(f"[bold]Estimated calls:[/bold] ~{est_calls}")
    console.print(f"[bold]Model:[/bold]           {model}")
    console.print(f"[bold]Judge:[/bold]           {judge_model}\n")

    if not click.confirm("Proceed?"):
        raise click.Abort()

    client = get_client()
    results = run_leave_one_out(client, model, judge_model, prompt, sections, inputs, task_desc=task)

    render_heatmap(results, console)
    render_recommendations(results, console)

    run_data = {
        "model": model,
        "judge_model": judge_model,
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
            for r in results
        ],
    }
    path = save_run(run_data)
    console.print(f"\n[dim]Results saved → {path}[/dim]")


@cli.command()
@click.argument("run_file", type=click.Path(), required=False)
def report(run_file):
    """Show heatmap from a previous run. Defaults to the most recent."""
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
    results = [_result_from_dict(r) for r in data["results"]]
    render_heatmap(results, console)
    render_recommendations(results, console)
