from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

from .ablation import SectionResult, CumulativeResult, PairwiseResult

_VERDICTS: list[tuple[float, str, str]] = [
    (0.7, "Critical — never remove", "red"),
    (0.4, "Important", "yellow"),
    (0.1, "Low value", "cyan"),
    (0.0, "Dead weight — safe to remove", "green"),
]


def _verdict_style(impact: float) -> tuple[str, str]:
    for threshold, label, color in _VERDICTS:
        if impact >= threshold:
            return label, color
    return "Dead weight — safe to remove", "green"


def _bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


# ── leave-one-out ─────────────────────────────────────────────────────────────

def render_heatmap(results: list[SectionResult], console: Console | None = None) -> None:
    if console is None:
        console = Console()

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold", title="Leave-One-Out Heatmap")
    table.add_column("Section")
    table.add_column("~Tokens", justify="right")
    table.add_column("Impact", justify="right")
    table.add_column("Verdict")

    for r in results:
        label, color = _verdict_style(r.impact_score)
        table.add_row(
            r.section.name,
            str(r.token_estimate),
            f"{r.impact_score:.2f} {_bar(r.impact_score)}",
            Text(label, style=color),
        )

    console.print()
    console.print(table)


def render_recommendations(results: list[SectionResult], console: Console | None = None) -> None:
    if console is None:
        console = Console()

    dead = [r for r in results if r.impact_score < 0.1]
    low = [r for r in results if 0.1 <= r.impact_score < 0.4]

    if dead:
        tokens = sum(r.token_estimate for r in dead)
        names = ", ".join(f'"{r.section.name}"' for r in dead)
        console.print(f"\n[green]Removing {names} saves ~{tokens} tokens with no measured quality loss.[/green]")

    if low:
        names = ", ".join(f'"{r.section.name}"' for r in low)
        console.print(f"[yellow]Low-value sections {names} are candidates for trimming — verify manually.[/yellow]")

    for r in results:
        if r.scores_per_input and r.impact_score >= 0.1:
            spread = max(r.scores_per_input) - min(r.scores_per_input)
            if spread > 0.5:
                console.print(
                    f'[blue]"{r.section.name}" has high variance ({spread:.2f}) — helps some inputs, hurts others.[/blue]'
                )


# ── cumulative ────────────────────────────────────────────────────────────────

def render_cumulative(result: CumulativeResult, console: Console | None = None) -> None:
    if console is None:
        console = Console()

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        title=f"Cumulative Ablation  (quality threshold: {result.threshold:.2f})",
    )
    table.add_column("Step", justify="right")
    table.add_column("Section Removed")
    table.add_column("~Tokens Saved", justify="right")
    table.add_column("Quality Remaining", justify="right")

    for i, step in enumerate(result.steps):
        last_removed = step.removed_section_names[-1]
        quality_str = f"{step.avg_quality:.2f} {_bar(step.avg_quality)}"
        ok = step.avg_quality >= result.threshold
        quality_cell = Text(quality_str, style="green" if ok else "red")
        table.add_row(str(i + 1), last_removed, f"~{step.tokens_removed}", quality_cell)

    console.print()
    console.print(table)

    safe = result.safe_steps
    if safe == 0:
        console.print("[red]Even removing the least impactful section drops quality below threshold.[/red]")
    elif safe == len(result.steps):
        last = result.steps[-1]
        console.print(
            f"[green]All {safe} sections can be removed while staying above threshold. "
            f"Saves ~{last.tokens_removed} tokens at {last.avg_quality:.2f} quality.[/green]"
        )
    else:
        cliff_step = result.steps[safe]
        safe_step = result.steps[safe - 1]
        console.print(
            f"[green]Safe to remove {safe} section(s): saves ~{safe_step.tokens_removed} tokens "
            f"at {safe_step.avg_quality:.2f} quality.[/green]"
        )
        console.print(
            f'[red]Quality cliff at step {safe + 1}: removing "{cliff_step.removed_section_names[-1]}" '
            f"drops quality to {cliff_step.avg_quality:.2f}.[/red]"
        )


# ── pairwise ──────────────────────────────────────────────────────────────────

def render_pairwise(result: PairwiseResult, console: Console | None = None) -> None:
    if console is None:
        console = Console()

    # Only show pairs with a meaningful interaction (|delta| > 0.1)
    notable = [p for p in result.pairs if abs(p.delta) > 0.1]

    if not notable:
        console.print("\n[dim]Pairwise: no notable section interactions detected.[/dim]")
        return

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        title="Pairwise Interactions  (|delta| > 0.10)",
    )
    table.add_column("Section A")
    table.add_column("Section B")
    table.add_column("Combined", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Finding")

    for p in notable:
        delta_str = f"{p.delta:+.2f}"
        if p.delta > 0:
            delta_cell = Text(delta_str, style="yellow")
            finding = Text("Synergy — neither alone is enough", style="yellow")
        else:
            delta_cell = Text(delta_str, style="cyan")
            finding = Text("Possible redundancy — consider keeping one", style="cyan")

        table.add_row(
            p.section_a,
            p.section_b,
            f"{p.combined_impact:.2f}",
            f"{p.expected_impact:.2f}",
            delta_cell,
            finding,
        )

    console.print()
    console.print(table)
