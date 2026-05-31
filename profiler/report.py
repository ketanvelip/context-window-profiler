from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

from .ablation import SectionResult

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


def render_heatmap(results: list[SectionResult], console: Console | None = None) -> None:
    if console is None:
        console = Console()

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
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
