from profiler.parser import Section, parse_sections
from profiler.ablation import (
    _ablate,
    _ablate_multiple,
    CumulativeStep,
    CumulativeResult,
    PairResult,
    SectionResult,
)


def _make_section(name, content, start, end):
    return Section(name=name, content=content, start=start, end=end)


# ── _ablate_multiple ──────────────────────────────────────────────────────────

def test_ablate_multiple_removes_both():
    prompt = "## A\nContent A.\n\n## B\nContent B.\n\n## C\nContent C."
    sections = parse_sections(prompt)
    assert len(sections) == 3
    ablated = _ablate_multiple(prompt, [sections[0], sections[2]])
    assert "Content A" not in ablated
    assert "Content C" not in ablated
    assert "Content B" in ablated


def test_ablate_multiple_single_is_same_as_ablate():
    prompt = "## X\nHello.\n\n## Y\nWorld."
    sections = parse_sections(prompt)
    assert _ablate(prompt, sections[0]) == _ablate_multiple(prompt, [sections[0]])


def test_ablate_multiple_empty_list_returns_prompt():
    prompt = "Hello world."
    assert _ablate_multiple(prompt, []) == prompt


def test_ablate_multiple_no_triple_newlines():
    prompt = "## A\nA.\n\n## B\nB.\n\n## C\nC.\n\n## D\nD."
    sections = parse_sections(prompt)
    ablated = _ablate_multiple(prompt, [sections[1], sections[2]])
    assert "\n\n\n" not in ablated


def test_ablate_multiple_order_independent():
    """Removing sections in any order gives the same result."""
    prompt = "## A\nContent A.\n\n## B\nContent B.\n\n## C\nContent C."
    sections = parse_sections(prompt)
    forward = _ablate_multiple(prompt, [sections[0], sections[1]])
    backward = _ablate_multiple(prompt, [sections[1], sections[0]])
    assert forward == backward


# ── CumulativeResult ──────────────────────────────────────────────────────────

def test_cumulative_safe_steps_all_above():
    steps = [
        CumulativeStep(["a"], 10, 0.95),
        CumulativeStep(["a", "b"], 20, 0.88),
        CumulativeStep(["a", "b", "c"], 35, 0.82),
    ]
    result = CumulativeResult(steps=steps, threshold=0.8)
    assert result.safe_steps == 3


def test_cumulative_safe_steps_cliff():
    steps = [
        CumulativeStep(["a"], 10, 0.95),
        CumulativeStep(["a", "b"], 20, 0.85),
        CumulativeStep(["a", "b", "c"], 35, 0.71),
    ]
    result = CumulativeResult(steps=steps, threshold=0.8)
    assert result.safe_steps == 2


def test_cumulative_safe_steps_none():
    steps = [CumulativeStep(["a"], 10, 0.60)]
    result = CumulativeResult(steps=steps, threshold=0.8)
    assert result.safe_steps == 0


# ── PairResult delta ──────────────────────────────────────────────────────────

def test_pair_delta_synergy():
    p = PairResult(section_a="A", section_b="B", combined_impact=0.8, expected_impact=0.4, delta=0.4)
    assert p.delta > 0  # combined hurts more than expected → synergy


def test_pair_delta_redundant():
    p = PairResult(section_a="A", section_b="B", combined_impact=0.2, expected_impact=0.5, delta=-0.3)
    assert p.delta < 0  # combined hurts less than expected → redundant


# ── SectionResult verdict ─────────────────────────────────────────────────────

def _make_result(impact: float) -> SectionResult:
    s = Section(name="x", content="x" * 100, start=0, end=100)
    return SectionResult(section=s, impact_score=impact, token_estimate=25)


def test_verdict_critical():
    assert _make_result(0.8).verdict == "Critical — never remove"


def test_verdict_important():
    assert _make_result(0.5).verdict == "Important"


def test_verdict_low_value():
    assert _make_result(0.15).verdict == "Low value"


def test_verdict_dead_weight():
    assert _make_result(0.05).verdict == "Dead weight — safe to remove"
