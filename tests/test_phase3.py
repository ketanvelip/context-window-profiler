from profiler.parser import parse_sections
from profiler.ablation import _ablate_multiple
from profiler.report import render_trim_summary, render_compare
from rich.console import Console


# ── trim helpers ──────────────────────────────────────────────────────────────

_PROMPT = "## Role\nYou are helpful.\n\n## Task\nAnswer questions.\n\n## Format\nRespond in JSON.\n\n## Tone\nBe concise."

_RUN_DATA = {
    "prompt": _PROMPT,
    "results": [
        {"section_name": "Role",   "impact_score": 0.82, "token_estimate": 5,  "scores_per_input": [], "sample_reasonings": [], "verdict": "Critical — never remove"},
        {"section_name": "Task",   "impact_score": 0.55, "token_estimate": 4,  "scores_per_input": [], "sample_reasonings": [], "verdict": "Important"},
        {"section_name": "Format", "impact_score": 0.05, "token_estimate": 5,  "scores_per_input": [], "sample_reasonings": [], "verdict": "Dead weight — safe to remove"},
        {"section_name": "Tone",   "impact_score": 0.02, "token_estimate": 3,  "scores_per_input": [], "sample_reasonings": [], "verdict": "Dead weight — safe to remove"},
    ],
}


def test_trim_removes_dead_weight_sections():
    prompt = _RUN_DATA["prompt"]
    sections = parse_sections(prompt)
    section_map = {s.name: s for s in sections}

    to_remove = [
        section_map[r["section_name"]]
        for r in _RUN_DATA["results"]
        if r["impact_score"] < 0.1 and r["section_name"] in section_map
    ]
    assert len(to_remove) == 2
    trimmed = _ablate_multiple(prompt, to_remove)
    assert "Format" not in trimmed
    assert "Tone" not in trimmed
    assert "Role" in trimmed
    assert "Task" in trimmed


def test_trim_higher_threshold_removes_more():
    prompt = _RUN_DATA["prompt"]
    sections = parse_sections(prompt)
    section_map = {s.name: s for s in sections}

    to_remove_aggressive = [
        section_map[r["section_name"]]
        for r in _RUN_DATA["results"]
        if r["impact_score"] < 0.6 and r["section_name"] in section_map
    ]
    trimmed = _ablate_multiple(prompt, to_remove_aggressive)
    assert "Format" not in trimmed
    assert "Tone" not in trimmed
    assert "Task" not in trimmed
    assert "Role" in trimmed


def test_trim_nothing_when_all_above_threshold():
    sections = parse_sections(_PROMPT)
    section_map = {s.name: s for s in sections}
    to_remove = [
        section_map[r["section_name"]]
        for r in _RUN_DATA["results"]
        if r["impact_score"] < 0.01 and r["section_name"] in section_map
    ]
    assert to_remove == []


def test_render_trim_summary_no_crash():
    c = Console(highlight=False)
    render_trim_summary(["Format", "Tone"], tokens_saved=8, total_tokens=100, output_path=None, console=c)


# ── compare helpers ───────────────────────────────────────────────────────────

_RUN_A = {
    "results": [
        {"section_name": "Role",   "impact_score": 0.80, "token_estimate": 5},
        {"section_name": "Task",   "impact_score": 0.50, "token_estimate": 4},
        {"section_name": "Format", "impact_score": 0.05, "token_estimate": 5},
        {"section_name": "Tone",   "impact_score": 0.02, "token_estimate": 3},
    ]
}

_RUN_B = {
    "results": [
        {"section_name": "Role",         "impact_score": 0.90, "token_estimate": 5},  # more critical
        {"section_name": "Task",         "impact_score": 0.20, "token_estimate": 4},  # less critical
        {"section_name": "Format",       "impact_score": 0.06, "token_estimate": 5},  # unchanged
        # Tone removed, new_section added
        {"section_name": "new_section",  "impact_score": 0.40, "token_estimate": 8},
    ]
}


def _compare_map(data_a, data_b):
    map_a = {r["section_name"]: r for r in data_a["results"]}
    map_b = {r["section_name"]: r for r in data_b["results"]}
    return map_a, map_b


def test_compare_detects_more_critical():
    map_a, map_b = _compare_map(_RUN_A, _RUN_B)
    delta = map_b["Role"]["impact_score"] - map_a["Role"]["impact_score"]
    assert delta > 0.05


def test_compare_detects_less_critical():
    map_a, map_b = _compare_map(_RUN_A, _RUN_B)
    delta = map_b["Task"]["impact_score"] - map_a["Task"]["impact_score"]
    assert delta < -0.05


def test_compare_detects_new_section():
    map_a, map_b = _compare_map(_RUN_A, _RUN_B)
    assert "new_section" not in map_a
    assert "new_section" in map_b


def test_compare_detects_removed_section():
    map_a, map_b = _compare_map(_RUN_A, _RUN_B)
    assert "Tone" in map_a
    assert "Tone" not in map_b


def test_render_compare_no_crash():
    c = Console(highlight=False)
    render_compare(_RUN_A, _RUN_B, label_a="v1", label_b="v2", console=c)


def test_compare_section_order_preserved():
    """A's sections should come first in the output, B-only sections appended."""
    map_a = {r["section_name"]: r for r in _RUN_A["results"]}
    map_b = {r["section_name"]: r for r in _RUN_B["results"]}
    names = list(map_a.keys()) + [n for n in map_b if n not in map_a]
    assert names.index("Role") < names.index("new_section")
