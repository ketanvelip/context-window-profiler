import pytest
from profiler.parser import parse_sections
from profiler.ablation import _ablate


def test_comment_markers():
    prompt = (
        "<!-- [SECTION: instructions] -->\nBe helpful and concise.\n<!-- [/SECTION] -->\n\n"
        "<!-- [SECTION: examples] -->\nQ: 2+2?\nA: 4\n<!-- [/SECTION] -->"
    )
    sections = parse_sections(prompt)
    assert len(sections) == 2
    assert sections[0].name == "instructions"
    assert sections[1].name == "examples"
    assert "Be helpful" in sections[0].content


def test_bracket_markers():
    prompt = "[SECTION: persona]\nYou are a pirate.\n[/SECTION]\n\n[SECTION: format]\nRespond in JSON.\n[/SECTION]"
    sections = parse_sections(prompt)
    assert len(sections) == 2
    assert sections[0].name == "persona"


def test_markdown_headings():
    prompt = "## Instructions\nBe helpful.\n\n## Examples\nQ: hi\nA: hello\n\n## Format\nRespond in JSON."
    sections = parse_sections(prompt)
    assert len(sections) == 3
    assert sections[0].name == "Instructions"
    assert "Be helpful" in sections[0].content


def test_paragraph_fallback():
    prompt = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."
    sections = parse_sections(prompt)
    assert len(sections) == 3
    assert sections[0].name == "paragraph_1"
    assert "First paragraph" in sections[0].content


def test_ablation_removes_section():
    prompt = "## Instructions\nBe helpful.\n\n## Examples\nQ: 2+2?\nA: 4"
    sections = parse_sections(prompt)
    assert len(sections) == 2
    ablated = _ablate(prompt, sections[0])
    assert "Instructions" not in ablated
    assert "Examples" in ablated


def test_ablation_no_triple_newlines():
    prompt = "## A\nContent A.\n\n## B\nContent B.\n\n## C\nContent C."
    sections = parse_sections(prompt)
    ablated = _ablate(prompt, sections[1])
    assert "\n\n\n" not in ablated


def test_section_start_end_are_valid_offsets():
    prompt = "## Role\nYou are helpful.\n\n## Task\nAnswer questions."
    sections = parse_sections(prompt)
    for s in sections:
        assert 0 <= s.start < s.end <= len(prompt)
