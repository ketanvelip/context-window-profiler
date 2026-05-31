import re
from dataclasses import dataclass


@dataclass
class Section:
    name: str
    content: str
    start: int
    end: int


def parse_sections(prompt: str) -> list[Section]:
    """Auto-detect sections using multiple strategies in priority order."""
    for strategy in [_try_comment_markers, _try_bracket_markers, _try_markdown_headings]:
        sections = strategy(prompt)
        if len(sections) >= 2:
            return sections
    return _try_paragraphs(prompt)


def _try_comment_markers(prompt: str) -> list[Section]:
    """Match <!-- [SECTION: name] --> ... <!-- [/SECTION] --> blocks."""
    pattern = re.compile(
        r'<!--\s*\[SECTION:\s*([^\]]+)\]\s*-->(.*?)<!--\s*\[/SECTION\]\s*-->',
        re.DOTALL | re.IGNORECASE,
    )
    return [
        Section(name=m.group(1).strip(), content=m.group(2).strip(), start=m.start(), end=m.end())
        for m in pattern.finditer(prompt)
    ]


def _try_bracket_markers(prompt: str) -> list[Section]:
    """Match [SECTION: name] ... [/SECTION] blocks."""
    pattern = re.compile(
        r'\[SECTION:\s*([^\]]+)\](.*?)\[/SECTION\]',
        re.DOTALL | re.IGNORECASE,
    )
    return [
        Section(name=m.group(1).strip(), content=m.group(2).strip(), start=m.start(), end=m.end())
        for m in pattern.finditer(prompt)
    ]


def _try_markdown_headings(prompt: str) -> list[Section]:
    """Split on ## / # / ### headings; each heading owns text until the next."""
    pattern = re.compile(r'^#{1,3}\s+(.+)$', re.MULTILINE)
    matches = list(pattern.finditer(prompt))
    if len(matches) < 2:
        return []
    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        content = prompt[m.end():end].strip()
        sections.append(Section(name=m.group(1).strip(), content=content, start=start, end=end))
    return sections


def _try_paragraphs(prompt: str) -> list[Section]:
    """Fallback: split on blank lines, name sections paragraph_1, paragraph_2, …"""
    separators = list(re.finditer(r'\n\n+', prompt))
    boundaries = [0] + [m.end() for m in separators] + [len(prompt)]
    sections = []
    for i in range(len(boundaries) - 1):
        chunk = prompt[boundaries[i]:boundaries[i + 1]]
        content = chunk.strip()
        if not content:
            continue
        actual_start = prompt.find(content, boundaries[i])
        sections.append(Section(
            name=f"paragraph_{len(sections) + 1}",
            content=content,
            start=actual_start,
            end=actual_start + len(content),
        ))
    return sections
