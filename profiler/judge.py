import json
import re
from together import Together
from .client import complete


_SYSTEM = "You are an impartial quality evaluator. Respond only with valid JSON."

_USER = """\
Compare these two AI responses.
{task_line}

BASELINE (generated from the complete prompt):
<baseline>
{baseline}
</baseline>

TEST (generated with section "{section_name}" removed):
<test>
{test_output}
</test>

How much quality was LOST by removing that section?
  1.0 = identical quality — the section had no measurable effect
  0.5 = noticeably worse
  0.0 = completely broken or off-task

Respond with JSON only, no other text:
{{"score": <float 0.0–1.0>, "reasoning": "<one sentence>"}}"""


def score_pair(
    client: Together,
    judge_model: str,
    baseline: str,
    test_output: str,
    section_name: str,
    task_desc: str = "",
) -> dict:
    task_line = f"Task: {task_desc}" if task_desc else ""
    prompt = _USER.format(
        task_line=task_line,
        baseline=baseline,
        section_name=section_name,
        test_output=test_output,
    )
    for attempt in range(3):
        raw = complete(client, judge_model, _SYSTEM, prompt)
        raw = raw.strip()
        raw = re.sub(r'^```\w*\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        try:
            result = json.loads(raw)
            # Clamp score to [0, 1]
            result["score"] = max(0.0, min(1.0, float(result["score"])))
            return result
        except (json.JSONDecodeError, KeyError, ValueError):
            if attempt == 2:
                return {"score": 0.5, "reasoning": "judge parse error"}
    return {"score": 0.5, "reasoning": "judge parse error"}
