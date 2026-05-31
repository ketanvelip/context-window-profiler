import json
from datetime import datetime
from pathlib import Path

_RUNS_DIR = Path.home() / ".context-profiler" / "runs"


def save_run(data: dict) -> Path:
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _RUNS_DIR / f"{ts}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def list_runs(limit: int = 10) -> list[Path]:
    if not _RUNS_DIR.exists():
        return []
    return sorted(_RUNS_DIR.glob("*.json"), reverse=True)[:limit]


def load_run(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
