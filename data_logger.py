import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

PROMPT_EVERY = 100
EXPORT_FILENAME = "Model.json"
STATE_FILENAME = "contribution_state.json"


@dataclass
class LogResult:
    chunk_file: str
    chunk_count: int
    completed_chunk: Optional[str] = None


class DataLogger:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.state_path = self.base_dir / STATE_FILENAME
        self.session_prompted_milestones = set()
        self.state = self._load_state()

    def _default_state(self) -> Dict[str, int]:
        return {
            "watch_count": 0,
            "last_submitted_milestone": 0,
        }

    def _load_state(self) -> Dict[str, int]:
        if not self.state_path.exists():
            state = self._default_state()
            self._save_state(state)
            return state
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            state = self._default_state()
            self._save_state(state)
            return state
        if not isinstance(payload, dict):
            state = self._default_state()
            self._save_state(state)
            return state
        state = self._default_state()
        for key in state:
            try:
                state[key] = max(0, int(payload.get(key, state[key]) or 0))
            except (TypeError, ValueError):
                continue
        return state

    def _save_state(self, state: Optional[Dict[str, int]] = None) -> None:
        payload = state if state is not None else self.state
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def export_path(self) -> Path:
        return self.base_dir / EXPORT_FILENAME

    @property
    def watch_count(self) -> int:
        return int(self.state.get("watch_count", 0) or 0)

    def _current_milestone(self) -> int:
        return self.watch_count // PROMPT_EVERY

    def _pending_milestones(self) -> int:
        return max(0, self._current_milestone() - int(self.state.get("last_submitted_milestone", 0) or 0))

    def _should_prompt(self) -> bool:
        milestone = self._current_milestone()
        if milestone <= int(self.state.get("last_submitted_milestone", 0) or 0):
            return False
        return milestone not in self.session_prompted_milestones

    def log_video(self, payload: Dict[str, Any]) -> LogResult:
        self.state["watch_count"] = self.watch_count + 1
        self._save_state()
        completed_chunk = None
        if self._should_prompt():
            milestone = self._current_milestone()
            self.session_prompted_milestones.add(milestone)
            completed_chunk = EXPORT_FILENAME
        return LogResult(EXPORT_FILENAME, self.watch_count, completed_chunk)

    def chunk_status(self) -> Dict[str, Any]:
        show_popup = self._should_prompt()
        if show_popup:
            self.session_prompted_milestones.add(self._current_milestone())
        return {
            "show_popup": show_popup,
            "chunk_file": EXPORT_FILENAME if show_popup else None,
            "pending_count": self._pending_milestones(),
            "privacy_notice": "Upload a copy of Model.json in the Data Contribution template to help train the shared base model. Keep Model.json after uploading because your app still uses it as your personal local model. 100 watched videos is a recommended starting point, not a requirement. Do not upload shorts_ai_data.json or trained_model.json.",
        }

    def mark_uploaded(self, filename: str) -> Dict[str, str]:
        if filename != EXPORT_FILENAME:
            raise ValueError("Invalid contribution model filename")
        export_file = self.export_path()
        if not export_file.exists():
            raise FileNotFoundError(filename)
        milestone = self._current_milestone()
        self.state["last_submitted_milestone"] = max(int(self.state.get("last_submitted_milestone", 0) or 0), milestone)
        self._save_state()
        return {"model_file": export_file.name, "watch_count": str(self.watch_count)}
