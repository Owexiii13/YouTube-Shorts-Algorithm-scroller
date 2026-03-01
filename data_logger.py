import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REQUIRED_FIELDS = [
    "title",
    "description",
    "channel",
    "tags",
    "category",
    "subtitles_snippet",
    "duration_seconds",
    "watch_percentage",
    "user_action",
    "algorithm_action",
    "day_of_week",
    "time_of_day_bucket",
]

MAX_ENTRIES_PER_CHUNK = 100
CHUNK_PATTERN = re.compile(r"^DataForTraining(\d+)\.json$")


class DataLogger:
    def __init__(self, working_directory: Optional[Path] = None):
        self.working_directory = Path(working_directory or Path.cwd())
        self.should_prompt_submission = False
        self.completed_chunk: Optional[str] = None
        self.last_recovery: Dict[str, str] = {}

    def _chunk_index(self, file_name: str) -> Optional[int]:
        if "Uploaded" in file_name:
            return None
        match = CHUNK_PATTERN.match(file_name)
        if not match:
            return None
        return int(match.group(1))

    def scan_chunk_files(self) -> List[Path]:
        chunk_files: List[Path] = []
        for path in self.working_directory.iterdir():
            if not path.is_file():
                continue
            if self._chunk_index(path.name) is not None:
                chunk_files.append(path)
        return sorted(chunk_files, key=lambda p: self._chunk_index(p.name) or 0)

    def get_or_create_active_chunk(self) -> Path:
        chunk_files = self.scan_chunk_files()
        if not chunk_files:
            chunk_path = self.working_directory / "DataForTraining1.json"
            self._write_chunk(chunk_path, [])
            return chunk_path

        for chunk_path in reversed(chunk_files):
            entries, _ = self._load_chunk_with_recovery(chunk_path)
            if len(entries) < MAX_ENTRIES_PER_CHUNK:
                return chunk_path

        highest_index = self._chunk_index(chunk_files[-1].name) or 1
        next_chunk = self.working_directory / f"DataForTraining{highest_index + 1}.json"
        self._write_chunk(next_chunk, [])
        return next_chunk

    def _backup_and_reset_malformed(self, chunk_path: Path) -> Dict[str, str]:
        stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        backup_path = chunk_path.with_suffix(f".malformed.{stamp}.bak")
        chunk_path.rename(backup_path)
        self._write_chunk(chunk_path, [])
        return {
            "recovered": "true",
            "backup": backup_path.name,
            "reason": "malformed_json",
        }

    def _sanitize_entry(self, raw_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(raw_entry, dict):
            return None

        sanitized: Dict[str, Any] = {}
        for field in REQUIRED_FIELDS:
            if field not in raw_entry:
                return None
            value = raw_entry[field]
            if field == "tags":
                if not isinstance(value, list):
                    return None
                sanitized[field] = [str(tag).strip() for tag in value if str(tag).strip()]
            elif field in {"duration_seconds", "watch_percentage"}:
                try:
                    sanitized[field] = float(value)
                except (TypeError, ValueError):
                    return None
            else:
                sanitized[field] = str(value).strip()

        return sanitized

    def _load_chunk_with_recovery(self, chunk_path: Path) -> (List[Dict[str, Any]], Dict[str, str]):
        if not chunk_path.exists():
            self._write_chunk(chunk_path, [])
            return [], {}

        try:
            raw_data = json.loads(chunk_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            recovery = self._backup_and_reset_malformed(chunk_path)
            self.last_recovery = {"chunk": chunk_path.name, **recovery}
            return [], recovery

        if not isinstance(raw_data, list):
            stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            backup_path = chunk_path.with_suffix(f".invalid-structure.{stamp}.bak")
            chunk_path.rename(backup_path)
            self._write_chunk(chunk_path, [])
            recovery = {
                "recovered": "true",
                "backup": backup_path.name,
                "reason": "invalid_structure",
            }
            self.last_recovery = {"chunk": chunk_path.name, **recovery}
            return [], recovery

        valid_entries: List[Dict[str, Any]] = []
        dropped = 0
        for item in raw_data:
            sanitized = self._sanitize_entry(item)
            if sanitized is None:
                dropped += 1
                continue
            valid_entries.append(sanitized)

        recovery_info: Dict[str, str] = {}
        if dropped:
            self._write_chunk(chunk_path, valid_entries)
            recovery_info = {
                "recovered": "true",
                "reason": "dropped_invalid_entries",
                "dropped_entries": str(dropped),
            }
            self.last_recovery = {"chunk": chunk_path.name, **recovery_info}

        return valid_entries, recovery_info

    def _write_chunk(self, chunk_path: Path, entries: List[Dict[str, Any]]) -> None:
        chunk_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")

    def append_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        sanitized_entry = self._sanitize_entry(entry)
        if sanitized_entry is None:
            raise ValueError("Entry must include all required fields with valid value types")

        chunk_path = self.get_or_create_active_chunk()
        entries, recovery_info = self._load_chunk_with_recovery(chunk_path)

        if len(entries) >= MAX_ENTRIES_PER_CHUNK:
            self.should_prompt_submission = True
            self.completed_chunk = chunk_path.name
            current_index = self._chunk_index(chunk_path.name) or 1
            chunk_path = self.working_directory / f"DataForTraining{current_index + 1}.json"
            entries = []

        entries.append(sanitized_entry)
        self._write_chunk(chunk_path, entries)

        prompt = False
        if len(entries) >= MAX_ENTRIES_PER_CHUNK:
            self.should_prompt_submission = True
            self.completed_chunk = chunk_path.name
            prompt = True

        visible_recovery = recovery_info
        if not visible_recovery and self.last_recovery.get("chunk") == chunk_path.name:
            visible_recovery = {k: v for k, v in self.last_recovery.items() if k != "chunk"}

        return {
            "active_chunk": chunk_path.name,
            "entry_count": len(entries),
            "max_entries": MAX_ENTRIES_PER_CHUNK,
            "prompt_submission": prompt,
            "completed_chunk": self.completed_chunk,
            "recovery": visible_recovery,
        }

    def get_status(self) -> Dict[str, Any]:
        chunk_path = self.get_or_create_active_chunk()
        entries, recovery_info = self._load_chunk_with_recovery(chunk_path)
        visible_recovery = recovery_info
        if not visible_recovery and self.last_recovery.get("chunk") == chunk_path.name:
            visible_recovery = {k: v for k, v in self.last_recovery.items() if k != "chunk"}

        return {
            "active_chunk": chunk_path.name,
            "entry_count": len(entries),
            "max_entries": MAX_ENTRIES_PER_CHUNK,
            "prompt_submission": self.should_prompt_submission,
            "completed_chunk": self.completed_chunk,
            "recovery": visible_recovery,
        }


__all__ = ["DataLogger", "REQUIRED_FIELDS", "MAX_ENTRIES_PER_CHUNK"]
