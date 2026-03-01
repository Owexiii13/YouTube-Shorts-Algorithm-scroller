import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

CHUNK_SIZE = 100
CHUNK_PATTERN = re.compile(r"^DataForTraining(\d+)\.json$")


@dataclass
class LogResult:
    chunk_file: str
    chunk_count: int
    completed_chunk: Optional[str] = None


class DataLogger:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.session_prompted_files: set[str] = set()
        self.pending_completed_chunks: List[str] = []
        self._refresh_pending_chunks()

    def _valid_unuploaded_chunks(self) -> List[Path]:
        files: List[Path] = []
        for path in self.base_dir.glob("DataForTraining*.json"):
            if "Uploaded" in path.name:
                continue
            if CHUNK_PATTERN.match(path.name):
                files.append(path)
        return sorted(files, key=lambda p: int(CHUNK_PATTERN.match(p.name).group(1)))

    def _chunk_number(self, path: Path) -> int:
        match = CHUNK_PATTERN.match(path.name)
        return int(match.group(1)) if match else 0

    def _load_entries(self, path: Path) -> List[Dict[str, Any]]:
        try:
            if not path.exists():
                return []
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
        except Exception:
            pass

        backup = path.with_suffix(".corrupt.json")
        try:
            path.rename(backup)
        except Exception:
            pass
        return []

    def _write_entries(self, path: Path, entries: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

    def _refresh_pending_chunks(self) -> None:
        for chunk in self._valid_unuploaded_chunks():
            entries = self._load_entries(chunk)
            if len(entries) >= CHUNK_SIZE and chunk.name not in self.pending_completed_chunks:
                self.pending_completed_chunks.append(chunk.name)

    def _active_chunk(self) -> Path:
        chunks = self._valid_unuploaded_chunks()
        if not chunks:
            path = self.base_dir / "DataForTraining1.json"
            self._write_entries(path, [])
            return path

        last = chunks[-1]
        entries = self._load_entries(last)
        if len(entries) >= CHUNK_SIZE:
            next_num = self._chunk_number(last) + 1
            new_path = self.base_dir / f"DataForTraining{next_num}.json"
            if not new_path.exists():
                self._write_entries(new_path, [])
            if last.name not in self.pending_completed_chunks:
                self.pending_completed_chunks.append(last.name)
            return new_path
        return last

    def _sanitize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            bucket = "morning"
        elif 12 <= hour < 17:
            bucket = "afternoon"
        elif 17 <= hour < 22:
            bucket = "evening"
        else:
            bucket = "night"

        user_action = payload.get("user_action", "neutral")
        if user_action not in {"liked", "disliked", "neutral"}:
            user_action = "neutral"

        algo_action = payload.get("algorithm_action", "none")
        if algo_action not in {"liked", "disliked", "scrolled", "none"}:
            algo_action = "none"

        tags = payload.get("tags") if isinstance(payload.get("tags"), list) else []
        tags = [str(t).strip()[:80] for t in tags if str(t).strip()][:20]

        subtitles = payload.get("subtitles_snippet")
        if subtitles is not None:
            subtitles = str(subtitles).strip()[:300] or None

        watch_percentage = float(payload.get("watch_percentage", 0.0) or 0.0)
        watch_percentage = max(0.0, min(1.0, watch_percentage))

        return {
            "title": str(payload.get("title", "") or "")[:300],
            "description": str(payload.get("description", "") or "")[:3000],
            "channel": str(payload.get("channel", "") or "")[:120],
            "tags": tags,
            "category": str(payload.get("category")).strip()[:80] if payload.get("category") else None,
            "subtitles_snippet": subtitles,
            "duration_seconds": int(max(0, payload.get("duration_seconds", 0) or 0)),
            "watch_percentage": watch_percentage,
            "user_action": user_action,
            "algorithm_action": algo_action,
            "day_of_week": now.weekday(),
            "time_of_day_bucket": bucket,
        }

    def log_video(self, payload: Dict[str, Any]) -> LogResult:
        active = self._active_chunk()
        entries = self._load_entries(active)
        entries.append(self._sanitize(payload))
        completed_chunk = None

        if len(entries) >= CHUNK_SIZE:
            entries = entries[:CHUNK_SIZE]
            completed_chunk = active.name
            if completed_chunk not in self.pending_completed_chunks:
                self.pending_completed_chunks.append(completed_chunk)

        self._write_entries(active, entries)

        if completed_chunk:
            next_num = self._chunk_number(active) + 1
            next_chunk = self.base_dir / f"DataForTraining{next_num}.json"
            if not next_chunk.exists():
                self._write_entries(next_chunk, [])

        return LogResult(active.name, len(entries), completed_chunk)

    def chunk_status(self) -> Dict[str, Any]:
        self._refresh_pending_chunks()
        pending = [name for name in self.pending_completed_chunks if name not in self.session_prompted_files]
        next_file = pending[0] if pending else None
        if next_file:
            self.session_prompted_files.add(next_file)
        return {
            "show_popup": bool(next_file),
            "chunk_file": next_file,
            "pending_count": len(pending),
            "privacy_notice": "Watch data only (titles/descriptions/likes/dislikes). No personal info is collected.",
        }

    def mark_uploaded(self, filename: str) -> Dict[str, str]:
        if "Uploaded" in filename:
            raise ValueError("File already marked as uploaded")
        if not CHUNK_PATTERN.match(filename):
            raise ValueError("Invalid chunk filename")

        src = self.base_dir / filename
        if not src.exists():
            raise FileNotFoundError(filename)

        target = src.with_name(src.stem + "Uploaded.json")
        src.rename(target)

        if filename in self.pending_completed_chunks:
            self.pending_completed_chunks.remove(filename)
        self.session_prompted_files.discard(filename)

        return {"original": src.name, "uploaded": target.name}
