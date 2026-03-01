import json
import os
import re
from datetime import datetime
from typing import Dict, Iterable, List, Optional

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


class FeatureBuilder:
    """Build fixed-size feature vectors from video metadata and context buckets."""

    def __init__(self, vocab_path: str = "vocab.json"):
        self.vocab_path = vocab_path
        self.token_to_idx = self._load_vocab(vocab_path)
        self.vocab_size = len(self.token_to_idx)

        # Bucket maps are fixed-size and deterministic.
        self.duration_buckets = ["unknown", "very_short", "short", "medium", "long", "very_long"]
        self.day_buckets = [
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
        ]
        self.time_buckets = ["overnight", "morning", "afternoon", "evening"]

        self.duration_offset = self.vocab_size
        self.day_offset = self.duration_offset + len(self.duration_buckets)
        self.time_offset = self.day_offset + len(self.day_buckets)
        self.total_dim = self.time_offset + len(self.time_buckets)

    def _load_vocab(self, vocab_path: str) -> Dict[str, int]:
        if not os.path.exists(vocab_path):
            return {}

        with open(vocab_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Supported shapes:
        # 1) {"word": 0, ...}
        # 2) {"token_to_idx": {...}}
        # 3) ["word1", "word2", ...]
        if isinstance(raw, dict) and "token_to_idx" in raw and isinstance(raw["token_to_idx"], dict):
            base = raw["token_to_idx"]
        elif isinstance(raw, dict):
            base = raw
        elif isinstance(raw, list):
            base = {str(token): idx for idx, token in enumerate(raw)}
        else:
            return {}

        out = {}
        for token, idx in base.items():
            try:
                out[str(token).lower()] = int(idx)
            except (TypeError, ValueError):
                continue
        return out

    def _tokenize(self, *text_parts: Optional[str]) -> Iterable[str]:
        combined = " ".join([(part or "") for part in text_parts]).lower()
        return TOKEN_PATTERN.findall(combined)

    def _duration_bucket(self, duration_seconds: Optional[float]) -> str:
        if duration_seconds is None or duration_seconds <= 0:
            return "unknown"
        if duration_seconds <= 15:
            return "very_short"
        if duration_seconds <= 30:
            return "short"
        if duration_seconds <= 60:
            return "medium"
        if duration_seconds <= 120:
            return "long"
        return "very_long"

    def _day_bucket(self, timestamp: Optional[float] = None) -> str:
        dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
        return self.day_buckets[dt.weekday()]

    def _time_bucket(self, timestamp: Optional[float] = None) -> str:
        dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
        hour = dt.hour
        if 0 <= hour < 6:
            return "overnight"
        if 6 <= hour < 12:
            return "morning"
        if 12 <= hour < 18:
            return "afternoon"
        return "evening"

    def build_vector(
        self,
        title: str = "",
        description: str = "",
        subtitles: str = "",
        category: str = "",
        duration_seconds: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> List[float]:
        vec = [0.0] * self.total_dim

        for token in self._tokenize(title, description, subtitles, category):
            idx = self.token_to_idx.get(token)
            if idx is not None and 0 <= idx < self.vocab_size:
                vec[idx] += 1.0

        # Binary context buckets.
        duration_bucket = self._duration_bucket(duration_seconds)
        day_bucket = self._day_bucket(timestamp)
        time_bucket = self._time_bucket(timestamp)

        vec[self.duration_offset + self.duration_buckets.index(duration_bucket)] = 1.0
        vec[self.day_offset + self.day_buckets.index(day_bucket)] = 1.0
        vec[self.time_offset + self.time_buckets.index(time_bucket)] = 1.0

        return vec
