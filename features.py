import json
import re
from pathlib import Path
from typing import Dict, List

DEFAULT_DIM = 535


def load_vocab(path: str = "vocab.json") -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict) and isinstance(data.get("vocab"), list):
            return [str(x) for x in data["vocab"]]
    except Exception:
        return []
    return []


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def extract_features(video: Dict, vocab: List[str]) -> List[float]:
    text = " ".join([
        str(video.get("title", "")),
        str(video.get("description", "")),
        str(video.get("subtitles_snippet", "") or ""),
    ])
    words = set(_tokenize(text))

    vocab = vocab[:500]
    if len(vocab) < 500:
        vocab = vocab + [f"__pad_{i}" for i in range(500 - len(vocab))]

    word_vec = [1.0 if token in words else 0.0 for token in vocab]

    categories = [
        "Film", "Autos", "Music", "Pets", "Sports", "Travel", "Gaming", "People",
        "Comedy", "Entertainment", "News", "Howto", "Education", "Science", "Nonprofits",
        "Tech", "Lifestyle", "Food", "Fashion", "Other"
    ]
    cat = str(video.get("category") or "Other")
    cat_vec = [1.0 if cat == c else 0.0 for c in categories]
    if sum(cat_vec) == 0:
        cat_vec[-1] = 1.0

    d = int(video.get("duration_seconds", 0) or 0)
    duration_vec = [0.0, 0.0, 0.0, 0.0]
    if d <= 15:
        duration_vec[0] = 1.0
    elif d <= 30:
        duration_vec[1] = 1.0
    elif d <= 45:
        duration_vec[2] = 1.0
    else:
        duration_vec[3] = 1.0

    day = int(video.get("day_of_week", 0) or 0)
    day = max(0, min(6, day))
    day_vec = [1.0 if i == day else 0.0 for i in range(7)]

    tb = str(video.get("time_of_day_bucket", "night"))
    time_labels = ["morning", "afternoon", "evening", "night"]
    time_vec = [1.0 if tb == label else 0.0 for label in time_labels]
    if sum(time_vec) == 0:
        time_vec[-1] = 1.0

    watch = float(video.get("watch_percentage", 0.5) or 0.5)
    watch = max(0.0, min(1.0, watch))

    features = word_vec + cat_vec + duration_vec + day_vec + time_vec + [watch]

    if len(features) < DEFAULT_DIM:
        features += [0.0] * (DEFAULT_DIM - len(features))

    return features[:DEFAULT_DIM]
