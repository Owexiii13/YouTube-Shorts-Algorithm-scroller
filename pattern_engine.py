from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
HASHTAG_RE = re.compile(r"#([a-z0-9_]{2,})")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "did",
    "do", "for", "from", "get", "got", "had", "has", "have", "he", "her",
    "here", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "just", "me", "my", "of", "on", "or", "our", "out", "she", "so", "that",
    "the", "their", "them", "there", "they", "this", "to", "up", "was", "we",
    "what", "when", "where", "who", "why", "with", "you", "your",
}
DEFAULT_BASE_MODEL_PATH = "trained_model.json"
SEMVER_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)")
MAX_TEXT_TOKENS = 96
MAX_BIGRAMS = 64
MAX_TITLE_TOKENS = 24
MAX_TAG_TOKENS = 24
ACTIONS = ("like", "skip")
DEFAULT_ACTION_BIAS = {"like": 0.12, "skip": -0.12}
FEATURE_ID_RE = re.compile(r"^f_[0-9a-f]{24}$")


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").split())


def tokenize(value: Any) -> List[str]:
    text = normalize_space(value).lower()
    if not text:
        return []
    tokens = TOKEN_RE.findall(text)
    return [token for token in tokens if token not in STOPWORDS]


def unique_preserve_order(values: Iterable[str], limit: int) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def duration_bucket(duration_seconds: Any) -> str:
    try:
        seconds = int(duration_seconds or 0)
    except (TypeError, ValueError):
        seconds = 0
    if seconds <= 15:
        return "micro"
    if seconds <= 30:
        return "short"
    if seconds <= 45:
        return "medium"
    return "long"


def normalize_bucket(value: Any, allowed: Iterable[str], fallback: str) -> str:
    normalized = normalize_space(value).lower().replace(" ", "_")
    allowed_values = {item.lower() for item in allowed}
    if normalized in allowed_values:
        return normalized
    return fallback


def feature_id(key: Any) -> str:
    text = str(key or "").strip()
    if not text:
        return ""
    if FEATURE_ID_RE.match(text):
        return text
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=12).hexdigest()
    return f"f_{digest}"


def sanitize_feature_weight_map(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    cleaned: Dict[str, float] = {}
    for key, raw in value.items():
        feature_key = feature_id(key)
        if not feature_key:
            continue
        try:
            cleaned[feature_key] = float(raw)
        except (TypeError, ValueError):
            continue
    return cleaned


def _add_pattern(target: Dict[str, float], key: str, value: float) -> None:
    feature_key = feature_id(key)
    if not feature_key:
        return
    target[feature_key] = target.get(feature_key, 0.0) + value


def extract_patterns(record: Dict[str, Any], mood: Optional[str] = None) -> Dict[str, float]:
    patterns: Dict[str, float] = {}

    title = normalize_space(record.get("title"))
    description = normalize_space(record.get("description"))
    captions = normalize_space(record.get("captions") or record.get("subtitles_snippet"))

    title_tokens = unique_preserve_order(tokenize(title), MAX_TITLE_TOKENS)
    body_tokens = tokenize(" ".join(part for part in [title, description, captions] if part))
    text_tokens = unique_preserve_order(body_tokens, MAX_TEXT_TOKENS)

    for token in text_tokens:
        _add_pattern(patterns, f"tok:{token}", 1.0)
    for token in title_tokens:
        _add_pattern(patterns, f"title:{token}", 1.2)

    bigrams: List[str] = []
    for index in range(len(body_tokens) - 1):
        left = body_tokens[index]
        right = body_tokens[index + 1]
        if left == right:
            continue
        bigrams.append(f"{left}_{right}")
    for bigram in unique_preserve_order(bigrams, MAX_BIGRAMS):
        _add_pattern(patterns, f"bi:{bigram}", 0.8)

    hashtags = unique_preserve_order(HASHTAG_RE.findall(f"{title} {description}".lower()), 12)
    for hashtag in hashtags:
        _add_pattern(patterns, f"hash:{hashtag}", 0.9)

    tag_tokens: List[str] = []
    for raw_tag in list(record.get("tags") or [])[:20]:
        tag_tokens.extend(tokenize(raw_tag))
    for token in unique_preserve_order(tag_tokens, MAX_TAG_TOKENS):
        _add_pattern(patterns, f"tag:{token}", 0.85)

    category = normalize_space(record.get("category")).lower().replace(" ", "_")
    if category:
        _add_pattern(patterns, f"ctx:category:{category}", 0.35)

    if mood:
        normalized_mood = normalize_space(mood).lower().replace(" ", "_")
        if normalized_mood:
            _add_pattern(patterns, f"ctx:mood:{normalized_mood}", 0.28)

    _add_pattern(patterns, f"ctx:duration:{duration_bucket(record.get('duration_seconds'))}", 0.22)
    bucket = normalize_bucket(record.get("time_of_day_bucket"), {"morning", "afternoon", "evening", "night"}, "night")
    _add_pattern(patterns, f"ctx:time:{bucket}", 0.16)

    if not patterns:
        return {}

    norm = math.sqrt(sum(value * value for value in patterns.values()))
    if norm <= 0:
        return {}
    return {key: value / norm for key, value in patterns.items()}


def empty_action_bias(fill: float = 0.0) -> Dict[str, float]:
    return {action: float(fill) for action in ACTIONS}


def empty_action_weights() -> Dict[str, Dict[str, float]]:
    return {action: {} for action in ACTIONS}


def sanitize_number_map(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    cleaned: Dict[str, float] = {}
    for key, raw in value.items():
        try:
            cleaned[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return cleaned


def sanitize_action_score_map(value: Any, fallback: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    cleaned = empty_action_bias()
    if fallback:
        for action in ACTIONS:
            try:
                cleaned[action] = float(fallback.get(action, cleaned[action]))
            except (TypeError, ValueError, AttributeError):
                continue
    if isinstance(value, dict):
        for action in ACTIONS:
            try:
                cleaned[action] = float(value.get(action, cleaned[action]))
            except (TypeError, ValueError):
                continue
        if "watch" in value and "like" not in value:
            try:
                cleaned["like"] += float(value.get("watch", 0.0)) * 0.4
            except (TypeError, ValueError):
                pass
    return cleaned


def normalize_action_counts(value: Any, fallback_total: int = 0) -> Dict[str, int]:
    counts = {action: 0 for action in ACTIONS}
    if isinstance(value, dict):
        for action in ACTIONS:
            try:
                counts[action] = max(0, int(float(value.get(action, 0) or 0)))
            except (TypeError, ValueError):
                continue
        if "watch" in value and not counts["like"]:
            try:
                counts["like"] = max(counts["like"], int(float(value.get("watch", 0) or 0) * 0.35))
            except (TypeError, ValueError):
                pass
    if sum(counts.values()) <= 0 and fallback_total > 0:
        counts["like"] = max(1, fallback_total // 2)
        counts["skip"] = max(0, fallback_total - counts["like"])
    return counts


def action_bias_from_counts(value: Any, fallback: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    counts = normalize_action_counts(value)
    total = sum(counts.values())
    if total <= 0:
        return sanitize_action_score_map(fallback, DEFAULT_ACTION_BIAS)
    smoothing = float(len(ACTIONS))
    derived = {
        action: math.log((counts[action] + 1.0) / (total + smoothing))
        for action in ACTIONS
    }
    return sanitize_action_score_map(derived, fallback or DEFAULT_ACTION_BIAS)


def sanitize_action_weight_maps(value: Any) -> Dict[str, Dict[str, float]]:
    cleaned = empty_action_weights()
    if not isinstance(value, dict):
        return cleaned

    if any(isinstance(v, dict) for v in value.values()):
        for action in ACTIONS:
            cleaned[action] = sanitize_feature_weight_map(value.get(action))
        watch_weights = sanitize_feature_weight_map(value.get("watch"))
        for key, weight in watch_weights.items():
            cleaned["like"][key] = cleaned["like"].get(key, 0.0) + (weight * 0.35)
        return cleaned

    return legacy_scalar_weights_to_action_weights(value)


def legacy_scalar_bias_to_action_bias(bias: Any) -> Dict[str, float]:
    try:
        scalar = float(bias or 0.0)
    except (TypeError, ValueError):
        scalar = 0.0
    return {
        "like": DEFAULT_ACTION_BIAS["like"] + (clamp(scalar, -4.0, 4.0) * 0.12),
        "skip": DEFAULT_ACTION_BIAS["skip"] + (clamp(-scalar, -4.0, 4.0) * 0.12),
    }


def legacy_scalar_weights_to_action_weights(weights: Any) -> Dict[str, Dict[str, float]]:
    converted = empty_action_weights()
    for key, value in sanitize_feature_weight_map(weights).items():
        scalar = float(value)
        if scalar > 0:
            converted["like"][key] = clamp(scalar, -6.0, 6.0)
            converted["skip"][key] = clamp(-scalar * 0.35, -6.0, 6.0)
        elif scalar < 0:
            converted["skip"][key] = clamp(-scalar, -6.0, 6.0)
            converted["like"][key] = clamp(scalar * 0.35, -6.0, 6.0)
    return converted


def legacy_scalar_video_scores_to_action_scores(scores: Any) -> Dict[str, Dict[str, float]]:
    converted: Dict[str, Dict[str, float]] = {}
    for video_id, raw in sanitize_number_map(scores).items():
        scalar = float(raw)
        converted[video_id] = {
            "like": clamp(max(scalar, 0.0) * 0.8, -18.0, 18.0),
            "skip": clamp(max(-scalar, 0.0) * 0.85, -18.0, 18.0),
        }
    return converted


def score_action_patterns(patterns: Dict[str, float], action_weights: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, float], int]:
    scores = empty_action_bias()
    matched = 0
    for key, feature_value in patterns.items():
        hit = False
        for action in ACTIONS:
            weight = action_weights.get(action, {}).get(key)
            if weight is None:
                continue
            scores[action] += weight * feature_value
            hit = True
        if hit:
            matched += 1
    return scores, matched


def reward_from_event(event_type: str, watched_percent: float) -> float:
    event = normalize_space(event_type).lower()
    watch_ratio = watched_percent / 100.0 if watched_percent > 1 else watched_percent
    watch_ratio = clamp(float(watch_ratio or 0.0), 0.0, 1.0)

    if event in {"trust_channel", "untrust_channel", "block_channel", "unblock_channel", "mood_change"}:
        return 0.0
    if event in {"user_like", "like", "completed", "undo_auto_dislike", "undo_ai_scroll"}:
        return 0.9 + (0.2 * watch_ratio)
    if event in {"user_dislike", "dislike", "manual_skip", "user_early_scroll_away", "undo_auto_like"}:
        return -1.0 + (0.1 * watch_ratio)
    if event in {"auto_like_confirmed", "auto_dislike_confirmed", "user_unlike", "user_undislike", "ai_scroll", "ai_scroll_skip", "ai_scroll_completion"}:
        return 0.0
    return 0.0


def action_from_record(record: Dict[str, Any]) -> str:
    try:
        watch_ratio = float(record.get("watch_percentage", 0.0) or 0.0)
    except (TypeError, ValueError):
        watch_ratio = 0.0
    watch_ratio = clamp(watch_ratio, 0.0, 1.0)

    user_action = normalize_space(record.get("user_action")).lower()
    if user_action == "liked":
        return "like"
    if user_action == "disliked":
        return "skip"

    algorithm_action = normalize_space(record.get("algorithm_action")).lower()
    if algorithm_action == "scrolled" and watch_ratio < 0.45:
        return "skip"
    if watch_ratio < 0.35:
        return "skip"
    return "like"


def target_from_record(record: Dict[str, Any]) -> float:
    return 1.0 if action_from_record(record) == "like" else -1.0


def prune_weights(weights: Dict[str, float], max_size: int = 16000, min_abs: float = 0.015) -> Dict[str, float]:
    filtered = {key: float(value) for key, value in weights.items() if abs(float(value)) >= min_abs}
    if len(filtered) <= max_size:
        return filtered
    strongest = sorted(filtered.items(), key=lambda item: abs(item[1]), reverse=True)[:max_size]
    return dict(strongest)


def prune_action_weights(weights: Dict[str, Dict[str, float]], max_size: int = 12000, min_abs: float = 0.015) -> Dict[str, Dict[str, float]]:
    pruned = empty_action_weights()
    for action in ACTIONS:
        pruned[action] = prune_weights(weights.get(action, {}), max_size=max_size, min_abs=min_abs)
    return pruned


def parse_semver(value: Any) -> Optional[Tuple[int, int, int]]:
    text = normalize_space(value)
    if not text:
        return None
    match = SEMVER_RE.search(text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def looks_like_model_payload(payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(payload, dict):
        return False
    if isinstance(payload.get("action_weights"), dict):
        return True
    return isinstance(payload.get("weights"), dict) and "bias" in payload


def resolve_base_model_candidate(path: Union[str, Path]) -> Tuple[Path, Optional[Dict[str, Any]]]:
    model_path = Path(path)
    if model_path.name != DEFAULT_BASE_MODEL_PATH:
        payload = read_json_file(model_path) if model_path.exists() else None
        return model_path, payload

    candidates: List[Tuple[int, Tuple[int, int, int], str, Path, Dict[str, Any]]] = []
    for candidate in sorted(model_path.parent.glob("*.json")):
        payload = read_json_file(candidate)
        if not looks_like_model_payload(payload):
            continue
        filename_version = parse_semver(candidate.stem)
        if candidate.name != model_path.name and filename_version is None:
            continue
        version = filename_version or parse_semver(payload.get("model_version")) or (-1, -1, -1)
        trained_at = normalize_space(payload.get("trained_at"))
        candidates.append((1 if version != (-1, -1, -1) else 0, version, trained_at, candidate, payload))

    if not candidates:
        return model_path, None

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3].name))
    _, _, _, resolved_path, payload = candidates[-1]
    return resolved_path, payload


def empty_base_model() -> Dict[str, Any]:
    return {
        "action_bias": dict(DEFAULT_ACTION_BIAS),
        "action_weights": empty_action_weights(),
        "record_count": 0,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "epochs": 0,
        "feature_count": 0,
        "trainer": "scratch",
        "device": "cpu",
        "notes": "Automatically created scratch binary action model with hashed feature IDs. Safe default until a trained model exists.",
        "model_version": None,
    }


def persist_base_model(path: Union[str, Path], model: Dict[str, Any]) -> None:
    model_path = Path(path)
    counts = normalize_action_counts(model.get("action_counts"), int(model.get("record_count", 0) or 0))
    payload: Dict[str, Any] = {
        "action_weights": prune_action_weights(model.get("action_weights", {})),
        "action_counts": counts,
    }
    model_version = normalize_space(model.get("model_version"))
    if model_version:
        payload["model_version"] = model_version
    model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_base_model(path: Union[str, Path] = DEFAULT_BASE_MODEL_PATH) -> Dict[str, Any]:
    requested_path = Path(path)
    default_model = empty_base_model()
    model_path, data = resolve_base_model_candidate(requested_path)

    if data is None:
        model = dict(default_model)
        model["source_path"] = str(requested_path)
        return model

    is_action_model = isinstance(data.get("action_weights"), dict)
    action_counts = normalize_action_counts(data.get("action_counts"), int(data.get("record_count", 0) or 0))
    if is_action_model:
        action_weights = prune_action_weights(sanitize_action_weight_maps(data.get("action_weights")))
        action_bias = sanitize_action_score_map(data.get("action_bias"), action_bias_from_counts(action_counts, DEFAULT_ACTION_BIAS))
    else:
        action_weights = prune_action_weights(legacy_scalar_weights_to_action_weights(data.get("weights")))
        action_bias = legacy_scalar_bias_to_action_bias(data.get("bias", 0.0))
        action_counts = normalize_action_counts(data.get("action_counts"), int(data.get("record_count", 0) or 0))

    parsed_version = parse_semver(data.get("model_version")) or parse_semver(model_path.stem)
    model = {
        "action_bias": action_bias,
        "action_weights": action_weights,
        "action_counts": action_counts,
        "record_count": int(data.get("record_count", sum(action_counts.values())) or sum(action_counts.values())),
        "trained_at": data.get("trained_at") or default_model["trained_at"],
        "epochs": int(data.get("epochs", 0) or 0),
        "feature_count": int(data.get("feature_count", sum(len(weights) for weights in action_weights.values())) or sum(len(weights) for weights in action_weights.values())),
        "trainer": str(data.get("trainer", "scratch") or "scratch"),
        "device": str(data.get("device", "cpu") or "cpu"),
        "notes": data.get("notes") or default_model["notes"],
        "model_version": data.get("model_version") or ('.'.join(str(part) for part in parsed_version) if parsed_version else None),
        "source_path": str(model_path),
        "model_format": "action" if is_action_model else "legacy-score",
    }
    if model_path == requested_path and not is_action_model:
        try:
            persist_base_model(model_path, model)
        except Exception:
            pass
    return model
