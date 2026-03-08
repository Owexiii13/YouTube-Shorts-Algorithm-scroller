from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Union

from pattern_engine import (
    ACTIONS,
    clamp,
    empty_action_bias,
    empty_action_weights,
    extract_patterns,
    load_base_model,
    prune_action_weights,
    reward_from_event,
    sanitize_action_score_map,
    sanitize_action_weight_maps,
    score_action_patterns,
)

MOODS = [
    "Neutral",
    "Happy",
    "Relaxed",
    "Focused",
    "Energetic",
    "Curious",
    "Creative",
    "Mad",
]

SETTINGS_SCHEMA_VERSION = 8
DEFAULT_LOCAL_MODEL_PATH = "Model.json"


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    peak = max(scores.values()) if scores else 0.0
    exp_scores = {action: math.exp(scores.get(action, 0.0) - peak) for action in ACTIONS}
    total = sum(exp_scores.values()) or 1.0
    return {action: exp_scores[action] / total for action in ACTIONS}


def _empty_action_counts() -> Dict[str, int]:
    return {action: 0 for action in ACTIONS}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ShortsAIModel:
    def __init__(
        self,
        data_file: str = "shorts_ai_data.json",
        base_model_file: str = "trained_model.json",
        user_model_file: str = DEFAULT_LOCAL_MODEL_PATH,
        export_model_file: str = DEFAULT_LOCAL_MODEL_PATH,
    ):
        self.data_file = Path(data_file)
        self.base_model_file = Path(base_model_file)
        self.user_model_file = Path(user_model_file)
        self.export_model_file = Path(export_model_file)
        self.base_model = load_base_model(self.base_model_file)
        self.buffer = deque(maxlen=40)  # type: Deque[Dict[str, Any]]
        self.learning_rate = 0.24
        self.last_mood_check = time.time()
        self.session_video_action_scores: Dict[str, Dict[str, float]] = {}
        self.session_recent_video_actions: Dict[str, Dict[str, Any]] = {}
        self.user_preferences = self._default_preferences()
        self.user_model = self._default_user_model()
        self.load_data()

    def _default_preferences(self) -> Dict[str, Any]:
        return {
            "schema_version": SETTINGS_SCHEMA_VERSION,
            "trusted_channels": set(),
            "blocked_channels": set(),
            "current_mood": "Neutral",
            "mood_last_changed": time.time(),
        }

    def _default_user_model(self) -> Dict[str, Any]:
        return {
            "action_weights": empty_action_weights(),
            "action_counts": _empty_action_counts(),
        }

    def _base_model_source_name(self) -> str:
        raw = str(self.base_model.get("source_path") or self.base_model_file)
        return Path(raw).name or self.base_model_file.name

    def _base_model_signature(self) -> str:
        payload = {
            "model_version": self.base_model.get("model_version"),
            "source": self._base_model_source_name(),
            "action_counts": self._sanitize_action_counts(self.base_model.get("action_counts")),
            "action_weights": prune_action_weights(self.base_model.get("action_weights", {}), max_size=12000),
        }
        digest = hashlib.blake2s(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            digest_size=12,
        ).hexdigest()
        return digest

    def _export_payload(self, watch_count: int = 0) -> Dict[str, Any]:
        counts = self._sanitize_action_counts(self.user_model.get("action_counts"))
        total = max(int(watch_count or 0), sum(counts.values()), 0)
        payload: Dict[str, Any] = {
            "model_role": "user_preference_delta",
            "base_model_version": self.base_model.get("model_version"),
            "base_model_source": self._base_model_source_name(),
            "base_model_signature": self._base_model_signature(),
            "action_weights": prune_action_weights(self.user_model.get("action_weights", {}), max_size=18000),
            "action_counts": counts,
        }
        if total > 0:
            payload["watch_count"] = total
        return payload

    def _sync_export_model(self, watch_count: int = 0) -> None:
        self.export_model_file.write_text(json.dumps(self._export_payload(watch_count), indent=2), encoding="utf-8")

    def _sanitize_action_counts(self, value: Any) -> Dict[str, int]:
        counts = _empty_action_counts()
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
        return counts

    def _compute_action_bias(self, counts: Dict[str, int]) -> Dict[str, float]:
        total = max(sum(max(0, int(counts.get(action, 0) or 0)) for action in ACTIONS), 1)
        bias = {}
        for action in ACTIONS:
            count = max(0, int(counts.get(action, 0) or 0))
            bias[action] = math.log((count + 1.0) / (total + len(ACTIONS)))
        return sanitize_action_score_map(bias, {"like": 0.0, "skip": 0.0})

    def _merge_model_fragment(
        self,
        target_weights: Dict[str, Dict[str, float]],
        target_counts: Dict[str, int],
        source_weights: Any,
        source_counts: Any,
        scale: float = 1.0,
    ) -> bool:
        cleaned_weights = sanitize_action_weight_maps(source_weights)
        cleaned_counts = self._sanitize_action_counts(source_counts)
        found = any(cleaned_weights.get(action) for action in ACTIONS) or any(cleaned_counts.values())
        if not found:
            return False
        for action in ACTIONS:
            target_counts[action] += cleaned_counts[action]
            action_weights = target_weights.setdefault(action, {})
            for feature, value in cleaned_weights.get(action, {}).items():
                action_weights[feature] = action_weights.get(feature, 0.0) + (float(value) * scale)
        return True

    def _load_user_model(self) -> bool:
        candidates = []
        seen = set()
        for path in (self.user_model_file, self.export_model_file):
            key = str(path)
            if key in seen or not path.exists():
                continue
            seen.add(key)
            candidates.append(path)

        for candidate in candidates:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            self.user_model = self._default_user_model()
            found = self._merge_model_fragment(
                self.user_model["action_weights"],
                self.user_model["action_counts"],
                payload.get("action_weights"),
                payload.get("action_counts"),
                1.0,
            )

            raw_mood_models = payload.get("mood_models") if isinstance(payload.get("mood_models"), dict) else payload.get("mood_preferences")
            if isinstance(raw_mood_models, dict):
                for mood in MOODS:
                    raw = raw_mood_models.get(mood, {}) if isinstance(raw_mood_models.get(mood), dict) else {}
                    if self._merge_model_fragment(
                        self.user_model["action_weights"],
                        self.user_model["action_counts"],
                        raw.get("action_weights") or raw.get("pattern_weights"),
                        raw.get("action_counts"),
                        0.55,
                    ):
                        found = True
            if found:
                return True
        return False

    def _legacy_model_from_settings(self, data: Dict[str, Any]) -> bool:
        found = self._merge_model_fragment(
            self.user_model["action_weights"],
            self.user_model["action_counts"],
            data.get("action_weights") or data.get("pattern_weights") or data.get("global_pattern_weights"),
            data.get("action_counts"),
            1.0,
        )
        raw_mood_prefs = data.get("mood_preferences", {}) if isinstance(data.get("mood_preferences"), dict) else {}
        for mood in MOODS:
            raw = raw_mood_prefs.get(mood, {}) if isinstance(raw_mood_prefs.get(mood), dict) else {}
            if self._merge_model_fragment(
                self.user_model["action_weights"],
                self.user_model["action_counts"],
                raw.get("action_weights") or raw.get("pattern_weights"),
                raw.get("action_counts"),
                0.55,
            ):
                found = True
        return found

    def load_data(self) -> None:
        settings: Dict[str, Any] = {}
        if self.data_file.exists():
            try:
                payload = json.loads(self.data_file.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    settings = payload
            except Exception:
                settings = {}

        self.user_preferences = self._default_preferences()
        self.user_preferences["trusted_channels"] = set(settings.get("trusted_channels", []))
        self.user_preferences["blocked_channels"] = set(settings.get("blocked_channels", []))
        current_mood = str(settings.get("current_mood", "Neutral") or "Neutral")
        self.user_preferences["current_mood"] = current_mood if current_mood in MOODS else "Neutral"
        self.user_preferences["mood_last_changed"] = _safe_float(settings.get("mood_last_changed", time.time()), time.time())

        self.user_model = self._default_user_model()
        loaded_user_model = self._load_user_model()
        if not loaded_user_model:
            self._legacy_model_from_settings(settings)
            self.save_user_model()

        needs_settings_save = not self.data_file.exists()
        if settings.get("schema_version") != SETTINGS_SCHEMA_VERSION:
            needs_settings_save = True
        if any(key in settings for key in ("action_weights", "pattern_weights", "global_pattern_weights", "video_action_scores", "video_scores", "previously_watched_videos", "mood_history", "action_counts", "mood_preferences")):
            needs_settings_save = True
        if needs_settings_save:
            self.save_data()
        self._sync_export_model()

    def save_data(self) -> None:
        data = {
            "schema_version": SETTINGS_SCHEMA_VERSION,
            "trusted_channels": sorted(self.user_preferences.get("trusted_channels", set())),
            "blocked_channels": sorted(self.user_preferences.get("blocked_channels", set())),
            "current_mood": self.user_preferences.get("current_mood", "Neutral"),
            "mood_last_changed": float(self.user_preferences.get("mood_last_changed", time.time())),
        }
        self.data_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save_user_model(self) -> None:
        payload = {
            "action_weights": prune_action_weights(self.user_model.get("action_weights", {}), max_size=18000),
            "action_counts": self._sanitize_action_counts(self.user_model.get("action_counts")),
        }
        self.user_model = payload
        watch_count = sum(payload["action_counts"].values())
        if self.user_model_file == self.export_model_file:
            combined_payload = self._export_payload(watch_count)
            self.user_model_file.write_text(json.dumps(combined_payload, indent=2), encoding="utf-8")
        else:
            self.user_model_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._sync_export_model(watch_count)

    def reload_base_model(self) -> None:
        self.base_model = load_base_model(self.base_model_file)

    def _record_context(
        self,
        title: str = "",
        description: str = "",
        captions: str = "",
        tags: Optional[List[str]] = None,
        duration_seconds: int = 0,
        watched_percent: float = 0.0,
    ) -> Dict[str, Any]:
        now = time.localtime()
        hour = now.tm_hour
        if 5 <= hour < 12:
            bucket = "morning"
        elif 12 <= hour < 17:
            bucket = "afternoon"
        elif 17 <= hour < 22:
            bucket = "evening"
        else:
            bucket = "night"

        watch_ratio = watched_percent / 100.0 if watched_percent > 1 else watched_percent
        return {
            "title": title,
            "description": description,
            "captions": captions,
            "tags": list(tags or []),
            "duration_seconds": max(0, int(duration_seconds or 0)),
            "day_of_week": max(0, min(6, now.tm_wday)),
            "time_of_day_bucket": bucket,
            "watch_percentage": clamp(float(watch_ratio or 0.0), 0.0, 1.0),
            "ctx_mood": self.get_current_mood(),
        }

    def get_current_mood(self) -> str:
        current_time = time.time()
        elapsed = current_time - float(self.user_preferences.get("mood_last_changed", current_time))
        current_mood = self.user_preferences.get("current_mood", "Neutral")

        if current_mood == "Mad":
            if elapsed > 600:
                self.set_mood("Happy")
            elif elapsed > 420:
                self.set_mood("Relaxed")
            elif elapsed > 180:
                self.set_mood("Neutral")

        return self.user_preferences.get("current_mood", "Neutral")

    def set_mood(self, mood: str) -> None:
        if mood not in MOODS:
            return
        old_mood = self.user_preferences.get("current_mood", "Neutral")
        if old_mood == mood:
            return
        self.user_preferences["current_mood"] = mood
        self.user_preferences["mood_last_changed"] = time.time()
        self.save_data()

    def _action_push(self, target_action: str, action: str) -> float:
        return 1.0 if action == target_action else -0.58

    def _update_action_weights(self, weights: Dict[str, Dict[str, float]], patterns: Dict[str, float], target_action: str, delta: float) -> None:
        for action in ACTIONS:
            direction = self._action_push(target_action, action)
            target_weights = weights.setdefault(action, {})
            for key, value in patterns.items():
                updated = target_weights.get(key, 0.0) + (delta * direction * value)
                target_weights[key] = clamp(updated, -6.0, 6.0)

    def _update_video_action_scores(self, store: Dict[str, Dict[str, float]], video_id: str, target_action: str, delta: float) -> None:
        current = sanitize_action_score_map(store.get(video_id))
        for action in ACTIONS:
            current[action] = clamp(current[action] + (delta * self._action_push(target_action, action) * 4.0), -18.0, 18.0)
        store[video_id] = current

    def _remember_video(self, video_id: str, action: str) -> None:
        self.session_recent_video_actions[video_id] = {"timestamp": time.time(), "action": action}
        if len(self.session_recent_video_actions) > 800:
            oldest = sorted(self.session_recent_video_actions.items(), key=lambda item: float(item[1].get("timestamp", 0.0) or 0.0))[:200]
            for key, _ in oldest:
                self.session_recent_video_actions.pop(key, None)
        if len(self.session_video_action_scores) > 1200:
            oldest_ids = list(self.session_video_action_scores.keys())[:300]
            for key in oldest_ids:
                self.session_video_action_scores.pop(key, None)

    def _event_learning_signal(self, event_type: str, watched_percent: float) -> Optional[Dict[str, Any]]:
        watch_ratio = watched_percent / 100.0 if watched_percent > 1 else watched_percent
        watch_ratio = clamp(float(watch_ratio or 0.0), 0.0, 1.0)
        event = str(event_type or "").strip().lower()

        if event == "user_like":
            return {"action": "like", "global_scale": 1.0, "video_scale": 1.25}
        if event == "like":
            return {"action": "like", "global_scale": 0.72, "video_scale": 0.95}
        if event in {"user_dislike", "dislike"}:
            return {"action": "skip", "global_scale": 1.0 if event == "user_dislike" else 0.78, "video_scale": 1.35}
        if event in {"manual_skip", "user_early_scroll_away"}:
            strength = 1.0 if watch_ratio < 0.18 else 0.82 if watch_ratio < 0.45 else 0.6
            return {"action": "skip", "global_scale": strength, "video_scale": strength + 0.25}
        if event == "undo_auto_like":
            return {"action": "skip", "global_scale": 1.2, "video_scale": 1.45}
        if event in {"undo_auto_dislike", "undo_ai_scroll", "completed"}:
            scale = 1.15 if event != "completed" else 0.28
            video_scale = 1.4 if event != "completed" else 0.8
            return {"action": "like", "global_scale": scale, "video_scale": video_scale}
        return None

    def process_event(
        self,
        video_id: str,
        channel_id: str,
        event_type: str,
        watched_percent: float,
        mood: str = "Neutral",
        title: str = "",
        description: str = "",
        captions: str = "",
        tags: Optional[List[str]] = None,
        duration_seconds: int = 0,
    ) -> Dict[str, Any]:
        if mood and mood in MOODS and mood != self.user_preferences.get("current_mood"):
            self.set_mood(mood)

        reward = reward_from_event(event_type, watched_percent)
        context = self._record_context(title, description, captions, tags, duration_seconds, watched_percent)
        patterns = extract_patterns(context, self.get_current_mood())
        signal = self._event_learning_signal(event_type, watched_percent)

        self.buffer.append(
            {
                "video_id": video_id,
                "channel_id": channel_id,
                "event_type": event_type,
                "watched_percent": watched_percent,
                "timestamp": time.time(),
                "reward": reward,
            }
        )

        if event_type == "trust_channel":
            self.user_preferences["trusted_channels"].add(channel_id)
            self.user_preferences["blocked_channels"].discard(channel_id)
        elif event_type == "untrust_channel":
            self.user_preferences["trusted_channels"].discard(channel_id)
        elif event_type == "block_channel":
            self.user_preferences["blocked_channels"].add(channel_id)
            self.user_preferences["trusted_channels"].discard(channel_id)
        elif event_type == "unblock_channel":
            self.user_preferences["blocked_channels"].discard(channel_id)

        if signal:
            self.user_model["action_counts"][signal["action"]] += 1

        if signal and patterns:
            learning_multiplier = 0.78 + (0.22 * clamp((watched_percent / 100.0) if watched_percent > 1 else watched_percent, 0.0, 1.0))
            pattern_delta = self.learning_rate * learning_multiplier * float(signal["global_scale"])
            self._update_action_weights(self.user_model["action_weights"], patterns, signal["action"], pattern_delta)
            self._update_video_action_scores(self.session_video_action_scores, video_id, signal["action"], float(signal["video_scale"]))

        if signal and (watched_percent > 10 or event_type == "undo_ai_scroll"):
            self._remember_video(video_id, signal["action"])

        self.save_user_model()
        self.save_data()
        return {"corrections_made": 0}

    def predict_action(
        self,
        video_id: str,
        channel_id: str,
        title: str = "",
        description: str = "",
        captions: str = "",
        tags: Optional[List[str]] = None,
        duration_seconds: int = 0,
        mood: str = "Neutral",
    ) -> Dict[str, Any]:
        if mood and mood in MOODS and mood != self.user_preferences.get("current_mood"):
            self.set_mood(mood)

        self.reload_base_model()
        context = self._record_context(title, description, captions, tags, duration_seconds, 0.0)
        patterns = extract_patterns(context, self.get_current_mood())

        base_scores, base_matches = score_action_patterns(patterns, self.base_model.get("action_weights", {}))
        user_scores, user_matches = score_action_patterns(patterns, self.user_model.get("action_weights", {}))
        base_bias = sanitize_action_score_map(self.base_model.get("action_bias"), {"like": 0.12, "skip": -0.12})
        user_bias = self._compute_action_bias(self.user_model.get("action_counts", {}))
        session_video_scores = sanitize_action_score_map(self.session_video_action_scores.get(video_id))

        combined = empty_action_bias()
        for action in ACTIONS:
            combined[action] = (
                base_bias[action]
                + user_bias[action]
                + (base_scores[action] * 0.94)
                + (user_scores[action] * 1.42)
                + (session_video_scores[action] * 0.18)
            )

        matched_patterns = base_matches + user_matches
        if matched_patterns == 0:
            combined["like"] += 0.16
        elif matched_patterns >= 8:
            combined["skip"] += 0.03

        recent = self.session_recent_video_actions.get(video_id)
        if isinstance(recent, dict):
            age_seconds = time.time() - _safe_float(recent.get("timestamp", 0.0), 0.0)
            if age_seconds < 3600:
                combined["skip"] += 0.48
                combined["like"] -= 0.12
            elif age_seconds > 604800:
                self.session_recent_video_actions.pop(video_id, None)

        if channel_id in self.user_preferences.get("trusted_channels", set()):
            combined["like"] += 0.75
        if channel_id in self.user_preferences.get("blocked_channels", set()):
            combined["skip"] += 1.4

        probabilities = _softmax(combined)
        skip_ready = probabilities["skip"] >= max(0.53, probabilities["like"] + 0.015)
        chosen_action = "skip" if skip_ready else "like"
        confidence = probabilities[chosen_action]

        return {
            "action": chosen_action,
            "confidence": round(clamp(confidence, 0.0, 0.99), 4),
            "matched_patterns": matched_patterns,
            "probabilities": {action: round(probabilities[action], 4) for action in ACTIONS},
            "components": {action: round(combined[action], 4) for action in ACTIONS},
            "mood_suggestion": self.suggest_mood_change(),
        }

    def predict_score(
        self,
        video_id: str,
        channel_id: str,
        title: str = "",
        description: str = "",
        captions: str = "",
        tags: Optional[List[str]] = None,
        duration_seconds: int = 0,
        mood: str = "Neutral",
    ) -> Dict[str, Any]:
        return self.predict_action(video_id, channel_id, title, description, captions, tags, duration_seconds, mood)

    def export_contribution_model(self, export_path: Union[str, Path], watch_count: int = 0) -> Dict[str, Any]:
        export_file = Path(export_path)
        payload = self._export_payload(watch_count)
        export_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if export_file != self.export_model_file:
            self._sync_export_model(int(payload.get("watch_count", 0) or 0))
        total = max(int(payload.get("watch_count", 0) or 0), 1)
        return {"model_file": export_file.name, "watch_count": total}

    def get_channel_status(self, channel_id: str) -> Dict[str, bool]:
        return {
            "trusted": channel_id in self.user_preferences.get("trusted_channels", set()),
            "blocked": channel_id in self.user_preferences.get("blocked_channels", set()),
        }

    def get_buffer_size(self) -> int:
        return len(self.buffer)

    def suggest_mood_change(self) -> str:
        current_time = time.time()
        if current_time - self.last_mood_check < 180:
            return self.user_preferences.get("current_mood", "Neutral")

        self.last_mood_check = current_time
        recent_events = [event for event in self.buffer if current_time - event["timestamp"] <= 240]
        if len(recent_events) < 4:
            return self.user_preferences.get("current_mood", "Neutral")

        reward_total = sum(float(event.get("reward", 0.0) or 0.0) for event in recent_events)
        if reward_total <= -1.5:
            return "Relaxed"
        if reward_total >= 1.5 and self.user_preferences.get("current_mood") == "Neutral":
            return "Curious"
        return self.user_preferences.get("current_mood", "Neutral")
