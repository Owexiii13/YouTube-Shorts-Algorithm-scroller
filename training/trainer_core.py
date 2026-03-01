"""Model training and serialization."""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from .vocab_builder import entry_to_token_ids


class MediumModel:
    """A compact multinomial Naive Bayes text model."""

    def __init__(self, class_log_priors: Dict[str, float], token_log_probs: Dict[str, Dict[int, float]]):
        self.class_log_priors = class_log_priors
        self.token_log_probs = token_log_probs

    def predict(self, token_ids: List[int]) -> str:
        scores = {}
        token_counts = Counter(token_ids)
        for label, prior in self.class_log_priors.items():
            score = prior
            label_probs = self.token_log_probs[label]
            default_prob = label_probs["__default__"]
            for token_id, count in token_counts.items():
                score += count * label_probs.get(token_id, default_prob)
            scores[label] = score
        return max(scores, key=scores.get)


def train_medium_model(
    entries: Iterable[dict],
    vocab: Dict[str, int],
    class_boost: Dict[str, float] | None = None,
) -> MediumModel:
    samples = list(entries)
    class_boost = class_boost or {}

    class_counts: Counter[str] = Counter()
    token_counts = defaultdict(Counter)

    for entry in samples:
        label = entry["label"]
        class_counts[label] += class_boost.get(label, 1.0)
        token_counts[label].update(entry_to_token_ids(entry["text"], vocab))

    total_weight = sum(class_counts.values())
    num_classes = len(class_counts)

    class_log_priors = {
        label: math.log((count + 1.0) / (total_weight + num_classes))
        for label, count in class_counts.items()
    }

    token_log_probs: Dict[str, Dict[int, float]] = {}
    vocab_size = max(len(vocab), 1)
    for label, counts in token_counts.items():
        total_tokens = sum(counts.values())
        denominator = total_tokens + vocab_size
        label_probs: Dict[int, float] = {
            token_id: math.log((count + 1.0) / denominator)
            for token_id, count in counts.items()
        }
        label_probs["__default__"] = math.log(1.0 / denominator)
        token_log_probs[label] = label_probs

    return MediumModel(class_log_priors=class_log_priors, token_log_probs=token_log_probs)


def save_model(model: MediumModel, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as handle:
        pickle.dump(
            {
                "class_log_priors": model.class_log_priors,
                "token_log_probs": model.token_log_probs,
            },
            handle,
        )
