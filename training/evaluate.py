"""Evaluation and model comparison routines."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from .train import MediumModel
from .vocab_builder import entry_to_token_ids


def evaluate_model(model: MediumModel, entries: Iterable[dict], vocab: Dict[str, int]) -> Dict[str, object]:
    entries = list(entries)
    if not entries:
        return {"accuracy": 0.0, "per_label": {}, "weak_labels": []}

    totals = Counter()
    correct = Counter()

    for entry in entries:
        label = entry["label"]
        prediction = model.predict(entry_to_token_ids(entry["text"], vocab))
        totals[label] += 1
        if prediction == label:
            correct[label] += 1

    total_correct = sum(correct.values())
    total_seen = sum(totals.values())
    per_label = {
        label: {
            "support": totals[label],
            "recall": (correct[label] / totals[label]) if totals[label] else 0.0,
        }
        for label in totals
    }

    weak_labels = [label for label, stats in per_label.items() if stats["recall"] < 0.65]

    return {
        "accuracy": total_correct / total_seen if total_seen else 0.0,
        "per_label": per_label,
        "weak_labels": weak_labels,
    }


def compare_with_previous(current_metrics: Dict[str, object], history_file: Path) -> Dict[str, float | None]:
    if not history_file.exists():
        return {"previous_accuracy": None, "delta_accuracy": None}

    history = json.loads(history_file.read_text())
    if not history:
        return {"previous_accuracy": None, "delta_accuracy": None}

    previous_accuracy = history[-1].get("accuracy")
    if previous_accuracy is None:
        return {"previous_accuracy": None, "delta_accuracy": None}

    delta = current_metrics["accuracy"] - previous_accuracy
    return {"previous_accuracy": previous_accuracy, "delta_accuracy": delta}


def append_metrics_history(history_file: Path, record: Dict[str, object]) -> None:
    history_file.parent.mkdir(parents=True, exist_ok=True)
    if history_file.exists():
        history = json.loads(history_file.read_text())
    else:
        history = []
    history.append(record)
    history_file.write_text(json.dumps(history, indent=2))
