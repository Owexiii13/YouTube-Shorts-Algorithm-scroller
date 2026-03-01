"""End-to-end training orchestration pipeline."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from .compress import compress_model
from .config import (
    DEFAULT_DATA_FILE,
    DEFAULT_VOCAB_FILE,
    MAX_VOCAB_SIZE,
    METRICS_HISTORY_FILE,
    MIN_ENTRIES,
    MIN_TOKEN_COUNT,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TRAIN_RATIO,
)
from .evaluate import append_metrics_history, compare_with_previous, evaluate_model
from .features import DataValidationError, validate_entries
from .train import save_model, train_medium_model
from .vocab_builder import build_vocab, save_vocab


def load_entries(data_file: Path) -> List[dict]:
    raw_data = json.loads(data_file.read_text())
    if isinstance(raw_data, dict):
        raw_data = raw_data.get("entries", [])
    if not isinstance(raw_data, list):
        raise DataValidationError("Training data must be a list of {text,label} entries.")

    entries = validate_entries(raw_data)

    if len(entries) < MIN_ENTRIES:
        raise DataValidationError(
            f"Minimum data requirement not met: got {len(entries)} entries, require >= {MIN_ENTRIES}."
        )
    return entries


def stratified_split(entries: List[dict], train_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["label"]].append(entry)

    rng = random.Random(seed)
    train_set, eval_set = [], []

    for label_entries in grouped.values():
        rng.shuffle(label_entries)
        split_idx = max(1, int(len(label_entries) * train_ratio))
        split_idx = min(split_idx, len(label_entries) - 1)
        train_set.extend(label_entries[:split_idx])
        eval_set.extend(label_entries[split_idx:])

    rng.shuffle(train_set)
    rng.shuffle(eval_set)
    return train_set, eval_set


def next_model_version() -> int:
    versions = []
    for path in MODELS_DIR.glob("Model*.pt"):
        suffix = path.stem.replace("Model", "")
        if suffix.isdigit():
            versions.append(int(suffix))
    return (max(versions) + 1) if versions else 1


def emit_report(report_path: Path, payload: Dict[str, object]) -> None:
    lines = [
        f"# Training Report v{payload['version']}",
        "",
        f"- Timestamp: {payload['timestamp']}",
        f"- Entries: {payload['entries']}",
        f"- Accuracy: {payload['metrics']['accuracy']:.4f}",
        f"- Previous Accuracy: {payload['comparison']['previous_accuracy']}",
        f"- Delta Accuracy: {payload['comparison']['delta_accuracy']}",
        f"- Weak Labels: {', '.join(payload['metrics']['weak_labels']) or 'None'}",
    ]
    report_path.write_text("\n".join(lines) + "\n")


def run_pipeline(data_file: Path) -> Dict[str, object]:
    entries = load_entries(data_file)
    train_set, eval_set = stratified_split(entries, TRAIN_RATIO, RANDOM_SEED)

    vocab = build_vocab(
        texts=[entry["text"] for entry in train_set],
        max_vocab_size=MAX_VOCAB_SIZE,
        min_token_count=MIN_TOKEN_COUNT,
    )
    save_vocab(vocab, DEFAULT_VOCAB_FILE)

    model = train_medium_model(train_set, vocab)
    compressed_model = compress_model(model)
    metrics = evaluate_model(compressed_model, eval_set, vocab)

    if metrics["weak_labels"]:
        class_boost = {label: 1.8 for label in metrics["weak_labels"]}
        fine_tuned = train_medium_model(train_set, vocab, class_boost=class_boost)
        compressed_model = compress_model(fine_tuned)
        metrics = evaluate_model(compressed_model, eval_set, vocab)

    version = next_model_version()
    model_path = MODELS_DIR / f"Model{version}.pt"
    save_model(compressed_model, model_path)

    comparison = compare_with_previous(metrics, METRICS_HISTORY_FILE)

    payload = {
        "version": version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "entries": len(entries),
        "artifacts": {
            "model": str(model_path),
            "vocab": str(DEFAULT_VOCAB_FILE),
        },
        "metrics": metrics,
        "comparison": comparison,
    }

    append_metrics_history(METRICS_HISTORY_FILE, {"version": version, "accuracy": metrics["accuracy"]})

    report_path = REPORTS_DIR / f"report_v{version}.md"
    emit_report(report_path, payload)
    payload["report"] = str(report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and package a distributable model.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    args = parser.parse_args()

    output = run_pipeline(args.data_file)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
