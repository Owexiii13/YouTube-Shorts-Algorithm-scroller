"""Builds and persists training vocabulary."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from .features import tokenize


def build_vocab(
    texts: Iterable[str],
    max_vocab_size: int,
    min_token_count: int,
) -> Dict[str, int]:
    token_counts = Counter()
    for text in texts:
        token_counts.update(tokenize(text))

    most_common = [
        token
        for token, count in token_counts.most_common(max_vocab_size)
        if count >= min_token_count
    ]
    return {token: idx for idx, token in enumerate(most_common)}


def save_vocab(vocab: Dict[str, int], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(vocab, indent=2, sort_keys=True))


def entry_to_token_ids(text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab[token] for token in tokenize(text) if token in vocab]
