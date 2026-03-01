"""Validation and feature extraction utilities."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_']+")


class DataValidationError(ValueError):
    """Raised when the provided training set does not meet minimum expectations."""


def normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single raw entry into the canonical training schema."""
    if not isinstance(entry, dict):
        raise DataValidationError("Each entry must be a JSON object.")

    text = str(entry.get("text", "")).strip()
    label = str(entry.get("label", "")).strip()

    if not text:
        raise DataValidationError("Entry is missing non-empty `text`.")
    if not label:
        raise DataValidationError("Entry is missing non-empty `label`.")

    return {
        "text": text,
        "label": label,
    }


def validate_entries(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and normalize all entries."""
    validated = [normalize_entry(entry) for entry in entries]

    labels = {entry["label"] for entry in validated}
    if len(labels) < 2:
        raise DataValidationError("At least 2 labels/classes are required for training.")

    return validated


def tokenize(text: str) -> List[str]:
    """Convert free text into simple lowercase tokens."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]
