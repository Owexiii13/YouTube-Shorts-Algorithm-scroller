"""Model compression helpers."""

from __future__ import annotations

from .train import MediumModel


def compress_model(model: MediumModel, keep_threshold: float = -12.0, ndigits: int = 5) -> MediumModel:
    """Drop very low-probability token params and round floats for compactness."""
    compressed_probs = {}
    for label, probs in model.token_log_probs.items():
        default_prob = round(probs["__default__"], ndigits)
        slim = {
            token_id: round(value, ndigits)
            for token_id, value in probs.items()
            if token_id != "__default__" and value >= keep_threshold
        }
        slim["__default__"] = default_prob
        compressed_probs[label] = slim

    compressed_priors = {label: round(value, ndigits) for label, value in model.class_log_priors.items()}
    return MediumModel(compressed_priors, compressed_probs)
