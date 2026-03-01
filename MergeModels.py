#!/usr/bin/env python3
import argparse
import copy
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(f"[MergeModels] PyTorch is required: {exc}")


MODEL_PATTERN = re.compile(r"^Model(?P<version>\d+(?:\.\d+)*)(?P<legacy>Legacy)?\.pt$")
USER_MODEL = "UserModel.pt"
USER_BACKUP = "UserModel_before_merge.pt"
DATA_PATTERN = "DataForTraining*.json"


@dataclass
class ModelFile:
    path: Path
    version_str: str
    version_key: Tuple[int, ...]
    is_legacy: bool


def version_to_key(version: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def parse_model_file(path: Path) -> Optional[ModelFile]:
    match = MODEL_PATTERN.match(path.name)
    if not match:
        return None
    version = match.group("version")
    return ModelFile(
        path=path,
        version_str=version,
        version_key=version_to_key(version),
        is_legacy=bool(match.group("legacy")),
    )


def discover_models(root: Path) -> Tuple[List[ModelFile], List[ModelFile]]:
    base_models: List[ModelFile] = []
    legacy_models: List[ModelFile] = []

    for file_path in root.glob("Model*.pt"):
        parsed = parse_model_file(file_path)
        if parsed is None:
            continue
        if parsed.is_legacy:
            legacy_models.append(parsed)
        else:
            base_models.append(parsed)

    base_models.sort(key=lambda item: item.version_key)
    legacy_models.sort(key=lambda item: item.version_key)
    return base_models, legacy_models


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and all(hasattr(v, "shape") for v in value.values()):
                return value
            if isinstance(value, nn.Module):
                return value.state_dict()

        if checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
            return checkpoint

    raise ValueError("Unsupported checkpoint format; could not locate state_dict")


def validate_state_dict_compatibility(base_sd: Dict[str, torch.Tensor], user_sd: Dict[str, torch.Tensor]) -> None:
    base_keys = set(base_sd.keys())
    user_keys = set(user_sd.keys())

    missing_in_user = sorted(base_keys - user_keys)
    missing_in_base = sorted(user_keys - base_keys)

    if missing_in_user or missing_in_base:
        msg_parts = []
        if missing_in_user:
            msg_parts.append(f"Missing in user model: {missing_in_user[:5]}{'...' if len(missing_in_user) > 5 else ''}")
        if missing_in_base:
            msg_parts.append(f"Missing in base model: {missing_in_base[:5]}{'...' if len(missing_in_base) > 5 else ''}")
        raise ValueError("State dict keys are incompatible. " + " | ".join(msg_parts))

    for key in sorted(base_keys):
        base_tensor = base_sd[key]
        user_tensor = user_sd[key]
        if getattr(base_tensor, "shape", None) != getattr(user_tensor, "shape", None):
            raise ValueError(
                f"Shape mismatch for '{key}': base={tuple(base_tensor.shape)} user={tuple(user_tensor.shape)}"
            )


def merge_state_dicts(
    base_sd: Dict[str, torch.Tensor],
    user_sd: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for key in user_sd:
        user_tensor = user_sd[key]
        base_tensor = base_sd[key]

        if torch.is_tensor(user_tensor) and user_tensor.is_floating_point():
            merged[key] = alpha * user_tensor + (1.0 - alpha) * base_tensor.to(user_tensor.dtype)
        else:
            merged[key] = user_tensor.clone() if torch.is_tensor(user_tensor) else copy.deepcopy(user_tensor)
    return merged


def inject_state_dict(checkpoint: Any, new_state_dict: Dict[str, torch.Tensor]) -> Any:
    if isinstance(checkpoint, nn.Module):
        checkpoint.load_state_dict(new_state_dict)
        return checkpoint

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint[key] = new_state_dict
                return checkpoint

        if "model" in checkpoint and isinstance(checkpoint["model"], nn.Module):
            checkpoint["model"].load_state_dict(new_state_dict)
            return checkpoint

        if checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
            return new_state_dict

    return new_state_dict


def load_checkpoint(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def materialize_model(checkpoint: Any) -> Optional[nn.Module]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint

    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("model"), nn.Module):
            return checkpoint["model"]

    return None


def get_recent_training_entries(root: Path, limit: int = 20) -> Tuple[Optional[Path], List[Any]]:
    candidates = list(root.glob(DATA_PATTERN))
    if not candidates:
        return None, []

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    most_recent = candidates[0]

    with most_recent.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        for key in ("entries", "data", "samples", "records"):
            if isinstance(payload.get(key), list):
                entries = payload[key]
                break
        else:
            entries = []
    else:
        entries = []

    return most_recent, entries[-limit:]


def _to_numeric_list(value: Any) -> Optional[List[float]]:
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, (int, float, bool)) for v in value):
        return [float(v) for v in value]
    return None


def extract_features(entry: Any) -> Optional[torch.Tensor]:
    if not isinstance(entry, dict):
        as_list = _to_numeric_list(entry)
        if as_list is None:
            return None
        return torch.tensor(as_list, dtype=torch.float32).unsqueeze(0)

    for key in ("input", "inputs", "x", "features", "feature_vector"):
        value = entry.get(key)
        as_list = _to_numeric_list(value)
        if as_list is not None:
            return torch.tensor(as_list, dtype=torch.float32).unsqueeze(0)

    features = entry.get("features")
    if isinstance(features, dict):
        numeric_items = [(k, v) for k, v in features.items() if isinstance(v, (int, float, bool))]
        if numeric_items:
            numeric_items.sort(key=lambda item: item[0])
            return torch.tensor([float(v) for _, v in numeric_items], dtype=torch.float32).unsqueeze(0)

    numeric_items = [(k, v) for k, v in entry.items() if isinstance(v, (int, float, bool)) and k != "label"]
    if numeric_items:
        numeric_items.sort(key=lambda item: item[0])
        return torch.tensor([float(v) for _, v in numeric_items], dtype=torch.float32).unsqueeze(0)

    return None


def extract_label(entry: Any) -> Optional[int]:
    if not isinstance(entry, dict):
        return None

    for key in ("label", "target", "y", "liked", "is_positive", "should_like"):
        if key in entry:
            value = entry[key]
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value > 0)
            if isinstance(value, str):
                low = value.strip().lower()
                if low in {"1", "true", "yes", "like", "liked", "positive", "keep"}:
                    return 1
                if low in {"0", "false", "no", "dislike", "negative", "skip"}:
                    return 0
    return None


def predict_binary(model: nn.Module, features: torch.Tensor) -> Optional[int]:
    with torch.no_grad():
        output = model(features)

    if not torch.is_tensor(output):
        return None

    output = output.detach().cpu()

    if output.numel() == 1:
        val = float(output.item())
        threshold = 0.5 if 0.0 <= val <= 1.0 else 0.0
        return int(val > threshold)

    if output.ndim >= 2 and output.shape[-1] == 1:
        val = float(output.reshape(-1)[0].item())
        threshold = 0.5 if 0.0 <= val <= 1.0 else 0.0
        return int(val > threshold)

    if output.ndim >= 2 and output.shape[-1] == 2:
        return int(torch.argmax(output, dim=-1).reshape(-1)[0].item())

    return None


def evaluate_models(old_model: Optional[nn.Module], merged_model: Optional[nn.Module], entries: Sequence[Any]) -> Tuple[Optional[float], Optional[float], int]:
    if old_model is None or merged_model is None or not entries:
        return None, None, 0

    old_model.eval()
    merged_model.eval()

    evaluated = 0
    old_correct = 0
    merged_correct = 0

    for entry in entries:
        features = extract_features(entry)
        label = extract_label(entry)
        if features is None or label is None:
            continue

        old_pred = predict_binary(old_model, features)
        merged_pred = predict_binary(merged_model, features)
        if old_pred is None or merged_pred is None:
            continue

        evaluated += 1
        old_correct += int(old_pred == label)
        merged_correct += int(merged_pred == label)

    if evaluated == 0:
        return None, None, 0

    return old_correct / evaluated, merged_correct / evaluated, evaluated


def collision_safe_legacy_name(base_model: ModelFile, root: Path) -> Path:
    base_stem = f"Model{base_model.version_str}Legacy"
    candidate = root / f"{base_stem}.pt"
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = root / f"{base_stem}_{index}.pt"
        if not candidate.exists():
            return candidate
        index += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge latest base model into UserModel.pt with compatibility checks.")
    parser.add_argument("--alpha", type=float, default=0.6, help="Merge ratio for UserModel weights (default: 0.6)")
    parser.add_argument("--root", type=str, default=".", help="Directory to scan for model/data files")
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Do not pause for Enter before exit (useful for terminal automation).",
    )
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        print("[MergeModels] alpha must be between 0 and 1.")
        return 1

    root = Path(args.root).resolve()

    base_models, legacy_models = discover_models(root)
    print(f"[MergeModels] Found {len(base_models)} base model(s) and {len(legacy_models)} legacy model(s).")

    if not base_models:
        print("[MergeModels] No base models found (Model{version}.pt). Nothing to merge.")
        return 0

    user_model_path = root / USER_MODEL
    if not user_model_path.exists():
        print("[MergeModels] UserModel.pt was not found. Exiting without changes.")
        return 0

    base_model = base_models[-1]
    print(f"[MergeModels] Selected base model: {base_model.path.name}")

    base_checkpoint = load_checkpoint(base_model.path)
    user_checkpoint = load_checkpoint(user_model_path)

    base_state = extract_state_dict(base_checkpoint)
    user_state = extract_state_dict(user_checkpoint)

    validate_state_dict_compatibility(base_state, user_state)
    print("[MergeModels] Architecture/state_dict compatibility check passed.")

    backup_path = root / USER_BACKUP
    shutil.copy2(user_model_path, backup_path)
    print(f"[MergeModels] Backup created: {backup_path.name}")

    merged_state = merge_state_dicts(base_state, user_state, alpha=args.alpha)
    merged_checkpoint = inject_state_dict(copy.deepcopy(user_checkpoint), merged_state)

    old_model_for_eval = materialize_model(user_checkpoint)
    merged_model_for_eval = materialize_model(copy.deepcopy(merged_checkpoint))

    torch.save(merged_checkpoint, user_model_path)
    print(f"[MergeModels] Merged model saved to {user_model_path.name} (alpha={args.alpha}).")

    data_file, entries = get_recent_training_entries(root, limit=20)
    if data_file is None:
        print("[MergeModels] No DataForTraining*.json files found; skipping evaluation.")
    else:
        print(f"[MergeModels] Evaluating on last {min(20, len(entries))} entries from {data_file.name}.")

    old_acc, merged_acc, evaluated = evaluate_models(old_model_for_eval, merged_model_for_eval, entries)
    if old_acc is None or merged_acc is None:
        print("[MergeModels] Could not compute accuracy comparison (model format or data schema unsupported).")
    else:
        print(f"[MergeModels] Accuracy (old user model):    {old_acc:.2%}")
        print(f"[MergeModels] Accuracy (merged user model): {merged_acc:.2%}")

        if merged_acc < old_acc:
            answer = input("[MergeModels] Merged model underperformed. Keep merged model? [y/N]: ").strip().lower()
            if answer not in {"y", "yes"}:
                shutil.copy2(backup_path, user_model_path)
                print("[MergeModels] Reverted UserModel.pt from backup.")
            else:
                print("[MergeModels] Keeping merged model by user choice.")

    legacy_target = collision_safe_legacy_name(base_model, root)
    base_model.path.rename(legacy_target)
    print(f"[MergeModels] Renamed base model to legacy file: {legacy_target.name}")

    print("[MergeModels] Done.")
    return 0


def maybe_pause_before_exit(no_pause: bool) -> None:
    """Keep the window open when script is launched by double-click.

    By default, if no command-line arguments are provided, we pause before
    exit so the user can read status/output in a GUI-launched console window.
    """
    if no_pause:
        return

    launched_without_args = len(sys.argv) == 1
    if launched_without_args:
        try:
            input("\n[MergeModels] Press Enter to close...")
        except EOFError:
            pass


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"[MergeModels] Fatal error: {exc}")
        exit_code = 1

    no_pause_flag = "--no-pause" in sys.argv
    maybe_pause_before_exit(no_pause=no_pause_flag)
    raise SystemExit(exit_code)
