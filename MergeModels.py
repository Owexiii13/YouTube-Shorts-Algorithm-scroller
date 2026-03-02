import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from features import extract_features, load_vocab

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

MODEL_RE = re.compile(r"^Model(\d+)\.(\d+)(Legacy)?\.pt$")


def _pause_for_double_click() -> None:
    """Keep terminal window open when script is launched by double-click."""
    if sys.stdin is None or not sys.stdin.isatty():
        return
    try:
        input("\nPress Enter to close...")
    except EOFError:
        pass


def ensure_torch() -> bool:
    """Ensure PyTorch is importable, attempting install for double-click workflows."""
    global torch  # noqa: PLW0603
    if torch is not None:
        return True

    print("PyTorch not found. Attempting automatic install...")
    install_commands = [
        [sys.executable, "-m", "pip", "install", "torch"],
        ["pip", "install", "torch"],
        ["pip3", "install", "torch"],
    ]

    for cmd in install_commands:
        try:
            subprocess.run(cmd, check=True)
            import torch as imported_torch  # local import after install

            torch = imported_torch
            print("PyTorch installed successfully.")
            return True
        except Exception:  # noqa: BLE001
            continue

    print("Could not install PyTorch automatically. Please install manually: pip install torch")
    return False


def parse_version(name: str) -> Optional[Tuple[int, int, bool]]:
    m = MODEL_RE.match(name)
    if not m:
        return None
    major = int(m.group(1))
    minor = int(m.group(2))
    legacy = bool(m.group(3))
    return major, minor, legacy


def list_models(base: Path) -> List[Tuple[Path, Tuple[int, int, bool]]]:
    found = []
    for p in base.glob("Model*.pt"):
        v = parse_version(p.name)
        if v:
            found.append((p, v))
    return sorted(found, key=lambda x: (x[1][0], x[1][1], x[1][2]))


def load_state(path: Path):
    if torch is None:
        raise RuntimeError("PyTorch is required for merging models.")
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj
    if isinstance(obj, dict):
        return obj, None
    raise RuntimeError(f"Unsupported model format: {path.name}")


def merge_states(new_state: Dict, user_state: Dict, alpha: float) -> Dict:
    if set(new_state.keys()) != set(user_state.keys()):
        missing_new = sorted(set(user_state.keys()) - set(new_state.keys()))
        missing_user = sorted(set(new_state.keys()) - set(user_state.keys()))
        raise RuntimeError(
            "Model parameters do not match between base and user model. "
            f"Missing in new: {missing_new[:5]}, missing in user: {missing_user[:5]}"
        )

    merged = {}
    for key in new_state:
        if tuple(new_state[key].shape) != tuple(user_state[key].shape):
            raise RuntimeError(f"Shape mismatch for tensor '{key}'.")
        merged[key] = alpha * new_state[key] + (1 - alpha) * user_state[key]
    return merged


def label_to_index(label: str) -> int:
    return {"liked": 0, "disliked": 1, "neutral": 2}.get(label, 2)


def _predict_with_linear_classifier(state: Dict, feature_vec: List[float]) -> int:
    if torch is None:
        return 2
    x = torch.tensor(feature_vec, dtype=torch.float32)
    if "classifier.weight" in state and "classifier.bias" in state:
        w = state["classifier.weight"]
        b = state["classifier.bias"]
        logits = torch.mv(w, x) + b
        return int(torch.argmax(logits).item())
    return 2


def evaluate_model(state: Dict, entries: List[Dict], vocab: List[str]) -> float:
    if not entries:
        return 0.0
    correct = 0
    for row in entries:
        pred = _predict_with_linear_classifier(state, extract_features(row, vocab))
        if pred == label_to_index(row.get("user_action", "neutral")):
            correct += 1
    return (correct / len(entries)) * 100.0


def load_eval_entries(base: Path) -> List[Dict]:
    files = sorted(base.glob("DataForTraining*.json"), key=lambda p: p.name)
    if not files:
        return []

    latest_data: List[Dict] = []
    for latest in reversed(files):
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            latest_data = [d for d in data if isinstance(d, dict)]
            if latest_data:
                break

    return latest_data[-20:] if latest_data else []


def choose_old_base(models: List[Tuple[Path, Tuple[int, int, bool]]], new_base: Path) -> Optional[Path]:
    previous_non_legacy = [p for p, v in models if p != new_base and not v[2]]
    if previous_non_legacy:
        return previous_non_legacy[-1]

    legacy_models = [p for p, v in models if p != new_base and v[2]]
    if legacy_models:
        return legacy_models[-1]
    return None


def maybe_prompt_revert(old_acc: float, merged_acc: float, user_model: Path, backup: Path) -> None:
    delta = merged_acc - old_acc
    if delta >= 0:
        print(f"Result: +{delta:.1f}% — The merged model is BETTER. Saving as UserModel.pt.")
        return

    print(f"Result: {delta:.1f}% — The merged model is WORSE on your data.")
    print("\nOptions:\n  [K] Keep merged model anyway (it may improve with more use)\n  [D] Delete merged model and restore old model")
    choice = input("Enter choice (K/D): ").strip().lower()
    if choice == "d":
        user_model.write_bytes(backup.read_bytes())
        print("Restored old user model from backup.")
    else:
        print("Keeping merged model.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge latest base model with UserModel.pt")
    parser.add_argument("--alpha", type=float, default=0.6, help="Blend ratio for new base model")
    args = parser.parse_args()

    if not ensure_torch():
        return 1

    if not (0.0 <= args.alpha <= 1.0):
        print("--alpha must be between 0.0 and 1.0")
        return 1

    base = Path(".")
    models = list_models(base)
    if not models:
        print("No base models found (ModelX.Y.pt).")
        return 1

    non_legacy = [m for m in models if not m[1][2]]
    if not non_legacy:
        print("Only a legacy model found. Please download the latest model from the GitHub releases page.")
        return 1

    new_base = non_legacy[-1][0]
    old_base = choose_old_base(models, new_base)

    user_model = base / "UserModel.pt"
    if not user_model.exists():
        print("No personal model found. Nothing to merge. The new base model will be used automatically.")
        return 0

    try:
        new_state, _ = load_state(new_base)
        user_state, user_raw = load_state(user_model)
        merged = merge_states(new_state, user_state, args.alpha)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not merge models: {exc}")
        return 1

    backup = base / "UserModel_before_merge.pt"
    backup.write_bytes(user_model.read_bytes())

    eval_entries = load_eval_entries(base)
    vocab = load_vocab("vocab.json")

    save_obj = user_raw if isinstance(user_raw, dict) else {}
    if save_obj:
        save_obj["state_dict"] = merged
        torch.save(save_obj, user_model)
    else:
        torch.save(merged, user_model)

    print("=== Merge Complete ===")
    if not eval_entries:
        print("No DataForTraining files found for evaluation. Merged model saved as UserModel.pt")
    else:
        old_acc = evaluate_model(user_state, eval_entries, vocab)
        merged_acc = evaluate_model(merged, eval_entries, vocab)
        print(f"Old personal model accuracy: {old_acc:.1f}%")
        print(f"Merged model accuracy:       {merged_acc:.1f}%")
        maybe_prompt_revert(old_acc, merged_acc, user_model, backup)

    print(f"Old model saved as: {backup.name}")

    if old_base and "Legacy" not in old_base.stem:
        legacy_target = old_base.with_name(f"{old_base.stem}Legacy.pt")
        if legacy_target.exists():
            legacy_target.unlink()
        old_base.rename(legacy_target)
        print(f"Renamed old base model to {legacy_target.name}")

    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error: {exc}")
        exit_code = 1
    finally:
        _pause_for_double_click()

    raise SystemExit(exit_code)
