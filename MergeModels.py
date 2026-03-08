from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from train import (
    ACTIONS,
    aggregate_components,
    load_model_json,
    normalize_action_counts,
    normalize_snapshot,
    parse_semver,
    prune_action_weights,
    resolve_local_path,
    sanitize_action_weight_maps,
)


def choose_base_model(base_dir: Path, explicit_base: Optional[str]) -> Optional[Path]:
    if explicit_base:
        candidate = resolve_local_path(explicit_base)
        return candidate if candidate.exists() else None

    candidates: List[Tuple[int, Tuple[int, int, int], str, Path]] = []
    for path in sorted(base_dir.glob('*.json')):
        payload = load_model_json(path)
        if not payload:
            continue
        role = str(payload.get('model_role') or '').strip().lower()
        version = parse_semver(payload.get('model_version')) or parse_semver(path.stem) or (-1, -1, -1)
        is_base = role == 'base_model' or path.name == 'trained_model.json' or 'basemodel' in path.name.lower() or version != (-1, -1, -1)
        if not is_base:
            continue
        candidates.append((1 if version != (-1, -1, -1) else 0, version, path.name.lower(), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[-1][3]


def load_preference_components(data_dir: Path) -> List[Dict[str, Any]]:
    components: List[Dict[str, Any]] = []
    for path in sorted(data_dir.rglob('*.json')):
        payload = load_model_json(path)
        if not payload:
            continue
        snapshot = normalize_snapshot(payload, path)
        if not snapshot:
            continue
        if snapshot.get('model_role') == 'base_model':
            continue
        components.append(snapshot)
    return components


def base_identity(payload: Dict[str, Any], path: Path) -> Tuple[str, str, str]:
    version = str(payload.get('model_version') or '').strip()
    source = path.name
    signature = str(payload.get('model_signature') or '').strip()
    return version, source, signature


def apply_preference_delta(
    base_payload: Optional[Dict[str, Any]],
    base_path: Optional[Path],
    preference_components: List[Dict[str, Any]],
    alpha: float,
) -> Dict[str, Any]:
    if base_payload:
        base_weights = sanitize_action_weight_maps(base_payload.get('action_weights'))
        base_counts = normalize_action_counts(base_payload.get('action_counts'))
        base_version, base_source, base_signature = base_identity(base_payload, base_path or Path('trained_model.json'))
    else:
        base_weights = {action: {} for action in ACTIONS}
        base_counts = {action: 0 for action in ACTIONS}
        base_version, base_source, base_signature = '', '', ''

    adjusted_components: List[Dict[str, Any]] = []
    for component in preference_components:
        adjusted = dict(component)
        weight = float(adjusted.get('weight', 1.0) or 1.0)
        same_signature = base_signature and adjusted.get('base_model_signature') == base_signature
        same_version = base_version and adjusted.get('base_model_version') == base_version
        same_source = base_source and adjusted.get('base_model_source') == base_source
        if same_signature or same_version or same_source:
            adjusted['weight'] = weight * 1.12
        elif base_payload:
            adjusted['weight'] = weight * 0.78
        adjusted_components.append(adjusted)

    merged_preferences = aggregate_components(adjusted_components)
    if not merged_preferences:
        return {
            'model_role': 'merged_model',
            'base_model_version': base_version or None,
            'base_model_source': base_source or None,
            'action_weights': prune_action_weights(base_weights, max_size=25000),
            'action_counts': normalize_action_counts(base_counts, sum(base_counts.values())),
            'merged_preference_models': 0,
        }

    merged_weights = {action: dict(base_weights.get(action, {})) for action in ACTIONS}
    for action in ACTIONS:
        for feature, delta in merged_preferences['action_weights'].get(action, {}).items():
            merged_weights[action][feature] = merged_weights[action].get(feature, 0.0) + (float(delta) * alpha)

    merged_counts = {action: base_counts.get(action, 0) + merged_preferences['action_counts'].get(action, 0) for action in ACTIONS}
    output: Dict[str, Any] = {
        'model_role': 'merged_model',
        'merge_strategy': 'base_plus_user_preference_delta_v1',
        'base_model_version': base_version or None,
        'base_model_source': base_source or None,
        'merged_preference_models': len(preference_components),
        'action_weights': prune_action_weights(merged_weights, max_size=25000),
        'action_counts': normalize_action_counts(merged_counts, sum(merged_counts.values())),
    }
    if base_payload and base_payload.get('model_version'):
        output['model_version'] = base_payload.get('model_version')
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description='Merge a base model with one or more user preference model snapshots.')
    parser.add_argument('--data-dir', default='data', help='Folder to scan for Model.json snapshots.')
    parser.add_argument('--base', default=None, help='Optional base model JSON file. Defaults to the newest base model in the current folder.')
    parser.add_argument('--output', default='MergedModel.json', help='Where to write the merged JSON model.')
    parser.add_argument('--alpha', type=float, default=1.0, help='How strongly to apply user preference deltas to the base model.')
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 2.0):
        print('--alpha must be between 0.0 and 2.0')
        return 1

    data_dir = resolve_local_path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        print(f'[merge] Data directory not found: {data_dir}')
        return 1

    base_path = choose_base_model(Path('.'), args.base)
    base_payload = load_model_json(base_path) if base_path else None
    preference_components = load_preference_components(data_dir)
    if not preference_components:
        print(f'[merge] No usable preference model snapshots found under {data_dir}')
        return 1

    merged = apply_preference_delta(base_payload, base_path, preference_components, args.alpha)
    output_path = resolve_local_path(args.output)
    if output_path.suffix.lower() != '.json':
        output_path = output_path.with_suffix('.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2), encoding='utf-8')

    feature_count = sum(len(weights) for weights in merged['action_weights'].values())
    print(f'[merge] wrote {output_path} with {feature_count} action features')
    if base_path:
        print(f'[merge] base model: {base_path.name}')
    print(f'[merge] merged preference snapshots: {len(preference_components)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
