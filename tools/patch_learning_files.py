"""
patch_learning_files.py
-----------------------
Retroactively upgrades all per-draw learning JSON files for both games so they
contain the three new fields introduced in the AI Learning tab rework:

  1. analysis.number_frequency.hot_numbers
       → actual winning draw numbers  (was missing — now the correct signal)

  2. analysis.number_frequency.penalized_numbers
       → alias of the existing missed_numbers list  (over-predicted but wrong)

  3. analysis.structural_fingerprint
       → zone_pattern, winning_sum, odd/even count, gap stats of the actual draw

All data needed to compute these fields already exists inside each file
(actual_results.numbers).  No external data or predictions are required.

Run from the project root:
    python tools/patch_learning_files.py

Safe to run multiple times — already-patched files are updated idempotently.
"""

import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Zone boundaries (must match the Structural Portfolio Generator in prediction_ai.py)
# ---------------------------------------------------------------------------
ZONE_BOUNDS = {
    'lotto_max': (17, 34),   # lo: 1-17 / mi: 18-34 / hi: 35-52
    'lotto_6_49': (16, 32),  # lo: 1-16 / mi: 17-32 / hi: 33-49
}


def _infer_game_key(file_path: Path) -> str:
    """Derive the zone-bound key from the learning file's parent folder name."""
    folder = file_path.parent.name.lower()
    if 'max' in folder:
        return 'lotto_max'
    return 'lotto_6_49'


def _compute_structural_fingerprint(numbers: list, game_key: str) -> dict:
    """Compute the structural fingerprint of a winning draw."""
    lo_cap, mi_cap = ZONE_BOUNDS[game_key]
    nums_sorted = sorted(int(n) for n in numbers)

    lo_count = sum(1 for n in nums_sorted if n <= lo_cap)
    mi_count = sum(1 for n in nums_sorted if lo_cap < n <= mi_cap)
    hi_count = sum(1 for n in nums_sorted if n > mi_cap)

    gaps = [nums_sorted[i + 1] - nums_sorted[i] for i in range(len(nums_sorted) - 1)]

    return {
        'zone_pattern': [lo_count, mi_count, hi_count],
        'winning_sum': int(sum(nums_sorted)),
        'odd_count': sum(1 for n in nums_sorted if n % 2 != 0),
        'even_count': sum(1 for n in nums_sorted if n % 2 == 0),
        'avg_gap': round(float(np.mean(gaps)), 4) if gaps else 0.0,
        'max_gap': int(max(gaps)) if gaps else 0,
        'min_gap': int(min(gaps)) if gaps else 0,
        'number_range': int(nums_sorted[-1] - nums_sorted[0]) if nums_sorted else 0,
    }


def patch_file(path: Path, dry_run: bool = False) -> str:
    """
    Patch a single learning JSON file.

    Returns a short status string ('patched', 'already_up_to_date', 'skipped').
    """
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception as exc:
        return f'ERROR reading: {exc}'

    actual = data.get('actual_results', {})
    winning_numbers = actual.get('numbers', [])
    if not winning_numbers:
        return 'skipped (no actual_results.numbers)'

    game_key = _infer_game_key(path)
    analysis = data.setdefault('analysis', {})

    changed = False

    # ------------------------------------------------------------------
    # 1. hot_numbers in number_frequency = actual winning draw numbers
    # ------------------------------------------------------------------
    num_freq = analysis.setdefault('number_frequency', {})
    hot_numbers_new = [
        {'number': int(n), 'frequency': 1.0}
        for n in sorted(winning_numbers)
    ]
    existing_hot = num_freq.get('hot_numbers', [])
    # Check if it's already the new format (list of dicts with 'frequency' key)
    already_new_format = (
        existing_hot
        and isinstance(existing_hot[0], dict)
        and 'frequency' in existing_hot[0]
    )
    if not already_new_format:
        num_freq['hot_numbers'] = hot_numbers_new
        changed = True

    # ------------------------------------------------------------------
    # 2. penalized_numbers = alias of missed_numbers (already present)
    # ------------------------------------------------------------------
    if 'penalized_numbers' not in num_freq:
        missed = num_freq.get('missed_numbers', [])
        num_freq['penalized_numbers'] = missed
        changed = True

    # ------------------------------------------------------------------
    # 3. structural_fingerprint
    # ------------------------------------------------------------------
    if 'structural_fingerprint' not in analysis:
        analysis['structural_fingerprint'] = _compute_structural_fingerprint(
            winning_numbers, game_key
        )
        changed = True

    if not changed:
        return 'already_up_to_date'

    if dry_run:
        return 'would_patch'

    try:
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2)
        return 'patched'
    except Exception as exc:
        return f'ERROR writing: {exc}'


def main():
    dry_run = '--dry-run' in sys.argv

    root = Path('data') / 'learning'
    if not root.exists():
        print(f'Learning directory not found: {root.resolve()}')
        sys.exit(1)

    learning_files = sorted(root.rglob('draw_*_learning.json'))
    if not learning_files:
        print('No draw_*_learning.json files found.')
        sys.exit(0)

    if dry_run:
        print('DRY RUN - no files will be written.\n')

    counts = {'patched': 0, 'already_up_to_date': 0, 'would_patch': 0, 'error': 0}

    for path in learning_files:
        status = patch_file(path, dry_run=dry_run)
        key = 'error' if status.startswith('ERROR') or status == 'skipped' else status
        counts[key] = counts.get(key, 0) + 1
        icon = {'patched': 'OK', 'already_up_to_date': '==', 'would_patch': '--'}.get(status, '!!')
        print(f'  {icon}  {path.parent.name}/{path.name}  [{status}]')

    print()
    if dry_run:
        print(f"Would patch {counts.get('would_patch', 0)} / {len(learning_files)} files.")
    else:
        print(f"Done. Patched: {counts.get('patched', 0)}  |  "
              f"Already up-to-date: {counts.get('already_up_to_date', 0)}  |  "
              f"Errors: {counts.get('error', 0)}")


if __name__ == '__main__':
    main()
