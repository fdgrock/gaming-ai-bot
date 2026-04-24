"""
Model Card and Feature Schema Utilities
========================================
Shared helpers for writing model_card.json and feature_schema.json files
alongside every trained model. These files are consumed at inference time
to validate that the feature matrix matches what the model was trained on.

Schema version history:
  2.0  — Phase A-E rewrite (random-target fix, draw_idx leakage fix, new
          combination features, window/lookback metadata, permutation baseline)
"""

import json
import sys
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Current schema version — bump MAJOR when feature columns change incompatibly
SCHEMA_VERSION = "2.0"

# Drop columns that are never features (labels / indices)
NON_FEATURE_COLS = {"draw_index", "draw_idx", "number", "target"}


# ──────────────────────────────────────────────────────────────────────────────
# Package version helper
# ──────────────────────────────────────────────────────────────────────────────

def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def collect_package_versions() -> Dict[str, str]:
    pkgs = ["numpy", "pandas", "tensorflow", "keras", "scikit-learn",
            "xgboost", "lightgbm", "catboost", "scipy"]
    return {p: _pkg_version(p) for p in pkgs}


# ──────────────────────────────────────────────────────────────────────────────
# Feature column derivation
# ──────────────────────────────────────────────────────────────────────────────

def derive_feature_cols(df) -> List[str]:
    """Return the ordered list of feature columns that will actually enter X."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# ──────────────────────────────────────────────────────────────────────────────
# Feature schema JSON
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_schema(
    *,
    model_type: str,
    game: str,
    feature_cols: List[str],
    n_samples: int,
    n_train: int,
    n_val: int,
    n_test: int,
    window_size: Optional[int] = None,
    lookback: Optional[int] = None,
    stride: Optional[int] = None,
    draw_index_range: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a fully-populated feature schema dict.

    This is written as ``feature_schema.json`` alongside the parquet files
    and also saved as ``feature_schema.json`` inside each model directory.
    At inference time the two are compared to catch mismatches.
    """
    schema = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "model_type": model_type,
        "game": game,
        # Feature identity
        "feature_names": feature_cols,
        "feature_count": len(feature_cols),
        "non_feature_cols_excluded": sorted(NON_FEATURE_COLS),
        # Sequence / window config
        "window_size": window_size,
        "lookback": lookback,
        "stride": stride,
        # Split sizes
        "n_samples_total": n_samples,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        # Draw index bounds (populated if available)
        "draw_index_range": draw_index_range or {},
        # Normalisation
        "normalization": "StandardScaler",
        # Runtime
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": collect_package_versions(),
    }
    if extra:
        schema.update(extra)
    return schema


def save_feature_schema(schema: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)


def load_feature_schema(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Schema compatibility check
# ──────────────────────────────────────────────────────────────────────────────

def check_schema_compatibility(
    training_schema: Dict[str, Any],
    inference_feature_cols: List[str],
) -> tuple:
    """
    Compare training feature schema against the columns available at inference.

    Returns:
        (ok: bool, issues: List[str])
    """
    issues: List[str] = []

    trained_cols = training_schema.get("feature_names", [])
    trained_count = training_schema.get("feature_count", len(trained_cols))
    infer_count = len(inference_feature_cols)

    if trained_count != infer_count:
        issues.append(
            f"Feature count mismatch: model trained on {trained_count} features, "
            f"inference data has {infer_count} features"
        )

    if trained_cols != inference_feature_cols:
        missing = [c for c in trained_cols if c not in inference_feature_cols]
        extra = [c for c in inference_feature_cols if c not in trained_cols]
        if missing:
            issues.append(f"Features missing at inference: {missing}")
        if extra:
            issues.append(f"Extra features at inference (ignored): {extra}")

    schema_ver = training_schema.get("schema_version", "0.0")
    if schema_ver != SCHEMA_VERSION:
        issues.append(
            f"Schema version mismatch: model schema={schema_ver}, "
            f"current={SCHEMA_VERSION}. Re-train models after feature regeneration."
        )

    return len(issues) == 0, issues


# ──────────────────────────────────────────────────────────────────────────────
# Model card JSON
# ──────────────────────────────────────────────────────────────────────────────

def build_model_card(
    *,
    architecture: str,
    game: str,
    position: Optional[int] = None,
    feature_schema: Dict[str, Any],
    metrics: Dict[str, float],
    permutation_baseline: Optional[float] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    target_alignment: str = "actual_lottery_numbers",
    notes: str = "",
) -> Dict[str, Any]:
    """
    Build a model card dict saved alongside every .pkl / .h5 model file.
    """
    card: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "architecture": architecture,
        "game": game,
        "position": position,
        # What the model was trained to predict
        "target_alignment": target_alignment,
        # Feature contract
        "feature_schema_version": feature_schema.get("schema_version", SCHEMA_VERSION),
        "feature_names": feature_schema.get("feature_names", []),
        "feature_count": feature_schema.get("feature_count", 0),
        "window_size": feature_schema.get("window_size"),
        "lookback": feature_schema.get("lookback"),
        "stride": feature_schema.get("stride"),
        "normalization": feature_schema.get("normalization", "StandardScaler"),
        # Evaluation
        "test_metrics": metrics,
        "permutation_baseline_composite": permutation_baseline,
        "beats_baseline": (
            (metrics.get("composite_score", 0) > (permutation_baseline or 0) + 0.005)
            if permutation_baseline is not None else None
        ),
        # Training config
        "hyperparams": hyperparams or {},
        "training_config": training_config or {},
        # Notes
        "notes": notes,
        "python_version": sys.version,
        "package_versions": collect_package_versions(),
    }
    return card


def save_model_card(card: Dict[str, Any], model_path: Path) -> None:
    """Save model_card.json alongside the model file."""
    card_path = model_path.with_suffix(".model_card.json")
    card_path.parent.mkdir(parents=True, exist_ok=True)
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)


def load_model_card(model_path: Path) -> Optional[Dict[str, Any]]:
    """Load model_card.json if it exists next to the model file."""
    card_path = model_path.with_suffix(".model_card.json")
    if not card_path.exists():
        return None
    with open(card_path) as f:
        return json.load(f)
