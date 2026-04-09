"""Model training, evaluation, and tensor-side helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
import torch

from src.targets import TargetSpec
from src.targets import get_target_spec


TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
RANDOM_SEED = 0
TARGET_TRANSFORM_NAMES = ("standard", "signed_log1p")


@dataclass
class SplitData:
    """Store tensors for one dataset split."""

    features: torch.Tensor
    targets: torch.Tensor
    frame: pd.DataFrame
    target_column: str


@dataclass
class Standardization:
    """Store training statistics used for feature scaling."""

    mean: torch.Tensor
    std: torch.Tensor


@dataclass
class TargetStandardization:
    """Store training statistics used for target scaling."""

    mean: torch.Tensor
    std: torch.Tensor
    transform_name: str


@dataclass
class ExperimentResult:
    """Store summary metrics for one experiment configuration."""

    target: str
    task_kind: str
    model_backend: str
    feature_blocks: tuple[str, ...]
    feature_count: int
    epochs: int
    device: str
    baseline_metrics: dict[str, dict[str, dict[str, float]]]
    persistence_valid_mae: float
    persistence_valid_rmse: float
    persistence_valid_r2: float
    persistence_test_mae: float
    persistence_test_rmse: float
    persistence_test_r2: float
    model_valid_mae: float
    model_valid_rmse: float
    model_valid_r2: float
    model_test_mae: float
    model_test_rmse: float
    model_test_r2: float


def resolve_device(requested_device: str) -> torch.device:
    """Resolve the requested torch device."""

    normalized = requested_device.strip().lower()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")

    if normalized == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError("MPS requested but not available.")
        return torch.device("mps")

    if normalized == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown device: {requested_device}")


def split_by_time(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset chronologically using unique timestamps."""

    timestamps = sorted(frame["timestamp"].unique())
    train_end = int(len(timestamps) * TRAIN_RATIO)
    valid_end = int(len(timestamps) * (TRAIN_RATIO + VALID_RATIO))
    train_times = set(timestamps[:train_end])
    valid_times = set(timestamps[train_end:valid_end])
    test_times = set(timestamps[valid_end:])
    train_frame = frame[frame["timestamp"].isin(train_times)].copy()
    valid_frame = frame[frame["timestamp"].isin(valid_times)].copy()
    test_frame = frame[frame["timestamp"].isin(test_times)].copy()
    return train_frame, valid_frame, test_frame


def validate_target_column(frame: pd.DataFrame, target_column: str) -> None:
    """Fail early if the requested target column is missing."""

    get_target_spec(target_column)


def validate_target_kind(target_spec: TargetSpec) -> None:
    """Reject unsupported task kinds for the current runner."""

    if target_spec.task_kind != "direction":
        return

    raise ValueError(
        "Direction targets are classification tasks. "
        "This runner currently supports regression only."
    )


def filter_target_rows(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Drop rows where the selected target is unavailable."""

    return frame.dropna(subset=[target_column]).reset_index(drop=True)


def to_split(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> SplitData:
    """Convert one DataFrame split into tensors."""

    feature_values = frame[feature_columns].to_numpy(dtype="float32")
    target_values = frame[target_column].to_numpy(dtype="float32")
    return SplitData(
        features=torch.tensor(feature_values),
        targets=torch.tensor(target_values),
        frame=frame,
        target_column=target_column,
    )


def compute_standardization(split: SplitData) -> Standardization:
    """Compute training-set scaling values."""

    reduce_dims = tuple(range(split.features.ndim - 1))
    mean = split.features.mean(dim=reduce_dims)
    std = split.features.std(dim=reduce_dims, correction=0)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return Standardization(mean=mean, std=std)


def standardize_split(split: SplitData, stats: Standardization) -> SplitData:
    """Apply training-set scaling to one split."""

    return SplitData(
        features=(split.features - stats.mean) / stats.std,
        targets=split.targets,
        frame=split.frame,
        target_column=split.target_column,
    )


def validate_target_transform_name(transform_name: str) -> None:
    """Fail early on unknown target transforms."""

    if transform_name in TARGET_TRANSFORM_NAMES:
        return

    available_names = ", ".join(TARGET_TRANSFORM_NAMES)
    raise ValueError(
        f"Unknown target transform: {transform_name}. "
        f"Available: {available_names}"
    )


def apply_target_transform(
    values: torch.Tensor,
    transform_name: str,
) -> torch.Tensor:
    """Apply one supported target transform."""

    validate_target_transform_name(transform_name)

    if transform_name == "standard":
        return values

    return torch.sign(values) * torch.log1p(values.abs())


def invert_target_transform(
    values: torch.Tensor,
    transform_name: str,
) -> torch.Tensor:
    """Invert one supported target transform."""

    validate_target_transform_name(transform_name)

    if transform_name == "standard":
        return values

    return torch.sign(values) * torch.expm1(values.abs())


def resolve_target_transform(
    model_backend: str,
    target_spec: TargetSpec,
    requested_transform: str | None,
) -> str:
    """Choose the target transform for one experiment."""

    if requested_transform is not None:
        validate_target_transform_name(requested_transform)
        return requested_transform

    return "standard"


def compute_target_standardization(
    split: SplitData,
    transform_name: str,
) -> TargetStandardization:
    """Compute training-set scaling values for the target."""

    transformed_targets = apply_target_transform(split.targets, transform_name)
    mean = transformed_targets.mean()
    std = transformed_targets.std(correction=0)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return TargetStandardization(
        mean=mean,
        std=std,
        transform_name=transform_name,
    )


def standardize_targets(
    split: SplitData,
    stats: TargetStandardization,
) -> SplitData:
    """Apply training-set target scaling to one split."""

    transformed_targets = apply_target_transform(split.targets, stats.transform_name)
    return SplitData(
        features=split.features,
        targets=(transformed_targets - stats.mean) / stats.std,
        frame=split.frame,
        target_column=split.target_column,
    )


def evaluate_predictions(
    predictions: torch.Tensor,
    actual_targets: torch.Tensor,
) -> dict[str, float]:
    """Compute a few easy-to-read regression metrics."""

    residuals = predictions - actual_targets
    mae = residuals.abs().mean().item()
    rmse = residuals.square().mean().sqrt().item()
    centered_targets = actual_targets - actual_targets.mean()
    total_sum_of_squares = centered_targets.square().sum()
    residual_sum_of_squares = residuals.square().sum()
    r2 = 1.0 - (residual_sum_of_squares / total_sum_of_squares).item()
    return {"mae": mae, "rmse": rmse, "r2": r2}


def make_persistence_predictions(
    frame: pd.DataFrame,
    target_spec: TargetSpec,
) -> torch.Tensor:
    """Build naive regression predictions for the selected target."""

    if target_spec.task_kind == "exact_count":
        values = frame[target_spec.source_column].to_numpy(dtype="float32")
        return torch.tensor(values)

    if target_spec.task_kind == "delta":
        return torch.zeros(len(frame), dtype=torch.float32)

    raise ValueError(f"Unsupported persistence baseline for {target_spec.task_kind}")


def make_baseline_predictions(
    frame: pd.DataFrame,
    target_spec: TargetSpec,
    baseline_name: str,
) -> torch.Tensor:
    """Build one named naive baseline for the selected target."""

    if baseline_name == "persistence":
        return make_persistence_predictions(frame, target_spec)

    raise ValueError(f"Unknown baseline: {baseline_name}")


def evaluate_persistence_baseline(
    split: SplitData,
    target_spec: TargetSpec,
) -> dict[str, float]:
    """Evaluate naive current-window persistence on one split."""

    predictions = make_persistence_predictions(split.frame, target_spec)
    actual_targets = torch.tensor(
        split.frame[split.target_column].to_numpy(dtype="float32")
    )
    return evaluate_predictions(predictions, actual_targets)


def evaluate_baseline_family(
    split: SplitData,
    target_spec: TargetSpec,
) -> dict[str, dict[str, float]]:
    """Evaluate the default family of naive baselines on one split."""

    baseline_names = ("persistence",)
    actual_targets = torch.tensor(
        split.frame[split.target_column].to_numpy(dtype="float32")
    )
    metrics: dict[str, dict[str, float]] = {}

    for baseline_name in baseline_names:
        predictions = make_baseline_predictions(
            split.frame,
            target_spec,
            baseline_name,
        )
        metrics[baseline_name] = evaluate_predictions(predictions, actual_targets)

    return metrics


def metrics_are_finite(metrics: dict[str, float]) -> bool:
    """Return whether all metric values are finite."""

    return all(math.isfinite(value) for value in metrics.values())
