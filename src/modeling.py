"""Model training, evaluation, and tensor-side helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from src.targets import TargetSpec
from src.targets import get_target_spec


TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
RANDOM_SEED = 0


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


@dataclass
class ExperimentResult:
    """Store summary metrics for one experiment configuration."""

    target: str
    task_kind: str
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


class LinearRegressionModel(nn.Module):
    """A single linear layer for regression."""

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_count, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        predictions = self.linear(features)
        return predictions.squeeze(1)


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


def move_target_stats(
    stats: TargetStandardization,
    device: torch.device,
) -> TargetStandardization:
    """Move target scaling tensors onto the selected device."""

    return TargetStandardization(
        mean=stats.mean.to(device),
        std=stats.std.to(device),
    )


def get_model_device(model: nn.Module) -> torch.device:
    """Return the device used by the model parameters."""

    return next(model.parameters()).device


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

    mean = split.features.mean(dim=0)
    std = split.features.std(dim=0)
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


def compute_target_standardization(split: SplitData) -> TargetStandardization:
    """Compute training-set scaling values for the target."""

    mean = split.targets.mean()
    std = split.targets.std()
    std = torch.where(std == 0, torch.ones_like(std), std)
    return TargetStandardization(mean=mean, std=std)


def standardize_targets(
    split: SplitData,
    stats: TargetStandardization,
) -> SplitData:
    """Apply training-set target scaling to one split."""

    return SplitData(
        features=split.features,
        targets=(split.targets - stats.mean) / stats.std,
        frame=split.frame,
        target_column=split.target_column,
    )


def make_loader(
    split: SplitData,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Create a DataLoader for one split."""

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    dataset = TensorDataset(split.features, split.targets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        pin_memory=device.type == "cuda",
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_split: SplitData,
    target_stats: TargetStandardization,
    epochs: int,
    learning_rate: float,
    report_progress: bool = True,
) -> None:
    """Fit the linear regression model."""

    device = get_model_device(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()

        for feature_batch, target_batch in train_loader:
            feature_batch = feature_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            predictions = model(feature_batch)
            loss = loss_fn(predictions, target_batch)
            loss.backward()
            optimizer.step()

        if report_progress and ((epoch + 1) % 25 == 0 or epoch == 0):
            valid_metrics = evaluate_model(model, valid_split, target_stats)
            print(
                f"epoch={epoch + 1} "
                f"valid_mae={valid_metrics['mae']:.2f} "
                f"valid_rmse={valid_metrics['rmse']:.2f} "
                f"valid_r2={valid_metrics['r2']:.4f}"
            )


def predict_scaled(model: nn.Module, split: SplitData) -> torch.Tensor:
    """Run the model and return standardized predictions."""

    device = get_model_device(model)
    model.eval()

    with torch.no_grad():
        features = split.features.to(device)
        return model(features)


def predict(
    model: nn.Module,
    split: SplitData,
    target_stats: TargetStandardization,
) -> torch.Tensor:
    """Run the model and return predictions on the original target scale."""

    device = get_model_device(model)
    scaled_predictions = predict_scaled(model, split)
    return scaled_predictions * target_stats.std.to(device) + target_stats.mean.to(device)


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


def evaluate_model(
    model: nn.Module,
    split: SplitData,
    target_stats: TargetStandardization,
) -> dict[str, float]:
    """Evaluate the learned model on one split."""

    predictions = predict(model, split, target_stats)
    actual_targets = torch.tensor(
        split.frame[split.target_column].to_numpy(dtype="float32"),
        device=predictions.device,
    )
    return evaluate_predictions(predictions, actual_targets)


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

    base_column = target_spec.source_column
    current_values = frame[base_column].astype("float32")
    persistence_predictions = pd.Series(
        make_persistence_predictions(frame, target_spec).numpy(),
        index=frame.index,
        dtype="float32",
    )

    if baseline_name == "persistence":
        return torch.tensor(persistence_predictions.to_numpy(dtype="float32"))

    if baseline_name == "moving_avg_3":
        rolling_values = frame[f"{base_column}_rolling_mean_3"].astype("float32")
        if target_spec.task_kind == "exact_count":
            predictions = rolling_values
        elif target_spec.task_kind == "delta":
            predictions = rolling_values - current_values
        else:
            raise ValueError(f"Unsupported baseline for {target_spec.task_kind}")
        predictions = predictions.fillna(persistence_predictions)
        return torch.tensor(predictions.to_numpy(dtype="float32"))

    if baseline_name == "moving_avg_12":
        rolling_values = frame[f"{base_column}_rolling_mean_12"].astype("float32")
        if target_spec.task_kind == "exact_count":
            predictions = rolling_values
        elif target_spec.task_kind == "delta":
            predictions = rolling_values - current_values
        else:
            raise ValueError(f"Unsupported baseline for {target_spec.task_kind}")
        predictions = predictions.fillna(persistence_predictions)
        return torch.tensor(predictions.to_numpy(dtype="float32"))

    if baseline_name == "lag_12":
        lag_values = frame[f"{base_column}_lag_12"].astype("float32")
        if target_spec.task_kind == "exact_count":
            predictions = lag_values
        elif target_spec.task_kind == "delta":
            predictions = lag_values - current_values
        else:
            raise ValueError(f"Unsupported baseline for {target_spec.task_kind}")
        predictions = predictions.fillna(persistence_predictions)
        return torch.tensor(predictions.to_numpy(dtype="float32"))

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

    baseline_names = (
        "persistence",
        "moving_avg_3",
        "moving_avg_12",
        "lag_12",
    )
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
