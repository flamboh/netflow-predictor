"""Train a simple PyTorch regression model for 5-minute netflow traffic."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from src.data import build_feature_frame
from src.feature_analysis import filter_ranked_features
from src.feature_analysis import get_linear_feature_ranking
from src.features import FEATURE_BLOCK_NAMES
from src.features import add_spectrum_features
from src.features import add_structure_features
from src.features import add_target_family_features
from src.features import choose_feature_columns
from src.features import filter_feature_rows
from src.features import structure_block_requested
from src.targets import TargetSpec
from src.targets import add_next_window_targets
from src.targets import describe_targets
from src.targets import get_target_spec


TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
RANDOM_SEED = 0
DEFAULT_TARGET_COLUMN = "next_sa_ipv4_count"
DEFAULT_FEATURE_BLOCKS = ("target", "general")


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
        predictions = predictions.squeeze(1)
        return predictions


def parse_args() -> argparse.Namespace:
    """Read command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/netflow_window.sqlite"),
        help="Path to the SQLite dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of optimization steps over the training data.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Mini-batch size used during training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--router",
        type=str,
        default=None,
        help="Optional router name for a single prediction lookup.",
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        default=None,
        help="Optional 5-minute trace timestamp for a single prediction lookup.",
    )
    parser.add_argument(
        "--describe-targets",
        action="store_true",
        help="Print derived next-window target summaries and exit.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET_COLUMN,
        help="Regression target column. Default: next_sa_ipv4_count.",
    )
    parser.add_argument(
        "--feature-blocks",
        type=str,
        default="target,general",
        help=(
            "Comma-separated feature blocks. "
            "Available: target,general,context,spectrum,structure,"
            "structure_summary,structure_tau_samples,structure_sd_samples."
        ),
    )
    parser.add_argument(
        "--show-feature-ranking",
        action="store_true",
        help="Print ranked learned feature weights after a single run.",
    )
    parser.add_argument(
        "--ranking-prefixes",
        type=str,
        default="",
        help="Comma-separated feature prefixes to keep in the ranking output.",
    )
    parser.add_argument(
        "--ranking-top-k",
        type=int,
        default=20,
        help="Number of ranked features to print.",
    )
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run a matrix of target/block experiments and print a summary table.",
    )
    parser.add_argument(
        "--experiment-targets",
        type=str,
        default=(
            "next_sa_ipv4_count;"
            "next_da_ipv4_count;"
            "next_sa_ipv4_count_delta;"
            "next_da_ipv4_count_delta"
        ),
        help="Semicolon-separated regression targets for experiment mode.",
    )
    parser.add_argument(
        "--experiment-feature-blocks",
        type=str,
        default=(
            "target,general;"
            "target,general,structure_summary;"
            "target,general,structure_tau_samples;"
            "target,general,structure_sd_samples;"
            "target,general,structure_summary,structure_tau_samples;"
            "target,general,structure_summary,structure_sd_samples"
        ),
        help="Semicolon-separated feature-block configs for experiment mode.",
    )
    args = parser.parse_args()
    return args


def parse_feature_blocks(raw_value: str) -> tuple[str, ...]:
    """Parse requested feature blocks in a stable order."""

    requested = [block.strip() for block in raw_value.split(",") if block.strip()]

    if not requested:
        return DEFAULT_FEATURE_BLOCKS

    ordered_blocks = [
        block
        for block in FEATURE_BLOCK_NAMES
        if block in requested
    ]
    unknown_blocks = sorted(set(requested) - set(FEATURE_BLOCK_NAMES))

    if unknown_blocks:
        block_list = ", ".join(unknown_blocks)
        raise ValueError(f"Unknown feature blocks: {block_list}")

    return tuple(ordered_blocks)


def parse_experiment_targets(raw_value: str) -> list[str]:
    """Parse semicolon-separated experiment targets."""

    targets = [value.strip() for value in raw_value.split(";") if value.strip()]

    if not targets:
        return [DEFAULT_TARGET_COLUMN]

    return targets


def parse_experiment_feature_blocks(raw_value: str) -> list[tuple[str, ...]]:
    """Parse semicolon-separated block configs for experiment mode."""

    block_configs = [
        parse_feature_blocks(value)
        for value in raw_value.split(";")
        if value.strip()
    ]

    if not block_configs:
        return [DEFAULT_FEATURE_BLOCKS]

    return block_configs


def format_feature_blocks(feature_blocks: tuple[str, ...]) -> str:
    """Format a block tuple for logs and tables."""

    return ",".join(feature_blocks)


def parse_ranking_prefixes(raw_value: str) -> list[str]:
    """Parse comma-separated feature prefixes for ranking output."""

    prefixes = [value.strip() for value in raw_value.split(",") if value.strip()]
    return prefixes


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

    moved = TargetStandardization(
        mean=stats.mean.to(device),
        std=stats.std.to(device),
    )
    return moved


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

    filtered = frame.dropna(subset=[target_column]).reset_index(drop=True)
    return filtered


def to_split(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> SplitData:
    """Convert one DataFrame split into tensors."""

    feature_values = frame[feature_columns].to_numpy(dtype="float32")
    target_values = frame[target_column].to_numpy(dtype="float32")
    features = torch.tensor(feature_values)
    targets = torch.tensor(target_values)
    split = SplitData(
        features=features,
        targets=targets,
        frame=frame,
        target_column=target_column,
    )
    return split


def compute_standardization(split: SplitData) -> Standardization:
    """Compute training-set scaling values."""

    mean = split.features.mean(dim=0)
    std = split.features.std(dim=0)
    std = torch.where(std == 0, torch.ones_like(std), std)
    stats = Standardization(mean=mean, std=std)
    return stats


def standardize_split(split: SplitData, stats: Standardization) -> SplitData:
    """Apply training-set scaling to one split."""

    scaled_features = (split.features - stats.mean) / stats.std
    scaled_split = SplitData(
        features=scaled_features,
        targets=split.targets,
        frame=split.frame,
        target_column=split.target_column,
    )
    return scaled_split


def compute_target_standardization(split: SplitData) -> TargetStandardization:
    """Compute training-set scaling values for the target."""

    mean = split.targets.mean()
    std = split.targets.std()
    std = torch.where(std == 0, torch.ones_like(std), std)
    stats = TargetStandardization(mean=mean, std=std)
    return stats


def standardize_targets(
    split: SplitData,
    stats: TargetStandardization,
) -> SplitData:
    """Apply training-set target scaling to one split."""

    scaled_targets = (split.targets - stats.mean) / stats.std
    scaled_split = SplitData(
        features=split.features,
        targets=scaled_targets,
        frame=split.frame,
        target_column=split.target_column,
    )
    return scaled_split


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
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    return loader


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
        predictions = model(features)

    return predictions


def predict(
    model: nn.Module,
    split: SplitData,
    target_stats: TargetStandardization,
) -> torch.Tensor:
    """Run the model and return predictions on the original target scale."""

    device = get_model_device(model)
    scaled_predictions = predict_scaled(model, split)
    predictions = (
        scaled_predictions * target_stats.std.to(device)
        + target_stats.mean.to(device)
    )
    return predictions


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
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    return metrics


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


def metrics_are_finite(metrics: dict[str, float]) -> bool:
    """Return whether all metric values are finite."""

    return all(math.isfinite(value) for value in metrics.values())


def build_modeling_frame(database_path: Path) -> pd.DataFrame:
    """Load the modeling frame once before running experiments."""

    frame = build_feature_frame(database_path)
    frame = add_next_window_targets(frame)
    return frame


def prepare_experiment_frame(
    frame: pd.DataFrame,
    target_spec: TargetSpec,
    target_column: str,
    feature_blocks: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    """Build feature columns and filter rows for one experiment."""

    experiment_frame = add_target_family_features(frame, target_spec.source_column)

    if "spectrum" in feature_blocks:
        experiment_frame = add_spectrum_features(
            experiment_frame,
            target_spec.source_column,
        )

    if structure_block_requested(feature_blocks):
        experiment_frame = add_structure_features(
            experiment_frame,
            target_spec.source_column,
        )

    feature_columns = choose_feature_columns(
        experiment_frame,
        target_spec.source_column,
        feature_blocks=feature_blocks,
    )
    experiment_frame = filter_feature_rows(experiment_frame, feature_columns)
    experiment_frame = filter_target_rows(experiment_frame, target_column)
    return experiment_frame, feature_columns


def run_regression_experiment_once(
    frame: pd.DataFrame,
    target_column: str,
    feature_blocks: tuple[str, ...],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
    report_progress: bool,
) -> tuple[ExperimentResult, SplitData, TargetStandardization, nn.Module, list[str]]:
    """Run one regression experiment on one device."""

    validate_target_column(frame, target_column)
    target_spec = get_target_spec(target_column)
    validate_target_kind(target_spec)
    experiment_frame, feature_columns = prepare_experiment_frame(
        frame=frame,
        target_spec=target_spec,
        target_column=target_column,
        feature_blocks=feature_blocks,
    )
    train_frame, valid_frame, test_frame = split_by_time(experiment_frame)
    train_split = to_split(train_frame, feature_columns, target_column)
    valid_split = to_split(valid_frame, feature_columns, target_column)
    test_split = to_split(test_frame, feature_columns, target_column)
    persistence_valid_metrics = evaluate_persistence_baseline(valid_split, target_spec)
    persistence_test_metrics = evaluate_persistence_baseline(test_split, target_spec)
    feature_stats = compute_standardization(train_split)
    target_stats = compute_target_standardization(train_split)
    train_split = standardize_split(train_split, feature_stats)
    valid_split = standardize_split(valid_split, feature_stats)
    test_split = standardize_split(test_split, feature_stats)
    train_split = standardize_targets(train_split, target_stats)
    valid_split = standardize_targets(valid_split, target_stats)
    test_split = standardize_targets(test_split, target_stats)
    train_loader = make_loader(train_split, batch_size, shuffle=True, device=device)
    target_stats = move_target_stats(target_stats, device)
    torch.manual_seed(RANDOM_SEED)
    model = LinearRegressionModel(feature_count=len(feature_columns))
    model = model.to(device)
    train_model(
        model=model,
        train_loader=train_loader,
        valid_split=valid_split,
        target_stats=target_stats,
        epochs=epochs,
        learning_rate=learning_rate,
        report_progress=report_progress,
    )
    valid_metrics = evaluate_model(model, valid_split, target_stats)
    test_metrics = evaluate_model(model, test_split, target_stats)
    result = ExperimentResult(
        target=target_column,
        task_kind=target_spec.task_kind,
        feature_blocks=feature_blocks,
        feature_count=len(feature_columns),
        epochs=epochs,
        device=device.type,
        persistence_valid_mae=persistence_valid_metrics["mae"],
        persistence_valid_rmse=persistence_valid_metrics["rmse"],
        persistence_valid_r2=persistence_valid_metrics["r2"],
        persistence_test_mae=persistence_test_metrics["mae"],
        persistence_test_rmse=persistence_test_metrics["rmse"],
        persistence_test_r2=persistence_test_metrics["r2"],
        model_valid_mae=valid_metrics["mae"],
        model_valid_rmse=valid_metrics["rmse"],
        model_valid_r2=valid_metrics["r2"],
        model_test_mae=test_metrics["mae"],
        model_test_rmse=test_metrics["rmse"],
        model_test_r2=test_metrics["r2"],
    )
    return result, test_split, target_stats, model, feature_columns


def run_regression_experiment(
    frame: pd.DataFrame,
    target_column: str,
    feature_blocks: tuple[str, ...],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
    report_progress: bool,
) -> tuple[ExperimentResult, SplitData, TargetStandardization, nn.Module, list[str]]:
    """Run one regression experiment and retry on CPU if MPS goes non-finite."""

    result, test_split, target_stats, model, feature_columns = run_regression_experiment_once(
        frame=frame,
        target_column=target_column,
        feature_blocks=feature_blocks,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        report_progress=report_progress,
    )

    model_metrics_are_finite = metrics_are_finite(
        {
            "valid_mae": result.model_valid_mae,
            "valid_rmse": result.model_valid_rmse,
            "valid_r2": result.model_valid_r2,
            "test_mae": result.model_test_mae,
            "test_rmse": result.model_test_rmse,
            "test_r2": result.model_test_r2,
        }
    )

    if device.type != "mps" or model_metrics_are_finite:
        return result, test_split, target_stats, model, feature_columns

    print(
        "Warning: non-finite metrics on MPS; retrying on CPU "
        f"for target={target_column} blocks={format_feature_blocks(feature_blocks)}",
        flush=True,
    )
    return run_regression_experiment_once(
        frame=frame,
        target_column=target_column,
        feature_blocks=feature_blocks,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=torch.device("cpu"),
        report_progress=report_progress,
    )


def print_experiment_summary(results: list[ExperimentResult]) -> None:
    """Print a compact experiment comparison table."""

    summary_rows = []

    for result in results:
        summary_rows.append(
            {
                "target": result.target,
                "blocks": format_feature_blocks(result.feature_blocks),
                "features": result.feature_count,
                "epochs": result.epochs,
                "device": result.device,
                "persist_val_mae": round(result.persistence_valid_mae, 2),
                "model_val_mae": round(result.model_valid_mae, 2),
                "val_mae_delta": round(
                    result.model_valid_mae - result.persistence_valid_mae,
                    2,
                ),
                "persist_val_r2": round(result.persistence_valid_r2, 6),
                "model_val_r2": round(result.model_valid_r2, 6),
                "val_r2_delta": round(
                    result.model_valid_r2 - result.persistence_valid_r2,
                    6,
                ),
                "persist_test_mae": round(result.persistence_test_mae, 2),
                "model_test_mae": round(result.model_test_mae, 2),
                "mae_delta": round(
                    result.model_test_mae - result.persistence_test_mae,
                    2,
                ),
                "persist_test_r2": round(result.persistence_test_r2, 6),
                "model_test_r2": round(result.model_test_r2, 6),
                "r2_delta": round(
                    result.model_test_r2 - result.persistence_test_r2,
                    6,
                ),
            }
        )

    summary_frame = pd.DataFrame(summary_rows)
    print(summary_frame.to_string(index=False))


def show_test_examples(
    model: nn.Module,
    split: SplitData,
    target_stats: TargetStandardization,
) -> None:
    """Print a few held-out predictions."""

    predictions = predict(model, split, target_stats).cpu().numpy()
    example_frame = split.frame[
        ["router_name", "timestamp", split.target_column]
    ].copy()
    example_frame = example_frame.rename(columns={"router_name": "router"})
    example_frame = example_frame.rename(
        columns={split.target_column: "actual_target"}
    )
    example_frame["predicted_target"] = predictions
    print()
    print("Sample test predictions:")
    print(example_frame.head(10).to_string(index=False))


def show_requested_prediction(
    model: nn.Module,
    split: SplitData,
    target_stats: TargetStandardization,
    router: str | None,
    timestamp: int | None,
) -> None:
    """Print the prediction for one requested 5-minute trace."""

    if router is None or timestamp is None:
        return

    matches = split.frame["router_name"].eq(router)
    matches &= split.frame["timestamp"].eq(timestamp)
    match_count = int(matches.sum())

    if match_count == 0:
        print()
        print("Requested trace was not found in the test split.")
        return

    prediction_frame = split.frame.loc[
        matches,
        ["router_name", "timestamp", split.target_column],
    ].copy()
    prediction_frame = prediction_frame.rename(columns={"router_name": "router"})
    prediction_split = SplitData(
        features=split.features[matches.to_numpy()],
        targets=split.targets[matches.to_numpy()],
        frame=prediction_frame,
        target_column=split.target_column,
    )
    prediction = predict(model, prediction_split, target_stats).item()
    actual = prediction_frame[split.target_column].iloc[0]
    print()
    print("Requested trace prediction:")
    print(f"router={router}")
    print(f"timestamp={timestamp}")
    print(f"target={split.target_column}")
    print(f"predicted_value={prediction:.2f}")
    print(f"actual_value={actual:.2f}")


def main() -> None:
    """Train and evaluate the regression baseline."""

    torch.manual_seed(RANDOM_SEED)
    args = parse_args()
    device = resolve_device(args.device)
    frame = build_modeling_frame(args.database)

    if args.describe_targets:
        target_summary = describe_targets(frame)
        print(target_summary.to_string(index=False))
        return

    if args.run_experiments:
        targets = parse_experiment_targets(args.experiment_targets)
        block_configs = parse_experiment_feature_blocks(
            args.experiment_feature_blocks
        )
        results: list[ExperimentResult] = []
        total_experiments = len(targets) * len(block_configs)
        completed_experiments = 0
        overall_start = time.perf_counter()

        print(
            "Running experiments: "
            f"{total_experiments} configs, epochs={args.epochs}, device={device.type}",
            flush=True,
        )

        for target_column in targets:
            for feature_blocks in block_configs:
                completed_experiments += 1
                experiment_start = time.perf_counter()
                print(
                    f"[{completed_experiments}/{total_experiments}] "
                    f"target={target_column} "
                    f"blocks={format_feature_blocks(feature_blocks)} "
                    "starting",
                    flush=True,
                )
                result, _, _, _, _ = run_regression_experiment(
                    frame=frame,
                    target_column=target_column,
                    feature_blocks=feature_blocks,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    device=device,
                    report_progress=False,
                )
                results.append(result)
                elapsed_seconds = time.perf_counter() - experiment_start
                print(
                    f"[{completed_experiments}/{total_experiments}] "
                    f"target={target_column} "
                    f"blocks={format_feature_blocks(feature_blocks)} "
                    f"done in {elapsed_seconds:.1f}s "
                    f"device={result.device} "
                    f"test_mae={result.model_test_mae:.2f} "
                    f"test_r2={result.model_test_r2:.6f}",
                    flush=True,
                )

        total_elapsed_seconds = time.perf_counter() - overall_start
        print(
            f"Completed {total_experiments} experiments "
            f"in {total_elapsed_seconds:.1f}s",
            flush=True,
        )
        print()
        print_experiment_summary(results)
        return

    feature_blocks = parse_feature_blocks(args.feature_blocks)
    result, test_split, target_stats, model, feature_columns = run_regression_experiment(
        frame=frame,
        target_column=args.target,
        feature_blocks=feature_blocks,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=device,
        report_progress=True,
    )
    print()
    print("Device:")
    print(result.device)
    print()
    print("Feature blocks:")
    print(format_feature_blocks(feature_blocks))
    print()
    print("Persistence baseline metrics:")
    print(f"target={args.target}")
    print(
        "validation="
        f"{{'mae': {result.persistence_valid_mae}, "
        f"'rmse': {result.persistence_valid_rmse}, "
        f"'r2': {result.persistence_valid_r2}}}"
    )
    print(
        "test="
        f"{{'mae': {result.persistence_test_mae}, "
        f"'rmse': {result.persistence_test_rmse}, "
        f"'r2': {result.persistence_test_r2}}}"
    )
    print()
    print("Validation metrics:")
    print(f"target={args.target}")
    print(
        {
            "mae": result.model_valid_mae,
            "rmse": result.model_valid_rmse,
            "r2": result.model_valid_r2,
        }
    )
    print()
    print("Test metrics:")
    print(
        {
            "mae": result.model_test_mae,
            "rmse": result.model_test_rmse,
            "r2": result.model_test_r2,
        }
    )
    if args.show_feature_ranking:
        ranking = get_linear_feature_ranking(model, feature_columns)
        ranking = filter_ranked_features(
            ranking,
            parse_ranking_prefixes(args.ranking_prefixes),
        )
        ranking = ranking.head(args.ranking_top_k)
        print()
        print("Feature ranking:")
        if ranking.empty:
            print("No features matched the requested ranking prefixes.")
        else:
            print(ranking.to_string(index=False))
    show_test_examples(model, test_split, target_stats)
    show_requested_prediction(
        model,
        test_split,
        target_stats,
        args.router,
        args.timestamp,
    )
