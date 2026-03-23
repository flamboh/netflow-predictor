"""Train a simple PyTorch regression model for 5-minute netflow traffic."""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


SECONDS_PER_HOUR = 60 * 60
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
RANDOM_SEED = 0


@dataclass
class SplitData:
    """Store tensors for one dataset split."""

    features: torch.Tensor
    targets: torch.Tensor
    frame: pd.DataFrame


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
    args = parser.parse_args()
    return args


def read_table(connection: sqlite3.Connection, query: str) -> pd.DataFrame:
    """Load a SQL query into a DataFrame."""

    frame = pd.read_sql_query(query, connection)
    return frame


def load_base_frame(database_path: Path) -> pd.DataFrame:
    """Load and join the small set of tables used by the baseline model."""

    connection = sqlite3.connect(database_path)

    netflow_query = """
    SELECT
        router,
        timestamp,
        flows,
        packets,
        bytes,
        flows_tcp,
        flows_udp,
        bytes_tcp,
        bytes_udp
    FROM netflow_stats
    ORDER BY router, timestamp
    """
    ip_query = """
    SELECT
        router,
        bucket_start AS timestamp,
        sa_ipv4_count,
        da_ipv4_count,
        sa_ipv6_count,
        da_ipv6_count
    FROM ip_stats
    WHERE granularity = '5m'
    """
    protocol_query = """
    SELECT
        router,
        bucket_start AS timestamp,
        unique_protocols_count_ipv4,
        unique_protocols_count_ipv6
    FROM protocol_stats
    WHERE granularity = '5m'
    """

    netflow = read_table(connection, netflow_query)
    ip_stats = read_table(connection, ip_query)
    protocol_stats = read_table(connection, protocol_query)
    connection.close()

    merged = netflow.merge(ip_stats, on=["router", "timestamp"], how="left")
    merged = merged.merge(protocol_stats, on=["router", "timestamp"], how="left")
    return merged


def validate_join_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail early if joined feature tables are missing rows."""

    validated = frame.copy()
    joined_feature_columns = [
        "sa_ipv4_count",
        "da_ipv4_count",
        "sa_ipv6_count",
        "da_ipv6_count",
        "unique_protocols_count_ipv4",
        "unique_protocols_count_ipv6",
    ]
    missing_counts = validated[joined_feature_columns].isna().sum()
    missing_columns = missing_counts[missing_counts > 0]

    if missing_columns.empty:
        return validated

    details = ", ".join(
        f"{column}={count}"
        for column, count in missing_columns.items()
    )
    raise ValueError(
        "Joined feature tables contain missing values. "
        f"Missing counts: {details}"
    )


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create simple clock features from the trace timestamp."""

    enriched = frame.copy()
    enriched["hour_of_day"] = (enriched["timestamp"] // SECONDS_PER_HOUR) % 24
    enriched["day_of_week"] = (
        enriched["timestamp"] // SECONDS_PER_DAY + 3
    ) % 7
    return enriched


def add_router_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one-hot router features."""

    encoded = frame.copy()
    encoded["router_name"] = encoded["router"]
    encoded = pd.get_dummies(encoded, columns=["router"], dtype=float)
    return encoded


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a few previous-trace features for each router."""

    enriched = frame.sort_values(["router", "timestamp"]).copy()
    router_groups = enriched.groupby("router")
    enriched["bytes_lag_1"] = router_groups["bytes"].shift(1)
    enriched["bytes_lag_12"] = router_groups["bytes"].shift(12)
    enriched["packets_lag_1"] = router_groups["packets"].shift(1)
    enriched["flows_lag_1"] = router_groups["flows"].shift(1)
    lag_columns = ["bytes_lag_1", "bytes_lag_12", "packets_lag_1", "flows_lag_1"]
    enriched = enriched.dropna(subset=lag_columns).reset_index(drop=True)
    return enriched


def build_feature_frame(database_path: Path) -> pd.DataFrame:
    """Build the final modeling frame."""

    frame = load_base_frame(database_path)
    frame = validate_join_features(frame)
    frame = add_time_features(frame)
    frame = add_lag_features(frame)
    frame = add_router_features(frame)
    return frame


def choose_feature_columns(frame: pd.DataFrame) -> list[str]:
    """List feature columns in one explicit place."""

    feature_columns = [
        "flows",
        "packets",
        "flows_tcp",
        "flows_udp",
        "sa_ipv4_count",
        "da_ipv4_count",
        "sa_ipv6_count",
        "da_ipv6_count",
        "unique_protocols_count_ipv4",
        "unique_protocols_count_ipv6",
        "hour_of_day",
        "day_of_week",
        "bytes_lag_1",
        "bytes_lag_12",
        "packets_lag_1",
        "flows_lag_1",
    ]

    router_columns = sorted(
        column
        for column in frame.columns
        if column.startswith("router_") and column != "router_name"
    )
    feature_columns.extend(router_columns)
    return feature_columns


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


def to_split(frame: pd.DataFrame, feature_columns: list[str]) -> SplitData:
    """Convert one DataFrame split into tensors."""

    feature_values = frame[feature_columns].to_numpy(dtype="float32")
    target_values = frame["bytes"].to_numpy(dtype="float32")
    features = torch.tensor(feature_values)
    targets = torch.tensor(target_values)
    split = SplitData(features=features, targets=targets, frame=frame)
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
    )
    return scaled_split


def make_loader(split: SplitData, batch_size: int, shuffle: bool) -> DataLoader:
    """Create a DataLoader for one split."""

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    dataset = TensorDataset(split.features, split.targets)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
    return loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_split: SplitData,
    target_stats: TargetStandardization,
    epochs: int,
    learning_rate: float,
) -> None:
    """Fit the linear regression model."""

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()

        for feature_batch, target_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(feature_batch)
            loss = loss_fn(predictions, target_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 25 == 0 or epoch == 0:
            valid_metrics = evaluate_model(model, valid_split, target_stats)
            print(
                f"epoch={epoch + 1} "
                f"valid_mae={valid_metrics['mae']:.2f} "
                f"valid_rmse={valid_metrics['rmse']:.2f} "
                f"valid_r2={valid_metrics['r2']:.4f}"
            )


def predict_scaled(model: nn.Module, split: SplitData) -> torch.Tensor:
    """Run the model and return standardized predictions."""

    model.eval()

    with torch.no_grad():
        predictions = model(split.features)

    return predictions


def predict(
    model: nn.Module,
    split: SplitData,
    target_stats: TargetStandardization,
) -> torch.Tensor:
    """Run the model and return predictions on the original target scale."""

    scaled_predictions = predict_scaled(model, split)
    predictions = scaled_predictions * target_stats.std + target_stats.mean
    return predictions


def evaluate_model(
    model: nn.Module,
    split: SplitData,
    target_stats: TargetStandardization,
) -> dict[str, float]:
    """Compute a few easy-to-read regression metrics."""

    predictions = predict(model, split, target_stats)
    actual_targets = torch.tensor(
        split.frame["bytes"].to_numpy(dtype="float32")
    )
    residuals = predictions - actual_targets
    mae = residuals.abs().mean().item()
    rmse = residuals.square().mean().sqrt().item()
    centered_targets = actual_targets - actual_targets.mean()
    total_sum_of_squares = centered_targets.square().sum()
    residual_sum_of_squares = residuals.square().sum()
    r2 = 1.0 - (residual_sum_of_squares / total_sum_of_squares).item()
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    return metrics


def show_test_examples(
    model: nn.Module,
    split: SplitData,
    target_stats: TargetStandardization,
) -> None:
    """Print a few held-out predictions."""

    predictions = predict(model, split, target_stats).cpu().numpy()
    example_frame = split.frame[["router_name", "timestamp", "bytes"]].copy()
    example_frame = example_frame.rename(columns={"router_name": "router"})
    example_frame["predicted_bytes"] = predictions
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
        ["router_name", "timestamp", "bytes"],
    ].copy()
    prediction_frame = prediction_frame.rename(columns={"router_name": "router"})
    prediction_split = SplitData(
        features=split.features[matches.to_numpy()],
        targets=split.targets[matches.to_numpy()],
        frame=prediction_frame,
    )
    prediction = predict(model, prediction_split, target_stats).item()
    actual = prediction_frame["bytes"].iloc[0]
    print()
    print("Requested trace prediction:")
    print(f"router={router}")
    print(f"timestamp={timestamp}")
    print(f"predicted_bytes={prediction:.2f}")
    print(f"actual_bytes={actual:.2f}")


def main() -> None:
    """Train and evaluate the regression baseline."""

    torch.manual_seed(RANDOM_SEED)
    args = parse_args()
    frame = build_feature_frame(args.database)
    feature_columns = choose_feature_columns(frame)
    train_frame, valid_frame, test_frame = split_by_time(frame)
    train_split = to_split(train_frame, feature_columns)
    valid_split = to_split(valid_frame, feature_columns)
    test_split = to_split(test_frame, feature_columns)
    feature_stats = compute_standardization(train_split)
    target_stats = compute_target_standardization(train_split)
    train_split = standardize_split(train_split, feature_stats)
    valid_split = standardize_split(valid_split, feature_stats)
    test_split = standardize_split(test_split, feature_stats)
    train_split = standardize_targets(train_split, target_stats)
    valid_split = standardize_targets(valid_split, target_stats)
    test_split = standardize_targets(test_split, target_stats)
    train_loader = make_loader(train_split, args.batch_size, shuffle=True)
    model = LinearRegressionModel(feature_count=len(feature_columns))
    train_model(
        model=model,
        train_loader=train_loader,
        valid_split=valid_split,
        target_stats=target_stats,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    valid_metrics = evaluate_model(model, valid_split, target_stats)
    test_metrics = evaluate_model(model, test_split, target_stats)
    print()
    print("Validation metrics:")
    print(valid_metrics)
    print()
    print("Test metrics:")
    print(test_metrics)
    show_test_examples(model, test_split, target_stats)
    show_requested_prediction(
        model,
        test_split,
        target_stats,
        args.router,
        args.timestamp,
    )
