"""Experiment assembly, execution, and output helpers."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.cli import format_feature_blocks
from src.data import build_feature_frame
from src.features import add_spectrum_features
from src.features import add_structure_features
from src.features import add_target_family_features
from src.features import choose_feature_columns
from src.features import filter_feature_rows
from src.features import structure_block_requested
from src.modeling import ExperimentResult
from src.modeling import LinearRegressionModel
from src.modeling import RANDOM_SEED
from src.modeling import SplitData
from src.modeling import TargetStandardization
from src.modeling import compute_standardization
from src.modeling import compute_target_standardization
from src.modeling import evaluate_model
from src.modeling import evaluate_persistence_baseline
from src.modeling import filter_target_rows
from src.modeling import make_loader
from src.modeling import metrics_are_finite
from src.modeling import move_target_stats
from src.modeling import predict
from src.modeling import split_by_time
from src.modeling import standardize_split
from src.modeling import standardize_targets
from src.modeling import to_split
from src.modeling import train_model
from src.modeling import validate_target_column
from src.modeling import validate_target_kind
from src.targets import TargetSpec
from src.targets import add_next_window_targets
from src.targets import get_target_spec


def build_modeling_frame(database_path: Path) -> pd.DataFrame:
    """Load the modeling frame once before running experiments."""

    frame = build_feature_frame(database_path)
    return add_next_window_targets(frame)


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
    model = LinearRegressionModel(feature_count=len(feature_columns)).to(device)
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

    print(pd.DataFrame(summary_rows).to_string(index=False))


def run_experiment_matrix(
    frame: pd.DataFrame,
    targets: list[str],
    block_configs: list[tuple[str, ...]],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
) -> list[ExperimentResult]:
    """Run and print a matrix of experiments."""

    results: list[ExperimentResult] = []
    total_experiments = len(targets) * len(block_configs)
    completed_experiments = 0
    overall_start = time.perf_counter()

    print(
        "Running experiments: "
        f"{total_experiments} configs, epochs={epochs}, device={device.type}",
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
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
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
    return results


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

    if int(matches.sum()) == 0:
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
