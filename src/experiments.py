"""Experiment assembly, execution, and output helpers."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from xgboost import XGBRegressor

from src.cli import format_feature_blocks
from src.data import build_feature_frame
from src.features import choose_feature_columns
from src.features import filter_feature_rows
from src.features import prepare_feature_frame
from src.modeling import ExperimentResult
from src.modeling import RANDOM_SEED
from src.modeling import SplitData
from src.modeling import TargetStandardization
from src.modeling import compute_standardization
from src.modeling import compute_target_standardization
from src.modeling import evaluate_baseline_family
from src.modeling import evaluate_persistence_baseline
from src.modeling import filter_target_rows
from src.modeling import metrics_are_finite
from src.modeling import resolve_target_transform
from src.modeling import split_by_time
from src.modeling import standardize_split
from src.modeling import standardize_targets
from src.modeling import to_split
from src.modeling import validate_target_column
from src.modeling import validate_target_kind
from src.regressors import evaluate_regressor
from src.regressors import train_regressor
from src.regressors import validate_model_backend
from src.reporting import print_experiment_summary
from src.sequence_data import to_sequence_split
from src.targets import TargetSpec
from src.targets import add_next_window_targets
from src.targets import get_target_spec


def build_modeling_frame(
    database_path: Path,
    train_router: str | None,
) -> pd.DataFrame:
    """Load the modeling frame once before running experiments."""

    frame = build_feature_frame(database_path, train_router=train_router)
    return add_next_window_targets(frame)


def prepare_experiment_frame(
    frame: pd.DataFrame,
    target_spec: TargetSpec,
    target_column: str,
    feature_blocks: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    """Build feature columns and filter rows for one experiment."""

    experiment_frame = prepare_feature_frame(
        frame=frame,
        target_base_column=target_spec.source_column,
        feature_blocks=feature_blocks,
    )
    feature_columns = choose_feature_columns(
        experiment_frame,
        feature_blocks=feature_blocks,
    )
    experiment_frame = filter_feature_rows(experiment_frame, feature_columns)
    experiment_frame = filter_target_rows(experiment_frame, target_column)
    return experiment_frame, feature_columns


def make_model_splits(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_backend: str,
    sequence_length: int,
) -> tuple[SplitData, SplitData, SplitData]:
    """Build train/validation/test tensors for one backend."""

    train_frame, valid_frame, test_frame = split_by_time(frame)

    if model_backend in ("gru", "mlp", "curve_gru"):
        train_split = to_sequence_split(
            train_frame,
            feature_columns,
            target_column,
            sequence_length,
        )
        valid_split = to_sequence_split(
            valid_frame,
            feature_columns,
            target_column,
            sequence_length,
            prior_history_rows=train_frame,
        )
        test_split = to_sequence_split(
            test_frame,
            feature_columns,
            target_column,
            sequence_length,
            prior_history_rows=pd.concat([train_frame, valid_frame], ignore_index=True),
        )
        return train_split, valid_split, test_split

    return (
        to_split(train_frame, feature_columns, target_column),
        to_split(valid_frame, feature_columns, target_column),
        to_split(test_frame, feature_columns, target_column),
    )


def run_regression_experiment_once(
    frame: pd.DataFrame,
    target_column: str,
    model_backend: str,
    feature_blocks: tuple[str, ...],
    sequence_length: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
    loss_name: str | None,
    target_transform: str | None,
    report_progress: bool,
) -> tuple[
    ExperimentResult,
    SplitData,
    SplitData,
    TargetStandardization,
    nn.Module | XGBRegressor,
    list[str],
]:
    """Run one regression experiment on one device."""

    validate_target_column(frame, target_column)
    validate_model_backend(model_backend)
    target_spec = get_target_spec(target_column)
    validate_target_kind(target_spec)
    experiment_frame, feature_columns = prepare_experiment_frame(
        frame=frame,
        target_spec=target_spec,
        target_column=target_column,
        feature_blocks=feature_blocks,
    )
    train_split, valid_split, test_split = make_model_splits(
        frame=experiment_frame,
        feature_columns=feature_columns,
        target_column=target_column,
        model_backend=model_backend,
        sequence_length=sequence_length,
    )
    baseline_valid_metrics = evaluate_baseline_family(valid_split, target_spec)
    baseline_test_metrics = evaluate_baseline_family(test_split, target_spec)
    persistence_valid_metrics = evaluate_persistence_baseline(valid_split, target_spec)
    persistence_test_metrics = evaluate_persistence_baseline(test_split, target_spec)
    target_transform = resolve_target_transform(
        model_backend=model_backend,
        target_spec=target_spec,
        requested_transform=target_transform,
    )

    if model_backend == "xgboost" and target_transform != "standard":
        raise ValueError(
            "Target transforms are currently supported only for torch backends."
        )

    target_stats = compute_target_standardization(train_split, target_transform)

    if model_backend != "xgboost":
        feature_stats = compute_standardization(train_split)
        train_split = standardize_split(train_split, feature_stats)
        valid_split = standardize_split(valid_split, feature_stats)
        test_split = standardize_split(test_split, feature_stats)
        train_split = standardize_targets(train_split, target_stats)
        valid_split = standardize_targets(valid_split, target_stats)
        test_split = standardize_targets(test_split, target_stats)

    torch.manual_seed(RANDOM_SEED)
    model, target_stats, model_device = train_regressor(
        model_backend=model_backend,
        train_split=train_split,
        valid_split=valid_split,
        target_stats=target_stats,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        feature_columns=feature_columns,
        loss_name=loss_name,
        report_progress=report_progress,
    )
    valid_metrics = evaluate_regressor(model, valid_split, target_stats)
    test_metrics = evaluate_regressor(model, test_split, target_stats)
    result = ExperimentResult(
        target=target_column,
        task_kind=target_spec.task_kind,
        model_backend=model_backend,
        feature_blocks=feature_blocks,
        feature_count=len(feature_columns),
        epochs=epochs,
        device=model_device,
        baseline_metrics={
            "validation": baseline_valid_metrics,
            "test": baseline_test_metrics,
        },
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
    return result, valid_split, test_split, target_stats, model, feature_columns


def run_regression_experiment(
    frame: pd.DataFrame,
    target_column: str,
    model_backend: str,
    feature_blocks: tuple[str, ...],
    sequence_length: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
    loss_name: str | None,
    target_transform: str | None,
    report_progress: bool,
) -> tuple[
    ExperimentResult,
    SplitData,
    SplitData,
    TargetStandardization,
    nn.Module | XGBRegressor,
    list[str],
]:
    """Run one regression experiment and retry on CPU if MPS goes non-finite."""

    result, valid_split, test_split, target_stats, model, feature_columns = run_regression_experiment_once(
        frame=frame,
        target_column=target_column,
        model_backend=model_backend,
        feature_blocks=feature_blocks,
        sequence_length=sequence_length,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        loss_name=loss_name,
        target_transform=target_transform,
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
        return result, valid_split, test_split, target_stats, model, feature_columns

    print(
        f"[WARN] non-finite metrics on MPS, retrying on CPU  "
        f"target={target_column}  blocks={format_feature_blocks(feature_blocks)}",
        flush=True,
    )
    return run_regression_experiment_once(
        frame=frame,
        target_column=target_column,
        model_backend=model_backend,
        feature_blocks=feature_blocks,
        sequence_length=sequence_length,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=torch.device("cpu"),
        loss_name=loss_name,
        target_transform=target_transform,
        report_progress=report_progress,
    )


def run_experiment_matrix(
    frame: pd.DataFrame,
    targets: list[str],
    block_configs: list[tuple[str, ...]],
    model_backend: str,
    sequence_length: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
    loss_name: str | None = None,
    target_transform: str | None = None,
) -> list[ExperimentResult]:
    """Run and print a matrix of experiments."""

    results: list[ExperimentResult] = []
    total_experiments = len(targets) * len(block_configs)
    completed_experiments = 0
    overall_start = time.perf_counter()

    print(
        f"Running {total_experiments} experiments  "
        f"backend={model_backend}  seq_len={sequence_length}  "
        f"epochs={epochs}  device={device.type}",
        flush=True,
    )

    for target_column in targets:
        for feature_blocks in block_configs:
            completed_experiments += 1
            experiment_start = time.perf_counter()
            print(
                f"[{completed_experiments}/{total_experiments}]  "
                f"{target_column}  blocks={format_feature_blocks(feature_blocks)}",
                flush=True,
            )
            result, _, _, _, _, _ = run_regression_experiment(
                frame=frame,
                target_column=target_column,
                model_backend=model_backend,
                feature_blocks=feature_blocks,
                sequence_length=sequence_length,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                device=device,
                loss_name=loss_name,
                target_transform=target_transform,
                report_progress=False,
            )
            results.append(result)
            elapsed_seconds = time.perf_counter() - experiment_start
            print(
                f"[{completed_experiments}/{total_experiments}]  done in {elapsed_seconds:.1f}s  "
                f"{target_column}  blocks={format_feature_blocks(feature_blocks)}  "
                f"test_mae={result.model_test_mae:.2f}  test_r2={result.model_test_r2:.4f}  "
                f"device={result.device}",
                flush=True,
            )

    total_elapsed_seconds = time.perf_counter() - overall_start
    print(
        f"Completed {total_experiments} experiments in {total_elapsed_seconds:.1f}s",
        flush=True,
    )
    print()
    print(f"Experiment summary ({total_experiments} runs):")
    print()
    print_experiment_summary(results)
    return results
