"""Experiment summary and prediction display helpers."""

from __future__ import annotations

import pandas as pd
from torch import nn
from xgboost import XGBRegressor

from src.cli import format_feature_blocks
from src.modeling import ExperimentResult
from src.modeling import SplitData
from src.modeling import TargetStandardization
from src.regressors import predict_regressor


def get_best_baseline_name(
    baseline_metrics: dict[str, dict[str, float]],
    metric_name: str,
    maximize: bool = False,
) -> str:
    """Return the best naive baseline for one metric."""

    metric_values = {
        name: metrics[metric_name]
        for name, metrics in baseline_metrics.items()
    }

    if maximize:
        return max(metric_values, key=metric_values.get)

    return min(metric_values, key=metric_values.get)


def format_baseline_name(baseline_name: str) -> str:
    """Shorten baseline labels for compact tables."""

    return {
        "persistence": "persist",
        "moving_avg_3": "ma3",
        "moving_avg_12": "ma12",
        "lag_12": "lag12",
    }.get(baseline_name, baseline_name)


def _signed(value: float, decimals: int = 2) -> str:
    """Format a float with an explicit leading sign."""

    formatted = f"{abs(value):.{decimals}f}"
    return f"+{formatted}" if value >= 0 else f"-{formatted}"


def print_run_config(
    target_column: str,
    model_backend: str,
    device: str,
    train_router: str,
    learning_rate: float,
    loss_name: str,
    target_transform: str,
    feature_blocks: tuple[str, ...],
    feature_count: int,
    epochs: int,
    sequence_length: int,
) -> None:
    """Print a compact run configuration block."""

    blocks_str = format_feature_blocks(feature_blocks)
    print("Config")
    print(f"  backend  {model_backend:<16}  device    {device}")
    print(f"  target   {target_column}")
    print(f"  router   {train_router}")
    print(f"  blocks   {blocks_str:<16}  features  {feature_count}")
    print(f"  lr       {learning_rate:<16.6g}  loss      {loss_name}")
    print(f"  tform    {target_transform}")
    print(f"  epochs   {epochs:<16}  seq_len   {sequence_length}")


def print_run_results(result: ExperimentResult) -> None:
    """Print a unified model-vs-baseline results table."""

    best_valid_name = get_best_baseline_name(
        result.baseline_metrics["validation"], "mae"
    )
    best_test_name = get_best_baseline_name(
        result.baseline_metrics["test"], "mae"
    )
    bv = result.baseline_metrics["validation"][best_valid_name]
    bt = result.baseline_metrics["test"][best_test_name]

    val_dmae  = result.model_valid_mae - bv["mae"]
    val_dr2   = result.model_valid_r2  - bv["r2"]
    test_dmae = result.model_test_mae  - bt["mae"]
    test_dr2  = result.model_test_r2   - bt["r2"]

    label_w = 22
    col_w = 9

    def row(label: str, a: str, b: str, c: str, d: str) -> str:
        return f"  {label:<{label_w}}  {a:>{col_w}}  {b:>{col_w}}  {c:>{col_w}}  {d:>{col_w}}"

    sep = "  " + "-" * (label_w + 4 * (col_w + 2) + 2)
    base_label = f"baseline ({format_baseline_name(best_test_name)})"

    print(f"Results  target={result.target}")
    print()
    print(row("", "val_mae", "val_r2", "test_mae", "test_r2"))
    print(sep)
    print(row(
        "model",
        f"{result.model_valid_mae:.2f}",
        f"{result.model_valid_r2:.4f}",
        f"{result.model_test_mae:.2f}",
        f"{result.model_test_r2:.4f}",
    ))
    print(row(
        base_label,
        f"{bv['mae']:.2f}",
        f"{bv['r2']:.4f}",
        f"{bt['mae']:.2f}",
        f"{bt['r2']:.4f}",
    ))
    print(sep)
    print(row(
        "Δ vs baseline",
        _signed(val_dmae),
        _signed(val_dr2, 4),
        _signed(test_dmae),
        _signed(test_dr2, 4),
    ))


def print_experiment_summary(results: list[ExperimentResult]) -> None:
    """Print a compact experiment comparison table."""

    summary_rows = []

    for result in results:
        best_valid_name = get_best_baseline_name(
            result.baseline_metrics["validation"], "mae"
        )
        best_test_name = get_best_baseline_name(
            result.baseline_metrics["test"], "mae"
        )
        best_test_r2_name = get_best_baseline_name(
            result.baseline_metrics["test"], "r2", maximize=True
        )
        bv = result.baseline_metrics["validation"][best_valid_name]
        bt = result.baseline_metrics["test"][best_test_name]
        bt_r2 = result.baseline_metrics["test"][best_test_r2_name]

        summary_rows.append(
            {
                "target":    result.target,
                "model":     result.model_backend,
                "blocks":    format_feature_blocks(result.feature_blocks),
                "feats":     result.feature_count,
                "val_mae":   round(result.model_valid_mae, 2),
                "val_Δmae":  round(result.model_valid_mae - bv["mae"], 2),
                "test_mae":  round(result.model_test_mae, 2),
                "test_Δmae": round(result.model_test_mae - bt["mae"], 2),
                "test_r2":   round(result.model_test_r2, 4),
                "test_Δr2":  round(result.model_test_r2 - bt_r2["r2"], 4),
            }
        )

    print(pd.DataFrame(summary_rows).to_string(index=False))


def show_test_examples(
    model: nn.Module | XGBRegressor,
    split: SplitData,
    target_stats: TargetStandardization,
) -> None:
    """Print a few held-out predictions with an error column."""

    predictions = predict_regressor(model, split, target_stats).cpu().numpy()
    example_frame = split.frame[
        ["router_name", "timestamp", split.target_column]
    ].copy()
    example_frame = example_frame.rename(
        columns={"router_name": "router", split.target_column: "actual"}
    )
    example_frame["actual"]    = example_frame["actual"].round(2)
    example_frame["predicted"] = predictions.round(2)
    example_frame["error"]     = (
        example_frame["predicted"] - example_frame["actual"]
    ).round(2)
    print()
    print("Sample test predictions (first 10):")
    print(example_frame.head(10).to_string(index=False))


def show_requested_prediction(
    model: nn.Module | XGBRegressor,
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
        print(
            f"Requested trace not found in test split  "
            f"(router={router}, timestamp={timestamp})"
        )
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
    prediction = predict_regressor(model, prediction_split, target_stats).item()
    actual = prediction_frame[split.target_column].iloc[0]
    print()
    print("Requested trace:")
    print(f"  router     {router}")
    print(f"  timestamp  {timestamp}")
    print(f"  target     {split.target_column}")
    print(f"  actual     {actual:.2f}")
    print(f"  predicted  {prediction:.2f}")
    print(f"  error      {_signed(prediction - actual)}")
