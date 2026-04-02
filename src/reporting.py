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
    }[baseline_name]


def print_experiment_summary(results: list[ExperimentResult]) -> None:
    """Print a compact experiment comparison table."""

    summary_rows = []

    for result in results:
        best_valid_baseline = get_best_baseline_name(
            result.baseline_metrics["validation"],
            "mae",
        )
        best_test_baseline = get_best_baseline_name(
            result.baseline_metrics["test"],
            "mae",
        )
        best_test_r2_baseline = get_best_baseline_name(
            result.baseline_metrics["test"],
            "r2",
            maximize=True,
        )
        best_valid_baseline_metrics = result.baseline_metrics["validation"][
            best_valid_baseline
        ]
        best_test_baseline_metrics = result.baseline_metrics["test"][
            best_test_baseline
        ]
        best_test_r2_metrics = result.baseline_metrics["test"][
            best_test_r2_baseline
        ]
        summary_rows.append(
            {
                "target": result.target,
                "model": result.model_backend,
                "blocks": format_feature_blocks(result.feature_blocks),
                "feats": result.feature_count,
                "ep": result.epochs,
                "dev": result.device,
                "base_mae": format_baseline_name(best_test_baseline),
                "base_r2": format_baseline_name(best_test_r2_baseline),
                "val_mae": round(result.model_valid_mae, 2),
                "val_vs_base": round(
                    result.model_valid_mae - best_valid_baseline_metrics["mae"],
                    2,
                ),
                "test_mae": round(result.model_test_mae, 2),
                "test_vs_base": round(
                    result.model_test_mae - best_test_baseline_metrics["mae"],
                    2,
                ),
                "test_r2": round(result.model_test_r2, 6),
                "r2_vs_base": round(
                    result.model_test_r2 - best_test_r2_metrics["r2"],
                    6,
                ),
            }
        )

    print(pd.DataFrame(summary_rows).to_string(index=False))


def show_test_examples(
    model: nn.Module | XGBRegressor,
    split: SplitData,
    target_stats: TargetStandardization,
) -> None:
    """Print a few held-out predictions."""

    predictions = predict_regressor(model, split, target_stats).cpu().numpy()
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
    prediction = predict_regressor(model, prediction_split, target_stats).item()
    actual = prediction_frame[split.target_column].iloc[0]
    print()
    print("Requested trace prediction:")
    print(f"router={router}")
    print(f"timestamp={timestamp}")
    print(f"target={split.target_column}")
    print(f"predicted_value={prediction:.2f}")
    print(f"actual_value={actual:.2f}")
