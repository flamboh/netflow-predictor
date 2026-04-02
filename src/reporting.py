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
                "features": result.feature_count,
                "epochs": result.epochs,
                "device": result.device,
                "best_val_baseline": best_valid_baseline,
                "best_test_baseline": best_test_baseline,
                "best_test_r2_baseline": best_test_r2_baseline,
                "persist_val_mae": round(result.persistence_valid_mae, 2),
                "best_val_mae": round(best_valid_baseline_metrics["mae"], 2),
                "model_val_mae": round(result.model_valid_mae, 2),
                "val_mae_delta": round(
                    result.model_valid_mae - result.persistence_valid_mae,
                    2,
                ),
                "vs_best_val_mae": round(
                    result.model_valid_mae - best_valid_baseline_metrics["mae"],
                    2,
                ),
                "persist_val_r2": round(result.persistence_valid_r2, 6),
                "model_val_r2": round(result.model_valid_r2, 6),
                "val_r2_delta": round(
                    result.model_valid_r2 - result.persistence_valid_r2,
                    6,
                ),
                "persist_test_mae": round(result.persistence_test_mae, 2),
                "best_test_mae": round(best_test_baseline_metrics["mae"], 2),
                "model_test_mae": round(result.model_test_mae, 2),
                "mae_delta": round(
                    result.model_test_mae - result.persistence_test_mae,
                    2,
                ),
                "vs_best_test_mae": round(
                    result.model_test_mae - best_test_baseline_metrics["mae"],
                    2,
                ),
                "persist_test_r2": round(result.persistence_test_r2, 6),
                "best_test_r2": round(best_test_r2_metrics["r2"], 6),
                "model_test_r2": round(result.model_test_r2, 6),
                "r2_delta": round(
                    result.model_test_r2 - result.persistence_test_r2,
                    6,
                ),
                "vs_best_test_r2": round(
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
