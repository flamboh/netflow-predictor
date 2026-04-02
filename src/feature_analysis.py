"""Helpers for inspecting learned linear feature weights."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import torch
from torch import nn

from src.features import make_spectrum_region_summary_feature_names
from src.features import make_structure_region_summary_feature_names
from src.features import make_structure_sd_sample_feature_names
from src.features import make_structure_summary_feature_names
from src.features import make_structure_tau_sample_feature_names
from src.modeling import SplitData
from src.modeling import TargetStandardization
from src.modeling import evaluate_predictions
from src.modeling import predict


def infer_feature_group(feature_name: str) -> str:
    """Assign one feature to a readable analysis group."""

    spectrum_region_summary = set(make_spectrum_region_summary_feature_names())
    structure_region_summary = set(make_structure_region_summary_feature_names())
    structure_summary = set(make_structure_summary_feature_names())
    structure_tau = set(make_structure_tau_sample_feature_names())
    structure_sd = set(make_structure_sd_sample_feature_names())

    if feature_name in spectrum_region_summary:
        return "spectrum_region_summary"

    if feature_name in structure_region_summary:
        return "structure_region_summary"

    if feature_name in structure_summary:
        return "structure_summary"

    if feature_name in structure_tau:
        return "structure_tau_samples"

    if feature_name in structure_sd:
        return "structure_sd_samples"

    if feature_name.startswith("structure_"):
        return "structure_other"

    if feature_name.startswith("spectrum_"):
        return "spectrum"

    if feature_name.startswith("router_"):
        return "router"

    return "base"


def get_linear_feature_ranking(
    model: nn.Module,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Build a sorted feature-weight table for the learned linear model."""

    if not hasattr(model, "linear"):
        raise ValueError("Expected a model with a .linear layer.")

    weights = model.linear.weight.detach().cpu().flatten().tolist()

    if len(weights) != len(feature_columns):
        raise ValueError("Feature column count does not match learned weights.")

    frame = pd.DataFrame(
        {
            "feature": list(feature_columns),
            "coefficient": weights,
        }
    )
    frame["abs_coefficient"] = frame["coefficient"].abs()
    frame["group"] = frame["feature"].map(infer_feature_group)
    frame = frame.sort_values(
        ["abs_coefficient", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return frame


def filter_ranked_features(
    ranking: pd.DataFrame,
    prefixes: Sequence[str],
) -> pd.DataFrame:
    """Filter ranked features by one or more string prefixes."""

    normalized_prefixes = [prefix for prefix in prefixes if prefix]

    if not normalized_prefixes:
        return ranking.copy()

    mask = ranking["feature"].apply(
        lambda feature: any(feature.startswith(prefix) for prefix in normalized_prefixes)
    )
    filtered = ranking.loc[mask].reset_index(drop=True)
    return filtered


def get_grouped_permutation_importance(
    model: nn.Module,
    split: SplitData,
    target_stats: TargetStandardization,
    feature_columns: Sequence[str],
    groups: Sequence[str],
    repeats: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Measure held-out performance change after grouped feature permutation."""

    if repeats < 1:
        raise ValueError("Permutation repeats must be at least 1.")

    ranking = get_linear_feature_ranking(model, feature_columns)
    actual_targets = torch.tensor(
        split.frame[split.target_column].to_numpy(dtype="float32")
    )
    baseline_predictions = predict(model, split, target_stats).detach().cpu()
    baseline_metrics = evaluate_predictions(baseline_predictions, actual_targets)
    group_rows: list[dict[str, float | str | int]] = []

    for group in groups:
        group_features = ranking.loc[ranking["group"].eq(group), "feature"].tolist()

        if not group_features:
            continue

        group_indices = [
            index
            for index, feature_name in enumerate(feature_columns)
            if feature_name in set(group_features)
        ]
        mae_deltas: list[float] = []
        rmse_deltas: list[float] = []
        r2_deltas: list[float] = []

        for repeat_index in range(repeats):
            generator = torch.Generator()
            generator.manual_seed(seed + repeat_index)
            permutation = torch.randperm(split.features.shape[0], generator=generator)
            permuted_features = split.features.clone()
            permuted_features[:, group_indices] = split.features[permutation][:, group_indices]
            permuted_split = SplitData(
                features=permuted_features,
                targets=split.targets,
                frame=split.frame,
                target_column=split.target_column,
            )
            permuted_predictions = predict(model, permuted_split, target_stats).detach().cpu()
            permuted_metrics = evaluate_predictions(permuted_predictions, actual_targets)
            mae_deltas.append(permuted_metrics["mae"] - baseline_metrics["mae"])
            rmse_deltas.append(permuted_metrics["rmse"] - baseline_metrics["rmse"])
            r2_deltas.append(permuted_metrics["r2"] - baseline_metrics["r2"])

        group_rows.append(
            {
                "group": group,
                "feature_count": len(group_indices),
                "baseline_mae": baseline_metrics["mae"],
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_r2": baseline_metrics["r2"],
                "mean_mae_delta": sum(mae_deltas) / len(mae_deltas),
                "mean_rmse_delta": sum(rmse_deltas) / len(rmse_deltas),
                "mean_r2_delta": sum(r2_deltas) / len(r2_deltas),
            }
        )

    frame = pd.DataFrame(group_rows)
    if frame.empty:
        return frame

    frame = frame.sort_values(
        ["mean_mae_delta", "group"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return frame
