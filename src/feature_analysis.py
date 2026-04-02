"""Helpers for inspecting learned linear feature weights."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from torch import nn

from src.features import make_structure_sd_sample_feature_names
from src.features import make_structure_summary_feature_names
from src.features import make_structure_tau_sample_feature_names


def infer_feature_group(feature_name: str) -> str:
    """Assign one feature to a readable analysis group."""

    structure_summary = set(make_structure_summary_feature_names())
    structure_tau = set(make_structure_tau_sample_feature_names())
    structure_sd = set(make_structure_sd_sample_feature_names())

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
