"""Feature construction for portable target-family experiments."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


GENERAL_FEATURE_COLUMNS = (
    "flows",
    "packets",
    "flows_tcp",
    "flows_udp",
    "unique_protocols_count_ipv4",
    "unique_protocols_count_ipv6",
    "hour_of_day",
    "day_of_week",
)
DEFAULT_TARGET_LAGS = (1, 2, 3, 12)
DEFAULT_ROLLING_WINDOWS = (3, 12)


def ordered_unique(columns: Iterable[str]) -> list[str]:
    """Keep first-seen order while removing duplicates."""

    ordered = list(dict.fromkeys(columns))
    return ordered


def get_router_column(frame: pd.DataFrame) -> str:
    """Return the router identity column used in the frame."""

    if "router" in frame.columns:
        return "router"

    if "router_name" in frame.columns:
        return "router_name"

    raise ValueError("Expected router or router_name column in feature frame.")


def make_target_feature_names(
    base_column: str,
    lags: tuple[int, ...] = DEFAULT_TARGET_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
) -> list[str]:
    """List target-family feature names for one base count column."""

    names = [base_column]
    names.extend(f"{base_column}_lag_{lag}" for lag in lags)
    names.extend(
        f"{base_column}_rolling_mean_{window}"
        for window in rolling_windows
    )
    names.extend(
        f"{base_column}_rolling_std_{window}"
        for window in rolling_windows
    )
    return names


def add_target_family_features(
    frame: pd.DataFrame,
    base_column: str,
    lags: tuple[int, ...] = DEFAULT_TARGET_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Add portable lag and rolling features for one target family."""

    if base_column not in frame.columns:
        raise ValueError(f"Unknown target feature base column: {base_column}")

    router_column = get_router_column(frame)
    enriched = frame.sort_values([router_column, "timestamp"]).copy()
    router_groups = enriched.groupby(router_column, sort=False)[base_column]

    for lag in lags:
        enriched[f"{base_column}_lag_{lag}"] = router_groups.shift(lag)

    for window in rolling_windows:
        rolling = (
            router_groups
            .rolling(window=window, min_periods=window)
            .agg(["mean", "std"])
            .reset_index(level=0, drop=True)
        )
        enriched[f"{base_column}_rolling_mean_{window}"] = rolling["mean"]
        enriched[f"{base_column}_rolling_std_{window}"] = rolling["std"]

    return enriched


def choose_feature_columns(
    frame: pd.DataFrame,
    target_base_column: str,
    general_columns: tuple[str, ...] = GENERAL_FEATURE_COLUMNS,
) -> list[str]:
    """Assemble ordered portable feature columns."""

    feature_columns = make_target_feature_names(target_base_column)
    feature_columns.extend(general_columns)

    router_columns = sorted(
        column
        for column in frame.columns
        if column.startswith("router_") and column != "router_name"
    )
    feature_columns.extend(router_columns)
    feature_columns = ordered_unique(feature_columns)
    return feature_columns


def filter_feature_rows(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Drop rows where required features are unavailable."""

    filtered = frame.dropna(subset=feature_columns).reset_index(drop=True)
    return filtered
