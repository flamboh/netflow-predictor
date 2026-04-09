"""Reduced feature assembly for tabular and sequence runners."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.curve_features import SPECTRUM_FEATURE_COLUMNS
from src.curve_features import STRUCTURE_FEATURE_COLUMNS
from src.curve_features import summarize_spectrum_curve
from src.curve_features import summarize_structure_curve


BASE_FEATURE_COLUMNS = (
    "flows",
    "packets",
    "bytes",
    "sa_ipv4_count",
    "da_ipv4_count",
    "unique_protocols_count_ipv4",
    "time_of_day_sin",
    "time_of_day_cos",
    "time_of_week_sin",
    "time_of_week_cos",
)
FEATURE_BLOCK_NAMES = (
    "base",
    "spectrum",
    "structure",
)


def ordered_unique(columns: Iterable[str]) -> list[str]:
    """Keep first-seen order while removing duplicates."""

    return list(dict.fromkeys(columns))


def get_router_column(frame: pd.DataFrame) -> str:
    """Return the router identity column used in the frame."""

    if "router" in frame.columns:
        return "router"

    if "router_name" in frame.columns:
        return "router_name"

    raise ValueError("Expected router or router_name column in feature frame.")


def get_target_axis_tokens(target_base_column: str) -> tuple[str, str]:
    """Map one target base column to MAAD direction and IP-family tokens."""

    if target_base_column.startswith("sa_"):
        axis_token = "sa"
    elif target_base_column.startswith("da_"):
        axis_token = "da"
    else:
        raise ValueError(f"Unrecognized target base column: {target_base_column}")

    if "_ipv4_" in f"_{target_base_column}_":
        family_token = "ipv4"
    elif "_ipv6_" in f"_{target_base_column}_":
        family_token = "ipv6"
    else:
        raise ValueError(f"Unrecognized target IP family: {target_base_column}")

    return axis_token, family_token


def make_base_feature_names(frame: pd.DataFrame) -> list[str]:
    """List current-window base features, including router indicators."""

    router_columns = sorted(
        column
        for column in frame.columns
        if column.startswith("router_") and column != "router_name"
    )
    return list(BASE_FEATURE_COLUMNS) + router_columns


def add_spectrum_features(
    frame: pd.DataFrame,
    target_base_column: str,
) -> pd.DataFrame:
    """Add reduced spectrum summaries for the selected target family."""

    axis_token, family_token = get_target_axis_tokens(target_base_column)

    if family_token != "ipv4":
        raise ValueError(
            "Spectrum features are currently available only for IPv4 targets."
        )

    spectrum_column = f"spectrum_json_{axis_token}"
    if spectrum_column not in frame.columns:
        raise ValueError(f"Missing spectrum column: {spectrum_column}")

    enriched = frame.copy()
    derived = enriched[spectrum_column].apply(summarize_spectrum_curve)

    for column in SPECTRUM_FEATURE_COLUMNS:
        enriched[column] = derived[column]

    return enriched


def add_structure_features(
    frame: pd.DataFrame,
    target_base_column: str,
) -> pd.DataFrame:
    """Add reduced structure summaries for the selected target family."""

    axis_token, family_token = get_target_axis_tokens(target_base_column)

    if family_token != "ipv4":
        raise ValueError(
            "Structure features are currently available only for IPv4 targets."
        )

    structure_column = f"structure_json_{axis_token}"
    if structure_column not in frame.columns:
        raise ValueError(f"Missing structure column: {structure_column}")

    enriched = frame.copy()
    derived = enriched[structure_column].apply(summarize_structure_curve)

    for column in STRUCTURE_FEATURE_COLUMNS:
        enriched[column] = derived[column]

    return enriched


def prepare_feature_frame(
    frame: pd.DataFrame,
    target_base_column: str,
    feature_blocks: tuple[str, ...],
) -> pd.DataFrame:
    """Materialize only the requested feature families."""

    prepared = frame

    if "spectrum" in feature_blocks:
        prepared = add_spectrum_features(prepared, target_base_column)

    if "structure" in feature_blocks:
        prepared = add_structure_features(prepared, target_base_column)

    return prepared


def choose_feature_columns(
    frame: pd.DataFrame,
    feature_blocks: tuple[str, ...] = ("base",),
) -> list[str]:
    """Assemble ordered feature columns for one experiment."""

    unknown_blocks = set(feature_blocks) - set(FEATURE_BLOCK_NAMES)

    if unknown_blocks:
        block_list = ", ".join(sorted(unknown_blocks))
        raise ValueError(f"Unknown feature blocks: {block_list}")

    feature_columns: list[str] = []

    if "base" in feature_blocks:
        feature_columns.extend(make_base_feature_names(frame))

    if "spectrum" in feature_blocks:
        feature_columns.extend(SPECTRUM_FEATURE_COLUMNS)

    if "structure" in feature_blocks:
        feature_columns.extend(STRUCTURE_FEATURE_COLUMNS)

    return ordered_unique(feature_columns)


def filter_feature_rows(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Drop rows where required features are unavailable."""

    return frame.dropna(subset=feature_columns).reset_index(drop=True)
