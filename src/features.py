"""Feature construction for portable target-family experiments."""

from __future__ import annotations

from collections.abc import Iterable
import json
import math

import pandas as pd

from src.targets import COUNT_TARGET_BASES


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
CURVE_SAMPLE_COUNT = 5
FEATURE_BLOCK_NAMES = (
    "target",
    "general",
    "context",
    "spectrum",
    "structure",
)


def compute_mean(values: list[float]) -> float:
    """Compute the mean of a non-empty float list."""

    return sum(values) / len(values)


def compute_std(values: list[float]) -> float:
    """Compute population standard deviation of a non-empty float list."""

    mean = compute_mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def sample_values(values: list[float], sample_count: int = CURVE_SAMPLE_COUNT) -> list[float]:
    """Sample a fixed number of ordered values from a variable-length list."""

    if len(values) == 1:
        return [values[0]] * sample_count

    sampled: list[float] = []

    for index in range(sample_count):
        position = round(index * (len(values) - 1) / (sample_count - 1))
        sampled.append(values[position])

    return sampled


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


def make_context_feature_names(target_base_column: str) -> list[str]:
    """List non-target count columns used as optional context."""

    names = [
        column
        for column in COUNT_TARGET_BASES
        if column != target_base_column
    ]
    return names


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


def make_spectrum_feature_names(
    sample_count: int = CURVE_SAMPLE_COUNT,
) -> list[str]:
    """List derived spectrum feature names."""

    names = [
        "spectrum_points",
        "spectrum_alpha_min",
        "spectrum_alpha_max",
        "spectrum_alpha_range",
        "spectrum_f_max",
        "spectrum_f_mean",
        "spectrum_f_std",
        "spectrum_alpha_at_f_max",
        "spectrum_area",
    ]
    names.extend(f"spectrum_alpha_sample_{index}" for index in range(sample_count))
    names.extend(f"spectrum_f_sample_{index}" for index in range(sample_count))
    return names


def make_structure_feature_names(
    sample_count: int = CURVE_SAMPLE_COUNT,
) -> list[str]:
    """List derived structure feature names."""

    names = [
        "structure_points",
        "structure_tau_min",
        "structure_tau_max",
        "structure_tau_mean",
        "structure_tau_std",
        "structure_sd_max",
        "structure_sd_mean",
        "structure_sd_std",
    ]
    names.extend(f"structure_tau_sample_{index}" for index in range(sample_count))
    names.extend(f"structure_sd_sample_{index}" for index in range(sample_count))
    return names


def summarize_spectrum_curve(curve_json: str) -> pd.Series:
    """Convert one spectrum JSON blob into compact numeric features."""

    curve = json.loads(curve_json)
    alphas = [float(point["alpha"]) for point in curve]
    f_values = [float(point["f"]) for point in curve]
    alpha_min = min(alphas)
    alpha_max = max(alphas)
    f_max_index = max(range(len(f_values)), key=f_values.__getitem__)
    ordered_pairs = sorted(zip(alphas, f_values))
    area = 0.0

    for (alpha_1, f_1), (alpha_2, f_2) in zip(ordered_pairs, ordered_pairs[1:]):
        area += (alpha_2 - alpha_1) * (f_1 + f_2) / 2.0

    values: dict[str, float] = {
        "spectrum_points": float(len(curve)),
        "spectrum_alpha_min": alpha_min,
        "spectrum_alpha_max": alpha_max,
        "spectrum_alpha_range": alpha_max - alpha_min,
        "spectrum_f_max": max(f_values),
        "spectrum_f_mean": compute_mean(f_values),
        "spectrum_f_std": compute_std(f_values),
        "spectrum_alpha_at_f_max": alphas[f_max_index],
        "spectrum_area": area,
    }

    for index, value in enumerate(sample_values(alphas)):
        values[f"spectrum_alpha_sample_{index}"] = value

    for index, value in enumerate(sample_values(f_values)):
        values[f"spectrum_f_sample_{index}"] = value

    return pd.Series(values)


def summarize_structure_curve(curve_json: str) -> pd.Series:
    """Convert one structure JSON blob into compact numeric features."""

    curve = json.loads(curve_json)
    tau_values = [float(point["tau"]) for point in curve]
    sd_values = [float(point["sd"]) for point in curve]
    values: dict[str, float] = {
        "structure_points": float(len(curve)),
        "structure_tau_min": min(tau_values),
        "structure_tau_max": max(tau_values),
        "structure_tau_mean": compute_mean(tau_values),
        "structure_tau_std": compute_std(tau_values),
        "structure_sd_max": max(sd_values),
        "structure_sd_mean": compute_mean(sd_values),
        "structure_sd_std": compute_std(sd_values),
    }

    for index, value in enumerate(sample_values(tau_values)):
        values[f"structure_tau_sample_{index}"] = value

    for index, value in enumerate(sample_values(sd_values)):
        values[f"structure_sd_sample_{index}"] = value

    return pd.Series(values)


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


def add_spectrum_features(
    frame: pd.DataFrame,
    target_base_column: str,
) -> pd.DataFrame:
    """Add derived spectrum features for the selected target family."""

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
    for column in make_spectrum_feature_names():
        enriched[column] = derived[column]
    return enriched


def add_structure_features(
    frame: pd.DataFrame,
    target_base_column: str,
) -> pd.DataFrame:
    """Add derived structure features for the selected target family."""

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
    for column in make_structure_feature_names():
        enriched[column] = derived[column]
    return enriched


def choose_feature_columns(
    frame: pd.DataFrame,
    target_base_column: str,
    feature_blocks: tuple[str, ...] = ("target", "general"),
    general_columns: tuple[str, ...] = GENERAL_FEATURE_COLUMNS,
) -> list[str]:
    """Assemble ordered portable feature columns."""

    unknown_blocks = set(feature_blocks) - set(FEATURE_BLOCK_NAMES)

    if unknown_blocks:
        block_list = ", ".join(sorted(unknown_blocks))
        raise ValueError(f"Unknown feature blocks: {block_list}")

    feature_columns: list[str] = []

    if "target" in feature_blocks:
        feature_columns.extend(make_target_feature_names(target_base_column))

    if "general" in feature_blocks:
        feature_columns.extend(general_columns)

    if "context" in feature_blocks:
        feature_columns.extend(make_context_feature_names(target_base_column))

    if "spectrum" in feature_blocks:
        feature_columns.extend(make_spectrum_feature_names())

    if "structure" in feature_blocks:
        feature_columns.extend(make_structure_feature_names())

    if "general" in feature_blocks:
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
