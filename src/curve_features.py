"""Compact spectrum and structure summaries."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


SPECTRUM_FEATURE_COLUMNS = (
    "spectrum_alpha_at_f_max",
    "spectrum_alpha_min",
    "spectrum_alpha_max",
    "spectrum_f_at_alpha_max",
    "spectrum_f_max",
    "spectrum_f_min",
)
STRUCTURE_FEATURE_COLUMNS = (
    "structure_tau_q0",
    "structure_tau_q2",
    "structure_sd_q0",
    "structure_sd_q2",
)
STRUCTURE_RAW_POINT_COUNT = 61
SPECTRUM_RAW_POINT_COUNT = 37


def _pointwise_feature_columns(
    feature_prefix: str,
    expected_length: int,
    point_fields: tuple[str, ...],
) -> tuple[str, ...]:
    """Build stable pointwise column names for one curve block."""

    return tuple(
        f"{feature_prefix}_point_{index:02d}_{field}"
        for index in range(expected_length)
        for field in point_fields
    )


SPECTRUM_RAW_FEATURE_COLUMNS = _pointwise_feature_columns(
    "spectrum",
    SPECTRUM_RAW_POINT_COUNT,
    ("alpha", "f"),
)
STRUCTURE_RAW_FEATURE_COLUMNS = _pointwise_feature_columns(
    "structure",
    STRUCTURE_RAW_POINT_COUNT,
    ("tau", "sd"),
)


def _empty_curve_series(columns: tuple[str, ...]) -> pd.Series:
    """Return an all-missing series for one curve feature block."""

    return pd.Series({column: pd.NA for column in columns})


def _pointwise_curve_series(
    curve_json: str,
    expected_length: int,
    point_fields: tuple[str, ...],
    feature_prefix: str,
) -> pd.Series:
    """Convert one JSON curve into stable resampled pointwise features."""

    if pd.isna(curve_json):
        return _empty_curve_series(
            _pointwise_feature_columns(
                feature_prefix,
                expected_length,
                point_fields,
            )
        )

    curve = json.loads(curve_json)
    if not curve:
        return _empty_curve_series(
            _pointwise_feature_columns(
                feature_prefix,
                expected_length,
                point_fields,
            )
        )

    source_positions = np.arange(len(curve), dtype="float32")
    target_positions = np.linspace(
        0,
        len(curve) - 1,
        expected_length,
        dtype="float32",
    )
    values: dict[str, float | object] = {}
    for field in point_fields:
        field_values = []

        for index, point in enumerate(curve):
            if field not in point:
                raise ValueError(
                    f"Missing {field} in {feature_prefix} point {index}."
                )
            field_values.append(float(point[field]))

        resampled_values = np.interp(
            target_positions,
            source_positions,
            np.array(field_values, dtype="float32"),
        )

        for index, value in enumerate(resampled_values):
            values[f"{feature_prefix}_point_{index:02d}_{field}"] = float(value)

    return pd.Series(values)


def summarize_spectrum_curve(curve_json: str) -> pd.Series:
    """Convert one spectrum JSON blob into the reduced summary set."""

    if pd.isna(curve_json):
        return pd.Series({column: pd.NA for column in SPECTRUM_FEATURE_COLUMNS})

    curve = json.loads(curve_json)
    if not curve:
        return pd.Series({column: pd.NA for column in SPECTRUM_FEATURE_COLUMNS})

    alphas = [float(point["alpha"]) for point in curve]
    f_values = [float(point["f"]) for point in curve]
    alpha_at_f_max_index = max(range(len(f_values)), key=f_values.__getitem__)
    f_at_alpha_max_index = max(range(len(alphas)), key=alphas.__getitem__)

    return pd.Series(
        {
            "spectrum_alpha_at_f_max": alphas[alpha_at_f_max_index],
            "spectrum_alpha_min": min(alphas),
            "spectrum_alpha_max": max(alphas),
            "spectrum_f_at_alpha_max": f_values[f_at_alpha_max_index],
            "spectrum_f_max": max(f_values),
            "spectrum_f_min": min(f_values),
        }
    )


def summarize_structure_curve(curve_json: str) -> pd.Series:
    """Convert one structure JSON blob into the reduced q-point summary set."""

    if pd.isna(curve_json):
        return pd.Series({column: pd.NA for column in STRUCTURE_FEATURE_COLUMNS})

    curve = json.loads(curve_json)
    if not curve:
        return pd.Series({column: pd.NA for column in STRUCTURE_FEATURE_COLUMNS})

    points_by_q = {float(point["q"]): point for point in curve}

    try:
        q0 = points_by_q[0.0]
        q2 = points_by_q[2.0]
    except KeyError as error:
        return pd.Series({column: pd.NA for column in STRUCTURE_FEATURE_COLUMNS})

    return pd.Series(
        {
            "structure_tau_q0": float(q0["tau"]),
            "structure_tau_q2": float(q2["tau"]),
            "structure_sd_q0": float(q0["sd"]),
            "structure_sd_q2": float(q2["sd"]),
        }
    )


def extract_spectrum_curve_points(curve_json: str) -> pd.Series:
    """Convert one spectrum JSON blob into raw pointwise features."""

    return _pointwise_curve_series(
        curve_json=curve_json,
        expected_length=SPECTRUM_RAW_POINT_COUNT,
        point_fields=("alpha", "f"),
        feature_prefix="spectrum",
    )


def extract_structure_curve_points(curve_json: str) -> pd.Series:
    """Convert one structure JSON blob into raw pointwise features."""

    return _pointwise_curve_series(
        curve_json=curve_json,
        expected_length=STRUCTURE_RAW_POINT_COUNT,
        point_fields=("tau", "sd"),
        feature_prefix="structure",
    )
