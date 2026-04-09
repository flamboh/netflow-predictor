"""Compact spectrum and structure summaries."""

from __future__ import annotations

import json

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
