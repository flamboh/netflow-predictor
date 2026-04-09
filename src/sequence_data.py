"""Sequence-window dataset builders for recurrent models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.features import get_router_column
from src.modeling import SplitData


TRACE_INTERVAL_SECONDS = 5 * 60


def to_sequence_split(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int,
    prior_history_rows: pd.DataFrame | None = None,
) -> SplitData:
    """Convert one split into fixed-length per-router windows.

    When prior history rows are provided, use them as left context but only
    emit windows whose target row belongs to the current split.
    """

    if sequence_length < 1:
        raise ValueError("Sequence length must be at least 1.")

    router_column = get_router_column(frame)
    current_rows = frame.copy()
    current_rows["_sequence_is_current"] = True
    if prior_history_rows is None:
        ordered = current_rows
    else:
        history_rows = prior_history_rows.copy()
        history_rows["_sequence_is_current"] = False
        ordered = pd.concat([history_rows, current_rows], ignore_index=True)

    ordered = ordered.sort_values(
        [router_column, "timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)
    sequence_rows: list[pd.Series] = []
    feature_windows: list[np.ndarray] = []
    target_values: list[float] = []

    for _, router_frame in ordered.groupby(router_column, sort=False):
        router_frame = router_frame.reset_index(drop=True)
        timestamp_diffs = router_frame["timestamp"].diff().fillna(TRACE_INTERVAL_SECONDS)
        segment_ids = timestamp_diffs.ne(TRACE_INTERVAL_SECONDS).cumsum()

        for _, segment_frame in router_frame.groupby(segment_ids, sort=False):
            if len(segment_frame) < sequence_length:
                continue

            segment_frame = segment_frame.reset_index(drop=True)
            is_current = segment_frame["_sequence_is_current"].to_numpy(dtype=bool)
            feature_values = segment_frame[feature_columns].to_numpy(dtype="float32")
            target_series = segment_frame[target_column].to_numpy(dtype="float32")

            for end_index in range(sequence_length - 1, len(segment_frame)):
                if not is_current[end_index]:
                    continue

                start_index = end_index - sequence_length + 1
                feature_windows.append(feature_values[start_index:end_index + 1])
                target_values.append(float(target_series[end_index]))
                sequence_rows.append(segment_frame.iloc[end_index])

    if not feature_windows:
        raise ValueError("No sequence windows available for the requested split.")

    sequence_frame = pd.DataFrame(sequence_rows).reset_index(drop=True)
    return SplitData(
        features=torch.tensor(np.stack(feature_windows), dtype=torch.float32),
        targets=torch.tensor(target_values, dtype=torch.float32),
        frame=sequence_frame.drop(columns=["_sequence_is_current"]).reset_index(drop=True),
        target_column=target_column,
    )
