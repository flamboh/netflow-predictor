"""Data loading and base feature construction for netflow experiments."""

from __future__ import annotations

import sqlite3
from pathlib import Path
import math

import pandas as pd


SECONDS_PER_MINUTE = 60
SECONDS_PER_DAY = 24 * 60 * SECONDS_PER_MINUTE
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY


def read_table(connection: sqlite3.Connection, query: str) -> pd.DataFrame:
    """Load a SQL query into a DataFrame."""

    frame = pd.read_sql_query(query, connection)
    return frame


def load_base_frame(database_path: Path) -> pd.DataFrame:
    """Load and join the small set of tables used by the baseline model."""

    connection = sqlite3.connect(database_path)

    netflow_query = """
    SELECT
        router,
        timestamp,
        flows,
        packets,
        bytes,
        flows_tcp,
        flows_udp,
        bytes_tcp,
        bytes_udp
    FROM netflow_stats
    ORDER BY router, timestamp
    """
    ip_query = """
    SELECT
        router,
        bucket_start AS timestamp,
        sa_ipv4_count,
        da_ipv4_count,
        sa_ipv6_count,
        da_ipv6_count
    FROM ip_stats
    WHERE granularity = '5m'
    """
    protocol_query = """
    SELECT
        router,
        bucket_start AS timestamp,
        unique_protocols_count_ipv4,
        unique_protocols_count_ipv6
    FROM protocol_stats
    WHERE granularity = '5m'
    """
    spectrum_query = """
    SELECT
        router,
        bucket_start AS timestamp,
        spectrum_json_sa,
        spectrum_json_da
    FROM spectrum_stats
    WHERE granularity = '5m' AND ip_version = 4
    """
    structure_query = """
    SELECT
        router,
        bucket_start AS timestamp,
        structure_json_sa,
        structure_json_da
    FROM structure_stats
    WHERE granularity = '5m' AND ip_version = 4
    """

    netflow = read_table(connection, netflow_query)
    ip_stats = read_table(connection, ip_query)
    protocol_stats = read_table(connection, protocol_query)
    spectrum_stats = read_table(connection, spectrum_query)
    structure_stats = read_table(connection, structure_query)
    connection.close()

    merged = netflow.merge(ip_stats, on=["router", "timestamp"], how="inner")
    merged = merged.merge(protocol_stats, on=["router", "timestamp"], how="inner")
    merged = merged.merge(spectrum_stats, on=["router", "timestamp"], how="left")
    merged = merged.merge(structure_stats, on=["router", "timestamp"], how="left")
    return merged


def validate_join_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail early if joined feature tables are missing rows."""

    validated = frame.copy()
    joined_feature_columns = [
        "sa_ipv4_count",
        "da_ipv4_count",
        "sa_ipv6_count",
        "da_ipv6_count",
        "unique_protocols_count_ipv4",
        "unique_protocols_count_ipv6",
    ]
    missing_counts = validated[joined_feature_columns].isna().sum()
    missing_columns = missing_counts[missing_counts > 0]

    if missing_columns.empty:
        return validated

    details = ", ".join(
        f"{column}={count}"
        for column, count in missing_columns.items()
    )
    raise ValueError(
        "Joined feature tables contain missing values. "
        f"Missing counts: {details}"
    )


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create cyclical time features from the trace timestamp."""

    enriched = frame.copy()
    minute_of_day = (enriched["timestamp"] % SECONDS_PER_DAY) / SECONDS_PER_MINUTE
    second_of_week = enriched["timestamp"] % SECONDS_PER_WEEK
    week_phase = second_of_week / SECONDS_PER_WEEK
    day_phase = minute_of_day / (SECONDS_PER_DAY / SECONDS_PER_MINUTE)
    enriched["time_of_day_sin"] = day_phase.map(lambda value: math.sin(2 * math.pi * value))
    enriched["time_of_day_cos"] = day_phase.map(lambda value: math.cos(2 * math.pi * value))
    enriched["time_of_week_sin"] = week_phase.map(lambda value: math.sin(2 * math.pi * value))
    enriched["time_of_week_cos"] = week_phase.map(lambda value: math.cos(2 * math.pi * value))
    return enriched


def add_router_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve router identity for reporting and sequence grouping."""

    encoded = frame.copy()
    encoded["router_name"] = encoded["router"]
    return encoded


def build_feature_frame(
    database_path: Path,
    train_router: str | None = None,
) -> pd.DataFrame:
    """Build the final modeling frame."""

    frame = load_base_frame(database_path)
    frame = validate_join_features(frame)

    if train_router is not None:
        frame = frame.loc[frame["router"].eq(train_router)].reset_index(drop=True)

    frame = add_time_features(frame)
    frame = add_router_features(frame)
    return frame
