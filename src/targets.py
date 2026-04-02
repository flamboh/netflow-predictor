"""Target construction for next-window netflow prediction tasks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


COUNT_TARGET_BASES = (
    "sa_ipv4_count",
    "da_ipv4_count",
    "sa_ipv6_count",
    "da_ipv6_count",
)


@dataclass(frozen=True)
class TargetSpec:
    """Describe one derived prediction target."""

    name: str
    source_column: str
    task_kind: str


def add_next_window_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Add next-window count, delta, and direction targets per router."""

    router_column = "router"

    if router_column not in frame.columns:
        router_column = "router_name"

    enriched = frame.sort_values([router_column, "timestamp"]).copy()
    router_groups = enriched.groupby(router_column, sort=False)

    for base_column in COUNT_TARGET_BASES:
        next_column = f"next_{base_column}"
        delta_column = f"{next_column}_delta"
        direction_column = f"{next_column}_direction"

        enriched[next_column] = router_groups[base_column].shift(-1)
        enriched[delta_column] = enriched[next_column] - enriched[base_column]
        direction = pd.Series(pd.NA, index=enriched.index, dtype="Int64")
        has_delta = enriched[delta_column].notna()
        direction.loc[has_delta] = (
            enriched.loc[has_delta, delta_column] > 0
        ).astype("int64")
        enriched[direction_column] = direction

    return enriched


def list_target_specs() -> list[TargetSpec]:
    """List supported next-window targets."""

    specs: list[TargetSpec] = []

    for base_column in COUNT_TARGET_BASES:
        next_column = f"next_{base_column}"
        specs.append(
            TargetSpec(
                name=next_column,
                source_column=base_column,
                task_kind="exact_count",
            )
        )
        specs.append(
            TargetSpec(
                name=f"{next_column}_delta",
                source_column=base_column,
                task_kind="delta",
            )
        )
        specs.append(
            TargetSpec(
                name=f"{next_column}_direction",
                source_column=base_column,
                task_kind="direction",
            )
        )

    return specs


def describe_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize supported targets after target construction."""

    summaries: list[dict[str, object]] = []

    for spec in list_target_specs():
        series = frame[spec.name]
        non_null = series.dropna()

        summary: dict[str, object] = {
            "target": spec.name,
            "task_kind": spec.task_kind,
            "source_column": spec.source_column,
            "rows": int(non_null.shape[0]),
            "missing": int(series.isna().sum()),
        }

        if spec.task_kind == "direction":
            value_counts = non_null.astype("int64").value_counts().to_dict()
            summary["zeros"] = int(value_counts.get(0, 0))
            summary["ones"] = int(value_counts.get(1, 0))
        else:
            summary["mean"] = float(non_null.mean())
            summary["std"] = float(non_null.std())
            summary["min"] = float(non_null.min())
            summary["max"] = float(non_null.max())

        summaries.append(summary)

    return pd.DataFrame(summaries)
