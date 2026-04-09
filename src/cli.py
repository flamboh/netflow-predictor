"""CLI parsing and small string helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.features import FEATURE_BLOCK_NAMES


DEFAULT_TARGET_COLUMN = "next_sa_ipv4_count_delta"
DEFAULT_FEATURE_BLOCKS = ("base",)
DEFAULT_DATABASE_PATH = Path("data/2025-03-01-to-2026-03-31/netflow_window.sqlite")
DEFAULT_TRAIN_ROUTER = "oh_ir1_gw"


def parse_args() -> argparse.Namespace:
    """Read command line arguments."""

    parser = argparse.ArgumentParser(
        description="Train netflow prediction baselines and GRU experiments."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="Path to the SQLite dataset.",
    )
    parser.add_argument(
        "--train-router",
        type=str,
        default=DEFAULT_TRAIN_ROUTER,
        help="Router to train on. Use all to disable router filtering.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of optimization steps over the training data.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optimizer learning rate. Default: 0.01 for linear/xgboost, 0.001 for gru.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Mini-batch size used during training.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=12,
        help="Sequence length used by the GRU backend.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--model-backend",
        type=str,
        default="linear",
        help="Regression backend to use: linear, xgboost, or gru.",
    )
    parser.add_argument(
        "--router",
        type=str,
        default=None,
        help="Optional router name for a single prediction lookup.",
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        default=None,
        help="Optional 5-minute trace timestamp for a single prediction lookup.",
    )
    parser.add_argument(
        "--describe-targets",
        action="store_true",
        help="Print derived next-window target summaries and exit.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET_COLUMN,
        help="Regression target column.",
    )
    parser.add_argument(
        "--feature-blocks",
        type=str,
        default="base",
        help="Comma-separated feature blocks. Available: base,spectrum,structure.",
    )
    parser.add_argument(
        "--show-feature-ranking",
        action="store_true",
        help="Print ranked learned feature weights after a single run.",
    )
    parser.add_argument(
        "--show-group-permutation-importance",
        action="store_true",
        help="Print grouped permutation importance after a single run.",
    )
    parser.add_argument(
        "--ranking-prefixes",
        type=str,
        default="",
        help="Comma-separated feature prefixes to keep in the ranking output.",
    )
    parser.add_argument(
        "--ranking-top-k",
        type=int,
        default=20,
        help="Number of ranked features to print.",
    )
    parser.add_argument(
        "--permutation-groups",
        type=str,
        default="base,time,router,spectrum,structure",
        help="Comma-separated feature groups for permutation importance.",
    )
    parser.add_argument(
        "--permutation-split",
        type=str,
        default="validation",
        help="Held-out split for permutation importance: validation or test.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=3,
        help="Number of grouped permutation repeats.",
    )
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run a matrix of target/block experiments and print a summary table.",
    )
    parser.add_argument(
        "--experiment-targets",
        type=str,
        default="next_sa_ipv4_count_delta;next_da_ipv4_count_delta",
        help="Semicolon-separated regression targets for experiment mode.",
    )
    parser.add_argument(
        "--experiment-feature-blocks",
        type=str,
        default="base;base,spectrum;base,structure;base,spectrum,structure",
        help="Semicolon-separated feature-block configs for experiment mode.",
    )
    return parser.parse_args()


def parse_feature_blocks(raw_value: str) -> tuple[str, ...]:
    """Parse requested feature blocks in a stable order."""

    requested = [block.strip() for block in raw_value.split(",") if block.strip()]

    if not requested:
        return DEFAULT_FEATURE_BLOCKS

    ordered_blocks = [
        block
        for block in FEATURE_BLOCK_NAMES
        if block in requested
    ]
    unknown_blocks = sorted(set(requested) - set(FEATURE_BLOCK_NAMES))

    if unknown_blocks:
        block_list = ", ".join(unknown_blocks)
        raise ValueError(f"Unknown feature blocks: {block_list}")

    return tuple(ordered_blocks)


def parse_experiment_targets(raw_value: str) -> list[str]:
    """Parse semicolon-separated experiment targets."""

    targets = [value.strip() for value in raw_value.split(";") if value.strip()]

    if not targets:
        return [DEFAULT_TARGET_COLUMN]

    return targets


def parse_experiment_feature_blocks(raw_value: str) -> list[tuple[str, ...]]:
    """Parse semicolon-separated block configs for experiment mode."""

    block_configs = [
        parse_feature_blocks(value)
        for value in raw_value.split(";")
        if value.strip()
    ]

    if not block_configs:
        return [DEFAULT_FEATURE_BLOCKS]

    return block_configs


def format_feature_blocks(feature_blocks: tuple[str, ...]) -> str:
    """Format a block tuple for logs and tables."""

    return ",".join(feature_blocks)


def parse_ranking_prefixes(raw_value: str) -> list[str]:
    """Parse comma-separated feature prefixes for ranking output."""

    return [value.strip() for value in raw_value.split(",") if value.strip()]


def parse_permutation_groups(raw_value: str) -> list[str]:
    """Parse comma-separated feature groups for permutation importance."""

    return [value.strip() for value in raw_value.split(",") if value.strip()]
