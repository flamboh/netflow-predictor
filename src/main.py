"""Train and evaluate a simple PyTorch regression model for 5-minute netflow traffic."""

from __future__ import annotations

import torch

from src.cli import format_feature_blocks
from src.cli import parse_args
from src.cli import parse_experiment_feature_blocks
from src.cli import parse_experiment_targets
from src.cli import parse_feature_blocks
from src.cli import parse_ranking_prefixes
from src.experiments import build_modeling_frame
from src.experiments import run_experiment_matrix
from src.experiments import run_regression_experiment
from src.experiments import show_requested_prediction
from src.experiments import show_test_examples
from src.feature_analysis import filter_ranked_features
from src.feature_analysis import get_linear_feature_ranking
from src.modeling import RANDOM_SEED
from src.modeling import resolve_device
from src.targets import describe_targets


def main() -> None:
    """Train and evaluate the regression baseline."""

    torch.manual_seed(RANDOM_SEED)
    args = parse_args()
    device = resolve_device(args.device)
    frame = build_modeling_frame(args.database)

    if args.describe_targets:
        target_summary = describe_targets(frame)
        print(target_summary.to_string(index=False))
        return

    if args.run_experiments:
        run_experiment_matrix(
            frame=frame,
            targets=parse_experiment_targets(args.experiment_targets),
            block_configs=parse_experiment_feature_blocks(
                args.experiment_feature_blocks
            ),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            device=device,
        )
        return

    feature_blocks = parse_feature_blocks(args.feature_blocks)
    result, test_split, target_stats, model, feature_columns = run_regression_experiment(
        frame=frame,
        target_column=args.target,
        feature_blocks=feature_blocks,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=device,
        report_progress=True,
    )
    print()
    print("Device:")
    print(result.device)
    print()
    print("Feature blocks:")
    print(format_feature_blocks(feature_blocks))
    print()
    print("Persistence baseline metrics:")
    print(f"target={args.target}")
    print(
        "validation="
        f"{{'mae': {result.persistence_valid_mae}, "
        f"'rmse': {result.persistence_valid_rmse}, "
        f"'r2': {result.persistence_valid_r2}}}"
    )
    print(
        "test="
        f"{{'mae': {result.persistence_test_mae}, "
        f"'rmse': {result.persistence_test_rmse}, "
        f"'r2': {result.persistence_test_r2}}}"
    )
    print()
    print("Validation metrics:")
    print(f"target={args.target}")
    print(
        {
            "mae": result.model_valid_mae,
            "rmse": result.model_valid_rmse,
            "r2": result.model_valid_r2,
        }
    )
    print()
    print("Test metrics:")
    print(
        {
            "mae": result.model_test_mae,
            "rmse": result.model_test_rmse,
            "r2": result.model_test_r2,
        }
    )
    if args.show_feature_ranking:
        ranking = get_linear_feature_ranking(model, feature_columns)
        ranking = filter_ranked_features(
            ranking,
            parse_ranking_prefixes(args.ranking_prefixes),
        )
        ranking = ranking.head(args.ranking_top_k)
        print()
        print("Feature ranking:")
        if ranking.empty:
            print("No features matched the requested ranking prefixes.")
        else:
            print(ranking.to_string(index=False))
    show_test_examples(model, test_split, target_stats)
    show_requested_prediction(
        model,
        test_split,
        target_stats,
        args.router,
        args.timestamp,
    )
