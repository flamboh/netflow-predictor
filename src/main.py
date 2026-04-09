"""Train and evaluate netflow prediction baselines and GRU models."""

from __future__ import annotations

import torch

from src.cli import parse_args
from src.cli import parse_experiment_feature_blocks
from src.cli import parse_experiment_targets
from src.cli import parse_feature_blocks
from src.cli import parse_permutation_groups
from src.cli import parse_ranking_prefixes
from src.experiments import build_modeling_frame
from src.experiments import run_experiment_matrix
from src.experiments import run_regression_experiment
from src.feature_analysis import filter_ranked_features
from src.feature_analysis import get_grouped_permutation_importance
from src.feature_analysis import get_model_feature_ranking
from src.modeling import RANDOM_SEED
from src.modeling import resolve_device
from src.reporting import print_run_config
from src.reporting import print_run_results
from src.reporting import show_requested_prediction
from src.reporting import show_test_examples
from src.targets import describe_targets


def main() -> None:
    """Train and evaluate one configured experiment."""

    torch.manual_seed(RANDOM_SEED)
    args = parse_args()
    device = resolve_device(args.device)
    train_router = None if args.train_router.strip().lower() == "all" else args.train_router
    learning_rate = 0.001 if args.model_backend == "gru" else 0.01
    if args.learning_rate is not None:
        learning_rate = args.learning_rate
    frame = build_modeling_frame(args.database, train_router)

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
            model_backend=args.model_backend,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            learning_rate=learning_rate,
            batch_size=args.batch_size,
            device=device,
        )
        return

    feature_blocks = parse_feature_blocks(args.feature_blocks)
    result, valid_split, test_split, target_stats, model, feature_columns = run_regression_experiment(
        frame=frame,
        target_column=args.target,
        model_backend=args.model_backend,
        feature_blocks=feature_blocks,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        learning_rate=learning_rate,
        batch_size=args.batch_size,
        device=device,
        report_progress=True,
    )
    print()
    print_run_config(
        target_column=args.target,
        model_backend=result.model_backend,
        device=result.device,
        train_router=args.train_router,
        learning_rate=learning_rate,
        feature_blocks=feature_blocks,
        feature_count=result.feature_count,
        epochs=result.epochs,
        sequence_length=args.sequence_length,
    )
    print()
    print_run_results(result)
    if args.show_feature_ranking:
        if args.model_backend == "gru":
            raise ValueError("Feature ranking is not supported for the gru backend.")

        ranking = get_model_feature_ranking(model, feature_columns)
        ranking = filter_ranked_features(
            ranking,
            parse_ranking_prefixes(args.ranking_prefixes),
        )
        ranking = ranking.head(args.ranking_top_k)
        print()
        print(f"Feature ranking  (top {args.ranking_top_k}):")
        if ranking.empty:
            print("No features matched the requested ranking prefixes.")
        else:
            print(ranking.to_string(index=False))
    if args.show_group_permutation_importance:
        if args.permutation_split == "validation":
            permutation_split = valid_split
        elif args.permutation_split == "test":
            permutation_split = test_split
        else:
            raise ValueError("Permutation split must be 'validation' or 'test'.")
        importance = get_grouped_permutation_importance(
            model=model,
            split=permutation_split,
            target_stats=target_stats,
            feature_columns=feature_columns,
            groups=parse_permutation_groups(args.permutation_groups),
            repeats=args.permutation_repeats,
        )
        print()
        print(f"Grouped permutation importance  (split={args.permutation_split})")
        if importance.empty:
            print("No groups matched the current feature set.")
        else:
            print(importance.to_string(index=False))
    show_test_examples(model, test_split, target_stats)
    show_requested_prediction(
        model,
        test_split,
        target_stats,
        args.router,
        args.timestamp,
    )


if __name__ == "__main__":
    main()
