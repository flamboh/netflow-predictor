"""Regression backends and prediction helpers."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from xgboost import XGBRegressor

from src.modeling import RANDOM_SEED
from src.modeling import SplitData
from src.modeling import TargetStandardization
from src.modeling import evaluate_predictions


MODEL_BACKENDS = ("linear", "xgboost", "gru")


class LinearRegressionModel(nn.Module):
    """A single linear layer for regression."""

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_count, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(1)


class GRURegressionModel(nn.Module):
    """A small GRU regressor over fixed-length windows."""

    def __init__(self, feature_count: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_count,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(features)
        return self.output(hidden[-1]).squeeze(1)


def validate_model_backend(model_backend: str) -> None:
    """Fail early on unknown regression backends."""

    if model_backend in MODEL_BACKENDS:
        return

    available_backends = ", ".join(MODEL_BACKENDS)
    raise ValueError(f"Unknown model backend: {model_backend}. Available: {available_backends}")


def move_target_stats(
    stats: TargetStandardization,
    device: torch.device,
) -> TargetStandardization:
    """Move target scaling tensors onto the selected device."""

    return TargetStandardization(
        mean=stats.mean.to(device),
        std=stats.std.to(device),
    )


def get_model_device(model: nn.Module) -> torch.device:
    """Return the device used by the model parameters."""

    return next(model.parameters()).device


def make_loader(
    split: SplitData,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Create a DataLoader for one split."""

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    dataset = TensorDataset(split.features, split.targets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        pin_memory=device.type == "cuda",
    )


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_split: SplitData,
    target_stats: TargetStandardization,
    epochs: int,
    learning_rate: float,
    report_progress: bool = True,
) -> None:
    """Fit one torch-based regression model."""

    device = get_model_device(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()

        for feature_batch, target_batch in train_loader:
            feature_batch = feature_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            predictions = model(feature_batch)
            loss = loss_fn(predictions, target_batch)
            loss.backward()
            optimizer.step()

        if report_progress and ((epoch + 1) % 25 == 0 or epoch == 0):
            valid_metrics = evaluate_regressor(model, valid_split, target_stats)
            print(
                f"epoch={epoch + 1} "
                f"valid_mae={valid_metrics['mae']:.2f} "
                f"valid_rmse={valid_metrics['rmse']:.2f} "
                f"valid_r2={valid_metrics['r2']:.4f}"
            )


def train_xgboost_model(
    train_split: SplitData,
    valid_split: SplitData,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    report_progress: bool,
) -> tuple[XGBRegressor, str]:
    """Fit an XGBoost regressor on the raw feature matrix."""

    if train_split.features.ndim != 2:
        raise ValueError("XGBoost requires a tabular 2D feature matrix.")

    xgb_device = "cuda" if device.type == "cuda" else "cpu"
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=epochs,
        learning_rate=learning_rate,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        tree_method="hist",
        device=xgb_device,
        eval_metric="mae",
        verbosity=1 if report_progress else 0,
    )
    model.fit(
        train_split.features.numpy(),
        train_split.targets.numpy(),
        eval_set=[(valid_split.features.numpy(), valid_split.targets.numpy())],
        verbose=25 if report_progress else False,
    )
    return model, xgb_device


def train_regressor(
    model_backend: str,
    train_split: SplitData,
    valid_split: SplitData,
    target_stats: TargetStandardization,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
    report_progress: bool,
) -> tuple[nn.Module | XGBRegressor, TargetStandardization, str]:
    """Train one supported regression backend."""

    validate_model_backend(model_backend)

    if model_backend == "xgboost":
        model, backend_device = train_xgboost_model(
            train_split=train_split,
            valid_split=valid_split,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            report_progress=report_progress,
        )
        return model, target_stats, backend_device

    train_loader = make_loader(train_split, batch_size, shuffle=True, device=device)
    moved_target_stats = move_target_stats(target_stats, device)
    torch.manual_seed(RANDOM_SEED)

    if model_backend == "linear":
        model = LinearRegressionModel(feature_count=train_split.features.shape[-1]).to(device)
    else:
        model = GRURegressionModel(feature_count=train_split.features.shape[-1]).to(device)

    train_torch_model(
        model=model,
        train_loader=train_loader,
        valid_split=valid_split,
        target_stats=moved_target_stats,
        epochs=epochs,
        learning_rate=learning_rate,
        report_progress=report_progress,
    )
    return model, moved_target_stats, device.type


def predict_regressor(
    model: nn.Module | XGBRegressor,
    split: SplitData,
    target_stats: TargetStandardization,
) -> torch.Tensor:
    """Run one supported backend and return predictions."""

    if isinstance(model, nn.Module):
        device = get_model_device(model)
        model.eval()

        with torch.no_grad():
            scaled_predictions = model(split.features.to(device))

        return scaled_predictions * target_stats.std.to(device) + target_stats.mean.to(device)

    predictions = model.predict(split.features.numpy())
    return torch.tensor(predictions, dtype=torch.float32)


def evaluate_regressor(
    model: nn.Module | XGBRegressor,
    split: SplitData,
    target_stats: TargetStandardization,
) -> dict[str, float]:
    """Evaluate one learned backend on one split."""

    predictions = predict_regressor(model, split, target_stats)
    actual_targets = torch.tensor(
        split.frame[split.target_column].to_numpy(dtype="float32"),
        device=predictions.device,
    )
    return evaluate_predictions(predictions, actual_targets)


def get_feature_ranking(
    model: nn.Module | XGBRegressor,
    feature_columns: Sequence[str],
    infer_feature_group: Callable[[str], str],
) -> pd.DataFrame:
    """Build a sorted feature-importance table for one learned backend."""

    if isinstance(model, nn.Module):
        if not hasattr(model, "linear"):
            raise ValueError(
                "Feature ranking is only supported for the linear and xgboost backends."
            )

        scores = model.linear.weight.detach().cpu().flatten().tolist()
        score_name = "coefficient"
        abs_score_name = "abs_coefficient"
    else:
        scores = model.feature_importances_.tolist()
        score_name = "importance"
        abs_score_name = "abs_importance"

    if len(scores) != len(feature_columns):
        raise ValueError("Feature column count does not match learned feature scores.")

    frame = pd.DataFrame(
        {
            "feature": list(feature_columns),
            score_name: scores,
        }
    )
    frame[abs_score_name] = frame[score_name].abs()
    frame["group"] = frame["feature"].map(infer_feature_group)
    return frame.sort_values(abs_score_name, ascending=False).reset_index(drop=True)
