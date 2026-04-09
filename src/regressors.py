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

from src.curve_features import SPECTRUM_RAW_FEATURE_COLUMNS
from src.curve_features import SPECTRUM_RAW_POINT_COUNT
from src.curve_features import STRUCTURE_RAW_FEATURE_COLUMNS
from src.curve_features import STRUCTURE_RAW_POINT_COUNT
from src.deep_models import CurveGRUInputSpec
from src.deep_models import CurveGRURegressionModel
from src.deep_models import GRURegressionModel
from src.deep_models import MLPRegressionModel
from src.modeling import RANDOM_SEED
from src.modeling import SplitData
from src.modeling import TargetStandardization
from src.modeling import evaluate_predictions
from src.modeling import invert_target_transform


MODEL_BACKENDS = ("linear", "xgboost", "gru", "mlp", "curve_gru")
GRU_PATIENCE = 75
GRU_GRAD_CLIP = 1.0
LOSS_NAMES = ("mse", "huber")


class LinearRegressionModel(nn.Module):
    """A single linear layer for regression."""

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_count, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(1)


def validate_model_backend(model_backend: str) -> None:
    """Fail early on unknown regression backends."""

    if model_backend in MODEL_BACKENDS:
        return

    available_backends = ", ".join(MODEL_BACKENDS)
    raise ValueError(f"Unknown model backend: {model_backend}. Available: {available_backends}")


def validate_loss_name(loss_name: str) -> None:
    """Fail early on unknown torch loss choices."""

    if loss_name in LOSS_NAMES:
        return

    available_losses = ", ".join(LOSS_NAMES)
    raise ValueError(f"Unknown loss: {loss_name}. Available: {available_losses}")


def build_curve_gru_input_spec(
    feature_columns: Sequence[str],
) -> CurveGRUInputSpec:
    """Partition one feature vector into scalar and raw-curve slices."""

    spectrum_lookup = set(SPECTRUM_RAW_FEATURE_COLUMNS)
    structure_lookup = set(STRUCTURE_RAW_FEATURE_COLUMNS)
    scalar_indices: list[int] = []
    spectrum_indices: list[int] = []
    structure_indices: list[int] = []

    for index, feature_name in enumerate(feature_columns):
        if feature_name in spectrum_lookup:
            spectrum_indices.append(index)
            continue
        if feature_name in structure_lookup:
            structure_indices.append(index)
            continue
        scalar_indices.append(index)

    if len(spectrum_indices) != len(SPECTRUM_RAW_FEATURE_COLUMNS):
        raise ValueError(
            "curve_gru requires the full spectrum_raw feature block."
        )
    if len(structure_indices) != len(STRUCTURE_RAW_FEATURE_COLUMNS):
        raise ValueError(
            "curve_gru requires the full structure_raw feature block."
        )
    if not scalar_indices:
        raise ValueError("curve_gru requires at least one non-raw feature.")

    return CurveGRUInputSpec(
        scalar_indices=tuple(scalar_indices),
        spectrum_raw_indices=tuple(spectrum_indices),
        structure_raw_indices=tuple(structure_indices),
        spectrum_point_count=SPECTRUM_RAW_POINT_COUNT,
        structure_point_count=STRUCTURE_RAW_POINT_COUNT,
    )


def move_target_stats(
    stats: TargetStandardization,
    device: torch.device,
) -> TargetStandardization:
    """Move target scaling tensors onto the selected device."""

    return TargetStandardization(
        mean=stats.mean.to(device),
        std=stats.std.to(device),
        transform_name=stats.transform_name,
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


def make_loss(loss_name: str) -> nn.Module:
    """Build the requested regression loss."""

    validate_loss_name(loss_name)

    if loss_name == "mse":
        return nn.MSELoss()

    return nn.HuberLoss()


def train_torch_model(
    model: nn.Module,
    model_backend: str,
    train_loader: DataLoader,
    valid_split: SplitData,
    target_stats: TargetStandardization,
    epochs: int,
    learning_rate: float,
    loss_name: str,
    report_progress: bool = True,
) -> None:
    """Fit one torch-based regression model."""

    device = get_model_device(model)
    weight_decay = 1e-4 if model_backend in ("gru", "mlp", "curve_gru") else 0.0
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = make_loss(loss_name)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )
    best_valid_rmse = float("inf")
    best_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }
    epochs_without_improvement = 0
    patience = GRU_PATIENCE if model_backend in ("gru", "mlp", "curve_gru") else epochs

    for epoch in range(epochs):
        model.train()

        for feature_batch, target_batch in train_loader:
            feature_batch = feature_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            predictions = model(feature_batch)
            loss = loss_fn(predictions, target_batch)
            loss.backward()
            if model_backend in ("gru", "mlp", "curve_gru"):
                nn.utils.clip_grad_norm_(model.parameters(), GRU_GRAD_CLIP)
            optimizer.step()

        valid_metrics = evaluate_regressor(model, valid_split, target_stats)
        scheduler.step(valid_metrics["rmse"])

        if valid_metrics["rmse"] < best_valid_rmse:
            best_valid_rmse = valid_metrics["rmse"]
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if report_progress and ((epoch + 1) % 25 == 0 or epoch == 0):
            print(
                f"epoch={epoch + 1} "
                f"valid_mae={valid_metrics['mae']:.2f} "
                f"valid_rmse={valid_metrics['rmse']:.2f} "
                f"valid_r2={valid_metrics['r2']:.4f}"
            )

        if epochs_without_improvement >= patience:
            if report_progress:
                print(
                    f"early_stop epoch={epoch + 1} "
                    f"best_valid_rmse={best_valid_rmse:.2f}"
                )
            break

    model.load_state_dict(best_state)


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
    feature_columns: Sequence[str] | None = None,
    loss_name: str | None = None,
    report_progress: bool = True,
) -> tuple[nn.Module | XGBRegressor, TargetStandardization, str]:
    """Train one supported regression backend."""

    validate_model_backend(model_backend)

    if loss_name is None:
        if model_backend in ("gru", "mlp", "curve_gru"):
            loss_name = "huber"
        else:
            loss_name = "mse"

    validate_loss_name(loss_name)

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
    elif model_backend == "gru":
        model = GRURegressionModel(feature_count=train_split.features.shape[-1]).to(device)
    elif model_backend == "curve_gru":
        if feature_columns is None:
            raise ValueError("curve_gru requires feature column metadata.")
        input_spec = build_curve_gru_input_spec(feature_columns)
        model = CurveGRURegressionModel(input_spec=input_spec).to(device)
    else:
        model = MLPRegressionModel(
            input_shape=tuple(train_split.features.shape[1:]),
        ).to(device)

    train_torch_model(
        model=model,
        model_backend=model_backend,
        train_loader=train_loader,
        valid_split=valid_split,
        target_stats=moved_target_stats,
        epochs=epochs,
        learning_rate=learning_rate,
        loss_name=loss_name,
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

        transformed_predictions = (
            scaled_predictions * target_stats.std.to(device)
            + target_stats.mean.to(device)
        )
        return invert_target_transform(
            transformed_predictions,
            target_stats.transform_name,
        )

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
