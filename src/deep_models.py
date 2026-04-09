"""Deep regression model definitions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CurveGRUInputSpec:
    """Describe how one feature vector is partitioned for the curve GRU."""

    scalar_indices: tuple[int, ...]
    spectrum_raw_indices: tuple[int, ...]
    structure_raw_indices: tuple[int, ...]
    spectrum_point_count: int
    structure_point_count: int


class GRURegressionModel(nn.Module):
    """A stacked GRU regressor with pooled sequence features."""

    def __init__(
        self,
        feature_count: int,
        hidden_size: int = 96,
        num_layers: int = 2,
        dropout: float = 0.3,
        head_hidden_size: int = 64,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.input_norm = nn.LayerNorm(feature_count)
        self.input_projection = nn.Linear(feature_count, hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, head_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.input_norm(features)
        features = self.input_projection(features)
        features = torch.nn.functional.silu(features)
        features = self.input_dropout(features)
        outputs, hidden = self.gru(features)
        last_hidden = hidden[-1]
        mean_pooled = outputs.mean(dim=1)
        max_pooled = outputs.amax(dim=1)
        pooled_features = torch.cat(
            [last_hidden, mean_pooled, max_pooled],
            dim=-1,
        )
        return self.head(pooled_features).squeeze(1)


class MLPRegressionModel(nn.Module):
    """A compact MLP that flattens fixed-size sequence windows."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        hidden_sizes: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        input_size = math.prod(input_shape)
        layers: list[nn.Module] = [nn.Flatten(start_dim=1), nn.LayerNorm(input_size)]
        in_features = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


class CurveGRURegressionModel(nn.Module):
    """Encode raw curves with Conv1D before temporal modeling with a GRU."""

    def __init__(
        self,
        input_spec: CurveGRUInputSpec,
        hidden_size: int = 96,
        num_layers: int = 2,
        dropout: float = 0.3,
        curve_channels: int = 32,
        scalar_hidden_size: int = 64,
        head_hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.input_spec = input_spec
        self.scalar_indices = torch.tensor(input_spec.scalar_indices, dtype=torch.long)
        self.spectrum_raw_indices = torch.tensor(
            input_spec.spectrum_raw_indices,
            dtype=torch.long,
        )
        self.structure_raw_indices = torch.tensor(
            input_spec.structure_raw_indices,
            dtype=torch.long,
        )
        scalar_feature_count = len(input_spec.scalar_indices)
        fused_feature_count = scalar_hidden_size + 2 * curve_channels

        self.scalar_norm = nn.LayerNorm(scalar_feature_count)
        self.scalar_projection = nn.Linear(scalar_feature_count, scalar_hidden_size)
        self.scalar_dropout = nn.Dropout(dropout)
        self.spectrum_encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, curve_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.structure_encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, curve_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.step_projection = nn.Sequential(
            nn.LayerNorm(fused_feature_count),
            nn.Linear(fused_feature_count, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, head_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, 1),
        )

    def _encode_curve_block(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        point_count: int,
        encoder: nn.Sequential,
    ) -> torch.Tensor:
        batch_size, sequence_length = features.shape[:2]
        curve_values = features.index_select(dim=2, index=indices.to(features.device))
        curve_values = curve_values.reshape(batch_size * sequence_length, point_count, 2)
        curve_values = curve_values.transpose(1, 2)
        encoded = encoder(curve_values).squeeze(-1)
        return encoded.reshape(batch_size, sequence_length, -1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scalar_values = features.index_select(
            dim=2,
            index=self.scalar_indices.to(features.device),
        )
        scalar_values = self.scalar_norm(scalar_values)
        scalar_values = self.scalar_projection(scalar_values)
        scalar_values = torch.nn.functional.silu(scalar_values)
        scalar_values = self.scalar_dropout(scalar_values)

        spectrum_values = self._encode_curve_block(
            features,
            self.spectrum_raw_indices,
            self.input_spec.spectrum_point_count,
            self.spectrum_encoder,
        )
        structure_values = self._encode_curve_block(
            features,
            self.structure_raw_indices,
            self.input_spec.structure_point_count,
            self.structure_encoder,
        )

        step_features = torch.cat(
            [scalar_values, spectrum_values, structure_values],
            dim=-1,
        )
        step_features = self.step_projection(step_features)
        outputs, hidden = self.gru(step_features)
        last_hidden = hidden[-1]
        mean_pooled = outputs.mean(dim=1)
        max_pooled = outputs.amax(dim=1)
        pooled_features = torch.cat(
            [last_hidden, mean_pooled, max_pooled],
            dim=-1,
        )
        return self.head(pooled_features).squeeze(1)
