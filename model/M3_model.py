"""Inference-only copy of the frozen fixed-window CNN architecture.

Layer names intentionally match the training implementation so checkpoint
state dictionaries load with ``strict=True``.  The model specification is
read from each frozen checkpoint instead of a mutable global training table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ArchitectureConfig:
    input_channels: int
    channels: tuple[int, ...]
    kernels: tuple[int, ...]
    ecl_strides: tuple[int, ...]
    current_strides: tuple[int, ...]
    ecl_masked_pool_bins: int
    current_masked_pool_bins: int
    signal_embedding_units: int
    scalar_embedding_units: int
    single_fusion_units: tuple[int, ...]
    dual_fusion_units: tuple[int, ...]
    dropout: float

    def validate(self) -> None:
        count = len(self.channels)
        if self.input_channels != 3 or count == 0:
            raise ValueError("The signal contract requires three channels")
        if not (
            count
            == len(self.kernels)
            == len(self.ecl_strides)
            == len(self.current_strides)
        ):
            raise ValueError("Encoder stage specifications must have equal length")
        if any(value <= 0 for value in self.channels + self.ecl_strides + self.current_strides):
            raise ValueError("Channels and strides must be positive")
        if any(value <= 0 or value % 2 == 0 for value in self.kernels):
            raise ValueError("Kernels must be positive odd integers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Dropout must be in [0, 1)")


def architecture_from_mapping(values: Mapping[str, Any]) -> ArchitectureConfig:
    return ArchitectureConfig(
        input_channels=int(values["input_channels"]),
        channels=tuple(int(value) for value in values["channels"]),
        kernels=tuple(int(value) for value in values["kernels"]),
        ecl_strides=tuple(int(value) for value in values["ecl_strides"]),
        current_strides=tuple(int(value) for value in values["current_strides"]),
        ecl_masked_pool_bins=int(values["ecl_masked_pool_bins"]),
        current_masked_pool_bins=int(values["current_masked_pool_bins"]),
        signal_embedding_units=int(values["signal_embedding_units"]),
        scalar_embedding_units=int(values["scalar_embedding_units"]),
        single_fusion_units=tuple(int(value) for value in values["single_fusion_units"]),
        dual_fusion_units=tuple(int(value) for value in values["dual_fusion_units"]),
        dropout=float(values["dropout"]),
    )


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.normalization(values.transpose(1, 2)).transpose(1, 2)


class MaskedConvStage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = kernel // 2
        self.first = nn.Conv1d(
            input_channels,
            output_channels,
            kernel,
            stride=stride,
            padding=self.padding,
            bias=False,
        )
        self.first_norm = ChannelLayerNorm(output_channels)
        self.second = nn.Conv1d(output_channels, output_channels, 3, padding=1, bias=False)
        self.second_norm = ChannelLayerNorm(output_channels)
        self.skip = nn.Conv1d(input_channels, output_channels, 1, stride=stride, bias=False)

    def _mask(self, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=2, dtype=torch.int64)
        output_lengths = (
            torch.div(
                lengths + 2 * self.padding - (self.kernel - 1) - 1,
                self.stride,
                rounding_mode="floor",
            )
            + 1
        )
        output_length = (
            mask.shape[2] + 2 * self.padding - (self.kernel - 1) - 1
        ) // self.stride + 1
        positions = torch.arange(output_length, device=mask.device).view(1, 1, -1)
        return positions < output_lengths.unsqueeze(2)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        next_mask = self._mask(mask)
        numeric = next_mask.to(values.dtype)
        main = F.gelu(self.first_norm(self.first(values))) * numeric
        main = self.second_norm(self.second(main))
        return F.gelu(main + self.skip(values)) * numeric, next_mask


class MaskedAdaptiveStatistics(nn.Module):
    def __init__(self, bins: int) -> None:
        super().__init__()
        self.bins = bins

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if bool(torch.all(mask)):
            return torch.cat(
                (
                    F.adaptive_avg_pool1d(values, self.bins),
                    F.adaptive_max_pool1d(values, self.bins),
                ),
                dim=1,
            )
        pooled = []
        for index in range(values.shape[0]):
            valid = values[index : index + 1, :, mask[index, 0]]
            if valid.shape[-1] < self.bins:
                raise ValueError("Downsampled record is too short")
            pooled.append(
                torch.cat(
                    (
                        F.adaptive_avg_pool1d(valid, self.bins),
                        F.adaptive_max_pool1d(valid, self.bins),
                    ),
                    dim=1,
                )
            )
        return torch.cat(pooled, dim=0)


class MaskedSignalEncoder(nn.Module):
    def __init__(self, architecture: ArchitectureConfig, strides: Sequence[int], pool_bins: int) -> None:
        super().__init__()
        stages = []
        input_channels = architecture.input_channels
        for output_channels, kernel, stride in zip(architecture.channels, architecture.kernels, strides):
            stages.append(MaskedConvStage(input_channels, output_channels, kernel, stride))
            input_channels = output_channels
        self.stages = nn.ModuleList(stages)
        self.pool = MaskedAdaptiveStatistics(pool_bins)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                2 * architecture.channels[-1] * pool_bins,
                architecture.signal_embedding_units,
            ),
            nn.LayerNorm(architecture.signal_embedding_units),
            nn.GELU(),
        )

    def forward(self, series: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if (
            series.ndim != 3
            or series.shape[1] != 3
            or mask.shape != (series.shape[0], 1, series.shape[2])
            or mask.dtype != torch.bool
        ):
            raise ValueError("Invalid series/mask contract")
        values = series * mask.to(series.dtype)
        current_mask = mask
        for stage in self.stages:
            values, current_mask = stage(values, current_mask)
        return self.projection(self.pool(values, current_mask))


class FrozenFixedWindowCNN(nn.Module):
    """Checkpoint-compatible inference model with an explicit saved contract."""

    def __init__(
        self,
        model_id: str,
        mode: str,
        variant: str,
        scalar_kind: str,
        scalar_features: int,
        target_names: Sequence[str],
        architecture: ArchitectureConfig,
        target_mean: Sequence[float],
        target_scale: Sequence[float],
    ) -> None:
        super().__init__()
        architecture.validate()
        if mode not in {"ecl", "dual"}:
            raise ValueError(f"Unsupported frozen mode: {mode}")
        if variant not in {"clean", "noisy"}:
            raise ValueError(f"Unsupported frozen variant: {variant}")
        if scalar_kind != "context" or scalar_features != 2:
            raise ValueError("Frozen M1-M6 require exactly CM0 and tau scalar features")
        targets = tuple(str(value) for value in target_names)
        mean = torch.as_tensor(target_mean, dtype=torch.float32)
        scale = torch.as_tensor(target_scale, dtype=torch.float32)
        if mean.shape != (len(targets),) or scale.shape != mean.shape or torch.any(scale <= 0):
            raise ValueError("Invalid target statistics")

        self.model_id = model_id
        self.mode = mode
        self.variant = variant
        self.scalar_kind = scalar_kind
        self.scalar_features = scalar_features
        self.target_names = targets
        self.architecture = architecture
        self.register_buffer("target_mean", mean)
        self.register_buffer("target_scale", scale)
        self.ecl_encoder = MaskedSignalEncoder(
            architecture,
            architecture.ecl_strides,
            architecture.ecl_masked_pool_bins,
        )
        self.current_encoder = (
            MaskedSignalEncoder(
                architecture,
                architecture.current_strides,
                architecture.current_masked_pool_bins,
            )
            if mode == "dual"
            else None
        )
        signal_features = architecture.signal_embedding_units * (2 if mode == "dual" else 1)
        fusion_units = architecture.dual_fusion_units if mode == "dual" else architecture.single_fusion_units
        self.scalar_projection = nn.Sequential(
            nn.Linear(scalar_features, architecture.scalar_embedding_units),
            nn.GELU(),
            nn.Linear(architecture.scalar_embedding_units, architecture.scalar_embedding_units),
            nn.GELU(),
        )
        previous = signal_features + architecture.scalar_embedding_units
        layers: list[nn.Module] = [nn.LayerNorm(previous)]
        for units in fusion_units:
            layers.extend((nn.Linear(previous, units), nn.GELU(), nn.Dropout(architecture.dropout)))
            previous = units
        self.fusion = nn.Sequential(*layers)
        self.output = nn.Linear(previous, len(targets))

    def forward(
        self,
        scalar_z: torch.Tensor,
        ecl_series: torch.Tensor,
        ecl_mask: torch.Tensor,
        current_series: torch.Tensor | None = None,
        current_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scalar_z.ndim != 2 or scalar_z.shape[1] != self.scalar_features:
            raise ValueError("Scalar feature count differs from frozen contract")
        embeddings = [self.ecl_encoder(ecl_series, ecl_mask)]
        if self.current_encoder is not None:
            if current_series is None or current_mask is None:
                raise ValueError("Current input is required")
            embeddings.append(self.current_encoder(current_series, current_mask))
        elif current_series is not None or current_mask is not None:
            raise ValueError("ECL-only model received current")
        embeddings.append(self.scalar_projection(scalar_z))
        return self.output(self.fusion(torch.cat(embeddings, dim=1)))


def build_from_checkpoint_config(config: Mapping[str, Any]) -> FrozenFixedWindowCNN:
    return FrozenFixedWindowCNN(
        model_id=str(config["model_id"]),
        mode=str(config["mode"]),
        variant=str(config["variant"]),
        scalar_kind=str(config["scalar_kind"]),
        scalar_features=int(config["scalar_features"]),
        target_names=tuple(config["target_names"]),
        architecture=architecture_from_mapping(config["architecture"]),
        target_mean=tuple(float(value) for value in config["target_mean"]),
        target_scale=tuple(float(value) for value in config["target_scale"]),
    )
