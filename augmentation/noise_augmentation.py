"""Noise augmentation used to prepare the noisy ECL/current training views."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ECLNoiseRealization:
    peak_photon_rate_s: float
    background_to_peak_rate: float
    gain_start: float
    gain_end_to_start: float
    nuisance_seed: int
    count_seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CurrentNoiseRealization:
    white_sigma_fraction: float
    ar1_sigma_fraction: float
    ar1_rho: float
    baseline_offset_fraction: float
    drift_end_fraction: float
    quantization_bits: int
    adc_range_fraction: float
    nuisance_seed: int
    noise_seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(int(base_seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def _log_uniform(rng: np.random.Generator, lower_log10: float, upper_log10: float) -> float:
    return float(10.0 ** rng.uniform(lower_log10, upper_log10))


def sample_ecl_realization(nuisance_seed: int, count_seed: int) -> ECLNoiseRealization:
    rng = np.random.default_rng(nuisance_seed)
    peak_rate = _log_uniform(rng, 5.0, 9.0)
    background = 0.0 if float(rng.random()) < 0.25 else _log_uniform(
        rng, -5.0, -1.3010299956639813
    )
    return ECLNoiseRealization(
        peak_photon_rate_s=peak_rate,
        background_to_peak_rate=background,
        gain_start=float(rng.uniform(0.8, 1.25)),
        gain_end_to_start=float(rng.uniform(0.95, 1.05)),
        nuisance_seed=int(nuisance_seed),
        count_seed=int(count_seed),
    )


def augment_ecl(
    clean_bin_average: np.ndarray,
    bin_width_s: float | np.ndarray,
    *,
    nuisance_seed: int,
    count_seed: int,
) -> tuple[np.ndarray, np.ndarray, ECLNoiseRealization]:
    """Convert a clean nonnegative ECL shape into Poisson-distributed counts.

    ``clean_bin_average`` is the clean ECL signal averaged within each bin and
    normalized to its positive maximum.  The returned arrays are integer counts
    and their expected values before Poisson sampling.
    """

    clean = np.maximum(np.asarray(clean_bin_average, dtype=np.float64), 0.0)
    widths = np.broadcast_to(np.asarray(bin_width_s, dtype=np.float64), clean.shape)
    if clean.ndim != 1 or clean.size < 2 or np.any(widths <= 0.0):
        raise ValueError("ECL input must be a one-dimensional record with positive bin widths")
    if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(widths)):
        raise ValueError("ECL input contains non-finite values")
    peak = float(np.max(clean))
    if peak <= 0.0:
        raise ValueError("ECL input has no positive signal")
    normalized = clean / peak
    realization = sample_ecl_realization(nuisance_seed, count_seed)
    progress = (np.arange(clean.size, dtype=np.float64) + 0.5) / clean.size
    gain = realization.gain_start * (
        1.0 + (realization.gain_end_to_start - 1.0) * progress
    )
    integrated = normalized * widths
    expected = np.maximum(
        gain
        * realization.peak_photon_rate_s
        * (integrated + realization.background_to_peak_rate * widths),
        0.0,
    )
    counts = np.random.default_rng(realization.count_seed).poisson(expected).astype(np.int64)
    return counts, expected, realization


def normalize_ecl_counts(counts: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(counts, dtype=np.float64), 0.0)
    peak = float(np.max(values))
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("ECL count record has no positive finite maximum")
    return (values / peak).astype(np.float32)


def sample_current_realization(nuisance_seed: int) -> CurrentNoiseRealization:
    rng = np.random.default_rng(nuisance_seed)
    bits = int(rng.choice(np.asarray((0, 14, 16, 18)), p=np.full(4, 0.25)))
    return CurrentNoiseRealization(
        white_sigma_fraction=_log_uniform(rng, -5.0, -2.5228787452803374),
        ar1_sigma_fraction=_log_uniform(rng, -5.0, -2.5228787452803374),
        ar1_rho=float(rng.uniform(0.8, 0.995)),
        baseline_offset_fraction=float(rng.uniform(-0.005, 0.005)),
        drift_end_fraction=float(rng.uniform(-0.01, 0.01)),
        quantization_bits=bits,
        adc_range_fraction=1.2,
        nuisance_seed=int(nuisance_seed),
        noise_seed=stable_seed(nuisance_seed, "current_noise"),
    )


def augment_current(
    clean_current_A: np.ndarray,
    *,
    nuisance_seed: int,
) -> tuple[np.ndarray, CurrentNoiseRealization]:
    """Add white, AR(1), offset, drift, and optional digitization effects."""

    clean = np.asarray(clean_current_A, dtype=np.float64)
    if clean.ndim != 1 or clean.size < 2 or not np.all(np.isfinite(clean)):
        raise ValueError("Current input must be a finite one-dimensional record")
    scale = float(np.max(np.abs(clean)))
    if scale <= 0.0:
        raise ValueError("Current input has no nonzero signal")
    realization = sample_current_realization(nuisance_seed)
    rng = np.random.default_rng(realization.noise_seed)
    white = rng.normal(
        scale=realization.white_sigma_fraction * scale,
        size=clean.size,
    )
    innovations = rng.normal(
        scale=realization.ar1_sigma_fraction
        * scale
        * math.sqrt(1.0 - realization.ar1_rho**2),
        size=clean.size,
    )
    ar1 = np.empty(clean.size, dtype=np.float64)
    ar1[0] = innovations[0]
    for index in range(1, clean.size):
        ar1[index] = realization.ar1_rho * ar1[index - 1] + innovations[index]
    progress = np.linspace(0.0, 1.0, clean.size, dtype=np.float64)
    observed = (
        clean
        + white
        + ar1
        + realization.baseline_offset_fraction * scale
        + realization.drift_end_fraction * scale * progress
    )
    if realization.quantization_bits > 0:
        adc_range = realization.adc_range_fraction * scale
        levels = 2**realization.quantization_bits - 1
        clipped = np.clip(observed, -adc_range, adc_range)
        observed = np.round((clipped + adc_range) * levels / (2.0 * adc_range))
        observed = observed * (2.0 * adc_range) / levels - adc_range
    return np.asarray(observed, dtype=np.float64), realization


def normalize_signed_current(current_A: np.ndarray) -> np.ndarray:
    values = np.asarray(current_A, dtype=np.float64)
    scale = float(np.max(np.abs(values)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Current record has no finite nonzero maximum")
    return (values / scale).astype(np.float32)
