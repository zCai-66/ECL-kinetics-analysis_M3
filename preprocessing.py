"""Convert two-column CSV files into M2/M3 input tensors.

Current must already use the model convention: negative first step, positive
second step. This module preserves its sign; it never chooses an orientation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ECL_BINS = 12_208
ECL_BIN_WIDTH_S = {0.5: 0.00002048, 1.0: 0.00004096, 2.0: 0.00008192}
CURRENT_DT_S = 0.0001


@dataclass(frozen=True)
class Inputs:
    context: np.ndarray
    ecl: np.ndarray
    current: np.ndarray | None


def read_trace(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a header followed by time in seconds and one measured signal."""
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if values.shape[1] != 2 or len(values) < 2:
        raise ValueError(f"{path}: expected two columns and at least two data rows")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: all values must be finite")
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError(f"{path}: time must increase continuously across the record")
    return values[:, 0], values[:, 1]


def ecl_grid(tau: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(ECL_BINS + 1, dtype=np.float64) * ECL_BIN_WIDTH_S[tau]
    edges[-1] = 0.5 * tau
    return edges, 0.5 * (edges[:-1] + edges[1:])


def align_time(time: np.ndarray, target: np.ndarray, duration: float) -> np.ndarray:
    """Keep zero-based windows; align other first samples to the model origin."""
    dt = float(np.median(np.diff(time)))
    tolerance = max(1e-12, dt * 1e-4)
    if time.shape == target.shape and np.allclose(time, target, rtol=0, atol=tolerance):
        return time.copy()
    if abs(float(time[0])) <= max(tolerance, 0.25 * dt) and time[-1] >= duration - 1.5 * dt:
        return time.copy()
    return time - time[0] + target[0]


def extend_window(time: np.ndarray, signal: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Fill a missing endpoint only when it is within one sampling interval."""
    dt = float(np.median(np.diff(time)))
    tolerance = max(dt * 1.51, 1e-12)
    if time[0] > tolerance or time[-1] < duration - tolerance:
        raise ValueError(f"The signal must cover the required {duration:g} s window")
    if time[0] > 0:
        time = np.concatenate(([0.0], time))
        signal = np.concatenate(([signal[0]], signal))
    if time[-1] < duration:
        time = np.concatenate((time, [duration]))
        signal = np.concatenate((signal, [signal[-1]]))
    return time, signal


def integral_at(time: np.ndarray, signal: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Integrate the piecewise-linear signal exactly at requested bin edges."""
    widths = np.diff(time)
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (signal[:-1] + signal[1:]) * widths)))
    indices = np.clip(np.searchsorted(time, query, side="right") - 1, 0, len(time) - 2)
    dx = query - time[indices]
    slopes = (signal[indices + 1] - signal[indices]) / widths[indices]
    return cumulative[indices] + signal[indices] * dx + 0.5 * slopes * np.square(dx)


def signal_channels(signal: np.ndarray, time: np.ndarray, tau: float) -> np.ndarray:
    """Three channels: normalized signal, time/period, and local interval/period."""
    delta = np.diff(np.concatenate(([0.0], np.asarray(time, dtype=np.float64))))
    result = np.empty((3, signal.size), dtype=np.float32)
    result[0] = signal
    result[1:] = np.stack((time / tau, delta / tau)).astype(np.float32)
    return result


def prepare_ecl(path: str | Path, tau: float) -> np.ndarray:
    time, counts = read_trace(path)
    edges, target = ecl_grid(tau)
    time = align_time(time, target, tau / 2)
    counts = np.maximum(counts, 0.0)
    if time.shape == target.shape and np.allclose(time, target, rtol=0, atol=max(1e-12, ECL_BIN_WIDTH_S[tau] * 1e-5)):
        binned = counts
    else:
        time, counts = extend_window(time, counts, tau / 2)
        binned = np.diff(integral_at(time, counts, edges)) / np.diff(edges)
    positive = np.maximum(binned, 0.0)
    maximum = float(np.max(positive))
    if maximum <= 0:
        raise ValueError("ECL needs a positive signal")
    return signal_channels((positive / maximum).astype(np.float32), target, tau)


def prepare_current(path: str | Path, tau: float) -> np.ndarray:
    """Resample signed current and normalize its magnitude without flipping it."""
    time, current = read_trace(path)
    target = np.linspace(0.0, tau, int(round(tau / CURRENT_DT_S)) + 1, dtype=np.float64)
    time = align_time(time, target, tau)
    exact = time.shape == target.shape and np.allclose(time, target, rtol=0, atol=max(1e-12, CURRENT_DT_S * 1e-5))
    extended_time, extended_current = extend_window(time, current, tau)
    sampled = current.copy() if exact else np.interp(target, extended_time, extended_current)
    maximum = float(np.max(np.abs(sampled)))
    if maximum <= 0:
        raise ValueError("Current needs a nonzero signal")
    return signal_channels((sampled / maximum).astype(np.float32), target, tau)


def prepare_inputs(ecl_path: str | Path, current_path: str | Path | None,
                   concentration_mM: float, frequency_hz: float) -> Inputs:
    if frequency_hz not in (0.5, 1.0, 2.0):
        raise ValueError("Use a cycle frequency of 0.5, 1, or 2 Hz")
    if not np.isfinite(concentration_mM) or not 0.05 <= concentration_mM <= 2.0:
        raise ValueError("Use a concentration between 0.05 and 2 mM")
    tau = 1.0 / frequency_hz
    # 1 mM is numerically equal to 1 mol/m^3, the training concentration unit.
    context = np.asarray((np.log10(concentration_mM), np.log10(tau)), dtype=np.float32)
    return Inputs(context, prepare_ecl(ecl_path, tau),
                  None if current_path is None else prepare_current(current_path, tau))
