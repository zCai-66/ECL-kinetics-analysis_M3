"""Acquisition-matched parsing, resampling, normalization, and QC for M3."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SUPPORTED_TAU_S = (0.5, 1.0, 2.0)
ECL_USED_BINS = 12_208
ECL_BIN_WIDTH_S = {0.5: 0.00002048, 1.0: 0.00004096, 2.0: 0.00008192}
CURRENT_DT_S = 0.0001
CM0_RANGE_MOL_PER_M3 = (0.05, 2.0)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class QCItem:
    severity: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedTrace:
    source_name: str
    source_sha256: str
    time_s: np.ndarray
    signal: np.ndarray


@dataclass(frozen=True)
class PreparedInputs:
    cm0_mol_per_m3: float
    tau_s: float
    context_z_raw: np.ndarray
    ecl_time_s: np.ndarray
    ecl_series: np.ndarray
    ecl_raw_time_s: np.ndarray
    ecl_raw_signal: np.ndarray
    current_time_s: np.ndarray | None
    current_series: np.ndarray | None
    current_raw_time_s: np.ndarray | None
    current_raw_signal: np.ndarray | None
    current_sign_action: str
    ecl_source_name: str
    ecl_source_sha256: str
    current_source_name: str | None
    current_source_sha256: str | None
    qc: tuple[QCItem, ...]


class InputContractError(ValueError):
    def __init__(self, message: str, qc: Iterable[QCItem] = ()) -> None:
        super().__init__(message)
        self.qc = tuple(qc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _path_from_upload(value: Any) -> Path:
    if value is None:
        raise InputContractError("No file was supplied")
    if isinstance(value, (str, Path)):
        path = Path(value)
    elif hasattr(value, "name"):
        path = Path(str(value.name))
    else:
        raise InputContractError("Unsupported uploaded-file object")
    if not path.is_file():
        raise InputContractError(f"Uploaded file is unavailable: {path.name}")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise InputContractError("Uploaded file exceeds the 50 MiB safety limit")
    return path


_SPLIT = re.compile(r"[,;\t ]+")


def read_two_column_trace(value: Any, label: str) -> ParsedTrace:
    """Read two numeric columns while allowing one or more header lines."""

    path = _path_from_upload(value)
    rows: list[tuple[float, float]] = []
    data_started = False
    with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text or text.startswith(("#", "%", "//")):
                continue
            tokens = [token for token in _SPLIT.split(text) if token]
            parsed: tuple[float, float] | None = None
            if len(tokens) == 2:
                try:
                    parsed = (float(tokens[0]), float(tokens[1]))
                except ValueError:
                    parsed = None
            if parsed is None:
                if not data_started:
                    continue
                raise InputContractError(
                    f"{label}: line {line_number} is not exactly two numeric columns"
                )
            data_started = True
            rows.append(parsed)
    if len(rows) < 2:
        raise InputContractError(f"{label}: at least two numeric rows are required")
    array = np.asarray(rows, dtype=np.float64)
    time_s = array[:, 0]
    signal = array[:, 1]
    if not np.all(np.isfinite(array)):
        raise InputContractError(f"{label}: non-finite values are not allowed")
    if np.any(np.diff(time_s) <= 0.0):
        raise InputContractError(f"{label}: time must be strictly increasing; files are never silently sorted")
    return ParsedTrace(
        source_name=path.name,
        source_sha256=sha256_file(path),
        time_s=time_s,
        signal=signal,
    )


_CHI_SAMPLE_INTERVAL = re.compile(
    r"Sample\s+Interval\s*\(s\)\s*=\s*([0-9.eE+\-]+)", re.IGNORECASE
)


def read_current_trace(value: Any, tau_s: float) -> ParsedTrace:
    """Read a CH Instruments export or an ordinary two-column current file.

    CHI potential-step exports can restart the displayed time at the second
    pulse.  When the locked 0.1 ms acquisition interval and expected row count
    are present, row order defines the continuous full-cycle time axis.
    """

    path = _path_from_upload(value)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()
    marker = next(
        (
            index
            for index, line in enumerate(lines)
            if "time/sec" in line.lower() and "current/a" in line.lower()
        ),
        None,
    )
    if marker is None:
        return read_two_column_trace(path, "Current")

    interval_match = _CHI_SAMPLE_INTERVAL.search(text)
    if interval_match is None:
        raise InputContractError("Current: CHI export is missing its sample interval")
    sample_interval_s = float(interval_match.group(1))
    rows: list[tuple[float, float]] = []
    for line_number, raw in enumerate(lines[marker + 1 :], start=marker + 2):
        stripped = raw.strip()
        if not stripped:
            continue
        tokens = [token for token in _SPLIT.split(stripped) if token]
        if len(tokens) != 2:
            raise InputContractError(
                f"Current: line {line_number} is not exactly two numeric columns"
            )
        try:
            rows.append((float(tokens[0]), float(tokens[1])))
        except ValueError as exc:
            raise InputContractError(
                f"Current: line {line_number} is not exactly two numeric columns"
            ) from exc
    if len(rows) < 2:
        raise InputContractError("Current: at least two numeric rows are required")
    array = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise InputContractError("Current: non-finite values are not allowed")

    expected_intervals = int(round(float(tau_s) / CURRENT_DT_S))
    if (
        np.isclose(sample_interval_s, CURRENT_DT_S, rtol=0.0, atol=1e-12)
        and array.shape[0] == expected_intervals
    ):
        time_s = np.arange(1, expected_intervals + 1, dtype=np.float64) * CURRENT_DT_S
    elif np.all(np.diff(array[:, 0]) > 0.0):
        time_s = array[:, 0]
    else:
        raise InputContractError(
            "Current: displayed time is not continuous and the locked 0.1 ms row contract was not met"
        )
    return ParsedTrace(
        source_name=path.name,
        source_sha256=sha256_file(path),
        time_s=time_s,
        signal=array[:, 1],
    )


def ecl_edges_and_centers(tau_s: float) -> tuple[np.ndarray, np.ndarray]:
    dt_s = ECL_BIN_WIDTH_S[tau_s]
    edges = np.arange(ECL_USED_BINS + 1, dtype=np.float64) * dt_s
    edges[-1] = 0.5 * tau_s
    if np.any(np.diff(edges) <= 0.0):
        raise RuntimeError("Frozen ECL grid is invalid")
    return edges, 0.5 * (edges[:-1] + edges[1:])


def current_grid(tau_s: float) -> np.ndarray:
    length = int(round(tau_s / CURRENT_DT_S)) + 1
    return np.linspace(0.0, tau_s, length, dtype=np.float64)


def time_features(time_s: np.ndarray, tau_s: float) -> np.ndarray:
    delta = np.diff(np.concatenate(([0.0], np.asarray(time_s, dtype=np.float64))))
    result = np.stack((time_s / tau_s, delta / tau_s)).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Time features are non-finite")
    return result


def _align_time_origin(
    source_time: np.ndarray,
    target_time: np.ndarray,
    duration_s: float,
) -> tuple[np.ndarray, str]:
    source_time = np.asarray(source_time, dtype=np.float64)
    median_dt = float(np.median(np.diff(source_time)))
    tolerance = max(1e-12, median_dt * 1e-4)
    if source_time.shape == target_time.shape and np.allclose(source_time, target_time, rtol=0.0, atol=tolerance):
        return source_time.copy(), "input time already matched the frozen grid"
    starts_at_zero_and_covers = (
        abs(float(source_time[0])) <= max(tolerance, 0.25 * median_dt)
        and float(source_time[-1]) >= duration_s - 1.5 * median_dt
    )
    if starts_at_zero_and_covers:
        return source_time.copy(), "input time already used a zero-based window"
    aligned = source_time - source_time[0] + target_time[0]
    return aligned, "first sample aligned to the frozen window origin"


def _antiderivative_at(time_s: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Exact integral of the piecewise-linear interpolant at query positions."""

    widths = np.diff(time_s)
    areas = 0.5 * (values[:-1] + values[1:]) * widths
    cumulative = np.concatenate(([0.0], np.cumsum(areas)))
    indices = np.searchsorted(time_s, query, side="right") - 1
    indices = np.clip(indices, 0, len(time_s) - 2)
    dx = query - time_s[indices]
    slopes = (values[indices + 1] - values[indices]) / widths[indices]
    return cumulative[indices] + values[indices] * dx + 0.5 * slopes * np.square(dx)


def _extend_to_window(
    time_s: np.ndarray,
    values: np.ndarray,
    duration_s: float,
    typical_dt: float,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    tolerance = max(typical_dt * 1.51, 1e-12)
    if time_s[0] > tolerance:
        raise InputContractError(f"{label}: the record does not cover the start of the required window")
    if time_s[-1] < duration_s - tolerance:
        raise InputContractError(
            f"{label}: record ends at {time_s[-1]:.9g} s but {duration_s:.9g} s is required"
        )
    extended_time = time_s
    extended_values = values
    if time_s[0] > 0.0:
        extended_time = np.concatenate(([0.0], extended_time))
        extended_values = np.concatenate(([extended_values[0]], extended_values))
    if extended_time[-1] < duration_s:
        extended_time = np.concatenate((extended_time, [duration_s]))
        extended_values = np.concatenate((extended_values, [extended_values[-1]]))
    return extended_time, extended_values


def _normalize_ecl(values: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    divisor = float(np.max(positive))
    if not math.isfinite(divisor) or divisor <= 0.0:
        raise InputContractError("ECL has no finite positive signal")
    return (positive / divisor).astype(np.float32)


def _normalize_signed_current_without_retaining_divisor(values: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    divisor = float(np.max(np.abs(numeric)))
    if not math.isfinite(divisor) or divisor <= 0.0:
        raise InputContractError("Current has no finite nonzero signal")
    normalized = (numeric / divisor).astype(np.float32)
    # The divisor is intentionally discarded here and never returned.
    del divisor
    return normalized


def _signal_series(signal: np.ndarray, time_s: np.ndarray, tau_s: float) -> np.ndarray:
    result = np.empty((3, signal.size), dtype=np.float32)
    result[0] = signal
    result[1:] = time_features(time_s, tau_s)
    return result


def prepare_ecl(trace: ParsedTrace, tau_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[QCItem]]:
    issues: list[QCItem] = []
    edges, target_time = ecl_edges_and_centers(tau_s)
    duration_s = 0.5 * tau_s
    aligned_time, origin_message = _align_time_origin(trace.time_s, target_time, duration_s)
    issues.append(QCItem("info", "ECL_TIME_ORIGIN", origin_message))
    negative_fraction = float(np.mean(trace.signal < 0.0))
    if negative_fraction > 0.0:
        issues.append(
            QCItem(
                "warning",
                "ECL_NEGATIVE_VALUES_CLAMPED",
                f"{negative_fraction:.1%} of ECL values were negative and were clamped to zero by the frozen contract.",
            )
        )
    positive = np.maximum(trace.signal, 0.0)
    source_dt = np.diff(aligned_time)
    median_dt = float(np.median(source_dt))
    max_gap = float(np.max(source_dt))
    target_dt = ECL_BIN_WIDTH_S[tau_s]
    if max_gap > 10.0 * target_dt:
        issues.append(QCItem("warning", "ECL_COARSE_GAPS", f"Maximum ECL gap is {max_gap / target_dt:.1f} frozen bin widths."))
    elif max_gap > 5.0 * target_dt:
        issues.append(QCItem("warning", "ECL_RESOLUTION_MARGINAL", f"Maximum ECL gap is {max_gap / target_dt:.1f} frozen bin widths."))
    if aligned_time[-1] > duration_s + max(median_dt, target_dt):
        issues.append(QCItem("info", "ECL_LONG_RECORD_CROPPED", "Only the positive-step half-window was used."))

    exact = (
        aligned_time.shape == target_time.shape
        and np.allclose(aligned_time, target_time, rtol=0.0, atol=max(1e-12, target_dt * 1e-5))
    )
    if exact:
        resampled = positive
        method = "exact frozen bin centers; no resampling"
    else:
        extended_time, extended_signal = _extend_to_window(
            aligned_time,
            positive,
            duration_s,
            median_dt,
            "ECL",
        )
        integral = _antiderivative_at(extended_time, extended_signal, edges)
        resampled = np.diff(integral) / np.diff(edges)
        method = "piecewise-linear integration to frozen bin averages"
    normalized = _normalize_ecl(resampled)
    issues.append(QCItem("pass", "ECL_PREPROCESSING", f"{method}; 12,208 bins; own-positive-maximum normalization."))

    peak_index = int(np.argmax(positive))
    peak = float(positive[peak_index])
    rise_points = 0
    if peak > 0.0:
        threshold_indices = np.flatnonzero(positive[: peak_index + 1] >= 0.1 * peak)
        if threshold_indices.size:
            rise_points = peak_index - int(threshold_indices[0]) + 1
    if rise_points < 10:
        issues.append(QCItem("warning", "ECL_RISE_UNDERSAMPLED", f"Only {rise_points} uploaded points span the 10%-to-peak rise region; at least 10 is preferred."))
    else:
        issues.append(QCItem("pass", "ECL_RISE_RESOLUTION", f"{rise_points} uploaded points span the 10%-to-peak rise region."))
    if peak_index >= max(1, int(0.98 * len(positive))):
        issues.append(QCItem("warning", "ECL_PEAK_NEAR_WINDOW_END", "The ECL maximum is near the uploaded window end; clipping is possible."))
    return target_time, _signal_series(normalized, target_time, tau_s), aligned_time, issues


def _robust_signed_extreme(values: np.ndarray) -> float:
    count = max(3, min(32, values.size // 4))
    indices = np.argpartition(np.abs(values), -count)[-count:]
    selected = values[indices]
    return float(np.median(selected))


def _orient_current_for_model(values: np.ndarray, tau_s: float) -> np.ndarray:
    """Choose the orientation that has a negative first-step transient."""

    half_index = int(round(0.5 * tau_s / CURRENT_DT_S))
    if half_index <= 0 or half_index >= values.size:
        raise InputContractError("Current: could not locate the potential switch")
    window = max(32, min(half_index // 4, int(round(0.02 / CURRENT_DT_S))))
    first = _robust_signed_extreme(values[:window])
    second = _robust_signed_extreme(values[half_index : half_index + window])
    scale = float(np.max(np.abs(values)))
    orientation_score = second - first
    if not math.isfinite(orientation_score) or abs(orientation_score) < 0.02 * scale:
        raise InputContractError("Current: automatic orientation was inconclusive")
    return values if orientation_score > 0.0 else -values


def prepare_current(
    trace: ParsedTrace,
    tau_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, list[QCItem]]:
    issues: list[QCItem] = []
    target_time = current_grid(tau_s)
    aligned_time, origin_message = _align_time_origin(trace.time_s, target_time, tau_s)
    issues.append(QCItem("info", "CURRENT_TIME_ORIGIN", origin_message))
    median_dt = float(np.median(np.diff(aligned_time)))
    max_gap = float(np.max(np.diff(aligned_time)))
    if max_gap > 5.0 * CURRENT_DT_S:
        issues.append(QCItem("warning", "CURRENT_RESOLUTION_COARSE", f"Maximum current gap is {max_gap / CURRENT_DT_S:.1f} frozen intervals."))
    if aligned_time[-1] > tau_s + median_dt:
        issues.append(QCItem("info", "CURRENT_LONG_RECORD_CROPPED", "Only the full two-step tau window was used."))

    extended_time, extended_signal = _extend_to_window(
        aligned_time,
        trace.signal,
        tau_s,
        median_dt,
        "Current",
    )
    exact = (
        aligned_time.shape == target_time.shape
        and np.allclose(aligned_time, target_time, rtol=0.0, atol=max(1e-12, CURRENT_DT_S * 1e-5))
    )
    if exact:
        resampled = trace.signal.copy()
        method = "exact 0.1 ms grid; no resampling"
    else:
        resampled = np.interp(target_time, extended_time, extended_signal)
        method = "linear interpolation to the exact 0.1 ms grid"

    signed = _orient_current_for_model(resampled, tau_s)
    changed = not np.shares_memory(signed, resampled) and np.allclose(signed, -resampled)
    raw_signed = -trace.signal if changed else trace.signal.copy()
    sign_action = "automatic"
    normalized = _normalize_signed_current_without_retaining_divisor(signed)
    issues.append(
        QCItem(
            "pass",
            "CURRENT_PREPROCESSING",
            f"{method}; own-maximum-absolute normalization; normalization magnitude discarded.",
        )
    )
    return (
        target_time,
        _signal_series(normalized, target_time, tau_s),
        aligned_time,
        raw_signed,
        sign_action,
        issues,
    )


def prepare_inputs(
    ecl_upload: Any,
    current_upload: Any,
    cm0_mol_per_m3: float,
    tau_s: float,
) -> PreparedInputs:
    qc: list[QCItem] = []
    tau_s = float(tau_s)
    if tau_s not in SUPPORTED_TAU_S:
        raise InputContractError("tau must be exactly 0.5, 1.0, or 2.0 s")
    cm0 = float(cm0_mol_per_m3)
    if not math.isfinite(cm0) or cm0 <= 0.0:
        raise InputContractError("CM0 must be finite and positive")
    if not CM0_RANGE_MOL_PER_M3[0] <= cm0 <= CM0_RANGE_MOL_PER_M3[1]:
        qc.append(QCItem("warning", "CM0_OUTSIDE_TRAINING_RANGE", "CM0 is outside the simulated 0.05-2 mol/m³ range; this is extrapolation."))
    else:
        qc.append(QCItem("pass", "CM0_IN_RANGE", "CM0 is inside the simulated 0.05-2 mol/m³ range (numerically identical to mM)."))
    qc.append(QCItem("pass", "TAU_SUPPORTED", f"tau={tau_s:g} s is a frozen training value."))

    ecl_trace = read_two_column_trace(ecl_upload, "ECL")
    ecl_time, ecl_series, ecl_raw_time, ecl_qc = prepare_ecl(ecl_trace, tau_s)
    qc.extend(ecl_qc)

    current_time: np.ndarray | None = None
    current_series: np.ndarray | None = None
    current_raw_time: np.ndarray | None = None
    current_raw_signal: np.ndarray | None = None
    current_name: str | None = None
    current_hash: str | None = None
    sign_action = "not applicable"
    if current_upload is None:
        qc.append(QCItem("info", "CURRENT_NOT_PROVIDED", "M3 requires a current record."))
    else:
        current_trace = read_current_trace(current_upload, tau_s)
        (
            current_time,
            current_series,
            current_raw_time,
            current_raw_signal,
            sign_action,
            current_qc,
        ) = prepare_current(current_trace, tau_s)
        qc.extend(current_qc)
        current_name = current_trace.source_name
        current_hash = current_trace.source_sha256

    return PreparedInputs(
        cm0_mol_per_m3=cm0,
        tau_s=tau_s,
        context_z_raw=np.asarray((np.log10(cm0), np.log10(tau_s)), dtype=np.float32),
        ecl_time_s=ecl_time,
        ecl_series=ecl_series,
        ecl_raw_time_s=ecl_raw_time,
        ecl_raw_signal=ecl_trace.signal,
        current_time_s=current_time,
        current_series=current_series,
        current_raw_time_s=current_raw_time,
        current_raw_signal=current_raw_signal,
        current_sign_action=sign_action,
        ecl_source_name=ecl_trace.source_name,
        ecl_source_sha256=ecl_trace.source_sha256,
        current_source_name=current_name,
        current_source_sha256=current_hash,
        qc=tuple(qc),
    )
