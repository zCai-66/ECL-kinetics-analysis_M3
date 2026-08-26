"""Run the frozen paper M3 model on paired ECL and current transients."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from model.M3_model import FrozenFixedWindowCNN, build_from_checkpoint_config
from inference.preprocessing import PreparedInputs, prepare_inputs


CHECKPOINT_PATH = REPOSITORY_ROOT / "model" / "M3_best.pt"
EXPECTED_CHECKPOINT_SHA256 = (
    "5AC1FA8A7CB24968DCB4B070CB1FAC3A883A09A6D2A643CF34B560BD892DCE61"
)
TARGETS = ("lnkann", "log10_Ru_ohm", "log10_Cdl_F_per_m2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_m3(checkpoint_path: Path = CHECKPOINT_PATH) -> tuple[FrozenFixedWindowCNN, Mapping[str, Any]]:
    checkpoint_path = checkpoint_path.resolve()
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"M3 checkpoint hash mismatch: {actual_hash}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = dict(checkpoint["model_config"])
    found = (
        str(checkpoint.get("base_architecture_model", config["model_id"])),
        str(config["mode"]),
        str(config["variant"]),
        tuple(str(value) for value in config["target_names"]),
        str(config["scalar_kind"]),
        int(config["scalar_features"]),
    )
    expected = ("M4", "dual", "noisy", TARGETS, "context", 2)
    if found != expected:
        raise RuntimeError(f"M3 checkpoint contract changed: expected {expected}, found {found}")
    model = build_from_checkpoint_config(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def predict_prepared(
    model: FrozenFixedWindowCNN,
    checkpoint: Mapping[str, Any],
    prepared: PreparedInputs,
) -> dict[str, float]:
    if prepared.current_series is None:
        raise ValueError("M3 requires both ECL and current")
    statistics = checkpoint["training_statistics"]
    context_mean = np.asarray(statistics["context"]["mean"], dtype=np.float32)
    context_scale = np.asarray(statistics["context"]["scale"], dtype=np.float32)
    scalar = torch.from_numpy(
        ((prepared.context_z_raw - context_mean) / context_scale)[None, :]
    )
    ecl = torch.from_numpy(prepared.ecl_series[None, :, :])
    current = torch.from_numpy(prepared.current_series[None, :, :])
    ecl_mask = torch.ones((1, 1, ecl.shape[2]), dtype=torch.bool)
    current_mask = torch.ones((1, 1, current.shape[2]), dtype=torch.bool)
    with torch.inference_mode():
        output_z = model(scalar, ecl, ecl_mask, current, current_mask)
    transformed = (
        output_z[0].cpu().numpy() * model.target_scale.cpu().numpy()
        + model.target_mean.cpu().numpy()
    )
    values = dict(zip(TARGETS, (float(value) for value in transformed)))
    kann_si = math.exp(values["lnkann"])
    cdl_si = 10.0 ** values["log10_Cdl_F_per_m2"]
    return {
        "ln_kANN": values["lnkann"],
        "kANN_m3_mol_s": kann_si,
        "kANN_M_inverse_s": 1000.0 * kann_si,
        "Ru_ohm": 10.0 ** values["log10_Ru_ohm"],
        "Cdl_F_per_m2": cdl_si,
        "Cdl_uF_per_cm2": 100.0 * cdl_si,
    }


def predict_record(
    model: FrozenFixedWindowCNN,
    checkpoint: Mapping[str, Any],
    *,
    ecl_path: Path,
    current_path: Path,
    tau_s: float,
    concentration_mm: float,
) -> tuple[dict[str, float], PreparedInputs]:
    prepared = prepare_inputs(
        ecl_path,
        current_path,
        cm0_mol_per_m3=float(concentration_mm),
        tau_s=float(tau_s),
    )
    return predict_prepared(model, checkpoint, prepared), prepared


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (manifest_path.parent / candidate)


def _check_manifest_hash(path: Path, expected: str, label: str) -> None:
    if expected and sha256_file(path) != expected.strip().upper():
        raise RuntimeError(f"{label} hash mismatch for {path}")


def predict_manifest(
    model: FrozenFixedWindowCNN,
    checkpoint: Mapping[str, Any],
    manifest_path: Path,
) -> list[dict[str, Any]]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError("The experimental-data manifest is empty")
    results: list[dict[str, Any]] = []
    for record in records:
        ecl_path = _resolve_manifest_path(manifest_path, record["ecl_file"])
        current_path = _resolve_manifest_path(manifest_path, record["current_file"])
        _check_manifest_hash(ecl_path, record.get("ecl_sha256", ""), "ECL")
        _check_manifest_hash(current_path, record.get("current_sha256", ""), "Current")
        prediction, prepared = predict_record(
            model,
            checkpoint,
            ecl_path=ecl_path,
            current_path=current_path,
            tau_s=float(record["tau_s"]),
            concentration_mm=float(record["concentration_mM"]),
        )
        results.append(
            {
                "record_id": record["record_id"],
                "compound": record.get("compound", ""),
                "frequency_Hz": record.get("frequency_Hz", ""),
                **prediction,
                "ecl_sha256": prepared.ecl_source_sha256,
                "current_sha256": prepared.current_source_sha256,
            }
        )
    return results


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError("No prediction rows were generated")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def print_prediction(values: Mapping[str, float]) -> None:
    print("M3 prediction")
    print(f"  ln(kANN):            {values['ln_kANN']:.6g}")
    print(f"  kANN:                {values['kANN_m3_mol_s']:.6g} m^3 mol^-1 s^-1")
    print(f"                       {values['kANN_M_inverse_s']:.6g} M^-1 s^-1")
    print(f"  Ru:                  {values['Ru_ohm']:.6g} ohm")
    print(f"  Cdl:                 {values['Cdl_F_per_m2']:.6g} F m^-2")
    print(f"                       {values['Cdl_uF_per_cm2']:.6g} uF cm^-2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict kANN, Ru, and Cdl from paired ECL and current transients."
    )
    parser.add_argument("--manifest", type=Path, help="Run every row in an experimental-data manifest")
    parser.add_argument("--ecl", type=Path, help="Two-column ECL time/count CSV")
    parser.add_argument("--current", type=Path, help="CHI current export or two-column current file")
    parser.add_argument("--frequency-hz", type=float, choices=(0.5, 1.0, 2.0))
    parser.add_argument("--concentration-mm", type=float, default=0.1)
    parser.add_argument("--output", type=Path, help="Optional CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model, checkpoint = load_m3()
    if args.manifest is not None:
        if args.ecl is not None or args.current is not None or args.frequency_hz is not None:
            raise SystemExit("Use --manifest by itself, or use --ecl/--current/--frequency-hz")
        rows = predict_manifest(model, checkpoint, args.manifest.resolve())
        output = args.output or Path("M3_predictions.csv")
        write_csv(output, rows)
        print(f"Predicted {len(rows)} records; results written to {output.resolve()}")
        return
    if args.ecl is None or args.current is None or args.frequency_hz is None:
        raise SystemExit("Single-record inference requires --ecl, --current, and --frequency-hz")
    prediction, prepared = predict_record(
        model,
        checkpoint,
        ecl_path=args.ecl.resolve(),
        current_path=args.current.resolve(),
        tau_s=1.0 / args.frequency_hz,
        concentration_mm=args.concentration_mm,
    )
    print_prediction(prediction)
    if args.output is not None:
        write_csv(
            args.output,
            [
                {
                    "record_id": args.ecl.stem,
                    "frequency_Hz": args.frequency_hz,
                    **prediction,
                    "ecl_sha256": prepared.ecl_source_sha256,
                    "current_sha256": prepared.current_source_sha256,
                }
            ],
        )


if __name__ == "__main__":
    main()
