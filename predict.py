"""Use the shared M3 or M2 model with your CSV files."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from model.architecture import build_from_checkpoint_config
from preprocessing import Inputs, prepare_inputs

ROOT = Path(__file__).resolve().parent


def load_model(name: str):
    checkpoint = torch.load(ROOT / "model" / f"{name}.pt", map_location="cpu", weights_only=True)
    config = checkpoint["model_config"]
    expected_mode = "dual" if name == "M3" else "ecl"
    expected_targets = ("lnkann", "log10_Ru_ohm", "log10_Cdl_F_per_m2") if name == "M3" else ("lnkann",)
    if config["mode"] != expected_mode or tuple(config["target_names"]) != expected_targets:
        raise ValueError(f"The checkpoint does not contain the expected {name} model")
    network = build_from_checkpoint_config(config)
    network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    network.eval()
    return network, checkpoint


def predict(network, checkpoint, inputs: Inputs) -> dict[str, float]:
    statistics = checkpoint["training_statistics"]["context"]
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    scale = np.asarray(statistics["scale"], dtype=np.float32)
    scalar = torch.from_numpy(((inputs.context - mean) / scale)[None, :])
    ecl = torch.from_numpy(inputs.ecl[None, :, :])
    ecl_mask = torch.ones((1, 1, ecl.shape[2]), dtype=torch.bool)
    with torch.inference_mode():
        if network.mode == "dual":
            if inputs.current is None:
                raise ValueError("M3 needs an ECL file and its paired current file")
            current = torch.from_numpy(inputs.current[None, :, :])
            current_mask = torch.ones((1, 1, current.shape[2]), dtype=torch.bool)
            output = network(scalar, ecl, ecl_mask, current, current_mask)
        else:
            if inputs.current is not None:
                raise ValueError("M2 uses only ECL; omit the current file")
            output = network(scalar, ecl, ecl_mask)
    transformed = output[0].cpu().numpy() * network.target_scale.cpu().numpy() + network.target_mean.cpu().numpy()
    values = dict(zip(network.target_names, map(float, transformed)))
    rate = math.exp(values["lnkann"])
    result = {"ln_kann": values["lnkann"], "kann_m3_mol_s": rate, "kann_M_inverse_s": 1000 * rate}
    if network.mode == "dual":
        capacitance = 10 ** values["log10_Cdl_F_per_m2"]
        result.update(Ru_ohm=10 ** values["log10_Ru_ohm"], Cdl_F_per_m2=capacitance,
                      Cdl_uF_per_cm2=100 * capacitance)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("M3", "M2"), default="M3", help="M3: ECL + current; M2: ECL only")
    parser.add_argument("--ecl", type=Path, required=True, help="CSV with time_s,ECL_counts")
    parser.add_argument("--current", type=Path, help="CSV with time_s,current_A; required for M3")
    parser.add_argument("--frequency-hz", type=float, choices=(0.5, 1.0, 2.0), required=True)
    parser.add_argument("--concentration-mm", type=float, required=True, help="Luminophore concentration in mM")
    parser.add_argument("--output", type=Path, help="Optionally save this prediction to a CSV")
    args = parser.parse_args()
    if args.model == "M3" and args.current is None:
        parser.error("M3 requires --current")
    if args.model == "M2" and args.current is not None:
        parser.error("M2 uses ECL only; omit --current")
    torch.set_num_threads(4)
    network, checkpoint = load_model(args.model)
    inputs = prepare_inputs(args.ecl, args.current, args.concentration_mm, args.frequency_hz)
    result = predict(network, checkpoint, inputs)
    print(f"{args.model} prediction")
    print(f"  ln(kann): {result['ln_kann']:.6g}")
    print(f"  kann:     {result['kann_m3_mol_s']:.6g} m^3 mol^-1 s^-1")
    print(f"            {result['kann_M_inverse_s']:.6g} M^-1 s^-1")
    if args.model == "M3":
        print(f"  Ru:       {result['Ru_ohm']:.6g} ohm")
        print(f"  Cdl:      {result['Cdl_F_per_m2']:.6g} F m^-2")
        print(f"            {result['Cdl_uF_per_cm2']:.6g} uF cm^-2")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result))
            writer.writeheader()
            writer.writerow(result)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
