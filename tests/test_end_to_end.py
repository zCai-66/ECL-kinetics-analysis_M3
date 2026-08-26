from __future__ import annotations

import csv
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class EndToEndInferenceTests(unittest.TestCase):
    def test_all_18_predictions_match_frozen_reference(self) -> None:
        from inference.predict_experimental import load_m3, predict_manifest

        manifest = ROOT / "experimental_data" / "manifest.csv"
        reference_path = ROOT / "experimental_data" / "reference_predictions.csv"
        with reference_path.open(newline="", encoding="utf-8-sig") as handle:
            reference = {row["record_id"]: row for row in csv.DictReader(handle)}

        model, checkpoint = load_m3()
        actual = predict_manifest(model, checkpoint, manifest)
        self.assertEqual(len(actual), 18)
        self.assertEqual({row["record_id"] for row in actual}, set(reference))
        for row in actual:
            expected = reference[row["record_id"]]
            self.assertAlmostEqual(row["ln_kANN"], float(expected["ln_kANN"]), delta=5e-5)
            self.assertAlmostEqual(
                row["kANN_m3_mol_s"],
                float(expected["kANN_m3_mol_s"]),
                delta=1e-4 * float(expected["kANN_m3_mol_s"]),
            )
            self.assertAlmostEqual(
                row["Ru_ohm"],
                float(expected["Ru_ohm"]),
                delta=1e-4 * float(expected["Ru_ohm"]),
            )
            self.assertAlmostEqual(
                row["Cdl_F_per_m2"],
                float(expected["Cdl_F_per_m2"]),
                delta=1e-4 * float(expected["Cdl_F_per_m2"]),
            )


if __name__ == "__main__":
    unittest.main()
