from __future__ import annotations

import csv
import hashlib
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from augmentation.noise_augmentation import augment_current, augment_ecl
from inference.preprocessing import ParsedTrace, prepare_current, prepare_inputs, read_current_trace


M3_SHA256 = "5AC1FA8A7CB24968DCB4B070CB1FAC3A883A09A6D2A643CF34B560BD892DCE61"
COMSOL_SHA256 = "A329466F975A259FD8104A533AE3656607B62C87F4FDE5B0427EB0D6496E5D56"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def manifest_rows() -> list[dict[str, str]]:
    path = ROOT / "experimental_data" / "manifest.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class RepositoryTests(unittest.TestCase):
    def test_frozen_binary_hashes(self) -> None:
        self.assertEqual(sha256_file(ROOT / "model" / "M3_best.pt"), M3_SHA256)
        self.assertEqual(
            sha256_file(ROOT / "comsol" / "ECL_CNN_training_model.mph"),
            COMSOL_SHA256,
        )

    def test_experimental_manifest_and_hashes(self) -> None:
        rows = manifest_rows()
        self.assertEqual(len(rows), 18)
        self.assertEqual(Counter(row["compound"] for row in rows), {"DPA": 9, "Rubrene": 9})
        self.assertEqual(
            Counter((row["compound"], row["frequency_Hz"]) for row in rows),
            {
                ("DPA", "0.5"): 3,
                ("DPA", "1"): 3,
                ("DPA", "2"): 3,
                ("Rubrene", "0.5"): 3,
                ("Rubrene", "1"): 3,
                ("Rubrene", "2"): 3,
            },
        )
        for row in rows:
            for file_field, hash_field in (
                ("ecl_file", "ecl_sha256"),
                ("current_file", "current_sha256"),
                ("raw_ecl_file", "raw_ecl_sha256"),
            ):
                path = ROOT / "experimental_data" / row[file_field]
                self.assertTrue(path.is_file(), path)
                self.assertEqual(sha256_file(path), row[hash_field])

    def test_all_records_meet_m3_input_shape(self) -> None:
        for row in manifest_rows():
            tau_s = float(row["tau_s"])
            prepared = prepare_inputs(
                ROOT / "experimental_data" / row["ecl_file"],
                ROOT / "experimental_data" / row["current_file"],
                cm0_mol_per_m3=float(row["concentration_mM"]),
                tau_s=tau_s,
            )
            self.assertEqual(prepared.ecl_series.shape, (3, 12_208))
            self.assertEqual(
                prepared.current_series.shape,
                (3, int(round(tau_s / 0.0001)) + 1),
            )

    def test_current_orientation_is_input_invariant(self) -> None:
        row = manifest_rows()[0]
        tau_s = float(row["tau_s"])
        trace = read_current_trace(
            ROOT / "experimental_data" / row["current_file"], tau_s
        )
        _, first, _, _, _, _ = prepare_current(trace, tau_s)
        reversed_trace = ParsedTrace(
            source_name=trace.source_name,
            source_sha256=trace.source_sha256,
            time_s=trace.time_s,
            signal=-trace.signal,
        )
        _, second, _, _, _, _ = prepare_current(reversed_trace, tau_s)
        np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-7)

    def test_noise_augmentation_is_deterministic(self) -> None:
        ecl = np.exp(-np.linspace(0.0, 8.0, 1000))
        counts_a, expected_a, settings_a = augment_ecl(
            ecl, 40.96e-6, nuisance_seed=123, count_seed=456
        )
        counts_b, expected_b, settings_b = augment_ecl(
            ecl, 40.96e-6, nuisance_seed=123, count_seed=456
        )
        np.testing.assert_array_equal(counts_a, counts_b)
        np.testing.assert_allclose(expected_a, expected_b, rtol=0.0, atol=0.0)
        self.assertEqual(settings_a, settings_b)

        current = np.concatenate((-np.exp(-np.linspace(0.0, 8.0, 500)), np.exp(-np.linspace(0.0, 8.0, 500))))
        noisy_a, current_settings_a = augment_current(current, nuisance_seed=789)
        noisy_b, current_settings_b = augment_current(current, nuisance_seed=789)
        np.testing.assert_allclose(noisy_a, noisy_b, rtol=0.0, atol=0.0)
        self.assertEqual(current_settings_a, current_settings_b)


if __name__ == "__main__":
    unittest.main()
