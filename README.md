# M3 and M2 models for ECL transients

Use these trained CNNs to estimate annihilation kinetics from ECL transients.
Predictions run on a CPU; COMSOL is not required.

| Model | Inputs | Outputs |
| --- | --- | --- |
| M3 | ECL, paired current, concentration, cycle frequency | kann, Ru, Cdl |
| M2 | ECL, concentration, cycle frequency | kann |

## Install

Use Python 3.11. In the repository folder, run:

```text
python -m pip install -r requirements.txt
```

## Run

Run the examples below with the supplied data, or replace the paths with your
own CSV files.

M3:

```text
python predict.py --model M3 --ecl data/DPA/0p5Hz/replicate_1/ecl.csv --current data/DPA/0p5Hz/replicate_1/current.csv --frequency-hz 0.5 --concentration-mm 0.1
```

M2:

```text
python predict.py --model M2 --ecl data/DPA/0p5Hz/replicate_1/ecl.csv --frequency-hz 0.5 --concentration-mm 0.1
```

Add `--output my_prediction.csv` to save the output. Rate constants are shown
in m³ mol⁻¹ s⁻¹ and M⁻¹ s⁻¹; ln(kann) uses the former convention. Ru is in Ω,
and Cdl is in F m⁻² and µF cm⁻².

## Input format

Supply CSV files with one header and two numeric columns:

- ECL: `time_s,ECL_counts`, covering the positive second step.
- Current for M3: `time_s,current_A`, covering the complete two-step cycle.

Time must increase continuously. Current must be negative in the first
step and positive in the second. The code preserves the sign and normalizes
signals internally. Use 0.05–2 mM and a cycle frequency of 0.5, 1, or 2 Hz;
each step lasts `1 / (2 × frequency)` seconds.

## Files

- `model/`: trained M3/M2 weights and CNN definition.
- `predict.py` and `preprocessing.py`: prediction and input preparation.
- `comsol/`: the original COMSOL 5.3 model.
- `data/`: two CV curves and 18 paired ECL/current records 

## License

Copyright © 2026 Zhenzhong Cai and Zhifeng Ding.

- **Python code:** [MIT License](LICENSE).
- **Model weights, data, and COMSOL model:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
