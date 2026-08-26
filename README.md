# ECL transient analysis with the M3 convolutional neural network

This repository accompanies a chemistry study in which electrochemiluminescence (ECL) and current transients are used to estimate the annihilation rate constant, uncompensated resistance, and double-layer capacitance.

The repository contains the paper model, the preprocessing and prediction code, the noise-augmentation method used during model development, 18 paired experimental records, and the COMSOL model used for the dataset generation.

## What M3 reports

M3 uses four experimental quantities:

- an ECL intensity-versus-time transient;
- a current-versus-time transient from the same potential-step experiment;
- the luminophore concentration, in mM;
- the potential-step frequency, 0.5, 1, or 2 Hz.

It reports:

| Output | Meaning | Units shown by the program |
| --- | --- | --- |
| `ln(kANN)` | Natural logarithm of the annihilation rate constant in the model's SI convention | dimensionless numerical value |
| `kANN` | Bimolecular annihilation rate constant | m³ mol⁻¹ s⁻¹ and M⁻¹ s⁻¹ |
| `Ru` | Uncompensated resistance | Ω |
| `Cdl` | Double-layer capacitance per electrode area | F m⁻² and µF cm⁻² |

## Quick start

Install Python 3.10 or 3.11, open a terminal in this folder, and install the two required scientific packages:

```text
python -m pip install -r requirements.txt
```

To analyze one record:

```text
python inference/predict_experimental.py --ecl experimental_data/DPA/0p5Hz/replicate_1/ECL_time_counts.csv --current experimental_data/DPA/0p5Hz/replicate_1/current.txt --frequency-hz 0.5 --concentration-mm 0.1
```

To reproduce predictions for all 18 supplied records:

```text
python inference/predict_experimental.py --manifest experimental_data/manifest.csv --output M3_predictions.csv
```

The new `M3_predictions.csv` can be compared with `experimental_data/reference_predictions.csv`.

Paths containing spaces should be placed inside quotation marks.

## Experimental input files

Each supplied record contains:

| File | Contents |
| --- | --- |
| `ECL_time_counts.csv` | Two columns: elapsed time in seconds and ECL counts; this is the file used for prediction |
| `current.txt` | CH Instruments current export for the paired potential-step experiment |
| `ECL_raw.lvm` | Preserved LabVIEW/SR430 acquisition export |

`experimental_data/manifest.csv` gives the compound, frequency, concentration, source replicate, and relative file paths for every record. The collection contains nine DPA and nine rubrene records: three replicates at each of 0.5, 1, and 2 Hz.

For a new experiment, provide an ECL file with two numeric columns (`time_s,ECL`) and either a CH Instruments current export or an ordinary two-column file (`time_s,current_A`). The supported concentration range is 0.05–2 mM, and the supported frequencies are 0.5, 1, and 2 Hz. Inputs outside this experimental contract should not be treated as validated predictions.

## Why the model has two transient branches

ECL and current contain complementary chemical information. The ECL transient is most directly connected to excited-state generation and annihilation kinetics. The current transient is especially sensitive to electrical effects such as uncompensated resistance and double-layer charging. M3 analyzes the two signals separately and combines them only after each signal has been converted into a compact set of temporal features. Concentration and pulse period are then added before the final predictions are made.

A one-dimensional convolutional neural network is suitable here because the informative features are changes in transient shape: onset, rise, peak, decay, switching behavior, and their relationships across different time scales. The architecture was fixed for this study as a suitable temporal model; it was not chosen by examining the held-out test set.

| M3 component | Configuration | Chemical role |
| --- | --- | --- |
| ECL branch | Three convolution stages; 24/48/96 channels; kernels 15/9/5 | Captures local and multiscale ECL shape features |
| Current branch | Three convolution stages with a finer initial time stride | Retains rapid charging and switching features |
| Temporal summary | Eight-bin average and maximum summaries for each branch | Preserves broad shape and prominent transient features |
| Experimental context | Concentration and pulse period | Places each transient in its acquisition condition |
| Joint prediction | Fused ECL, current, and context features | Reports `ln(kANN)`, `Ru`, and `Cdl` together |

The M3 model architecture is provided in `model/M3_model.py`, and its trained parameters are stored in `model/M3_best.pt`.

## Noise augmentation

The simulated chemistry was solved first. Noise was then added to create acquisition-like training presentations without changing the underlying chemical parameters.

| Signal | Added variation |
| --- | --- |
| ECL | Poisson counting statistics, variable photon count level, optional background, gain variation, and slow gain change across the record |
| Current | White noise, correlated AR(1) noise, baseline offset, linear drift, and optional 14/16/18-bit digitization |

The implementation is in `augmentation/noise_augmentation.py`; the numerical ranges are listed in `augmentation/noise_settings.json`.

## Model evaluation quantities

For measured values \(y_i\), predictions \(\hat{y}_i\), and \(n\) records:

\[
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

\[
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

\[
R^2=1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}
\]

MAE gives the average absolute prediction error, RMSE gives greater weight to large errors, and \(R^2\) describes how much of the variation in the reference values is reproduced by the predictions.

The included simulated validation report contains 4,176 cases. Its overall values are:

| Quantity | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| `ln(kANN)` | 0.5596 | 0.9556 | 0.9630 |
| `Ru` (Ω) | 135.48 | 276.03 | 0.9914 |
| `Cdl` (F m⁻²) | 0.01544 | 0.02766 | 0.9609 |

These are simulation-domain validation values, not experimental ground truth.

## COMSOL model and supporting records

`comsol/ECL_CNN_training_model.mph` is the preserved COMSOL 5.3 model used in this work. The source model was copied without modification.

`model/M3_provenance.json` retains the frozen training identity and model contract. `model/M3_validation_report.json` contains the complete validation summary. The held-out test records are not included in this package.

## Scientific scope

M3 is an inverse model for the simulation and acquisition conditions represented during training. It does not replace inspection of the experimental transients, calibration of the acquisition system, or forward electrochemical validation. Predictions should be interpreted within the concentration, frequency, geometry, and chemical-mechanism domain represented by the accompanying COMSOL model.

## License and citation

The Python source code is available under the MIT License in `LICENSE`. The experimental data, trained M3 parameters, COMSOL model, and other non-code research materials are available under the Creative Commons Attribution 4.0 International License described in `DATA_LICENSE.md`.

If you use this repository, please cite it using the author and title information in `CITATION.cff`.
