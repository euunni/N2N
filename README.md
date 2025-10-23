## Noise2Noise — 1D Waveform Denoising

#### Introduction

This package now focuses on denoising 1D waveforms (e.g., per-event vectors of length 1000) with a Noise2Noise 1D model.

#### Usage
The package is designed for Python 3.10+ on Linux/Windows. GPU acceleration is used where available.
To create a new environment and install the package:

```bash
conda env create -f environment.yml
conda activate N2N
pip install -e .
```
### Train on 1D Waveforms
Input: NumPy file with shape `(n_samples, n_points)` containing clean waveforms. The script
generates noisy inputs automatically and standardises inputs/targets.

```bash
python scripts/train_waveforms.py \
  --input /path/to/waveforms.npy \
  --output_dir /path/to/out \
  --epochs 200 --batch_size 256 --lr 1e-3
```

Outputs in `--output_dir`:
- `waveform_n2n_weights.pt` — trained weights
- `waveform_feature_scaler.joblib` — input scaler
- `waveform_target_scaler.joblib` — target scaler (if used)
- `training_curve.png` — loss curves
