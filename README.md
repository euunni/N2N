## Noise2Noise — 1D Waveform Denoising (Quantum TCN)

#### Introduction

This package provides Noise2Noise denoising for 1D waveforms and now includes a Quantum Machine Learning (QML) variant. The QML model integrates a variational quantum circuit (VQC) as the final temporal block in a Temporal Convolutional Network (TCN), implemented with PennyLane. When a compatible GPU is available, the simulation uses `pennylane-lightning-gpu` automatically; otherwise it falls back to CPU.

Key module: `n2n.tcn.Noise2Noise1DTCN` uses a `QuantumTemporalBlock` as the last layer. You can adjust quantum hyperparameters (`n_qubits`, `vqc_depth`, `shots`) in the model constructor.

#### Installation
The environment targets Python 3.11 on Linux and includes Torch, PennyLane, and GPU-accelerated quantum simulators.

```bash
conda env create -f environment.yml
conda activate N2N_QML
pip install -e .
```

#### Training
Training uses a runlist that enumerates input `.npz` files and towers per split. The script automatically generates two independent noisy realizations per sample (Noise2Noise) and standardises both features and targets.

Example command:
```bash
python scripts/train_waveforms.py \
  --runlist scripts/runlist.txt \
  --npz_dir /path/to/npz \
  --output_dir /path/to/out \
  --epochs 200 --batch_size 200 --lr 1e-3
```

#### Outputs
Saved to `--output_dir`:
- `waveform_n2n_weights.pt` — best model weights (updated on validation improvement)
- `waveform_feature_scaler.joblib` — feature scaler
- `waveform_target_scaler.joblib` — target scaler
- `training_curve.png` — loss curves

#### Batch jobs (SLURM)
Batch scripts are provided under `batch/` for multi-node, multi-GPU training with resume.

- Scripts:
  - `batch/n2n_script.sl` — single SLURM job using `torchrun` (DDP)
  - `batch/submit_chain.sh` — submits a chain of jobs (1 epoch per job) with dependency-resume

- Submit a single job (uses defaults inside the script):
```bash
sbatch QML/denoising/N2N/TCN_QML/N2N/batch/n2n_script.sl
```

- Submit a chained run (N jobs × 1 epoch, auto-resume):
```bash
bash QML/denoising/N2N/TCN_QML/N2N/batch/submit_chain.sh <exp_name> <total_epochs>
# e.g.
bash QML/denoising/N2N/TCN_QML/N2N/batch/submit_chain.sh exp1 100
```
