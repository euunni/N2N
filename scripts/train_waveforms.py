#!/usr/bin/env python3
"""
Train Noise2Noise1D on 1D waveforms (e.g., per-event 1000-length vectors).

Input: a NumPy file (.npy or .npz) containing clean waveforms shaped (n_samples, n_points),
or a 3D array shaped (E, C, L) which will be flattened to (E*C, L) at load time.
The script generates noisy inputs, splits train/val/test, standardises features/targets,
trains the model, and saves weights and scalers.

Example:
  python scripts/train_waveforms.py \
    --input /path/to/waveforms.npy \
    --output_dir /path/to/output \
    --epochs 200 --batch_size 256 --lr 1e-3
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from joblib import dump

from n2n.tcn import Noise2Noise1DTCN
from n2n.model_functions import check_available_device, train, validate
from n2n.preprocessing import (
    standardise_array,
    generate_noisy_waveforms,
    split_train_val_test_indices,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Noise2Noise on 1D waveforms")
    p.add_argument("--array_key", default="waves_tower", help="npz key if using .npz (default: waveforms). Common: 'waveforms' or 'waves_tower'")
    p.add_argument("--output_dir", required=True, help="Directory to save weights/scaler/plots")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--noise_level", type=float, default=0.02)
    p.add_argument("--min_sigma", type=float, default=0.0, help="Soft-min noise std in raw units; if >0, use sigma = sqrt((noise_level*s)^2 + min_sigma^2) with s=1.4826*MAD per trace")
    p.add_argument("--scaler", choices=["RobustScaler","StandardScaler","MinMaxScaler"], default="RobustScaler")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--events_per_file", type=int, default=0, help="If input is 3D (E,C,L), randomly sample up to this many events per file before flattening; 0=all")
    # Runlist mode only: read split/run/tower from a text file and build inputs
    p.add_argument("--runlist", required=True, help="Path to run list txt: <split_flag> <run_number> <tower> per line; split_flag in {0=train,1=val,2=test}")
    p.add_argument("--npz_dir", default="/pscratch/sd/h/haeun/TB2025", help="Directory containing per-run npz files (used with --runlist)")
    p.add_argument("--npz_pattern", default="run_{run}_merged.npz", help="Filename pattern with {run} placeholder (used with --runlist)")
    return p.parse_args()

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def load_waveforms(path: str, key: str, events_limit: int = 0, seed: int = 42, channels: str = ""):
    if path.endswith(".npy"):
        arr = np.load(path)
        chan_names = None
    elif path.endswith(".npz"):
        data = np.load(path)
        if key not in data:
            raise KeyError(f"Key '{key}' not found in npz file. Available: {list(data.keys())}")
        arr = data[key]
        chan_names = data.get("channel_names", None)
    else:
        raise ValueError("Input must be .npy or .npz")
    # Accept 2D (n_samples, n_points) or 3D (E, C, L) -> flatten to (E*C, L)
    if arr.ndim == 3:
        # Optional channel filtering by names (npz path only)
        if channels:
            if chan_names is None:
                raise ValueError("Channel filtering requested but 'channel_names' not found in npz.")
            try:
                names = [str(x) for x in chan_names]
            except Exception:
                names = [str(x) for x in chan_names.tolist()]
            want = [s.strip() for s in channels.split(',') if s.strip()]
            idxs = []
            missing = []
            for w in want:
                if w in names:
                    idxs.append(names.index(w))
                else:
                    missing.append(w)
            if missing:
                raise ValueError(f"Channels not found in file: {missing}. Available example: {names[:8]} ...")
            arr = arr[:, idxs, :]
        e, c, l = arr.shape
        if events_limit and events_limit > 0 and e > events_limit:
            rng = np.random.default_rng(seed)
            idx = rng.choice(e, size=events_limit, replace=False)
            arr = arr[idx]
        # Recompute shape after potential subsampling
        e, c, l = arr.shape
        arr = arr.reshape(e * c, l)
    elif arr.ndim != 2:
        raise ValueError(f"Expected 2D (n_samples, n_points) or 3D (E, C, L), got shape {arr.shape}")
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print("Starting training...", flush=True)

    def _load_concat_pairs(pairs: list[tuple[str, str]], npz_key: str) -> np.ndarray:
        """Load multiple (path, channels) pairs and concatenate (rows)."""
        if not pairs:
            raise ValueError("Empty input pair list provided for a split.")
        arrays = []
        for p, ch in pairs:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Input file not found: {p}")
            arrays.append(load_waveforms(p, npz_key, events_limit=args.events_per_file, seed=args.seed, channels=ch))
        return np.ascontiguousarray(np.concatenate(arrays, axis=0))

    def _read_runlist(path: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
        if not args.npz_dir:
            raise ValueError("--npz_dir is required when using --runlist")
        train_pairs: list[tuple[str, str]] = []
        val_pairs: list[tuple[str, str]] = []
        test_pairs: list[tuple[str, str]] = []
        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                cols = line.split()
                if len(cols) < 3:
                    raise ValueError(f"Invalid runlist line: '{line}' (expected: flag run tower [tower ...])")
                flag = int(cols[0]); run = int(cols[1]); towers = cols[2:]
                file_path = os.path.join(args.npz_dir, args.npz_pattern.format(run=run))
                for tower in towers:
                    # Only S channel for each specified tower
                    ch = f"{tower}S"
                    pair = (file_path, ch)
                    if flag == 0:
                        train_pairs.append(pair)
                    elif flag == 1:
                        val_pairs.append(pair)
                    elif flag == 2:
                        test_pairs.append(pair)
                    else:
                        raise ValueError(f"Invalid split flag {flag} in runlist (use 0,1,2)")
        return train_pairs, val_pairs, test_pairs

    train_pairs, val_pairs, test_pairs = _read_runlist(args.runlist)
    X_train_orig = _load_concat_pairs(train_pairs, args.array_key)
    X_val_orig   = _load_concat_pairs(val_pairs,   args.array_key)
    X_test_orig  = _load_concat_pairs(test_pairs,  args.array_key)

    n_points = X_train_orig.shape[1]
    print(
        f"Loaded shapes (runlist) - train:{X_train_orig.shape}, val:{X_val_orig.shape}, test:{X_test_orig.shape}",
        flush=True,
    )
    # Noise2Noise: two independent noisy realizations (same clean, different seeds)
    X_train_noisy1 = generate_noisy_waveforms(X_train_orig, noise_level=args.noise_level, random_seed=args.seed, min_sigma=args.min_sigma)
    X_val_noisy1   = generate_noisy_waveforms(X_val_orig,   noise_level=args.noise_level, random_seed=args.seed, min_sigma=args.min_sigma)
    X_test_noisy1  = generate_noisy_waveforms(X_test_orig,  noise_level=args.noise_level, random_seed=args.seed, min_sigma=args.min_sigma)

    X_train_noisy2 = generate_noisy_waveforms(X_train_orig, noise_level=args.noise_level, random_seed=args.seed + 1, min_sigma=args.min_sigma)
    X_val_noisy2   = generate_noisy_waveforms(X_val_orig,   noise_level=args.noise_level, random_seed=args.seed + 2, min_sigma=args.min_sigma)
    X_test_noisy2  = generate_noisy_waveforms(X_test_orig,  noise_level=args.noise_level, random_seed=args.seed + 3, min_sigma=args.min_sigma)

    # Residual Noise2Noise: learn residual r = noisy1 - noisy2, then ŷ = noisy1 - r
    X_train, y_train = X_train_noisy1, (X_train_noisy1 - X_train_noisy2)
    X_val,   y_val   = X_val_noisy1,   (X_val_noisy1   - X_val_noisy2)
    X_test,  y_test  = X_test_noisy1,  (X_test_noisy1  - X_test_noisy2)

    # Standardise features and targets with separate scalers (robust by default)
    X_train_s, X_scaler = standardise_array(X_train, method=args.scaler)
    X_val_s, _ = standardise_array(X_val, method=args.scaler, scaler=X_scaler)
    X_test_s, _ = standardise_array(X_test, method=args.scaler, scaler=X_scaler)

    y_train_s, y_scaler = standardise_array(y_train, method=args.scaler)
    y_val_s, _   = standardise_array(y_val, method=args.scaler, scaler=y_scaler)
    y_test_s, _  = standardise_array(y_test, method=args.scaler, scaler=y_scaler)

    # Torch datasets
    train_ds = TensorDataset(torch.from_numpy(X_train_s).float(), torch.from_numpy(y_train_s).float())
    val_ds   = TensorDataset(torch.from_numpy(X_val_s).float(),   torch.from_numpy(y_val_s).float())
    test_ds  = TensorDataset(torch.from_numpy(X_test_s).float(),  torch.from_numpy(y_test_s).float())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # Model
    # TCN with channel=1, causal (from tcn.py), reasonable defaults
    model = Noise2Noise1DTCN(in_channels=1, num_channels=None, kernel_size=3, dropout=0.1)
    device = check_available_device()
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train loop with early stopping
    best_val = float("inf")
    patience = args.patience
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        tr = train(model, optimizer, criterion, train_loader, device)
        vl = validate(model, criterion, val_loader, device)
        train_losses.append(tr); val_losses.append(vl)

        if vl < best_val:
            best_val = vl
            patience = args.patience
            torch.save(model.state_dict(), os.path.join(args.output_dir, "waveform_n2n_weights.pt"))
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.", flush=True)
                break

        # Save curve
        plt.figure(figsize=(8, 3))
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Val")
        plt.xlabel("Epoch"); plt.ylabel("MSE (scaled)"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "training_curve.png"))
        plt.close()

    # Test
    test_loss = validate(model, criterion, test_loader, device)
    print(f"Test loss (scaled): {test_loss:.6g}")

    # Save scalers
    dump(X_scaler, os.path.join(args.output_dir, "waveform_feature_scaler.joblib"))
    dump(y_scaler, os.path.join(args.output_dir, "waveform_target_scaler.joblib"))
    print("Saved weights and scalers to:", args.output_dir, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)
