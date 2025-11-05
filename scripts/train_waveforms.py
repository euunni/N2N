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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from joblib import dump

from n2n.tcn import Noise2Noise1DTCN
from n2n.model_functions import train, validate
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
    p.add_argument("--min_sigma", type=float, default=5.0, help="Soft-min noise std in raw units; if >0, use sigma = sqrt((noise_level*s)^2 + min_sigma^2) with s=1.4826*MAD per trace")
    p.add_argument("--scaler", choices=["RobustScaler","StandardScaler","MinMaxScaler"], default="RobustScaler")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--events_per_file", type=int, default=2000, help="If input is 3D (E,C,L), randomly sample up to this many events per file before flattening; 0=all")
    # Runlist mode only: read split/run/tower from a text file and build inputs
    p.add_argument("--runlist", required=True, help="Path to run list txt: <split_flag> <run_number> <tower> per line; split_flag in {0=train,1=val,2=test}")
    p.add_argument("--npz_dir", default="/pscratch/sd/h/haeun/TB2025", help="Directory containing per-run npz files (used with --runlist)")
    p.add_argument("--npz_pattern", default="run_{run}_merged.npz", help="Filename pattern with {run} placeholder (used with --runlist)")
    p.add_argument("--plot_noise", action="store_true", help="Save original vs noisy plots for event 0 per channel (train split)")
    # Checkpoint/resume options
    p.add_argument("--checkpoint_dir", default=None, help="Directory to save/load checkpoints (default: <output_dir>/checkpoints)")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in --checkpoint_dir if available")
    p.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    p.add_argument("--total-epochs", dest="total_epochs", type=int, default=0, help="Stop when this many total epochs are reached across resumes; 0 = ignore")
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

    # Detect distributed context (torchrun exports LOCAL_RANK/RANK/WORLD_SIZE)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1 or ("LOCAL_RANK" in os.environ)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0

    if (not is_distributed) or rank == 0:
        print("Starting training...", flush=True)

    # Checkpoint directory
    ckpt_dir = args.checkpoint_dir or os.path.join(args.output_dir, "checkpoints")
    if (not is_distributed) or rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

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

    def _load_arrays_with_labels(pairs: list[tuple[str, str]], npz_key: str) -> tuple[list[np.ndarray], list[str]]:
        """Load each (path, channel) pair separately, returning arrays and labels for logging."""
        arrays: list[np.ndarray] = []
        labels: list[str] = []
        for p, ch in pairs:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Input file not found: {p}")
            arr = load_waveforms(p, npz_key, events_limit=args.events_per_file, seed=args.seed, channels=ch)
            arrays.append(arr)
            labels.append(f"{os.path.basename(p)}:{ch}")
        return arrays, labels

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
    # Load per-pair arrays to enable per-channel noise logging
    train_arrays, train_labels = _load_arrays_with_labels(train_pairs, args.array_key)
    val_arrays,   val_labels   = _load_arrays_with_labels(val_pairs,   args.array_key)
    test_arrays,  test_labels  = _load_arrays_with_labels(test_pairs,  args.array_key)

    # Also keep concatenated originals for shape reporting (optional)
    X_train_orig = np.ascontiguousarray(np.concatenate(train_arrays, axis=0))
    X_val_orig   = np.ascontiguousarray(np.concatenate(val_arrays,   axis=0))
    X_test_orig  = np.ascontiguousarray(np.concatenate(test_arrays,  axis=0))

    n_points = X_train_orig.shape[1]
    print(
        f"Loaded shapes (runlist) - train:{X_train_orig.shape}, val:{X_val_orig.shape}, test:{X_test_orig.shape}",
        flush=True,
    )
    # Noise2Noise: two independent noisy realizations per pair (per-channel logging)
    def _gen_noisy_pairwise(arrays: list[np.ndarray], labels: list[str], seed_base: int) -> tuple[np.ndarray, np.ndarray]:
        noisy1_list: list[np.ndarray] = []
        noisy2_list: list[np.ndarray] = []
        for i, arr in enumerate(arrays):
            seed1 = seed_base + i * 2
            seed2 = seed_base + i * 2 + 1
            label = labels[i]
            n1 = generate_noisy_waveforms(
                arr,
                noise_level=args.noise_level,
                random_seed=seed1,
                min_sigma=args.min_sigma,
            )
            n2 = generate_noisy_waveforms(
                arr,
                noise_level=args.noise_level,
                random_seed=seed2,
                min_sigma=args.min_sigma,
            )
            noisy1_list.append(n1)
            noisy2_list.append(n2)
        return (
            np.ascontiguousarray(np.concatenate(noisy1_list, axis=0)),
            np.ascontiguousarray(np.concatenate(noisy2_list, axis=0)),
        )

    X_train_noisy1, X_train_noisy2 = _gen_noisy_pairwise(train_arrays, train_labels, seed_base=args.seed)
    X_val_noisy1,   X_val_noisy2   = _gen_noisy_pairwise(val_arrays,   val_labels,   seed_base=args.seed + 100000)
    X_test_noisy1,  X_test_noisy2  = _gen_noisy_pairwise(test_arrays,  test_labels,  seed_base=args.seed + 200000)
    # Order for concatenation matches train_arrays order
    train_labels_ordered = list(train_labels)

    # Residual Noise2Noise: learn residual r = noisy1 - noisy2, then ŷ = noisy1 - r
    X_train, y_train = X_train_noisy1, (X_train_noisy1 - X_train_noisy2)
    # Optional: Plot event 0 original vs noisy per channel (train split)
    if args.plot_noise:
        os.makedirs(args.output_dir, exist_ok=True)
        # Build offsets for each label according to concatenation order
        size_per_label = {lab: arr.shape[0] for lab, arr in zip(train_labels, train_arrays)}
        offsets = {}
        cur = 0
        for lab in train_labels_ordered:
            offsets[lab] = cur
            cur += size_per_label[lab]
        # Without group info, skip special-case y-limits per first channel or set none.
        first_labels_set = set()
        # Plot each channel's event 0
        for lab in train_labels_ordered:
            idx0 = offsets[lab]
            arr_idx = train_labels.index(lab)
            orig = train_arrays[arr_idx][0]
            noisy = X_train_noisy1[idx0]
            plt.figure(figsize=(8, 3))
            plt.plot(orig, label="original", lw=1.2)
            plt.plot(noisy, label="noisy1", lw=1.0, alpha=0.8)
            if lab in first_labels_set:
                plt.ylim(-20, 20)
            plt.title(f"Event 0 original vs noisy1\n{lab}")
            plt.xlabel("sample"); plt.ylabel("amplitude")
            plt.legend(); plt.tight_layout()
            out_path = os.path.join(args.output_dir, f"event0_{lab.replace(':','_')}_orig_vs_noisy1.png")
            plt.savefig(out_path)
            plt.close()

            # Also plot original vs noisy2
            noisy2 = X_train_noisy2[idx0]
            plt.figure(figsize=(8, 3))
            plt.plot(orig, label="original", lw=1.2)
            plt.plot(noisy2, label="noisy2", lw=1.0, alpha=0.8)
            if lab in first_labels_set:
                plt.ylim(-20, 20)
            plt.title(f"Event 0 original vs noisy2\n{lab}")
            plt.xlabel("sample"); plt.ylabel("amplitude")
            plt.legend(); plt.tight_layout()
            out_path2 = os.path.join(args.output_dir, f"event0_{lab.replace(':','_')}_orig_vs_noisy2.png")
            plt.savefig(out_path2)
            plt.close()
        print(f"Saved event 0 noise example plots to {args.output_dir}", flush=True)
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

    if is_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)
        test_sampler  = DistributedSampler(test_ds,  shuffle=False, drop_last=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, sampler=train_sampler)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, sampler=val_sampler)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, sampler=test_sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # Model
    # TCN with channel=1, causal (from tcn.py), reasonable defaults
    model = Noise2Noise1DTCN(in_channels=1, num_channels=None, kernel_size=5, dropout=0.1)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    if is_distributed:
        # One process per GPU
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, output_device=local_rank if torch.cuda.is_available() else None, find_unused_parameters=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Utility: checkpoint helpers
    def _latest_checkpoint_path(directory: str):
        try:
            candidates = [f for f in os.listdir(directory) if f.endswith(".pt") and f.startswith("epoch_")]
            if not candidates:
                return None
            candidates.sort()
            return os.path.join(directory, candidates[-1])
        except Exception:
            return None

    def _save_checkpoint(epoch_idx: int, done: bool = False):
        if (is_distributed and rank != 0):
            return
        to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        ckpt = {
            "epoch": epoch_idx,
            "model": to_save,
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
            "patience_left": patience,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "done": done,
        }
        path_epoch = os.path.join(ckpt_dir, f"epoch_{epoch_idx:04d}.pt")
        torch.save(ckpt, path_epoch)
        torch.save(ckpt, os.path.join(ckpt_dir, "latest.pt"))

    def _load_checkpoint_if_any():
        nonlocal start_epoch, best_val, patience, train_losses, val_losses
        latest = _latest_checkpoint_path(ckpt_dir) if args.resume else None
        if latest is None:
            return None
        if (not is_distributed) or rank == 0:
            print(f"Resuming from checkpoint: {latest}", flush=True)
        map_location = torch.device("cpu") if not torch.cuda.is_available() else None
        ckpt = torch.load(latest, map_location=map_location)
        state = ckpt.get("model", ckpt)
        # Load into model (all ranks load)
        if hasattr(model, "module"):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])  # optimizer states are safe to restore
        best_val = ckpt.get("best_val", float("inf"))
        patience = ckpt.get("patience_left", args.patience)
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        last_epoch = int(ckpt.get("epoch", 0))
        done_flag = bool(ckpt.get("done", False))
        start_epoch = last_epoch + 1
        return done_flag, last_epoch

    # Train loop with early stopping + resume
    best_val = float("inf")
    patience = args.patience
    train_losses, val_losses = [], []
    start_epoch = 1

    # Optional resume
    done_on_load = None
    resume_info = _load_checkpoint_if_any()
    if resume_info is not None:
        done_on_load, last_epoch_total = resume_info
        if args.total_epochs and last_epoch_total >= args.total_epochs:
            if (not is_distributed) or rank == 0:
                print(f"Total epochs already reached ({last_epoch_total} >= {args.total_epochs}). Exiting.", flush=True)
            if is_distributed:
                dist.destroy_process_group()
            return
        if done_on_load:
            if (not is_distributed) or rank == 0:
                print("Training already marked as done in latest checkpoint. Exiting.", flush=True)
            if is_distributed:
                dist.destroy_process_group()
            return

    # Determine end epoch for this invocation
    if args.total_epochs and args.total_epochs > 0:
        end_epoch = min(start_epoch + args.epochs - 1, args.total_epochs)
    else:
        end_epoch = start_epoch + args.epochs - 1

    for epoch in range(start_epoch, end_epoch + 1):
        if is_distributed:
            # Ensure all processes shuffle differently each epoch
            train_loader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]
        if (not is_distributed) or rank == 0:
            print(f"Epoch {epoch}/{args.epochs}", flush=True)
        tr = train(model, optimizer, criterion, train_loader, device)
        vl = validate(model, criterion, val_loader, device)

        # Average losses across ranks for consistent early stopping
        if is_distributed:
            t = torch.tensor([tr, vl], device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            tr = (t[0].item() / world_size)
            vl = (t[1].item() / world_size)
        train_losses.append(tr); val_losses.append(vl)

        stop_training = False
        if (not is_distributed) or rank == 0:
            if vl < best_val:
                best_val = vl
                patience = args.patience
                to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(to_save, os.path.join(args.output_dir, "waveform_n2n_weights.pt"))
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping.", flush=True)
                    stop_training = True
        if is_distributed:
            flag = torch.tensor([1 if stop_training else 0], device=device, dtype=torch.int)
            dist.broadcast(flag, src=0)
            if flag.item() == 1:
                # Save a final checkpoint marking completion, rank 0 only
                _save_checkpoint(epoch, done=True)
                break
        else:
            if stop_training:
                _save_checkpoint(epoch, done=True)
                break

        # Save curve
        if (not is_distributed) or rank == 0:
            plt.figure(figsize=(8, 3))
            plt.plot(train_losses, label="Train")
            plt.plot(val_losses, label="Val")
            plt.xlabel("Epoch"); plt.ylabel("MSE (scaled)"); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "training_curve.png"))
            plt.close()

        # Periodic checkpoint save
        if (not is_distributed) or rank == 0:
            if args.save_every > 0 and (epoch % args.save_every == 0):
                _save_checkpoint(epoch, done=False)

    # Test
    test_loss = validate(model, criterion, test_loader, device)
    if is_distributed:
        t = torch.tensor([test_loss], device=device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        test_loss = t[0].item() / world_size
    if (not is_distributed) or rank == 0:
        print(f"Test loss (scaled): {test_loss:.6g}")

    # Save scalers
    if (not is_distributed) or rank == 0:
        dump(X_scaler, os.path.join(args.output_dir, "waveform_feature_scaler.joblib"))
        dump(y_scaler, os.path.join(args.output_dir, "waveform_target_scaler.joblib"))
        print("Saved weights and scalers to:", args.output_dir, flush=True)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)
