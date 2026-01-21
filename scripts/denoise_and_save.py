#!/usr/bin/env python3
"""
Denoise full waveform NPZ files with a trained Noise2Noise1D TCN and save
results in the original NPZ structure. You can choose which channel groups to
include (default: towers + mcp_c + mcp_s). Unknown/other channels can be
optionally included as well.

Features:
- Loads a trained model + scalers from an output directory
- Reads input .npz with key (default: waves_tower) shaped (E, C, L)
- Optionally filters channels based on a runlist (towers -> channels)
- Streams events to limit memory, configurable chunk/shard sizes
- Saves NPZ(s) with the same metadata keys as the original files

Example:
  python n2n/scripts/denoise_and_save.py \
    --input_dir /pscratch/sd/h/haeun/TB2025 \
    --model_dir /global/homes/h/haeun/QML/denoising/N2N/TCN/n2n/output/251024_val \
    --output_dir /pscratch/sd/h/haeun/TB2025_denoised \
    --runlist /global/homes/h/haeun/QML/denoising/N2N/TCN/n2n/scripts/runlist.txt \
    --chunk_events 20000

Notes:
- Runlist should contain one run number per line (comments with '#' allowed).
- Use --channel_groups to select among: towers, mcp_c, mcp_s, dwc, aux, others, all
- If no runlist is provided, this script will error; provide the runs to process.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from joblib import load

from n2n import Noise2Noise1DTCN, denoise_waveforms
from n2n.model_functions import check_available_device
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # Fallback: disable progress bar if tqdm not available


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Denoise 1D waveforms and save NPZ in original structure")
    p.add_argument("--input_dir", default="/pscratch/sd/h/haeun/TB2025/1024bins", help="Directory containing per-run npz files")
    p.add_argument("--npz_pattern", default="run_{run}.npz", help="Filename pattern with {run} placeholder")
    p.add_argument("--array_key", default="waves_tower", help="Array key in NPZ (default: waves_tower)")
    p.add_argument("--model_dir", required=True, help="Directory containing waveform_n2n_weights.pt and scalers")
    p.add_argument("--output_dir", default="/pscratch/sd/h/haeun/TB2025_denoised/1024bins", help="Directory to write denoised NPZ(s)")
    p.add_argument("--runlist", default="runlist_denoised_test.txt", help="Runlist path (one run number per non-comment line)")
    p.add_argument("--channel_groups", default="towers", help="Comma-separated groups among {towers,mcp_c,mcp_s,dwc,aux,others,all}")
    p.add_argument("--batch_size", type=int, default=1000, help="Batch size for inference")
    p.add_argument("--chunk_events", type=int, default=0, help="Stream events in chunks of this size (0=load all)")
    p.add_argument("--shard_events", type=int, default=0, help="If >0, write each chunk as its own NPZ shard of up to this many events")
    p.add_argument("--output_dtype", default="float32", choices=["float16","float32"], help="Output dtype for saved waves")
    return p.parse_args()


def _load_npz(input_path: Path, array_key: str) -> tuple[np.ndarray, dict]:
    data = np.load(input_path, allow_pickle=True)
    if array_key not in data:
        raise KeyError(f"Key '{array_key}' not found. Available: {list(data.keys())}")
    waves = data[array_key]
    if waves.ndim != 3:
        raise ValueError(f"Expected 3D (E,C,L) array in '{array_key}', got {waves.shape}")
    meta = {
        "tower_names": data.get("tower_names", None),
        "channel_names": data.get("channel_names", None),
        "run_number": int(data.get("run_number", -1)),
        "num_events": int(data.get("num_events", waves.shape[0])),
        "num_channels": int(data.get("num_channels", waves.shape[1])),
        "wave_length": int(data.get("wave_length", waves.shape[2])),
        "channel_rule": data.get("channel_rule", None),
        "baseline_config": data.get("baseline_config", None),
        "tree_name": data.get("tree_name", None),
        "source_file": data.get("source_file", str(input_path)),
        "channel_groups": data.get("channel_groups", None),
        "waves_dtype": data.get("waves_dtype", str(waves.dtype)),
    }
    return waves, meta


def _read_runlist(path: Path) -> list[int]:
    """Parse runlist file of runs.
    Accepts lines with either:
      - '<run>'
      - '<flag> <run>' (legacy; flag ignored)
    Returns list of run integers.
    """
    if not path.exists():
        raise FileNotFoundError(f"Runlist not found: {path}")
    runs: list[int] = []
    with path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            # Try single-column run first
            try:
                runs.append(int(cols[0]))
                continue
            except Exception:
                pass
            # Legacy: first column is a flag, second is run
            if len(cols) >= 2:
                try:
                    runs.append(int(cols[1]))
                    continue
                except Exception:
                    pass
            raise ValueError(f"Invalid runlist line: '{line}' (expected: '<run>' or '<flag> <run>')")
    return runs


def _generate_tower_names() -> list[str]:
    modules = [f"M{m}" for m in range(1, 10)]
    towers = [f"T{t}" for t in range(1, 5)]
    out: list[str] = []
    for m in modules:
        for t in towers:
            out.append(f"{m}{t}")
    return out


def _build_known_groups() -> dict[str, set[str]]:
    tower_bases = _generate_tower_names()
    towers = {f"{b}S" for b in tower_bases} | {f"{b}C" for b in tower_bases}
    mcp_c = {f"C{i}" for i in range(1, 65)}
    mcp_s = {f"S{i}" for i in range(1, 65)}
    dwc = {
        "DWC1R", "DWC1L", "DWC1U", "DWC1D",
        "DWC2R", "DWC2L", "DWC2U", "DWC2D",
    }
    aux = {"PS", "MC", "CC1", "CC2", "T1", "T2"}
    return {
        "towers": towers,
        "mcp_c": mcp_c,
        "mcp_s": mcp_s,
        "dwc": dwc,
        "aux": aux,
    }


def _resolve_selected_channels(npz_channel_names: Iterable, groups_csv: str) -> list[int]:
    """Return channel indices filtered by desired groups. If 'all' is present, select all.
    Supported groups: towers, mcp_c, mcp_s, dwc, aux, others, all
    """
    names = [str(x) for x in np.asarray(npz_channel_names).astype(str)]
    tokens = [t.strip().lower() for t in groups_csv.split(",") if t.strip()]
    if not tokens:
        tokens = ["towers", "mcp_c", "mcp_s"]
    if "all" in tokens:
        return list(range(len(names)))
    group_map = _build_known_groups()
    known_union = set().union(*group_map.values()) if group_map else set()
    selected_sets: list[set[str]] = [group_map[t] for t in tokens if t in group_map]
    include_others = "others" in tokens
    selected_names = set().union(*selected_sets) if selected_sets else set()
    idxs: list[int] = []
    for i, n in enumerate(names):
        if n in selected_names:
            idxs.append(i)
        elif include_others and n not in known_union:
            idxs.append(i)
    return idxs


def _save_npz(
    out_path: Path,
    waves: np.ndarray,
    meta_in: dict,
    selected_channel_names: list[str],
    model_dir: Path,
    output_dtype: str,
):
    # Build metadata preserving original where possible
    baseline_cfg = meta_in.get("baseline_config")
    if isinstance(baseline_cfg, (bytes, np.ndarray)):
        try:
            baseline_cfg = str(baseline_cfg)
        except Exception:
            baseline_cfg = None
    channel_groups = meta_in.get("channel_groups")
    if isinstance(channel_groups, (bytes, np.ndarray)):
        try:
            channel_groups = str(channel_groups)
        except Exception:
            channel_groups = None

    np.savez_compressed(
        out_path,
        waves_tower=waves,
        tower_names=(np.asarray(meta_in.get("tower_names")) if meta_in.get("tower_names") is not None else np.array([])),
        channel_names=np.asarray(selected_channel_names),
        run_number=np.int32(meta_in.get("run_number", -1)),
        num_events=np.int64(waves.shape[0]),
        num_channels=np.int32(waves.shape[1]),
        wave_length=np.int32(waves.shape[2]),
        channel_rule=(meta_in.get("channel_rule") if meta_in.get("channel_rule") is not None else ""),
        baseline_config=(baseline_cfg if baseline_cfg is not None else ""),
        tree_name=(meta_in.get("tree_name") if meta_in.get("tree_name") is not None else ""),
        source_file=str(meta_in.get("source_file", "")),
        channel_groups=(channel_groups if channel_groups is not None else ""),
        waves_dtype=str(output_dtype),
        denoise_model_dir=str(model_dir),
    )


def _load_model_and_scalers(model_dir: Path) -> tuple[Noise2Noise1DTCN, object, object | None]:
    fscaler = load(model_dir / "waveform_feature_scaler.joblib")
    try:
        tscaler = load(model_dir / "waveform_target_scaler.joblib")
    except Exception:
        tscaler = None
    model = Noise2Noise1DTCN(in_channels=1, num_channels=None, kernel_size=3, dropout=0.1)
    # Load weights on CPU first to avoid CUDA OOM during deserialization
    state = torch.load(model_dir / "waveform_n2n_weights.pt", map_location="cpu")
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    # Try moving to best available device; fall back to CPU on CUDA OOM
    try:
        model.to(torch.device(check_available_device()))
    except RuntimeError as e:
        if "CUDA" in str(e):
            print("Warning: CUDA OOM or device error while placing model. Falling back to CPU.", flush=True)
            model.to(torch.device("cpu"))
        else:
            raise
    return model, fscaler, tscaler


def _process_one_npz(in_path: Path, run: int, args: argparse.Namespace, model, fscaler, tscaler):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    waves, meta = _load_npz(in_path, args.array_key)
    if meta.get("channel_names") is None:
        raise ValueError("Input NPZ must include 'channel_names' for channel filtering and metadata.")
    # Ensure meta has run_number for saving
    meta["run_number"] = int(run)
    chan_names_in = [str(x) for x in np.asarray(meta["channel_names"]).astype(str)]

    # Determine selected channel indices via channel_groups
    idx_sel = _resolve_selected_channels(chan_names_in, args.channel_groups)
    chan_names_sel = [chan_names_in[i] for i in idx_sel]

    E, C, L = waves.shape
    if not idx_sel:
        raise RuntimeError("No channels selected to denoise.")

    # Output dtype
    out_dtype = np.float16 if args.output_dtype == "float16" else np.float32

    # Streaming or full-materialize
    chunk_events = int(args.chunk_events) if args.chunk_events and args.chunk_events > 0 else 0
    shard_events = int(args.shard_events) if args.shard_events and args.shard_events > 0 else 0

    base_name = in_path.stem
    if shard_events > 0:
        total_written = 0
        part = 0
        pbar = tqdm(total=E, unit="events", desc=f"Run {run} (shards)") if tqdm else None
        for start in range(0, E, shard_events):
            stop = min(start + shard_events, E)
            sub = np.asarray(waves[start:stop, :, :], dtype=np.float32)
            sub = sub[:, idx_sel, :]
            e_chunk = sub.shape[0]
            sub2d = sub.reshape(-1, L)
            preds2d = denoise_waveforms(sub2d, fscaler, model, batch_size=args.batch_size, target_scaler=tscaler)
            deno = preds2d.reshape(e_chunk, len(idx_sel), L).astype(out_dtype, copy=False)
            out_path = out_dir / f"{base_name}_denoised_part{part:03d}.npz"
            _save_npz(out_path, deno, meta, chan_names_sel, model_dir, args.output_dtype)
            total_written += e_chunk
            if pbar is not None:
                pbar.update(e_chunk)
            part += 1
            print(f"[{run}] Saved shard {part} ({total_written}/{E} events)", flush=True)
        # Save index JSON for shards
        if pbar is not None:
            pbar.close()
        index = {
            "source_npz": str(in_path),
            "run_number": int(run),
            "num_events": int(E),
            "num_channels": int(len(idx_sel)),
            "wave_length": int(L),
            "parts": part,
            "pattern": f"{base_name}_denoised_part%03d.npz",
            "array_key": args.array_key,
            "model_dir": str(model_dir),
            "output_dtype": args.output_dtype,
        }
        (out_dir / f"{base_name}_denoised_index.json").write_text(json.dumps(index, indent=2))
        return

    # Non-sharded: optionally stream by chunk_events to limit peak memory, else load all
    if chunk_events > 0:
        preds_out = np.empty((E, len(idx_sel), L), dtype=out_dtype)
        write_cursor = 0
        pbar = tqdm(total=E, unit="events", desc=f"Run {run}") if tqdm else None
        for start in range(0, E, chunk_events):
            stop = min(start + chunk_events, E)
            sub = np.asarray(waves[start:stop, :, :], dtype=np.float32)
            sub = sub[:, idx_sel, :]
            e_chunk = sub.shape[0]
            sub2d = sub.reshape(-1, L)
            preds2d = denoise_waveforms(sub2d, fscaler, model, batch_size=args.batch_size, target_scaler=tscaler)
            preds_out[write_cursor:write_cursor+e_chunk] = preds2d.reshape(e_chunk, len(idx_sel), L).astype(out_dtype, copy=False)
            write_cursor += e_chunk
            if pbar is not None:
                pbar.update(e_chunk)
            else:
                print(f"[{run}] Denoised {write_cursor}/{E} events", end="\r", flush=True)
        if pbar is not None:
            pbar.close()
        else:
            print("", flush=True)
    else:
        sub = np.asarray(waves[:, idx_sel, :], dtype=np.float32)
        sub2d = sub.reshape(-1, L)
        preds2d = denoise_waveforms(sub2d, fscaler, model, batch_size=args.batch_size, target_scaler=tscaler)
        preds_out = preds2d.reshape(E, len(idx_sel), L).astype(out_dtype, copy=False)

    out_path = out_dir / f"{base_name}_denoised.npz"
    _save_npz(out_path, preds_out, meta, chan_names_sel, model_dir, args.output_dtype)
    print(f"[{run}] Saved: {out_path}")


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build run list and unique runs
    runs = _read_runlist(Path(args.runlist))
    unique_runs = sorted(set(runs))

    # Load model once
    model, fscaler, tscaler = _load_model_and_scalers(model_dir)

    for run in unique_runs:
        in_path = input_dir / args.npz_pattern.format(run=run)
        if not in_path.exists():
            raise FileNotFoundError(f"Input NPZ not found for run {run}: {in_path}")
        _process_one_npz(in_path, int(run), args, model, fscaler, tscaler)


if __name__ == "__main__":
    main()
