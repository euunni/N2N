#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from joblib import load

# Optional tqdm progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # fallback: no progress bar

# Ensure local `n2n` package is importable when running the script directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import n2n  # noqa: E402
from n2n import Noise2Noise1DTCN  # noqa: E402
from n2n.n2n_denoise import denoise_waveforms  # noqa: E402
from n2n.model_functions import check_available_device  # noqa: E402


def parse_event_slice(arg: Optional[str]) -> Optional[slice]:
    if arg is None:
        return None
    arg = arg.strip()
    if arg == "":
        return None
    # single index
    if ":" not in arg:
        idx = int(arg)
        return slice(idx, idx + 1)
    # "start:stop"
    start_str, stop_str = arg.split(":", 1)
    start = int(start_str) if start_str != "" else None
    stop = int(stop_str) if stop_str != "" else None
    return slice(start, stop)


def parse_ylim(arg: Optional[str]) -> Optional[Tuple[float, float]]:
    if arg is None or arg.strip() == "":
        return None
    lo, hi = arg.split(":", 1)
    return float(lo), float(hi)


def resolve_channel_indices_npz(meta_channel_names: Optional[np.ndarray], select_names: Optional[Iterable[str]]) -> Optional[np.ndarray]:
    if select_names is None:
        return None
    if meta_channel_names is None:
        raise ValueError("Channel selection by name requested but NPZ has no 'channel_names' metadata.")
    name_to_idx = {str(n): i for i, n in enumerate(meta_channel_names.astype(str))}
    indices = []
    for nm in select_names:
        if nm not in name_to_idx:
            raise KeyError(f"Channel name not found in NPZ: {nm}")
        indices.append(name_to_idx[nm])
    if not indices:
        return None
    # deduplicate preserving order
    return np.asarray(list(dict.fromkeys(indices)), dtype=int)


def load_waveforms(
    wave_path: str,
    event_slice: Optional[slice],
    select_channel_names: Optional[Iterable[str]],
    chunk_events: Optional[int],
):
    """
    Returns:
      X: Optional[np.ndarray] flattened (N, L) when chunking disabled; else None
      sub: np.ndarray selection view (E_sel, C_sel, L) when chunking enabled; else unflattened
      L: int waveform length
      sel_e: int selected events
      sel_c: int selected channels
      ch_names_sel: Optional[np.ndarray] channel names for selected channels
    """
    ch_index = None
    ch_names_sel = None

    if wave_path.endswith(".npz"):
        data = np.load(wave_path)
        full = data["waves_tower"]
        meta_channel_names = data.get("channel_names", None)
        if meta_channel_names is not None:
            ch_index = resolve_channel_indices_npz(meta_channel_names, select_channel_names)
            if ch_index is not None:
                ch_names_sel = meta_channel_names.astype(str)[ch_index]
        if full.ndim == 3:
            ev_idx = event_slice if event_slice is not None else slice(None)
            ch_idx = ch_index if ch_index is not None else slice(None)
            sub = full[ev_idx, ch_idx, :]
            sel_e, sel_c, L = int(sub.shape[0]), int(sub.shape[1]), int(sub.shape[2])
            if chunk_events is None:
                sub = sub.astype(np.float32)
                X = sub.reshape(-1, L)
            else:
                X = None
        elif full.ndim == 2:
            L = int(full.shape[1])
            sub = full.astype(np.float32)
            sel_e, sel_c = int(sub.shape[0]), 1
            X = sub
        else:
            raise ValueError(f"Unsupported waves_tower shape: {full.shape}")
    else:
        full = np.load(wave_path, mmap_mode="r")
        if full.ndim == 3:
            ev_idx = event_slice if event_slice is not None else slice(None)
            ch_idx = slice(None)
            sub = full[ev_idx, ch_idx, :]
            sel_e, sel_c, L = int(sub.shape[0]), int(sub.shape[1]), int(sub.shape[2])
            if chunk_events is None:
                sub = np.asarray(sub, dtype=np.float32)
                X = sub.reshape(-1, L)
            else:
                X = None
        elif full.ndim == 2:
            L = int(full.shape[1])
            sub = np.asarray(full, dtype=np.float32)
            sel_e, sel_c = int(sub.shape[0]), 1
            X = sub
        else:
            raise ValueError(f"Unsupported array shape: {full.shape}")

    if X is not None:
        print(f"Loaded waveforms: {X.shape} (flattened 2D)")
    else:
        print(f"Streaming mode: E={sel_e}, C={sel_c}, L={L}, chunk_events={chunk_events}")
    return X, sub, L, sel_e, sel_c, ch_names_sel


def build_model_and_scalers(out_dir: str) -> tuple[Noise2Noise1DTCN, object, Optional[object]]:
    fscaler = load(os.path.join(out_dir, "waveform_feature_scaler.joblib"))
    try:
        tscaler = load(os.path.join(out_dir, "waveform_target_scaler.joblib"))
    except Exception:
        tscaler = None
    model = Noise2Noise1DTCN(in_channels=1, num_channels=None, kernel_size=5, dropout=0.1)
    device = torch.device(check_available_device())
    state = torch.load(os.path.join(out_dir, "waveform_n2n_weights.pt"), map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval().to(device)
    print("Model loaded.")
    return model, fscaler, tscaler


def run_denoise(
    wave_path: str,
    out_dir: str,
    select_channel_names: Optional[Iterable[str]],
    event_slice_str: Optional[str],
    chunk_events: Optional[int],
    batch_size: int,
    save_preds: Optional[str],
    save_preds_reshaped: Optional[str],
    progress: bool,
):
    event_slice = parse_event_slice(event_slice_str)
    X, sub, L, sel_e, sel_c, ch_names_sel = load_waveforms(
        wave_path, event_slice, select_channel_names, chunk_events
    )
    model, fscaler, tscaler = build_model_and_scalers(out_dir)

    if X is not None:
        preds = denoise_waveforms(
            X, fscaler, model, batch_size=batch_size, show_progress=bool(progress), target_scaler=tscaler
        )
    else:
        assert chunk_events is not None and chunk_events > 0
        preds_list = []
        E_sel = sub.shape[0]
        it = range(0, E_sel, chunk_events)
        pbar = None
        if progress and tqdm is not None:
            pbar = tqdm(total=E_sel, unit="event", desc="Events", dynamic_ncols=True)
        for start in it:
            stop = min(start + chunk_events, E_sel)
            chunk = np.asarray(sub[start:stop, :, :], dtype=np.float32)
            chunk2d = chunk.reshape(-1, L)
            chunk_raw = denoise_waveforms(
                chunk2d, fscaler, model, batch_size=batch_size, show_progress=False, target_scaler=tscaler
            )
            preds_list.append(chunk_raw)
            if pbar is not None:
                pbar.update(stop - start)
        if pbar is not None:
            pbar.close()
        preds = np.concatenate(preds_list, axis=0)

    print("Denoised:", preds.shape)

    # Save outputs if requested
    if save_preds:
        np.save(save_preds, preds.astype(np.float32))
        print(f"Saved flattened preds to: {save_preds}")
    if save_preds_reshaped:
        try:
            preds_ecl = preds.reshape(sel_e, sel_c, L)
        except Exception:
            raise ValueError("Cannot reshape preds to (E, C, L); missing selection dims.")
        np.save(save_preds_reshaped, preds_ecl.astype(np.float32))
        print(f"Saved reshaped preds to: {save_preds_reshaped}")

    # Small summary line that can be grepped in logs
    print(f"[DONE] path={wave_path} E={sel_e} C={sel_c} L={L} out={out_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Denoise 1D waveforms with Noise2Noise1DTCN (streaming supported).")
    p.add_argument("--wave-path", required=True, help="Path to .npz (key 'waves_tower') or .npy 3D/2D array")
    p.add_argument("--out-dir", required=True, help="Training output directory with weights/scalers")
    p.add_argument("--select-channels", type=str, default=None, help="Comma-separated channel names (NPZ only)")
    p.add_argument("--event-slice", type=str, default=None, help="Event range 'start:stop' or single index 'k'")
    p.add_argument("--chunk-events", type=int, default=100, help="Chunk size by events for streaming (None disables)")
    p.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    p.add_argument("--save-preds", type=str, default=None, help="Save flattened (N, L) predictions to .npy")
    p.add_argument("--save-preds-reshaped", type=str, default=None, help="Save reshaped (E, C, L) predictions to .npy")
    p.add_argument("--progress", action="store_true", help="Show progress bar (events for streaming, batches otherwise)")
    return p


def main():
    args = build_argparser().parse_args()
    select_channel_names = None
    if args.select_channels:
        select_channel_names = [s for s in args.select_channels.split(",") if s]

    # Sanitize paths
    wave_path = os.path.abspath(args.wave_path)
    out_dir = os.path.abspath(args.out_dir)
    save_preds = os.path.abspath(args.save_preds) if args.save_preds else None
    save_preds_reshaped = os.path.abspath(args.save_preds_reshaped) if args.save_preds_reshaped else None

    os.makedirs(os.path.dirname(save_preds), exist_ok=True) if save_preds else None
    os.makedirs(os.path.dirname(save_preds_reshaped), exist_ok=True) if save_preds_reshaped else None

    run_denoise(
        wave_path=wave_path,
        out_dir=out_dir,
        select_channel_names=select_channel_names,
        event_slice_str=args.event_slice,
        chunk_events=args.chunk_events,
        batch_size=args.batch_size,
        save_preds=save_preds,
        save_preds_reshaped=save_preds_reshaped,
        progress=bool(args.progress),
    )


if __name__ == "__main__":
    main()
