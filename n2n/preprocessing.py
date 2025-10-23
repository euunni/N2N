import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 1D waveform helpers

def standardise_array(
    data: np.ndarray,
    method: str = "RobustScaler",
    scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None,
) -> tuple[np.ndarray, StandardScaler | MinMaxScaler | RobustScaler]:

    if scaler is None:
        if method == "StandardScaler":
            scaler = StandardScaler()
        elif method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif method == "RobustScaler":
            scaler = RobustScaler()
        else:
            raise ValueError("Unsupported scaler method")
        scaler.fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled, scaler


def generate_noisy_waveforms(
    waveforms: np.ndarray,
    noise_level: float = 0.02,
    random_seed: int | None = 42,
    min_sigma: float | None = None,
) -> np.ndarray:

    rng = np.random.default_rng(random_seed)
    wf = np.asarray(waveforms, dtype=float)

    if min_sigma is not None and float(min_sigma) > 0.0:
        # Robust per-trace scale via MAD -> std (1.4826 factor for Gaussian consistency)
        med = np.median(wf, axis=1, keepdims=True)
        mad = np.median(np.abs(wf - med), axis=1, keepdims=True)
        s = 1.4826 * mad
        base = max(noise_level, 1e-6) * s
        sigma = np.sqrt(base * base + (float(min_sigma) ** 2))
    else:
        # Legacy: per-trace std scaled by noise_level, with 0.01 fallback
        per_std = np.std(wf, axis=1, keepdims=True)
        sigma = np.where(per_std > 0, per_std * max(noise_level, 1e-3), 0.01)

    noise = rng.normal(0.0, sigma, size=wf.shape)
    return wf + noise


def split_train_val_test_indices(
    n_samples: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return train_idx, val_idx, test_idx
