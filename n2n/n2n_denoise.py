import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from n2n.model_functions import predict, check_available_device
from torch.nn import Module

def create_dataloader(
    spectra: np.ndarray,
    batch_size: int = 1000,
) -> DataLoader:
    """Create a DataLoader from the spectra.

    Parameters
    ----------
    spectra : np.ndarray
        The spectra to create a DataLoader from.
        Shape (n_samples, n_bands)
    batch_size : int, optional
        The batch size for the DataLoader.
        Default 1000

    Returns
    -------
    data_loader : DataLoader
        The DataLoader for the spectra.
    """
    dataset = TensorDataset(Tensor(spectra))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


# ------------ 1D waveform inference ------------
def denoise_waveforms(
    waveforms: np.ndarray,
    feature_scaler: BaseEstimator,
    model: Module,
    batch_size: int = 1024,
    show_progress: bool = True,
    residual_mode: bool = True,
    target_scaler: BaseEstimator | None = None,
) -> np.ndarray:
    """Denoise 1D waveforms already shaped (n_samples, n_points).

    Steps:
    - Scale inputs with feature_scaler
    - Predict
    - If residual_mode:
        * If target_scaler is provided, interpret model output as residual in target-scaled
          space, inverse-transform it to raw units, and subtract from raw input: y_hat = X_raw - r_raw
        * Otherwise, fall back to subtracting in scaled feature space: y_hat_s = X_s - r_s
      If not residual_mode, return model outputs (scaled space)
    """
    spectra = np.asarray(waveforms, dtype=float)
    spectra_s = feature_scaler.transform(spectra)
    loader = create_dataloader(spectra_s, batch_size=batch_size)
    preds = predict(
        model,
        loader,
        torch.device(check_available_device()),
        show_progress=show_progress,
        desc="Denoising",
    )
    preds = preds.cpu().numpy()
    if residual_mode:
        # Prefer performing subtraction in raw space when target scaler available
        if target_scaler is not None:
            r_raw = target_scaler.inverse_transform(preds)
            return spectra - r_raw
        # Fallback: subtract in scaled feature space
        return spectra_s - preds
    return preds
