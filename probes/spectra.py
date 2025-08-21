import numpy as np
def fft2_spectrum(x2d):
    F = np.fft.fftshift(np.fft.fft2(x2d))
    return np.abs(F)**2
def band_rmse(pred, true, high_k_mask):
    err = (pred - true)**2
    return float(np.sqrt(np.mean(err[high_k_mask])))
