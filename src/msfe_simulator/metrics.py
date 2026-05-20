from __future__ import annotations
from typing import Dict
import numpy as np


def _interp_crossing(x1: float, y1: float, x2: float, y2: float, y_target: float) -> float:
    """Linear interpolation of x at y=y_target between two points."""
    if y2 == y1:
        return float(x1)
    return float(x1 + (y_target - y1) * (x2 - x1) / (y2 - y1))


def _gaussian_fwhm_from_sigma(sigma: float) -> float:
    """Convert Gaussian sigma to FWHM."""
    return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma)


def histogram_fwhm(
    x: np.ndarray,
    weights: np.ndarray | None = None,
    bins: int = 60,
) -> float:
    """
    Estimate FWHM from a weighted histogram.
    Best used for sampled frequency components fk.
    """
    x = np.asarray(x, dtype=float)
    if weights is None:
        weights = np.ones_like(x, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    m = np.isfinite(x) & np.isfinite(weights)
    x = x[m]
    weights = weights[m]

    if x.size < 5:
        return float("nan")

    hist, edges = np.histogram(x, bins=bins, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if hist.size < 3 or np.max(hist) <= 0:
        return float("nan")

    i_max = int(np.argmax(hist))
    hmax = float(hist[i_max])
    half = 0.5 * hmax

    left = float("nan")
    for i in range(i_max, 0, -1):
        y0, y1 = float(hist[i - 1]), float(hist[i])
        if (y0 <= half <= y1) or (y1 <= half <= y0):
            left = _interp_crossing(float(centers[i - 1]), y0, float(centers[i]), y1, half)
            break

    right = float("nan")
    for i in range(i_max, hist.size - 1):
        y0, y1 = float(hist[i]), float(hist[i + 1])
        if (y0 >= half >= y1) or (y1 >= half >= y0):
            right = _interp_crossing(float(centers[i]), y0, float(centers[i + 1]), y1, half)
            break

    if np.isfinite(left) and np.isfinite(right) and right >= left:
        return float(right - left)
    return float("nan")


def fft_peak_metrics(
    f: np.ndarray,
    amp: np.ndarray,
    f0_expected: float,
    sigma_f: float,
    band_sigma: float = 3.0,
) -> Dict[str, float]:
    """
    FFT-based peak metrics in band [f0_expected ± band_sigma*sigma_f].
    """
    if f.size != amp.size or f.size == 0:
        raise ValueError("f and amp must have same non-zero length.")
    if sigma_f <= 0 or band_sigma <= 0:
        raise ValueError("sigma_f and band_sigma must be > 0.")

    f = np.asarray(f, dtype=float)
    amp = np.asarray(amp, dtype=float)

    valid = np.isfinite(f) & np.isfinite(amp)
    f = f[valid]
    amp = amp[valid]

    if f.size == 0:
        raise ValueError("No finite FFT values found.")

    df_fft = float(np.median(np.diff(f))) if f.size > 1 else float("nan")

    lo = max(f0_expected - band_sigma * sigma_f, 0.0)
    hi = f0_expected + band_sigma * sigma_f
    band = (f >= lo) & (f <= hi)

    empty = {
        "peak_center": float("nan"),
        "peak_center_error": float("nan"),
        "peak_center_error_pct": float("nan"),
        "peak_amp": float("nan"),
        "peak_fwhm_fft": float("nan"),
        "peak_energy": float("nan"),
        "peak_prominence": float("nan"),
        "df_fft": df_fft,
    }

    if not np.any(band):
        return empty

    f_b = f[band]
    a_b = amp[band]

    i_max = int(np.argmax(a_b))
    peak_center = float(f_b[i_max])
    peak_amp = float(a_b[i_max])

    half = 0.5 * peak_amp

    left = float("nan")
    for i in range(i_max, 0, -1):
        y0, y1 = float(a_b[i - 1]), float(a_b[i])
        if (y0 <= half <= y1) or (y1 <= half <= y0):
            left = _interp_crossing(float(f_b[i - 1]), y0, float(f_b[i]), y1, half)
            break

    right = float("nan")
    for i in range(i_max, a_b.size - 1):
        y0, y1 = float(a_b[i]), float(a_b[i + 1])
        if (y0 >= half >= y1) or (y1 >= half >= y0):
            right = _interp_crossing(float(f_b[i]), y0, float(f_b[i + 1]), y1, half)
            break

    peak_fwhm_fft = float(right - left) if np.isfinite(left) and np.isfinite(right) and right >= left else float("nan")
    peak_energy = float(np.sum(a_b ** 2) * df_fft) if np.isfinite(df_fft) else float("nan")

    outside = ~band
    if np.any(outside):
        bg95 = float(np.percentile(amp[outside], 95))
    else:
        bg95 = float(np.percentile(amp, 50))

    peak_prominence = float(peak_amp - bg95)
    peak_center_error = float(peak_center - f0_expected)
    peak_center_error_pct = float(100.0 * peak_center_error / f0_expected) if f0_expected > 0 else float("nan")

    return {
        "peak_center": peak_center,
        "peak_center_error": peak_center_error,
        "peak_center_error_pct": peak_center_error_pct,
        "peak_amp": peak_amp,
        "peak_fwhm_fft": peak_fwhm_fft,
        "peak_energy": peak_energy,
        "peak_prominence": peak_prominence,
        "df_fft": df_fft,
    }


def component_peak_metrics(
    fk: np.ndarray,
    sigma_f: float,
    weights: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    Metrics from sampled frequency components fk.
    This is the best estimate of the realized model peak width.
    """
    fk = np.asarray(fk, dtype=float)
    fk = fk[np.isfinite(fk)]

    if fk.size < 5:
        return {
            "fwhm_expected": float("nan"),
            "fwhm_fk_est": float("nan"),
            "fk_mean": float("nan"),
            "fk_std": float("nan"),
        }

    fk_mean = float(np.mean(fk))
    fk_std = float(np.std(fk, ddof=0))
    fwhm_expected = _gaussian_fwhm_from_sigma(sigma_f)
    fwhm_fk_est = histogram_fwhm(fk, weights=weights, bins=min(60, max(20, fk.size // 5)))

    return {
        "fwhm_expected": fwhm_expected,
        "fwhm_fk_est": fwhm_fk_est,
        "fk_mean": fk_mean,
        "fk_std": fk_std,
    }


def surface_metrics(Z_um: np.ndarray) -> Dict[str, float]:
    """RMS and PV inside aperture (finite values only)."""
    z = Z_um[np.isfinite(Z_um)]
    if z.size == 0:
        return {
            "surface_rms_um": float("nan"),
            "surface_pv_um": float("nan"),
        }

    rms = float(np.sqrt(np.mean(z ** 2)))
    pv = float(np.max(z) - np.min(z))
    return {
        "surface_rms_um": rms,
        "surface_pv_um": pv,
    }