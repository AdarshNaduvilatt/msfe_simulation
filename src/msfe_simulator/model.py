# model.py
"""

User parameters (ModelParams)
- R_mm: aperture radius [mm]
- v_mm_per_min: path speed [mm/min]
- n_rpm: tool rotation [rpm]
- peak_type: "gaussian" (disc-like) or "skewed" (pot-like right-skew)
- FWHMrel: relative peak width (FWHM / f0)
- alpha: skew strength (only for skewed peak)
- a, b: Uniform(a,b) multiplier inside peak: A_rel = A_env * U
- z_rms_um: target RMS amplitude of z(s) [µm]
- Contact footprint: r_mm & h_um (preferred) OR sigma_xy_mm (fallback)
- random_seed: reproducibility

Fixed-by-convergence numerical settings (ModelSettings)
- Nf (# components), m_trunc (±mσ truncation)
- ds_mm chosen automatically by rule ds <= λ0 / ds_per_wavelength with clamps
- grid_pitch_mm (2D grid resolution)

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any, Tuple
import numpy as np

PeakType = Literal["gaussian", "skewed"]


@dataclass(frozen=True)
class ModelParams:
    # Geometry / part
    R_mm: float

    # Kinematics
    v_mm_per_min: float
    n_rpm: float

    # Peak (machine/tool signature)
    peak_type: PeakType
    FWHMrel: float
    alpha: float = 0.0  # only for peak_type="skewed"

    # Amplitude model
    a: float = 0.8
    b: float = 1.2
    z_rms_um: float = 0.12

    # Contact footprint (prefer r,h; fallback sigma)
    r_mm: Optional[float] = None
    h_um: Optional[float] = None
    sigma_xy_mm: Optional[float] = None

    # Reproducibility
    random_seed: int = 0


@dataclass(frozen=True)
class ModelSettings:
    # Fixed by convergence
    spiral_step_mm: float = 0.05
    m_trunc: float = 3.0
    Nf: int = 300

    # ds selection rule (if ds_mm is None)
    ds_mm: Optional[float] = None
    ds_per_wavelength: float = 20.0
    ds_min_mm: float = 0.002
    ds_max_mm: float = 0.05

    grid_pitch_mm: float = 0.05


def f0_from_kinematics(v_mm_per_min: float, n_rpm: float) -> float:
    """Dominant spatial frequency along toolpath [1/mm]."""
    if v_mm_per_min <= 0 or n_rpm <= 0:
        raise ValueError("v_mm_per_min and n_rpm must be > 0.")
    return n_rpm / v_mm_per_min


def lambda0_from_f0(f0_1_per_mm: float) -> float:
    """Dominant wavelength [mm]."""
    if f0_1_per_mm <= 0:
        raise ValueError("f0 must be > 0.")
    return 1.0 / f0_1_per_mm


def fwhm_to_sigma(fwhm: float) -> float:
    """Gaussian: FWHM = 2*sqrt(2 ln 2)*sigma."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def choose_ds_mm(f0: float, settings: ModelSettings) -> float:
    """Pick ds by convergence rule ds <= λ0 / ds_per_wavelength, then clamp."""
    if settings.ds_mm is not None:
        return float(settings.ds_mm)
    lam0 = lambda0_from_f0(f0)
    ds_target = lam0 / max(settings.ds_per_wavelength, 1.0)
    return float(np.clip(ds_target, settings.ds_min_mm, settings.ds_max_mm))


def contact_width_s_perp(r_mm: float, h_mm: float) -> float:
    """Chord width: s⊥ = 2*sqrt(2 r h − h²)."""
    if r_mm <= 0 or h_mm < 0:
        raise ValueError("r_mm must be > 0 and h_mm must be >= 0.")
    inside = 2.0 * r_mm * h_mm - h_mm * h_mm
    return 2.0 * np.sqrt(inside) if inside > 0 else 0.0


def sigma_xy_from_rh(r_mm: float, h_mm: float) -> float:
    """Approximate σ from width using 6σ ≈ width (±3σ)."""
    s_perp = contact_width_s_perp(r_mm, h_mm)
    return s_perp / 6.0 if s_perp > 0 else 0.0


def effective_sigma_xy(params: ModelParams) -> float:
    """Use r,h if provided; else sigma_xy fallback."""
    if params.r_mm is not None and params.h_um is not None:
        return float(sigma_xy_from_rh(params.r_mm, params.h_um / 1000.0))
    if params.sigma_xy_mm is not None:
        return float(params.sigma_xy_mm)
    raise ValueError("Provide either (r_mm and h_um) OR sigma_xy_mm.")


def spiral_path(R_mm: float, step_mm: float, ds_mm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spiral: r(θ)=bθ, b=step/(2π).
    ds/dθ = sqrt((bθ)^2 + b^2). Choose Δθ ≈ ds / (ds/dθ).
    Returns arrays: s_mm, x_mm, y_mm.
    """
    if R_mm <= 0 or step_mm <= 0 or ds_mm <= 0:
        raise ValueError("R_mm, step_mm, ds_mm must be > 0.")

    b = step_mm / (2.0 * np.pi)
    theta_end = R_mm / b

    thetas = [0.0]
    theta = 0.0
    while theta < theta_end:
        ds_dtheta = np.sqrt((b * theta) ** 2 + b ** 2) + 1e-15
        dtheta = ds_mm / ds_dtheta
        theta = min(theta + dtheta, theta_end)
        thetas.append(theta)

    theta = np.array(thetas, dtype=float)
    r = b * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    ds_seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    s = np.concatenate([[0.0], np.cumsum(ds_seg)])
    return s, x, y


def sample_frequencies(
    f0: float,
    FWHMrel: float,
    Nf: int,
    peak_type: PeakType,
    alpha: float,
    m: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    
    if f0 <= 0 or FWHMrel <= 0 or m <= 0:
        raise ValueError("Require f0>0, FWHMrel>0, m>0.")
    if Nf < 5:
        raise ValueError("Nf must be >= 5.")

    sigma_f = fwhm_to_sigma(FWHMrel * f0) + 1e-15

    if peak_type == "gaussian":
        fk = rng.normal(f0, sigma_f, size=Nf)
    elif peak_type == "skewed":
        base = rng.normal(f0, sigma_f, size=Nf)
        skew = np.abs(rng.normal(0.0, max(alpha, 0.0) * sigma_f, size=Nf))
        fk = base + skew
    else:
        raise ValueError(f"Unknown peak_type: {peak_type}")

    lo = max(f0 - m * sigma_f, 1e-12)
    hi = f0 + m * sigma_f
    fk = np.clip(fk, lo, hi)
    return fk, float(sigma_f)


def amplitude_model(
    fk: np.ndarray,
    f0: float,
    sigma_f: float,
    a: float,
    b: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """A(f)=exp(-(f-f0)^2/(2σ_f^2)) * U, U~Uniform(a,b)."""
    if not (0 < a <= b):
        raise ValueError("Require 0 < a <= b.")
    if sigma_f <= 0:
        raise ValueError("sigma_f must be > 0.")
    A_env = np.exp(-0.5 * ((fk - f0) / sigma_f) ** 2)
    U = rng.uniform(a, b, size=fk.shape)
    return A_env * U


def synthesize_z_of_s(
    s_mm: np.ndarray,
    fk: np.ndarray,
    Ak: np.ndarray,
    z_rms_um: float,
    rng: np.random.Generator,
) -> np.ndarray:
    
    if z_rms_um <= 0:
        raise ValueError("z_rms_um must be > 0.")
    phases = rng.uniform(0.0, 2.0 * np.pi, size=fk.size)

    z = np.zeros_like(s_mm, dtype=float)
    for k in range(fk.size):
        z += Ak[k] * np.sin(2.0 * np.pi * fk[k] * s_mm + phases[k])

    z -= np.mean(z)
    rms = float(np.sqrt(np.mean(z ** 2)) + 1e-15)
    return z * (z_rms_um / rms)


def fft_spectrum_1d(s_mm: np.ndarray, z_um: np.ndarray, use_hann: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum; Hann window optional."""
    ds = float(np.median(np.diff(s_mm)))
    z = z_um - np.mean(z_um)
    n = z.size

    if use_hann:
        w = np.hanning(n)
        Z = np.fft.rfft(z * w)
        amp = (2.0 / (np.sum(w) + 1e-15)) * np.abs(Z)
    else:
        Z = np.fft.rfft(z)
        amp = (2.0 / n) * np.abs(Z)

    f = np.fft.rfftfreq(n, d=ds)
    return f, amp


def paint_to_grid(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_um: np.ndarray,
    R_mm: float,
    grid_pitch_mm: float,
    sigma_xy_mm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if grid_pitch_mm <= 0:
        raise ValueError("grid_pitch_mm must be > 0.")
    if sigma_xy_mm < 0:
        raise ValueError("sigma_xy_mm must be >= 0.")

    xs = np.arange(-R_mm, R_mm + grid_pitch_mm, grid_pitch_mm)
    ys = np.arange(-R_mm, R_mm + grid_pitch_mm, grid_pitch_mm)
    X, Y = np.meshgrid(xs, ys)

    Zacc = np.zeros_like(X, dtype=float)
    Wacc = np.zeros_like(X, dtype=float)

    if sigma_xy_mm == 0.0:
        for xp, yp, zp in zip(x_mm, y_mm, z_um):
            ix = int(round((xp - xs[0]) / grid_pitch_mm))
            iy = int(round((yp - ys[0]) / grid_pitch_mm))
            if 0 <= iy < X.shape[0] and 0 <= ix < X.shape[1]:
                Zacc[iy, ix] += zp
                Wacc[iy, ix] += 1.0
    else:
        rad = int(np.ceil(3.0 * sigma_xy_mm / grid_pitch_mm))
        two_sigma2 = 2.0 * sigma_xy_mm * sigma_xy_mm + 1e-15

        for xp, yp, zp in zip(x_mm, y_mm, z_um):
            ix = int(round((xp - xs[0]) / grid_pitch_mm))
            iy = int(round((yp - ys[0]) / grid_pitch_mm))

            x0 = max(ix - rad, 0)
            x1 = min(ix + rad + 1, X.shape[1])
            y0 = max(iy - rad, 0)
            y1 = min(iy + rad + 1, X.shape[0])

            XX = X[y0:y1, x0:x1] - xp
            YY = Y[y0:y1, x0:x1] - yp
            w = np.exp(-(XX * XX + YY * YY) / two_sigma2)

            Zacc[y0:y1, x0:x1] += w * zp
            Wacc[y0:y1, x0:x1] += w

    Z = np.divide(Zacc, Wacc, out=np.zeros_like(Zacc), where=Wacc > 1e-12)

    mask = (X * X + Y * Y) <= (R_mm * R_mm)
    Z[~mask] = np.nan
    return X, Y, Z


def simulate_msfe(
    params: ModelParams,
    settings: ModelSettings = ModelSettings(),
    use_hann_fft: bool = True,
) -> Dict[str, Any]:
    """Run full forward simulation and return all intermediate results."""
    if params.R_mm <= 0:
        raise ValueError("R_mm must be > 0.")
    if params.v_mm_per_min <= 0 or params.n_rpm <= 0:
        raise ValueError("v_mm_per_min and n_rpm must be > 0.")
    if params.FWHMrel <= 0:
        raise ValueError("FWHMrel must be > 0.")
    if not (0 < params.a <= params.b):
        raise ValueError("Require 0 < a <= b.")
    if params.z_rms_um <= 0:
        raise ValueError("z_rms_um must be > 0.")

    rng = np.random.default_rng(int(params.random_seed))

    f0 = f0_from_kinematics(params.v_mm_per_min, params.n_rpm)
    lam0 = lambda0_from_f0(f0)
    ds_mm = choose_ds_mm(f0, settings)

    s_mm, x_mm, y_mm = spiral_path(params.R_mm, settings.spiral_step_mm, ds_mm)

    fk, sigma_f = sample_frequencies(
        f0=f0,
        FWHMrel=params.FWHMrel,
        Nf=settings.Nf,
        peak_type=params.peak_type,
        alpha=params.alpha,
        m=settings.m_trunc,
        rng=rng,
    )

    Ak = amplitude_model(fk, f0, sigma_f, params.a, params.b, rng)
    z_um = synthesize_z_of_s(s_mm, fk, Ak, params.z_rms_um, rng)

    f, amp = fft_spectrum_1d(s_mm, z_um, use_hann=use_hann_fft)

    sigma_xy = effective_sigma_xy(params)
    X, Y, Z = paint_to_grid(x_mm, y_mm, z_um, params.R_mm, settings.grid_pitch_mm, sigma_xy)

    return {
        "meta": {
            "f0_1_per_mm": float(f0),
            "lambda0_mm": float(lam0),
            "sigma_f_1_per_mm": float(sigma_f),
            "ds_mm": float(ds_mm),
            "spiral_step_mm": float(settings.spiral_step_mm),
            "m_trunc": float(settings.m_trunc),
            "Nf": int(settings.Nf),
            "sigma_xy_mm": float(sigma_xy),
        },
        "toolpath": {
            "s_mm": s_mm,
            "x_mm": x_mm,
            "y_mm": y_mm,
        },
        "toolpath_1d": {
            "s_mm": s_mm,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "z_um": z_um,
        },
        "signal_1d": {
            "s_mm": s_mm,
            "z_um": z_um,
            "fk_1_per_mm": fk,
            "Ak_rel": Ak,
        },
        "spectrum_1d": {
            "f_1_per_mm": f,
            "amp_um": amp,
        },
        "surface_2d": {
            "X_mm": X,
            "Y_mm": Y,
            "Z_um": Z,
        },
        "params": params,
        "settings": settings,
    }
