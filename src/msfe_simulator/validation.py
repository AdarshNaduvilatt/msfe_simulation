# validation.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator

from .model import ModelParams, ModelSettings, simulate_msfe


def parse_mahr_polar_txt(filepath: str | Path) -> Dict[str, object]:
    filepath = Path(filepath)
    lines = filepath.read_text(encoding="utf-8", errors="ignore").splitlines()

    header: Dict[str, str] = {}
    rows: List[Tuple[int, float, float, float, float, int]] = []

    in_header = False
    in_values = False

    line_re = re.compile(
        r"^\s*(\d+)\s*=\s*"
        r"([+-]?\d+(?:\.\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?)\s+"
        r"(\d+)\s*$"
    )

    for line in lines:
        s = line.strip()
        if not s:
            continue

        if s == "[PROFILE_HEADER]":
            in_header = True
            in_values = False
            continue

        if s == "[PROFILE_VALUES]":
            in_header = False
            in_values = True
            continue

        if s.startswith("[") and s.endswith("]"):
            in_header = False
            in_values = False
            continue

        if in_header and "=" in s:
            k, v = s.split("=", 1)
            header[k.strip()] = v.strip()
            continue

        if in_values:
            if s.startswith("//"):
                continue
            m = line_re.match(s)
            if m:
                rows.append((
                    int(m.group(1)),
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                    float(m.group(5)),
                    int(m.group(6)),
                ))

    if not rows:
        raise ValueError(f"No profile data found in {filepath}")

    df = pd.DataFrame(rows, columns=["idx", "X", "Y", "Z", "C_deg", "Lock"])
    return {"header": header, "data": df}



def _uniform_angle_resample(c_deg: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = np.asarray(c_deg, dtype=float)
    z = np.asarray(z, dtype=float)

    m = np.isfinite(c) & np.isfinite(z)
    c = c[m]
    z = z[m]

    order = np.argsort(c)
    c = c[order]
    z = z[order]

    c_unique, uniq_idx = np.unique(c, return_index=True)
    z_unique = z[uniq_idx]

    if c_unique.size < 4:
        return c_unique, z_unique

    dc = np.diff(c_unique)
    dc = dc[np.isfinite(dc) & (dc > 0)]
    if dc.size == 0:
        return c_unique, z_unique

    dcu = float(np.median(dc))
    c_u = np.arange(c_unique[0], c_unique[-1] + 0.5 * dcu, dcu)
    z_u = np.interp(c_u, c_unique, z_unique)
    return c_u, z_u


def _detrend_vs_angle(c_deg: np.ndarray, z: np.ndarray, poly_order: int = 1) -> np.ndarray:
    c = np.asarray(c_deg, dtype=float)
    z = np.asarray(z, dtype=float)

    m = np.isfinite(c) & np.isfinite(z)
    c = c[m]
    z = z[m]

    if c.size < poly_order + 2:
        return z - np.mean(z)

    if np.nanmax(c) - np.nanmin(c) <= 1e-12:
        return z - np.mean(z)

    try:
        coeff = np.polyfit(c, z, poly_order)
        trend = np.polyval(coeff, c)
        out = z - trend
        out -= np.mean(out)
        return out
    except Exception:
        return z - np.mean(z)


def load_measured_rings(
    folder: str | Path,
    pattern: str = "emmav_3d_pol_*.txt",
    aperture_diameter_mm: float = 45.0,
    nrings_expected: int = 68,
    z_scale_to_um: float = 1.0,
) -> List[Dict[str, object]]:
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {folder}")

    R = aperture_diameter_mm / 2.0
    dr = R / max(nrings_expected - 1, 1)

    rings: List[Dict[str, object]] = []

    for i, fp in enumerate(files):
        parsed = parse_mahr_polar_txt(fp)
        df = parsed["data"].copy()
        df = df[df["Lock"] == 0]

        c_u, z_u = _uniform_angle_resample(
            df["C_deg"].to_numpy(),
            df["Z"].to_numpy() * z_scale_to_um,
        )
        z_det = _detrend_vs_angle(c_u, z_u, poly_order=1)

        r_i = i * dr
        theta = np.deg2rad(c_u)

        x = r_i * np.cos(theta)
        y = r_i * np.sin(theta)

        rings.append({
            "ring_index": i,
            "radius_mm": float(r_i),
            "C_deg_uniform": c_u,
            "X_mm": x,
            "Y_mm": y,
            "Z_detrended_um": z_det,
            "file": fp.name,
        })

    return rings


def reconstruct_ground_truth_surface(
    rings: List[Dict[str, object]],
    aperture_radius_mm: float = 22.5,
    grid_step_mm: float = 0.1,
) -> Dict[str, np.ndarray]:
    xs = np.concatenate([r["X_mm"] for r in rings])
    ys = np.concatenate([r["Y_mm"] for r in rings])
    zs = np.concatenate([r["Z_detrended_um"] for r in rings])

    gx = np.arange(-aperture_radius_mm, aperture_radius_mm + grid_step_mm, grid_step_mm)
    gy = np.arange(-aperture_radius_mm, aperture_radius_mm + grid_step_mm, grid_step_mm)
    Xg, Yg = np.meshgrid(gx, gy)

    Zg = griddata((xs, ys), zs, (Xg, Yg), method="linear")

    mask_nan = ~np.isfinite(Zg)
    if np.any(mask_nan):
        Znn = griddata((xs, ys), zs, (Xg, Yg), method="nearest")
        Zg[mask_nan] = Znn[mask_nan]

    m = np.isfinite(Zg)
    A = np.c_[np.ones(np.count_nonzero(m)), Xg[m], Yg[m]]
    b = Zg[m]
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    plane = coef[0] + coef[1] * Xg + coef[2] * Yg
    Zgt = Zg - plane

    Rg = np.sqrt(Xg**2 + Yg**2)
    Zgt[Rg > aperture_radius_mm] = np.nan

    return {
        "X_mm": Xg,
        "Y_mm": Yg,
        "Z_um": Zgt,
    }



def _sample_circle(
    X_mm: np.ndarray,
    Y_mm: np.ndarray,
    Z_um: np.ndarray,
    radius_mm: float,
    n_theta: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = X_mm[0, :]
    ys = Y_mm[:, 0]

    interp = RegularGridInterpolator(
        (ys, xs),
        Z_um,
        bounds_error=False,
        fill_value=np.nan,
    )

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    xq = radius_mm * np.cos(theta)
    yq = radius_mm * np.sin(theta)

    pts = np.c_[yq, xq]
    zq = interp(pts)

    s_mm = radius_mm * theta
    return s_mm, zq


def _detrend_vs_arc_length(s_mm: np.ndarray, z_um: np.ndarray, poly_order: int = 1) -> np.ndarray:
    s = np.asarray(s_mm, dtype=float)
    z = np.asarray(z_um, dtype=float)

    m = np.isfinite(s) & np.isfinite(z)
    s = s[m]
    z = z[m]

    if s.size < poly_order + 2:
        return z - np.mean(z)

    if np.nanmax(s) - np.nanmin(s) <= 1e-12:
        return z - np.mean(z)

    try:
        coeff = np.polyfit(s, z, poly_order)
        trend = np.polyval(coeff, s)
        out = z - trend
        out -= np.mean(out)
        return out
    except Exception:
        return z - np.mean(z)


def _fft_arc_length(s_mm: np.ndarray, z_um: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s_mm, dtype=float)
    z = np.asarray(z_um, dtype=float)

    m = np.isfinite(s) & np.isfinite(z)
    s = s[m]
    z = z[m]

    if s.size < 8:
        return np.array([]), np.array([])

    ds = float(np.median(np.diff(s)))
    if not np.isfinite(ds) or ds <= 0:
        return np.array([]), np.array([])

    # normalized, unity window
    Zf = np.fft.rfft(z)
    f = np.fft.rfftfreq(len(z), d=ds)
    amp = 2.0 * np.abs(Zf) / len(z)
    return f, amp


def _peak_fwhm(f: np.ndarray, amp: np.ndarray, i_peak: int) -> float:
    a_peak = amp[i_peak]
    half = 0.5 * a_peak

    left = np.nan
    for i in range(i_peak, 0, -1):
        y0, y1 = amp[i - 1], amp[i]
        if (y0 <= half <= y1) or (y1 <= half <= y0):
            x0, x1 = f[i - 1], f[i]
            left = x0 if y1 == y0 else x0 + (half - y0) * (x1 - x0) / (y1 - y0)
            break

    right = np.nan
    for i in range(i_peak, len(amp) - 1):
        y0, y1 = amp[i], amp[i + 1]
        if (y0 >= half >= y1) or (y1 >= half >= y0):
            x0, x1 = f[i], f[i + 1]
            right = x0 if y1 == y0 else x0 + (half - y0) * (x1 - x0) / (y1 - y0)
            break

    if np.isfinite(left) and np.isfinite(right) and right >= left:
        return float(right - left)
    return np.nan


def collect_ring_spectra(
    X_mm: np.ndarray,
    Y_mm: np.ndarray,
    Z_um: np.ndarray,
    aperture_radius_mm: float = 22.5,
    nrings: int = 68,
    min_radius_mm: float = 2.5,
    max_radius_mm: float | None = None,
    n_theta: int = 4096,
) -> List[Dict[str, object]]:
    dr = aperture_radius_mm / (nrings - 1)
    radii = np.arange(1, nrings, dtype=float) * dr
    radii = radii[radii >= min_radius_mm]
    if max_radius_mm is not None:
        radii = radii[radii <= max_radius_mm]

    spectra = []

    for ring_idx, r in enumerate(radii, start=1):
        s_mm, z_ring = _sample_circle(X_mm, Y_mm, Z_um, radius_mm=float(r), n_theta=n_theta)

        m = np.isfinite(z_ring)
        if np.count_nonzero(m) < max(64, n_theta // 3):
            continue

        s_use = s_mm[m]
        z_use = z_ring[m]

        if s_use.size < 16 or (np.nanmax(s_use) - np.nanmin(s_use) <= 1e-12):
            continue

        z_det = _detrend_vs_arc_length(s_use, z_use, poly_order=1)
        f, amp = _fft_arc_length(s_use, z_det)

        if f.size == 0:
            continue

        spectra.append({
            "radius_mm": float(r),
            "s_mm": s_use,
            "z_um": z_det,
            "f_1_per_mm": f,
            "amp_um": amp,
        })

    return spectra


def build_representative_spectrum(
    spectra: List[Dict[str, object]],
    band: Tuple[float, float] = (0.4, 3.4),
) -> Dict[str, np.ndarray]:
    if len(spectra) == 0:
        return {
            "f_1_per_mm": np.array([]),
            "amp_um": np.array([]),
        }

    flow, fhigh = band

    f_ref = None
    amp_stack = []

    for sp in spectra:
        f = np.asarray(sp["f_1_per_mm"], dtype=float)
        a = np.asarray(sp["amp_um"], dtype=float)

        m = np.isfinite(f) & np.isfinite(a) & (f >= flow) & (f <= fhigh)
        f = f[m]
        a = a[m]

        if f.size < 10:
            continue

        if f_ref is None:
            f_ref = f
            amp_stack.append(a)
        else:
            a_i = np.interp(f_ref, f, a, left=np.nan, right=np.nan)
            amp_stack.append(a_i)

    if f_ref is None or len(amp_stack) == 0:
        return {
            "f_1_per_mm": np.array([]),
            "amp_um": np.array([]),
        }

    A = np.vstack(amp_stack)
    amp_mean = np.nanmean(A, axis=0)

    return {
        "f_1_per_mm": f_ref,
        "amp_um": amp_mean,
    }


def summarize_spectrum_peak(
    f: np.ndarray,
    amp: np.ndarray,
    band: Tuple[float, float] = (0.4, 3.4),
) -> Dict[str, float]:
    flow, fhigh = band

    m = np.isfinite(f) & np.isfinite(amp) & (f >= flow) & (f <= fhigh)
    f_use = f[m]
    a_use = amp[m]

    if f_use.size == 0:
        return {
            "dominant_frequency_1_per_mm": np.nan,
            "dominant_wavelength_mm": np.nan,
            "bandwidth_fwhm_1_per_mm": np.nan,
            "peak_amplitude_um": np.nan,
        }

    i_local = int(np.argmax(a_use))
    i_peak = np.where(m)[0][i_local]

    f0 = float(f[i_peak])
    lam = float(1.0 / f0) if f0 > 0 else np.nan
    a0 = float(amp[i_peak])
    fwhm = _peak_fwhm(f, amp, i_peak)

    return {
        "dominant_frequency_1_per_mm": f0,
        "dominant_wavelength_mm": lam,
        "bandwidth_fwhm_1_per_mm": fwhm,
        "peak_amplitude_um": a0,
    }



def plot_peak_overlay(
    meas_spec: Dict[str, np.ndarray],
    sim_spec: Dict[str, np.ndarray],
    meas_peak: Dict[str, float],
    sim_peak: Dict[str, float],
    band: Tuple[float, float] = (0.4, 3.4),
    zoom_halfwidth: float = 0.08,
) -> None:
    f_meas = meas_spec["f_1_per_mm"]
    a_meas = meas_spec["amp_um"]
    f_sim = sim_spec["f_1_per_mm"]
    a_sim = sim_spec["amp_um"]

    if f_meas.size == 0 or f_sim.size == 0:
        raise RuntimeError("Measured or simulated representative spectrum is empty.")

    center_candidates = [
        meas_peak["dominant_frequency_1_per_mm"],
        sim_peak["dominant_frequency_1_per_mm"],
    ]
    center_candidates = [v for v in center_candidates if np.isfinite(v)]
    f_center = float(np.mean(center_candidates)) if center_candidates else 0.6

    xmin = max(0.0, f_center - zoom_halfwidth)
    xmax = f_center + zoom_halfwidth

    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    ax.plot(f_meas, a_meas, label="Measured", linewidth=1.2)
    ax.plot(f_sim, a_sim, label="Simulated", linewidth=1.2)

    if np.isfinite(meas_peak["dominant_frequency_1_per_mm"]):
        ax.axvline(
            meas_peak["dominant_frequency_1_per_mm"],
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
        )

    if np.isfinite(sim_peak["dominant_frequency_1_per_mm"]):
        ax.axvline(
            sim_peak["dominant_frequency_1_per_mm"],
            linestyle=":",
            linewidth=1.3,
            alpha=0.8,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Spatial frequency [1/mm]")
    ax.set_ylabel("FFT amplitude [µm]")
    ax.set_title("Measured vs simulated MSF spectrum around the dominant peak")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()



if __name__ == "__main__":
    folder = "."

    aperture_radius_mm = 22.5
    nrings = 68

    # use only reasonably stable rings
    min_radius_mm = 2.5
    max_radius_mm = 20.0

    # MSFE window
    msfe_band = (0.4, 3.4)

    # keep independent from simulation
    z_scale_to_um = 1.0

    # -------- measured branch --------
    print("Loading measured rings...")
    rings = load_measured_rings(
        folder=folder,
        pattern="emmav_3d_pol_*.txt",
        aperture_diameter_mm=45.0,
        nrings_expected=nrings,
        z_scale_to_um=z_scale_to_um,
    )

    print("Reconstructing measured ground-truth surface...")
    gt = reconstruct_ground_truth_surface(
        rings=rings,
        aperture_radius_mm=aperture_radius_mm,
        grid_step_mm=0.1,
    )

    print("Extracting measured circular spectra...")
    meas_ring_spectra = collect_ring_spectra(
        gt["X_mm"],
        gt["Y_mm"],
        gt["Z_um"],
        aperture_radius_mm=aperture_radius_mm,
        nrings=nrings,
        min_radius_mm=min_radius_mm,
        max_radius_mm=max_radius_mm,
        n_theta=4096,
    )

    meas_spec = build_representative_spectrum(
        meas_ring_spectra,
        band=msfe_band,
    )
    meas_peak = summarize_spectrum_peak(
        meas_spec["f_1_per_mm"],
        meas_spec["amp_um"],
        band=msfe_band,
    )

    # -------- simulated branch --------
    params = ModelParams(
        R_mm=31.0,
        v_mm_per_min=5000.0,
        n_rpm=2971.0,
        peak_type="gaussian",
        FWHMrel=0.20,
        alpha=0.0,
        a=0.8,
        b=1.2,
        z_rms_um=0.12,
        r_mm=100.0,
        h_um=10.0,
        sigma_xy_mm=None,
        random_seed=42,
    )

    settings = ModelSettings(
        spiral_step_mm=0.05,
        Nf=300,
        m_trunc=3.0,
        grid_pitch_mm=0.05,
    )

    print("Running independent simulation...")
    sim = simulate_msfe(params, settings)
    surf_sim = sim["surface_2d"]

    print("Extracting simulated circular spectra...")
    sim_ring_spectra = collect_ring_spectra(
        surf_sim["X_mm"],
        surf_sim["Y_mm"],
        surf_sim["Z_um"],
        aperture_radius_mm=aperture_radius_mm,
        nrings=nrings,
        min_radius_mm=min_radius_mm,
        max_radius_mm=max_radius_mm,
        n_theta=4096,
    )

    sim_spec = build_representative_spectrum(
        sim_ring_spectra,
        band=msfe_band,
    )
    sim_peak = summarize_spectrum_peak(
        sim_spec["f_1_per_mm"],
        sim_spec["amp_um"],
        band=msfe_band,
    )

    # -------- comparison table --------
    summary = pd.DataFrame([
        {"source": "Measured", **meas_peak},
        {"source": "Simulated", **sim_peak},
    ])

    summary["dominant_frequency_1_per_mm"] = summary["dominant_frequency_1_per_mm"].round(6)
    summary["dominant_wavelength_mm"] = summary["dominant_wavelength_mm"].round(6)
    summary["bandwidth_fwhm_1_per_mm"] = summary["bandwidth_fwhm_1_per_mm"].round(6)
    summary["peak_amplitude_um"] = summary["peak_amplitude_um"].round(6)

    summary.to_csv("validation_summary.csv", index=False)

    print("\nValidation comparison table:")
    print(summary)

    if np.isfinite(meas_peak["dominant_frequency_1_per_mm"]) and np.isfinite(sim_peak["dominant_frequency_1_per_mm"]):
        f0_err_pct = 100.0 * (
            sim_peak["dominant_frequency_1_per_mm"] - meas_peak["dominant_frequency_1_per_mm"]
        ) / meas_peak["dominant_frequency_1_per_mm"]
        print(f"\nDominant-frequency error: {f0_err_pct:.2f}%")

    if np.isfinite(meas_peak["bandwidth_fwhm_1_per_mm"]) and np.isfinite(sim_peak["bandwidth_fwhm_1_per_mm"]):
        bw_err_pct = 100.0 * (
            sim_peak["bandwidth_fwhm_1_per_mm"] - meas_peak["bandwidth_fwhm_1_per_mm"]
        ) / meas_peak["bandwidth_fwhm_1_per_mm"]
        print(f"Bandwidth error: {bw_err_pct:.2f}%")

    # -------- one final validation plot --------
    plot_peak_overlay(
        meas_spec=meas_spec,
        sim_spec=sim_spec,
        meas_peak=meas_peak,
        sim_peak=sim_peak,
        band=msfe_band,
        zoom_halfwidth=0.08,
    )

    plt.show()