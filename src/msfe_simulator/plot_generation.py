# plot_generation.py

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _as_2d_grid(X_mm: np.ndarray, Y_mm: np.ndarray, Z_um: np.ndarray):
    """Accept either 2D meshgrid or 1D axes + 2D Z."""
    X = np.asarray(X_mm)
    Y = np.asarray(Y_mm)
    Z = np.asarray(Z_um)

    if X.ndim == 1 and Y.ndim == 1 and Z.ndim == 2:
        X2, Y2 = np.meshgrid(X, Y)
        return X2, Y2, Z
    return X, Y, Z


def fig_zs(s_mm: np.ndarray, z_um: np.ndarray):
    fig, ax = plt.subplots()
    ax.plot(np.asarray(s_mm), np.asarray(z_um), linewidth=1.0)
    ax.set_xlabel("s [mm]")
    ax.set_ylabel("z [µm]")
    ax.set_title("Simulated MSFE signal z(s)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def fig_spectrum(
    f_per_mm: np.ndarray,
    amp_um: np.ndarray,
    f0_1_per_mm: float | None = None,
    sigma_f_1_per_mm: float | None = None,
):


    import numpy as np
    import matplotlib.pyplot as plt

    f = np.asarray(f_per_mm, dtype=float)
    a = np.asarray(amp_um, dtype=float)

    m = np.isfinite(f) & np.isfinite(a)
    f, a = f[m], a[m]

    fig, ax = plt.subplots(figsize=(8.6, 5.8))

    if f.size == 0:
        ax.set_title("Simulated MSFE spectrum")
        ax.set_xlabel("Spatial frequency [mm⁻¹]")
        ax.set_ylabel("Amplitude [µm]")
        ax.grid(True, alpha=0.25)
        return fig

    # Ignore DC when finding dominant peak
    non_dc = f > 0
    f_ndc, a_ndc = (f[non_dc], a[non_dc]) if np.any(non_dc) else (f, a)

    if f0_1_per_mm is None:
        i_pk_global = int(np.nanargmax(a_ndc))
        f0 = float(f_ndc[i_pk_global])
    else:
        f0 = float(f0_1_per_mm)

    # Find realized FFT peak near expected location
    i_pk = int(np.argmin(np.abs(f - f0)))
    amp_pk = float(a[i_pk])
    f_pk = float(f[i_pk])

    # Zoom window
    if sigma_f_1_per_mm is not None and sigma_f_1_per_mm > 0:
        sigma_f = float(sigma_f_1_per_mm)
        xmin = max(0.0, f0 - 4 * sigma_f)
        xmax = f0 + 4 * sigma_f
    else:
        sigma_f = None
        half = max(0.03, 0.20 * f0)
        xmin = max(0.0, f0 - half)
        xmax = f0 + half

    # Plot spectrum
    ax.plot(
        f,
        a,
        linewidth=1.3,
        alpha=0.9,
        label="FFT spectrum",
    )

    # ±1σ band shading
    if sigma_f is not None:
        ax.axvspan(
            f0 - sigma_f,
            f0 + sigma_f,
            alpha=0.18,
            label=f"Expected band around f₀",
        )

    # Peak marker
    ax.scatter(
        [f_pk],
        [amp_pk],
        color="black",
        s=20,
        zorder=5,
    )

    # Annotation
    lambda_pk = 1 / f_pk if f_pk > 0 else np.nan
    annotation = f"Peak ≈ {amp_pk:.4f} µm"
    if np.isfinite(lambda_pk):
        annotation += f"\nλ ≈ {lambda_pk:.2f} mm"

    x_text = f_pk + 0.03 * (xmax - xmin)
    y_text = amp_pk * 1.05

    ax.annotate(
        annotation,
        xy=(f_pk, amp_pk),
        xytext=(x_text, y_text),
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=10,
    )

    # Labels
    ax.set_title("Simulated MSFE spectrum around the dominant spatial frequency")
    ax.set_xlabel("Spatial frequency [mm⁻¹]")
    ax.set_ylabel("Amplitude [µm]")

    ax.set_xlim(xmin, xmax)

    # Y limits
    in_win = (f >= xmin) & (f <= xmax)
    if np.any(in_win):
        ymax = np.nanmax(a[in_win])
        ax.set_ylim(0, ymax * 1.12)

    ax.grid(True, alpha=0.25)

    # Legend INSIDE plot
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
    )

    return fig


def fig_surface(X_mm: np.ndarray, Y_mm: np.ndarray, Z_um: np.ndarray):
    """
    Surface wavefronts as black dotted iso-height contours (no color map).
    """

    import numpy as np
    import matplotlib.pyplot as plt

    X = np.asarray(X_mm, dtype=float)
    Y = np.asarray(Y_mm, dtype=float)
    Z = np.asarray(Z_um, dtype=float)

    fig, ax = plt.subplots()

    # Valid points inside aperture
    m = np.isfinite(Z)
    if np.count_nonzero(m) < 100:
        ax.set_title("Simulated surface — black-dot wavefronts")
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.grid(False)
        return fig

    Zv = Z[m]

    # Robust Z range
    z_lo, z_hi = np.percentile(Zv, [2, 98])
    if not np.isfinite(z_lo) or not np.isfinite(z_hi) or z_hi <= z_lo:
        z_lo, z_hi = float(np.nanmin(Zv)), float(np.nanmax(Zv))

    n_levels = 24
    levels = np.linspace(z_lo, z_hi, n_levels)

    cs = ax.contour(X, Y, Z, levels=levels, colors="none", linewidths=0.0)

    dot_spacing_mm = 0.35
    dot_size = 3.0
    dot_alpha = 0.85

    xs_all, ys_all = [], []

    for seglist in cs.allsegs:
        for seg in seglist:
            if seg.shape[0] < 5:
                continue

            dx = np.diff(seg[:, 0])
            dy = np.diff(seg[:, 1])
            ds = np.sqrt(dx * dx + dy * dy)
            s = np.concatenate([[0.0], np.cumsum(ds)])

            total = s[-1]
            if total < 2.0:
                continue

            n = int(np.floor(total / dot_spacing_mm))
            if n < 5:
                continue

            s_query = np.linspace(0.0, total, n, endpoint=True)
            xq = np.interp(s_query, s, seg[:, 0])
            yq = np.interp(s_query, s, seg[:, 1])

            xs_all.append(xq)
            ys_all.append(yq)

    for c in cs.collections:
        c.remove()

    if xs_all:
        x_pts = np.concatenate(xs_all)
        y_pts = np.concatenate(ys_all)

        ax.scatter(
            x_pts,
            y_pts,
            s=dot_size,
            c="black",
            alpha=dot_alpha,
            linewidths=0,
            marker=".",
            rasterized=True,
        )

    # ---- Aperture outline (NOW BLACK) ----
    r = np.sqrt(X[m] ** 2 + Y[m] ** 2)
    R = float(np.nanmax(r)) if r.size else 0.0

    if R > 0:
        t = np.linspace(0, 2 * np.pi, 600)
        ax.plot(
            R * np.cos(t),
            R * np.sin(t),
            color="black",   # <-- changed from default blue
            linewidth=1.2,
            alpha=0.9
        )

        ax.set_xlim(-R * 1.05, R * 1.05)
        ax.set_ylim(-R * 1.05, R * 1.05)

    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Simulated surface — black-dot wavefronts")

    fig.tight_layout()
    return fig


def fig_toolpath(x_mm: np.ndarray, y_mm: np.ndarray, title: str = "Toolpath"):

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.asarray(x_mm, dtype=float)
    y = np.asarray(y_mm, dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    if x.size == 0 or y.size == 0:
        ax.set_title(title)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.text(0.5, 0.5, "No toolpath data", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, alpha=0.25)
        return fig

    n = min(x.size, y.size)
    x, y = x[:n], y[:n]

    # Decimate strongly so the spiral becomes visible
    max_draw = 6000
    if n > max_draw:
        idx = np.linspace(0, n - 1, max_draw).astype(int)
        x_plot = x[idx]
        y_plot = y[idx]
    else:
        x_plot = x
        y_plot = y

    # Draw spiral path
    ax.plot(
        x_plot,
        y_plot,
        linewidth=0.7,
        alpha=0.9,
    )

    # Optional sparse points to show sampling locations
    step_pts = max(1, len(x_plot) // 800)
    ax.scatter(
        x_plot[::step_pts],
        y_plot[::step_pts],
        s=2,
        alpha=0.5,
    )

    # Start / end markers
    ax.scatter([x[0]], [y[0]], s=80, marker="o", zorder=4, label="Start")
    ax.scatter([x[-1]], [y[-1]], s=90, marker="X", zorder=4, label="End")

    # Aperture outline
    r = np.sqrt(x**2 + y**2)
    R = float(np.nanmax(r)) if r.size else 0.0
    if R > 0:
        t = np.linspace(0, 2 * np.pi, 500)
        ax.plot(
            R * np.cos(t),
            R * np.sin(t),
            color="black",
            linewidth=1.0,
            alpha=0.5,
        )
        ax.set_xlim(-R * 1.05, R * 1.05)
        ax.set_ylim(-R * 1.05, R * 1.05)

    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title)
    ax.grid(True, alpha=0.20)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    return fig
