# app.py
"""Streamlit UI for MSFE simulation (clean workflow + separated plotting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from model import ModelParams, ModelSettings, simulate_msfe
from metrics import fft_peak_metrics, component_peak_metrics, surface_metrics
from sensitivity import sweep_parameter
from plot_generation import fig_surface, fig_spectrum, fig_toolpath, fig_zs


st.set_page_config(page_title="MSFE Simulator", layout="wide")

st.markdown("""
<style>
.block-container{
    max-width:100% !important;
    padding-top:0rem !important;
    padding-left:1rem;
    padding-right:1rem;
}
[data-testid="stHeader"]{
    height:0rem;
}
section.main > div{
    padding-top:0rem !important;
}
</style>
""", unsafe_allow_html=True)

if "mode" not in st.session_state:
    st.session_state.mode = "Single simulation"
if "view" not in st.session_state:
    st.session_state.view = "Surface"
if "sim" not in st.session_state:
    st.session_state.sim = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "sensitivity" not in st.session_state:
    st.session_state.sensitivity = None



def _st_df(df: pd.DataFrame, *, hide_index: bool = True) -> None:
    try:
        st.dataframe(df, hide_index=hide_index, width="stretch")
    except TypeError:
        st.dataframe(df, hide_index=hide_index, use_container_width=True)


def _st_pyplot(fig) -> None:
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        st.pyplot(fig, use_container_width=True)


def _fixed_defaults_table(settings: ModelSettings) -> pd.DataFrame:
    rows = [
        ("spiral step", settings.spiral_step_mm, "mm", "Radial step Δr"),
        ("frequency samples", settings.Nf, "-", "# frequency components"),
        ("m_trunc", settings.m_trunc, "-", "Truncation ±mσ"),
        ("ds_per_wavelength", settings.ds_per_wavelength, "-", "Rule: ds ≤ λ0/ds_per_wavelength (clamped)"),
        ("ds_min_mm", settings.ds_min_mm, "mm", "Minimum ds clamp"),
        ("ds_max_mm", settings.ds_max_mm, "mm", "Maximum ds clamp"),
        ("grid_pitch_mm", settings.grid_pitch_mm, "mm", "Surface grid spacing"),
    ]
    return pd.DataFrame(rows, columns=["Setting", "Value", "Unit", "Notes"])


def _derived_table(meta: dict) -> pd.DataFrame:
    rows = [
        ("Expected dominant frequency", meta.get("f0_1_per_mm", np.nan), "1/mm", "Frequency predicted from tool speed and path speed"),
        ("Expected dominant wavelength", meta.get("lambda0_mm", np.nan), "mm", "Wavelength corresponding to the dominant frequency"),
        ("Input peak width (σf)", meta.get("sigma_f_1_per_mm", np.nan), "1/mm", "Width parameter used to generate the frequency distribution"),
        ("Path sampling step", meta.get("ds_mm", np.nan), "mm", "Sampling step along the spiral path"),
        ("Spiral radial step", meta.get("spiral_step_mm", np.nan), "mm", "Radial increment between neighboring spiral turns"),
        ("Peak truncation", meta.get("m_trunc", np.nan), "σ", "Frequency distribution truncated at ±mσ"),
        ("Number of frequency components", meta.get("Nf", np.nan), "-", "Number of sinusoidal components used in the simulation"),
        ("Effective footprint width (σxy)", meta.get("sigma_xy_mm", np.nan), "mm", "Controls spatial smoothing during 1D to 2D mapping"),
    ]
    df = pd.DataFrame(rows, columns=["Quantity", "Value", "Unit", "Notes"])

    def _fmt(v):
        if v is None:
            return np.nan
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return v
            av = abs(float(v))
            if av == 0:
                return 0.0
            if av < 1e-3 or av >= 1e4:
                return float(f"{v:.6g}")
            return float(f"{v:.6f}".rstrip("0").rstrip("."))
        return v

    df["Value"] = df["Value"].apply(_fmt)
    return df


def _metrics_tables(sim: dict) -> dict[str, pd.DataFrame]:
    meta = sim["meta"]
    f = sim["spectrum_1d"]["f_1_per_mm"]
    amp = sim["spectrum_1d"]["amp_um"]
    Z = sim["surface_2d"]["Z_um"]

    sig = sim.get("signal_1d", {})
    z_path = np.asarray(sig.get("z_um", []), dtype=float)
    fk = np.asarray(sig.get("fk_1_per_mm", []), dtype=float)
    Ak = np.asarray(sig.get("Ak_rel", []), dtype=float) if "Ak_rel" in sig else None

    fft_m = fft_peak_metrics(
        f=f,
        amp=amp,
        f0_expected=float(meta["f0_1_per_mm"]),
        sigma_f=float(meta["sigma_f_1_per_mm"]),
    )

    comp_m = component_peak_metrics(
        fk=fk,
        sigma_f=float(meta["sigma_f_1_per_mm"]),
        weights=Ak,
    )

    surf_m = surface_metrics(Z)

    path_rms = float(np.sqrt(np.mean(z_path ** 2))) if z_path.size > 0 else float("nan")
    surface_rms = surf_m.get("surface_rms_um", np.nan)
    surface_rms_ratio = float(surface_rms / path_rms) if np.isfinite(path_rms) and path_rms > 0 else float("nan")

    peak_center = fft_m.get("peak_center", np.nan)
    dominant_wavelength = float(1.0 / peak_center) if np.isfinite(peak_center) and peak_center > 0 else float("nan")

    fwhm_fk_est = comp_m.get("fwhm_fk_est", np.nan)
    relative_bandwidth = (
        float(fwhm_fk_est / peak_center)
        if np.isfinite(fwhm_fk_est) and np.isfinite(peak_center) and peak_center > 0
        else float("nan")
    )

    peak_prominence = fft_m.get("peak_prominence", np.nan)
    pattern_strength = (
        float(peak_prominence / surface_rms)
        if np.isfinite(peak_prominence) and np.isfinite(surface_rms) and surface_rms > 0
        else float("nan")
    )

    df_pattern = pd.DataFrame(
        [
            ("Dominant frequency", peak_center, "1/mm", "Main spatial frequency observed in the simulated spectrum"),
            ("Dominant wavelength", dominant_wavelength, "mm", "Surface spacing associated with the dominant frequency"),
            ("Theoretical peak width", comp_m.get("fwhm_expected", np.nan), "1/mm", "Width implied by the input peak-width parameter"),
            ("Realized peak width", fwhm_fk_est, "1/mm", "Width realized by the sampled frequency components"),
            ("Relative bandwidth", relative_bandwidth, "-", "Realized peak width divided by dominant frequency"),
            ("Mean sampled frequency", comp_m.get("fk_mean", np.nan), "1/mm", "Mean of the sampled frequency components"),
            ("Std. dev. of sampled frequencies", comp_m.get("fk_std", np.nan), "1/mm", "Spread of the sampled frequency components"),
        ],
        columns=["Metric", "Value", "Unit", "Notes"],
    )

    df_strength = pd.DataFrame(
        [
            ("Peak amplitude", fft_m.get("peak_amp", np.nan), "µm", "Height of the dominant peak in the FFT spectrum"),
            ("Peak prominence", peak_prominence, "µm", "How far the dominant peak rises above the spectral background"),
            ("Band energy", fft_m.get("peak_energy", np.nan), "µm²/mm", "Energy concentrated in the dominant spectral band"),
            ("Pattern strength", pattern_strength, "-", "Peak prominence relative to surface RMS"),
            ("FFT frequency resolution", fft_m.get("df_fft", np.nan), "1/mm", "Frequency bin spacing of the FFT"),
            ("FFT line width", fft_m.get("peak_fwhm_fft", np.nan), "1/mm", "Observed narrow line width in the realized FFT"),
        ],
        columns=["Metric", "Value", "Unit", "Notes"],
    )

    df_surface = pd.DataFrame(
        [
            ("Path RMS", path_rms, "µm", "RMS of the 1D simulated signal before surface mapping"),
            ("Surface RMS", surface_rms, "µm", "RMS of the final 2D surface inside the aperture"),
            ("Surface PV", surf_m.get("surface_pv_um", np.nan), "µm", "Peak-to-valley of the final 2D surface"),
            ("Surface smoothing factor", surface_rms_ratio, "-", "Remaining RMS after 1D to 2D mapping"),
        ],
        columns=["Metric", "Value", "Unit", "Notes"],
    )

    def _fmt_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        def _fmt(v):
            if v is None:
                return np.nan
            if isinstance(v, (int, np.integer)):
                return int(v)
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    return v
                av = abs(float(v))
                if av == 0:
                    return 0.0
                if av < 1e-6 or av >= 1e6:
                    return float(f"{v:.6g}")
                return float(f"{v:.6f}".rstrip("0").rstrip("."))
            return v

        out["Value"] = out["Value"].apply(_fmt)
        return out

    return {
        "Pattern characteristics": _fmt_df(df_pattern),
        "Pattern strength": _fmt_df(df_strength),
        "Surface response": _fmt_df(df_surface),
    }


def _bandwidth_label(rb: float) -> str:
    if not np.isfinite(rb):
        return "not available"
    if rb < 0.10:
        return "narrow-band and highly regular"
    if rb < 0.25:
        return "moderately broad"
    return "broad and more diffuse"


def _strength_label(ps: float) -> str:
    if not np.isfinite(ps):
        return "not available"
    if ps < 0.5:
        return "weak"
    if ps < 1.5:
        return "moderate"
    return "strong"


def _smoothing_label(sf: float) -> str:
    if not np.isfinite(sf):
        return "not available"
    if sf < 0.15:
        return "strongly smoothed during 2D mapping"
    if sf < 0.40:
        return "moderately smoothed during 2D mapping"
    return "only lightly smoothed during 2D mapping"


def _interpretation_text(sim: dict) -> str:
    meta = sim["meta"]
    tables = _metrics_tables(sim)

    pattern_df = tables["Pattern characteristics"]
    strength_df = tables["Pattern strength"]
    surface_df = tables["Surface response"]

    def get_val(df: pd.DataFrame, name: str) -> float:
        row = df.loc[df["Metric"] == name, "Value"]
        if row.empty:
            return float("nan")
        val = row.iloc[0]
        try:
            return float(val)
        except Exception:
            return float("nan")

    f0_expected = float(meta.get("f0_1_per_mm", np.nan))
    peak_center = get_val(pattern_df, "Dominant frequency")
    wavelength = get_val(pattern_df, "Dominant wavelength")
    rel_bw = get_val(pattern_df, "Relative bandwidth")
    peak_amp = get_val(strength_df, "Peak amplitude")
    pattern_strength = get_val(strength_df, "Pattern strength")
    smoothing = get_val(surface_df, "Surface smoothing factor")
    surface_rms = get_val(surface_df, "Surface RMS")
    surface_pv = get_val(surface_df, "Surface PV")

    lines = []

    lines.append(
        f"The simulation produced a dominant mid-spatial-frequency pattern at "
        f"**{peak_center:.4f} 1/mm**"
        + (f", corresponding to a surface spacing of **{wavelength:.3f} mm**." if np.isfinite(wavelength) else ".")
    )

    if np.isfinite(f0_expected):
        lines.append(
            f"The kinematic expectation from the input parameters is **{f0_expected:.4f} 1/mm**, "
            f"so the simulated spectrum is centered close to the process-driven target."
        )

    lines.append(
        f"The realized spectral band is **{_bandwidth_label(rel_bw)}**"
        + (f" with a relative bandwidth of **{rel_bw:.3f}**." if np.isfinite(rel_bw) else ".")
    )

    lines.append(
        f"The dominant peak amplitude is **{peak_amp:.4f} µm**, and the overall pattern strength is "
        f"**{_strength_label(pattern_strength)}**"
        + (f" (strength metric = **{pattern_strength:.3f}**)." if np.isfinite(pattern_strength) else ".")
    )

    lines.append(
        f"After mapping the 1D signal to the 2D surface, the result is **{_smoothing_label(smoothing)}**"
        + (f". The final surface RMS is **{surface_rms:.4f} µm** and the peak-to-valley value is **{surface_pv:.4f} µm**."
           if np.isfinite(surface_rms) and np.isfinite(surface_pv) else ".")
    )

    return "\n\n".join(lines)

def _default_sweep_values(param_name: str, params: ModelParams) -> list[float]:
    if param_name == "v_mm_per_min":
        base = params.v_mm_per_min
        return [0.7 * base, 0.85 * base, base, 1.15 * base, 1.3 * base]

    if param_name == "n_rpm":
        base = params.n_rpm
        return [0.7 * base, 0.85 * base, base, 1.15 * base, 1.3 * base]

    if param_name == "FWHMrel":
        return [0.05, 0.10, 0.15, 0.20, 0.30]

    if param_name == "alpha":
        return [0.0, 0.5, 1.0, 2.0, 3.0]

    if param_name == "a":
        return [0.5, 0.7, 0.8, 0.9, 1.0]

    if param_name == "b":
        return [1.0, 1.1, 1.2, 1.4, 1.6]

    if param_name == "z_rms_um":
        return [0.05, 0.08, 0.12, 0.16, 0.20]

    if param_name == "r_mm":
        base = params.r_mm if params.r_mm is not None else 100.0
        return [0.5 * base, 0.75 * base, base, 1.25 * base, 1.5 * base]

    if param_name == "h_um":
        base = params.h_um if params.h_um is not None else 10.0
        return [0.5 * base, 0.75 * base, base, 1.25 * base, 1.5 * base]

    if param_name == "sigma_xy_mm":
        base = params.sigma_xy_mm if params.sigma_xy_mm is not None else 0.15
        return [0.5 * base, 0.75 * base, base, 1.25 * base, 1.5 * base]

    return []


def _sensitivity_dataframe(result: dict) -> pd.DataFrame:
    rows = []
    for rec in result["records"]:
        rows.append({
            "Parameter value": rec["value"],
            "Peak center [1/mm]": rec["spec"]["peak_center"],
            "Peak amplitude [µm]": rec["spec"]["peak_amp"],
            "Peak FWHM FFT [1/mm]": rec["spec"]["peak_fwhm_fft"],
            "Peak energy [µm²/mm]": rec["spec"]["peak_energy"],
            "Peak prominence [µm]": rec["spec"]["peak_prominence"],
            "Surface RMS [µm]": rec["surf"]["surface_rms_um"],
            "Surface PV [µm]": rec["surf"]["surface_pv_um"],
        })

    return pd.DataFrame(rows).sort_values("Parameter value").reset_index(drop=True)


def _fig_sensitivity(df: pd.DataFrame, param_label: str, y_col: str):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(df["Parameter value"], df[y_col], marker="o", linewidth=1.5)
    ax.set_xlabel(param_label)
    ax.set_ylabel(y_col)
    ax.set_title(f"Sensitivity of {y_col} to {param_label}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _run_single(params: ModelParams, settings: ModelSettings) -> dict:
    sim = simulate_msfe(params, settings)
    st.session_state.metrics = _metrics_tables(sim)
    return sim


col_params, col_view, col_ctrl = st.columns([1, 2, 1], gap="large")

with col_params:
    st.subheader("Parameters")

    with st.expander("Aperture / geometry", expanded=False):
        R_mm = st.number_input("Aperture radius [mm]", min_value=1.0, value=31.0, step=1.0)

    with st.expander("Kinematics", expanded=False):
        v_mm_per_min = st.number_input("Path speed [mm/min]", min_value=1.0, value=5000.0, step=100.0)
        n_rpm = st.number_input("Tool speed [rpm]", min_value=1.0, value=1912.0, step=10.0)

    with st.expander("Toolpath", expanded=False):
        spiral_step_mm = st.number_input(
        "Spiral radial step Δr [mm]",
        min_value=0.001,
        value=0.05,
        step=0.01,
        format="%.3f",
        help="Radial increment between spiral turns (controls toolpath density)"
    )

    with st.expander("Peak distribution", expanded=False):
        peak_type = st.selectbox("Peak type", ["gaussian", "skewed"], index=0)
        FWHMrel = st.number_input("FWHMrel (FWHM / f0)", min_value=0.01, value=0.20, step=0.01, format="%.3f")
        alpha = 0.0
        if peak_type == "skewed":
            alpha = st.number_input("alpha (skew strength)", min_value=0.0, value=2.0, step=0.1, format="%.2f")

    with st.expander("Amplitude model", expanded=False):
        a = st.number_input("Uniform lower bound", min_value=0.01, value=0.8, step=0.05, format="%.2f")
        b = st.number_input("Uniform upper bound", min_value=0.01, value=1.2, step=0.05, format="%.2f")
        z_rms_um = st.number_input("Target z RMS [µm]", min_value=0.001, value=0.12, step=0.01, format="%.3f")

    with st.expander("Contact footprint", expanded=False):
        use_rh = st.radio("Specify footprint via", ["r & h", "σxy"], horizontal=True, index=0)
        r_mm = h_um = sigma_xy_mm = None
        if use_rh == "r & h":
            r_mm = st.number_input("Tool curvature radius [mm]", min_value=1.0, value=100.0, step=10.0)
            h_um = st.number_input("Depth of cut [µm]", min_value=0.0, value=10.0, step=1.0)
        else:
            sigma_xy_mm = st.number_input("σxy [mm]", min_value=0.0, value=0.15, step=0.01, format="%.3f")

    with st.expander("Reproducibility", expanded=False):
        random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    params = ModelParams(
        R_mm=float(R_mm),
        v_mm_per_min=float(v_mm_per_min),
        n_rpm=float(n_rpm),
        peak_type=str(peak_type),
        FWHMrel=float(FWHMrel),
        alpha=float(alpha),
        a=float(a),
        b=float(b),
        z_rms_um=float(z_rms_um),
        r_mm=None if r_mm is None else float(r_mm),
        h_um=None if h_um is None else float(h_um),
        sigma_xy_mm=None if sigma_xy_mm is None else float(sigma_xy_mm),
        random_seed=int(random_seed),
    )

    settings = ModelSettings(
    spiral_step_mm=float(spiral_step_mm)
    )

with col_ctrl:
    st.subheader("Controls")

    if st.button("Calculate", use_container_width=True):
        st.session_state.sim = _run_single(params, settings)
        st.session_state.view = "Surface"

    st.markdown("### Output view")
    if st.button("Surface", use_container_width=True):
        st.session_state.view = "Surface"
    if st.button("Spectrum", use_container_width=True):
        st.session_state.view = "Spectrum"
    if st.button("z(s)", use_container_width=True):
        st.session_state.view = "z(s)"
    if st.button("Derived quantities", use_container_width=True):
        st.session_state.view = "Derived quantities"
    if st.button("Metrics", use_container_width=True):
        st.session_state.view = "Metrics"
    if st.button("Interpretation", use_container_width=True):
        st.session_state.view = "Interpretation"
    if st.button("Sensitivity analysis", use_container_width=True):
        st.session_state.view = "Sensitivity analysis"

with col_view:
    st.subheader("Output")

    sim = st.session_state.sim

    if sim is None:
        st.info("Set parameters and click Calculate.")
    else:
        view = st.session_state.view

        if view == "Surface":
            surf = sim["surface_2d"]
            fig = fig_surface(surf["X_mm"], surf["Y_mm"], surf["Z_um"])
            _st_pyplot(fig)

        elif view == "Spectrum":
            spec = sim["spectrum_1d"]
            meta = sim["meta"]
            fig = fig_spectrum(
                spec["f_1_per_mm"],
                spec["amp_um"],
                f0_1_per_mm=meta["f0_1_per_mm"],
                sigma_f_1_per_mm=meta["sigma_f_1_per_mm"],
            )
            _st_pyplot(fig)

        elif view == "Toolpath":
            tp = sim["toolpath_1d"]
            fig = fig_toolpath(tp["x_mm"], tp["y_mm"], title="Simulated spiral toolpath")
            _st_pyplot(fig)

        elif view == "z(s)":
            sig = sim.get("signal_1d", sim.get("toolpath_1d", {}))
            fig = fig_zs(sig["s_mm"], sig["z_um"])
            _st_pyplot(fig)

        elif view == "Derived quantities":
            _st_df(_derived_table(sim["meta"]))

        elif view == "Metrics":
            if st.session_state.metrics is None:
                st.session_state.metrics = _metrics_tables(sim)

            st.markdown("### Pattern characteristics")
            _st_df(st.session_state.metrics["Pattern characteristics"])

            st.markdown("### Pattern strength")
            _st_df(st.session_state.metrics["Pattern strength"])

            st.markdown("### Surface response")
            _st_df(st.session_state.metrics["Surface response"])

        elif view == "Interpretation":
            st.markdown("### User-friendly summary")
            st.markdown(_interpretation_text(sim))

        elif view == "Sensitivity analysis":
            st.markdown("### One-factor-at-a-time sensitivity analysis")

            if params.r_mm is not None and params.h_um is not None:
                allowed_params = [
                    "v_mm_per_min",
                    "n_rpm",
                    "FWHMrel",
                    "alpha",
                    "a",
                    "b",
                    "z_rms_um",
                    "r_mm",
                    "h_um",
                ]
            else:
                allowed_params = [
                    "v_mm_per_min",
                    "n_rpm",
                    "FWHMrel",
                    "alpha",
                    "a",
                    "b",
                    "z_rms_um",
                    "sigma_xy_mm",
                ]

            pretty_names = {
                "v_mm_per_min": "Path speed v [mm/min]",
                "n_rpm": "Tool speed n [rpm]",
                "FWHMrel": "Relative peak width FWHMrel [-]",
                "alpha": "Skew parameter α [-]",
                "a": "Amplitude lower bound a [-]",
                "b": "Amplitude upper bound b [-]",
                "z_rms_um": "Target RMS z_rms [µm]",
                "r_mm": "Tool curvature radius r [mm]",
                "h_um": "Depth of cut h [µm]",
                "sigma_xy_mm": "Effective footprint σxy [mm]",
            }

            sweep_param = st.selectbox(
                "Parameter to sweep",
                options=allowed_params,
                format_func=lambda x: pretty_names.get(x, x),
                key="sens_param_select",
            )

            default_vals = _default_sweep_values(sweep_param, params)
            default_vals_str = ", ".join(f"{v:.6g}" for v in default_vals)

            vals_str = st.text_input(
                "Sweep values (comma-separated)",
                value=default_vals_str,
                key="sens_values_input",
            )

            metric_to_plot = st.selectbox(
                "Metric to plot",
                options=[
                    "Peak center [1/mm]",
                    "Peak amplitude [µm]",
                    "Peak FWHM FFT [1/mm]",
                    "Peak energy [µm²/mm]",
                    "Peak prominence [µm]",
                    "Surface RMS [µm]",
                    "Surface PV [µm]",
                ],
                index=5,
                key="sens_metric_plot",
            )

            if st.button("Run sensitivity analysis", use_container_width=True):
                try:
                    values = [float(v.strip()) for v in vals_str.split(",") if v.strip() != ""]
                    if len(values) < 2:
                        st.error("Please provide at least two sweep values.")
                    else:
                        result = sweep_parameter(
                            base_params=params,
                            settings=settings,
                            param_name=sweep_param,
                            values=values,
                            fixed_seed=True,
                        )
                        st.session_state.sensitivity = {
                            "result": result,
                            "df": _sensitivity_dataframe(result),
                            "param_name": sweep_param,
                        }
                except Exception as e:
                    st.error(f"Sensitivity analysis failed: {e}")

            sens = st.session_state.sensitivity
            if sens is not None:
                df = sens["df"]
                pname = sens["param_name"]

                st.markdown("#### Results table")
                _st_df(df)

                st.markdown("#### Sensitivity plot")
                fig = _fig_sensitivity(df, pretty_names.get(pname, pname), metric_to_plot)
                _st_pyplot(fig)