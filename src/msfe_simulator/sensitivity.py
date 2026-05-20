
from __future__ import annotations
from dataclasses import replace
from typing import Any, Dict, List
import numpy as np

from model import ModelParams, ModelSettings, simulate_msfe
from metrics import fft_peak_metrics, surface_metrics


def run_with_metrics(params: ModelParams, settings: ModelSettings) -> Dict[str, Any]:
    out = simulate_msfe(params, settings=settings)
    meta = out["meta"]
    f = out["spectrum_1d"]["f_1_per_mm"]
    amp = out["spectrum_1d"]["amp_um"]
    Z = out["surface_2d"]["Z_um"]

    spec = fft_peak_metrics(
        f,
        amp,
        f0_expected=meta["f0_1_per_mm"],
        sigma_f=meta["sigma_f_1_per_mm"],
    )
    surf = surface_metrics(Z)
    return {"out": out, "spec": spec, "surf": surf}


def sweep_parameter(
    base_params: ModelParams,
    settings: ModelSettings,
    param_name: str,
    values: List[float],
    fixed_seed: bool = True,
    n_seeds: int = 5,
) -> Dict[str, Any]:

    if not hasattr(base_params, param_name):
        raise ValueError(f"Unknown parameter '{param_name}' in ModelParams.")

    records = []
    for v in values:
        if fixed_seed:
            p = replace(base_params, **{param_name: v})
            r = run_with_metrics(p, settings)
            records.append({"value": v, "spec": r["spec"], "surf": r["surf"]})
        else:
            specs, surfs = [], []
            for j in range(n_seeds):
                p = replace(base_params, **{param_name: v, "random_seed": base_params.random_seed + j})
                r = run_with_metrics(p, settings)
                specs.append(r["spec"])
                surfs.append(r["surf"])

            def agg(dlist, key):
                arr = np.array([d[key] for d in dlist], dtype=float)
                return float(np.mean(arr)), float(np.std(arr))

            spec_mean, spec_std = {}, {}
            for k in specs[0].keys():
                mu, sd = agg(specs, k)
                spec_mean[k] = mu
                spec_std[k] = sd

            surf_mean, surf_std = {}, {}
            for k in surfs[0].keys():
                mu, sd = agg(surfs, k)
                surf_mean[k] = mu
                surf_std[k] = sd

            records.append({
                "value": v,
                "spec_mean": spec_mean,
                "spec_std": spec_std,
                "surf_mean": surf_mean,
                "surf_std": surf_std,
            })

    return {
        "param_name": param_name,
        "values": values,
        "fixed_seed": fixed_seed,
        "n_seeds": n_seeds,
        "records": records,
    }


def convergence_test_ds(
    base_params: ModelParams,
    base_settings: ModelSettings,
    ds_values_mm: List[float],
    metric_key: str = "peak_fwhm_fft",
) -> Dict[str, Any]:
    rows = []
    for ds in ds_values_mm:
        settings = replace(base_settings, ds_mm=float(ds))
        r = run_with_metrics(base_params, settings)
        rows.append({"ds_mm": float(ds), metric_key: float(r["spec"][metric_key])})
    return {"parameter": "ds_mm", "metric_key": metric_key, "rows": rows}


def convergence_test_Nf(
    base_params: ModelParams,
    base_settings: ModelSettings,
    Nf_values: List[int],
    metric_key: str = "peak_fwhm_fft",
) -> Dict[str, Any]:
    rows = []
    for n in Nf_values:
        settings = replace(base_settings, Nf=int(n))
        r = run_with_metrics(base_params, settings)
        rows.append({"Nf": int(n), metric_key: float(r["spec"][metric_key])})
    return {"parameter": "Nf", "metric_key": metric_key, "rows": rows}


def convergence_test_m_trunc(
    base_params: ModelParams,
    base_settings: ModelSettings,
    m_values: List[float],
    metric_key: str = "peak_fwhm_fft",
) -> Dict[str, Any]:
    rows = []
    for m in m_values:
        settings = replace(base_settings, m_trunc=float(m))
        r = run_with_metrics(base_params, settings)
        rows.append({"m_trunc": float(m), metric_key: float(r["spec"][metric_key])})
    return {"parameter": "m_trunc", "metric_key": metric_key, "rows": rows}