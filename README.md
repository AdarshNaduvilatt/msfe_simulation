# MSFE Simulator

Simulation tool for **mid-spatial-frequency error (MSFE)** patterns generated from a grinding process. The project includes a Streamlit user interface, core simulation model, plotting utilities, sensitivity analysis, convergence helpers, and validation tools for measured circular-profile data.

## Features

- Process-parameter-driven MSFE simulation
- Spiral toolpath generation with approximately constant arc-length sampling
- Gaussian or skewed spectral peak distribution
- Contact-footprint smoothing using either `r` and `h` or direct `sigma_xy`
- 1D signal, FFT spectrum, and 2D surface generation
- Streamlit UI for single simulations and one-factor-at-a-time sensitivity analysis
- Validation utilities for measured Mahr polar profile text files

## Project structure

```text
msfe-simulator/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── streamlit_app.py
└── src/
    └── msfe_simulator/
        ├── __init__.py
        ├── app.py
        ├── cli.py
        ├── model.py
        ├── metrics.py
        ├── plot_generation.py
        ├── sensitivity.py
        └── validation.py
```

## Installation

## 1. Clone the repository

```bash
git clone https://github.com/AdarshNaduvilatt/msfe_simulation.git
cd msfe-simulation
```

## 2. Create a Conda environment

```bash
conda create -n msfe python=3.10 -y
conda activate msfe
```

(You can use Python 3.10–3.12, but 3.11 is recommended.)

## 3. Install the package

```bash
pip install -e .
```

This installs the package and all required dependencies from `pyproject.toml`.


## Run the Streamlit app

After installation, launch the app with:

```bash
msfe-app
```

Alternative local command:

```bash
streamlit run streamlit_app.py
```

## Minimal Python usage

```python
from msfe_simulator import ModelParams, ModelSettings, simulate_msfe

params = ModelParams(
    R_mm=31.0,
    v_mm_per_min=5000.0,
    n_rpm=1912.0,
    peak_type="gaussian",
    FWHMrel=0.20,
    alpha=0.0,
    a=0.8,
    b=1.2,
    z_rms_um=0.12,
    r_mm=100.0,
    h_um=10.0,
    random_seed=42,
)

settings = ModelSettings(
    spiral_step_mm=0.05,
    Nf=300,
    m_trunc=3.0,
    grid_pitch_mm=0.05,
)

result = simulate_msfe(params, settings)
print(result["meta"])
```

## Main parameters

| Parameter | Meaning |
|---|---|
| `R_mm` | Aperture radius in mm |
| `v_mm_per_min` | Tool path/feed speed in mm/min |
| `n_rpm` | Tool rotational speed in rpm |
| `FWHMrel` | Relative spectral peak width, `FWHM / f0` |
| `a`, `b` | Lower and upper bounds of the random amplitude multiplier |
| `z_rms_um` | Target RMS of the generated 1D MSFE signal |
| `r_mm`, `h_um` | Contact-footprint parameters used to derive effective smoothing width |
| `sigma_xy_mm` | Direct smoothing-width input, used when `r_mm` and `h_um` are not supplied |
| `spiral_step_mm` | Radial step between neighboring spiral turns |
| `Nf` | Number of sampled frequency components |
| `m_trunc` | Frequency truncation range in multiples of sigma |
| `grid_pitch_mm` | 2D surface grid spacing |
