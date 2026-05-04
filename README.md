# SolarCar_MPC_Optimizer

This repository contains a Python simulation and control script for an electric-vehicle endurance race strategy problem:

> Drive as far as possible in **50 hours** using at most **50 kWh** of usable energy.

The main script, `ev_race_50h_mpc_hybrid67_updated_coeffs.py`, evaluates three pacing strategies on the ASC 2024 full base route:

1. **Best constant-speed sweep**
2. **Adaptive constant-power baseline**
3. **Hybrid 67 grade-aware MPC**

The Hybrid 67 controller is a receding-horizon, candidate-search MPC. At each control update, it evaluates both terrain-aware power-shaping candidates and direct grade-based speed-law candidates, then applies only the first command from the best predicted candidate.

---

## Repository layout

Recommended layout:

```text
.
├── ev_race_50h_mpc_hybrid67_updated_coeffs.py
├── requirements.txt
├── README.md
└── data/
    └── asc_24/
        └── 0_FullBaseRoute.gpx
```

The GPX file can either be provided locally or downloaded automatically by the script.

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick start

### Option 1: Run with a local GPX file

If `0_FullBaseRoute.gpx` is in the current directory:

```bash
python ev_race_50h_mpc_hybrid67_updated_coeffs.py \
  --gpx 0_FullBaseRoute.gpx \
  --out-prefix hybrid67_updated \
  --no-plots
```

If it is stored under `data/asc_24/`:

```bash
python ev_race_50h_mpc_hybrid67_updated_coeffs.py \
  --gpx data/asc_24/0_FullBaseRoute.gpx \
  --out-prefix hybrid67_updated \
  --no-plots
```

### Option 2: Let the script download the ASC 2024 route

```bash
python ev_race_50h_mpc_hybrid67_updated_coeffs.py \
  --download-asc24 \
  --out-prefix hybrid67_updated \
  --no-plots
```

Remove `--no-plots` to also generate PNG plots.

---

## Full run with plots

```bash
python ev_race_50h_mpc_hybrid67_updated_coeffs.py \
  --gpx data/asc_24/0_FullBaseRoute.gpx \
  --out-prefix hybrid67_updated
```

This writes CSV summaries and plot images using the selected output prefix.

---

## Output files

For `--out-prefix hybrid67_updated`, the script writes:

```text
hybrid67_updated_summary.csv
hybrid67_updated_constant_speed_sweep.csv
hybrid67_updated_route_profile.csv
hybrid67_updated_mpc_history.csv
```

When plots are enabled, it also writes:

```text
hybrid67_updated_distance.png
hybrid67_updated_energy.png
hybrid67_updated_speed.png
hybrid67_updated_power.png
hybrid67_updated_grade.png
hybrid67_updated_speed_sweep.png
```

### Key output files

#### `hybrid67_updated_summary.csv`

High-level comparison of the three strategies:

- strategy name
- distance traveled
- final energy
- finish time
- average speed

This is the primary file to inspect when comparing controller performance.

#### `hybrid67_updated_mpc_history.csv`

Detailed controller diagnostics for the Hybrid 67 MPC. This file records the chosen candidate type, command speed, state of charge, power, and MPC prediction diagnostics at each control update.

Useful columns include:

- `time_s`
- `position_m`
- `energy_kwh`
- `candidate_kind`
- `command_speed_mps`
- `instant_power_w`
- `power_bias`
- `power_grade_gain`
- `speed_base_mps`
- `speed_grade_gain`
- `objective`

---

## Controller description

The Hybrid 67 MPC evaluates two candidate families at each MPC update.

### 1. Power-aware terrain candidates

These start from the adaptive constant-power rule:

```text
adaptive_power = remaining_energy / remaining_time
```

Then they apply a grade-dependent terrain multiplier:

```text
target_power = adaptive_power * power_bias * exp(-power_grade_gain * grade)
```

Negative `power_grade_gain` values spend more power on uphills and less on downhills.

### 2. Direct speed-law candidates

These use a direct grade-aware speed rule:

```text
speed = base_speed_mps - grade_gain * grade
```

Positive grades reduce speed, while negative grades increase speed.

### Hybrid selection

For every control update, the controller rolls each candidate forward over the prediction horizon, scores the predicted trajectory, and applies only the first command from the best candidate. This is a receding-horizon MPC design.

---

## Current vehicle model

The updated script uses the following vehicle coefficients:

```python
mass_kg = 405.0
drag_coefficient = 0.31
frontal_area_m2 = 1.228
air_density_kg_m3 = 1.225
rolling_resistance_coeff = 0.010

drivetrain_efficiency = 0.88

motor_copper_loss_coeff_w_per_kw_exp = 55.0
motor_copper_loss_exponent = 2.0

regen_efficiency = 0.35
max_regen_power_w = 500.0
aux_power_w = 0.0
```

---

## Important command-line options

```text
--gpx PATH                     Path to a GPX route file.
--gpx-dir DIR                  Directory containing 0_FullBaseRoute.gpx.
--download-asc24               Download the ASC 2024 full base route.
--overwrite-download           Redownload the GPX file even if it already exists.
--gpx-spacing-m VALUE          Route resampling spacing in meters. Default: 100.
--gpx-grade-smoothing-m VALUE  Elevation smoothing window before grade calculation. Default: 500.
--out-prefix PREFIX            Prefix for all output files. Default: hybrid67_updated.
--no-plots                     Skip PNG plot generation.
```

The script also includes `--tune-mpc`, but the Hybrid 67 candidate grid is already configured as the main controller. Use tuning only for additional experimentation.

---

## Example workflow

Run the controller:

```bash
python ev_race_50h_mpc_hybrid67_updated_coeffs.py \
  --gpx data/asc_24/0_FullBaseRoute.gpx \
  --out-prefix hybrid67_updated \
  --no-plots
```

Inspect the summary:

```bash
cat hybrid67_updated_summary.csv
```

Inspect how often each MPC candidate family was selected:

```bash
python - <<'PY'
import pandas as pd
h = pd.read_csv("hybrid67_updated_mpc_history.csv")
print(h["candidate_kind"].value_counts())
PY
```

Generate plots by rerunning without `--no-plots`:

```bash
python ev_race_50h_mpc_hybrid67_updated_coeffs.py \
  --gpx data/asc_24/0_FullBaseRoute.gpx \
  --out-prefix hybrid67_updated
```

---

## Notes on runtime

Hybrid 67 is more expensive than a simple constant-power strategy because it evaluates multiple candidate controllers at each MPC update. Runtime depends on hardware, route resolution, horizon length, and whether plots are generated.

For faster experiments, use:

```bash
--no-plots
```

For route-resolution experiments, adjust:

```bash
--gpx-spacing-m
--gpx-grade-smoothing-m
```

Larger route spacing generally runs faster but may smooth out important terrain detail.
