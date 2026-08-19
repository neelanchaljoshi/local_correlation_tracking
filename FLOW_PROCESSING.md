# flow_processing — Detailed Documentation

Stage 3 of the pipeline. Concatenates the per-year LCT flow-map HDF5
files (from `lct_pipeline/`) into one 2010–2024 time series per flow
component, cleans it, and writes the `.npy` files that
`inertial_mode_pipeline/` loads.

Like `get_hmi_keys/`, this folder is an unrefactored script: no CLI
flags beyond two positional arguments, no config file, and — as
documented below — its one test is currently broken. This document
describes it exactly as it is.

## Contents

- [What it does](#what-it-does)
- [Files](#files)
- [The FlowData pipeline, step by step](#the-flowdata-pipeline-step-by-step)
- [How to run it](#how-to-run-it)
- [Input data](#input-data)
- [Output data](#output-data)
- [How this feeds into inertial_mode_pipeline](#how-this-feeds-into-inertial_mode_pipeline)
- [Known rough edges](#known-rough-edges)

---

## What it does

`lct_pipeline/` writes one HDF5 file per year (or per month/chunk, if
using the newer chunked pipeline — but this stage currently only knows
how to read the older per-year layout, see
[Known rough edges](#known-rough-edges)). This stage:

1. Loads and concatenates 15 years (2010–2024) of one flow component
   (`uphi` or `utheta`) into a single `(nt, nlat, nlng)` array.
2. Removes the per-pixel time-median (the static flow pattern).
3. Rejects >3σ outliers (median-absolute-deviation based) as NaN.
4. Fits and removes an annual + semi-annual sinusoidal systematic
   (caused by the spacecraft's varying B0 angle/orbital velocity) from
   every pixel's time series independently.
5. Saves the cleaned array as a single `.npy` file, and optionally
   produces diagnostic histogram/time-series/map plots along the way.

## Files

| File | Role |
|---|---|
| `main.py` | Entrypoint: `python main.py <which_flow> <which_data>`. Hardcodes the processing sequence and plot indices. |
| `flow_data.py` | The `FlowData` class — all the actual logic. |
| `utils/io_utils.py` | `save_flow_array()` — the output-naming convention. |
| `utils/fitting.py` | `sin_fit()` — the annual+semi-annual model function. |
| `utils/plotting.py` | `make_plot()` — the four diagnostic plot types. |
| `tests/test_flow_data.py` | One integration-style test — see [Known rough edges](#known-rough-edges), it does not currently pass. |

No `.ini`/`.yaml`/`.json` config, no SLURM script, no `requirements.txt`,
no `conftest.py`, not run by CI.

## The FlowData pipeline, step by step

```python
flow = FlowData(which_flow, which_data)
flow.getdata()
flow.remove_median()
flow.outlier_rejection(3)
flow.remove_yearly_variation()
flow.save()
```
(`main.py` also calls `.plot()` between several of these steps — see
[Output data](#output-data).)

### `getdata()`

Loops years 2010–2024, opening
`/scratch/seismo/joshin/pipeline-test/IterativeLCT/{which_data}/20{YY}_ntry_3_grid_len_5_dspan_6_dstep_30_extent_73.hdf5`
for each, reading `tstart`, `<which_flow>`, `latitude`, `longitude`,
and concatenating along the time axis. Converts `tstart` (TAI strings)
to decimal years via `astropy.time.Time(..., scale='tai').decimalyear`.

**This filename template is hardcoded to one specific LCT run naming
convention (granulation-tracking parameters) and must be hand-edited**
to switch datasets — there's a second, commented-out template line
directly above it for `hmi.m_720s`-style runs
(`20{YY}_dt_1h_dspan_6h_dstep_120m.hdf5`). Whichever one is active
must match both the actual files on disk under `{which_data}/` and the
corresponding line in `utils/io_utils.py` (see below) — the two are
not derived from a shared config and can drift out of sync.

### `remove_median()`

`self.median = nanmedian(flow_array, axis=0)`; subtracts it in place —
removes the static, time-averaged flow pattern (differential rotation
residual, meridional circulation) so what remains is the time-varying
signal.

### `outlier_rejection(threshold)`

Computes the median absolute deviation per pixel
(`scipy.stats.median_abs_deviation`), converts it to an
approximately-Gaussian-equivalent σ via `k = 1.4826`, and sets any
point with `|flow| > threshold · k · MAD` to NaN. Called with
`threshold=3` from `main.py` (a ~3σ clip). Note this thresholds the
**median-removed** flow directly — it only gives a mean-centered clip
because `remove_median()` already ran.

### `remove_yearly_variation()`

Fits `sin_fit(t, a, b, c, d, e) = a·sin(2πt) + b·cos(2πt) + c·sin(4πt) + d·cos(4πt) + e`
(period-1-year + period-6-month harmonics, `t` in decimal years) to
the observed `crlt_obs` time series first, to get a starting-guess
`p0` for every pixel's fit. Then, independently for each of the
73×73 = 5,329 spatial pixels: skips the pixel (sets its whole time
series to NaN) if fewer than 10 valid (non-NaN) points remain after
outlier rejection; otherwise fits `sin_fit` to that pixel's valid
points via `scipy.optimize.curve_fit` and subtracts the fitted curve,
evaluated over the *full* time axis, from the pixel's data in place.
Serial, one `curve_fit` call per pixel (5,329 total).

### `save()`

Delegates to `utils.io_utils.save_flow_array()` — see
[Output data](#output-data) for the exact naming convention.

## How to run it

Flat imports (`from flow_data import FlowData`,
`from utils.fitting import sin_fit`), so this must run with this
folder as the working directory:

```bash
cd flow_processing
# Before running: check that flow_data.py::getdata's active HDF5
# filename template and utils/io_utils.py's naming suffix both match
# the dataset you're about to process (see "Known rough edges" below).
python main.py uphi hmi.ic_45s
python main.py utheta hmi.ic_45s
```

Run it **once per flow component** (`uphi` and `utheta` separately) —
`inertial_mode_pipeline/` needs both. There's no `--help`; wrong
argument count just prints a usage line and exits.

## Input data

- `/scratch/seismo/joshin/pipeline-test/IterativeLCT/{which_data}/20{YY}_....hdf5`
  for `YY` in `10..24` — one file per year, in the schema
  `lct_pipeline/lct_pipeline/io.py::create_output_hdf5` writes
  (`tstart`, `uphi`, `utheta`, `latitude`, `longitude` datasets).
- `data/{crln_obs,crlt_obs,rsun_obs}.npy` — loaded once at import time
  as `FlowData` class attributes (shared across all instances). Only
  `crlt_obs` is actually used (as the sinusoid-fit seed).

## Output data

- **Flow array**: `data/processed_data/{which_flow}_{which_data_with_underscores}{suffix}_processed.npy`
  — a plain `np.save` of the `(nt, nlat, nlng)` `float32` array
  (currently `(21436, 73, 73)`, ~457 MB, NaN at rejected/gapped
  points). `{suffix}` is either `_granule` or `_dt_1h` depending on
  which line is active in `utils/io_utils.py` — see
  [Known rough edges](#known-rough-edges).
- **Diagnostic plots** (relative to cwd, so `flow_processing/figures_processing/`
  when run as intended): `flow_histogram_{1,2,3}.png`,
  `time_series_{2,4}.png`, `flow_data_plot_5.png`. The numeric suffix
  is just the hardcoded call-site index from `main.py`, not a
  meaningful version number. `time_series` always plots the single
  hardcoded pixel `[:, 35, 35]` (center of the 73×73 grid);
  `flow_data_plot` always plots the hardcoded frame index `1576`.

## How this feeds into inertial_mode_pipeline

`inertial_mode_pipeline/inertial_mode_pipeline/io.py::load_flow_data(data_name)`
loads exactly
`PROC_DATA / f'uphi_{data_name}_processed.npy'` and the `utheta`
equivalent, where `PROC_DATA` =
`data/processed_data/`. So `data_name` must match this stage's output
naming exactly, including the `_granule`/`_dt_1h` suffix — e.g. to use
the granulation-tracking output, `run_pipeline.py`'s `data` argument
must be `hmi.ic_45s_granule` (dots become underscores, so
`data_name = hmi_ic_45s_granule`).

## Known rough edges

- **Two hardcoded, must-match-by-hand switches.** Selecting which LCT
  dataset to process requires toggling a commented-out line in both
  `flow_data.py::getdata` (the input filename template) and
  `utils/io_utils.py::save_flow_array` (the output suffix) — nothing
  ties them together, so it's possible to read one dataset's HDF5
  files and accidentally label the output as the other.
- **Input path doesn't match either `.ini`'s configured output.**
  This reads from `/scratch/seismo/joshin/pipeline-test/IterativeLCT/{which_data}/`,
  which is where the legacy (pre-refactor) LCT code wrote its output —
  not `lct_pipeline/config/granulation.ini`'s `rootdir_out`
  (`/data/seismo/joshin/pipeline-test/pmi_test/supergranular_flow/final_sg_compare/data`)
  or `magnetic.ini`'s. If you generate new LCT output with the current
  `lct_pipeline/`, you'll need to either copy/symlink it into the
  `IterativeLCT/` layout this stage expects, or update `getdata()`'s
  path template to point at your actual `rootdir_out`.
- **Only reads the older per-year, all-months-in-one-file HDF5
  layout** — it has no support for the monthly files
  `lct_pipeline/pipeline.py` (MPI mode) produces, nor the
  one-row-per-chunk files `pipeline_chunk.py` (non-MPI mode) produces.
  Bridging either of those newer layouts into what this stage expects
  would need new concatenation logic here.
- **The one test doesn't currently pass.** `tests/test_flow_data.py`
  calls `FlowData("uphi", "hmi.m_720s").getdata()`, but the *active*
  filename template in `getdata()` is the granulation-tracking one —
  the actual files under `hmi.m_720s/` use different naming, so
  `getdata()` raises before the test's first assertion. It's also not
  a unit test in the usual sense: it reads all 15 real HDF5 files, runs
  the full 5,329-pixel curve-fit loop, and **overwrites the real
  production `.npy` output** — running it is not side-effect-free.
  Not part of CI.
- **Hardcoded diagnostic-plot parameters**: histogram range `±2000
  m/s`, time-series pixel `(35, 35)`, sample-frame index `1576` — none
  configurable without editing `utils/plotting.py`.
- **No config file, no CLI flags beyond the two positional
  arguments** — cadence, outlier threshold, minimum-valid-points
  cutoff, and the input/output path templates all require editing
  source directly.
