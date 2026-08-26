# flow_processing — Detailed Documentation

Stage 3 of the pipeline. Concatenates the LCT flow-map HDF5 files
(from `lct_pipeline/` — any granularity: legacy per-year, or the
current pipeline's per-month/per-chunk output) into one time series per
flow component, cleans it, and writes the `.npy` files that
`inertial_mode_pipeline/` loads.

Like `get_hmi_keys/`, this folder started as an unrefactored script —
no config file, minimal structure. `getdata()`/`save()` now take
explicit `data_root`/`pattern`/`suffix` parameters (`main.py`:
`--data-root`/`--pattern`/`--out-suffix`) instead of source lines that
had to be hand-edited to switch datasets; see
[Known rough edges](#known-rough-edges) for what's still rough.

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

`lct_pipeline/` writes flow-map HDF5 files at one of three
granularities depending on which pipeline mode produced them: one file
per year (legacy, pre-refactor), one file per month (`pipeline.py`/MPI
mode), or one file per chunk (`pipeline_chunk.py`/non-MPI mode). This
stage:

1. Loads and concatenates however many files match `pattern` under
   `data_root` (any of the three granularities, sorted by time) of one
   flow component (`uphi` or `utheta`) into a single `(nt, nlat, nlng)`
   array.
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
| `main.py` | Entrypoint: `python main.py <which_flow> <which_data> [--data-root] [--pattern] [--out-suffix]`. Hardcodes the processing sequence and plot indices. |
| `flow_data.py` | The `FlowData` class — all the actual logic, plus `LEGACY_ROOT`/`LEGACY_GLOB_PATTERNS` (default input location/selection for known legacy datasets). |
| `utils/io_utils.py` | `save_flow_array()` — the output-naming convention, plus `LEGACY_OUTPUT_SUFFIX` (default output suffix for known legacy datasets). |
| `utils/fitting.py` | `sin_fit()` — the annual+semi-annual model function. |
| `utils/plotting.py` | `make_plot()` — the four diagnostic plot types. |
| `tests/test_flow_data.py` | One integration-style test against real data — see [Known rough edges](#known-rough-edges), it now passes but is not side-effect-free. |
| `tests/test_getdata_synthetic.py` | `getdata()`'s glob/sort/error-handling logic against synthetic fixtures — fast, safe to run anytime. |
| `tests/test_io_utils.py` | `save_flow_array()`'s suffix logic, patched to a temp directory — fast, safe to run anytime. |

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

### `getdata(data_root=None, pattern=None)`

Globs `data_root/pattern`, reads `tstart`, `<which_flow>`, `latitude`,
`longitude` from every matching file, and concatenates them along the
time axis — **sorted by each file's own recorded timestamps, not by
filename**, so it works the same way regardless of whether the source
is one file per year (the legacy pre-refactor LCT pipeline), one file
per month (the current `lct_pipeline`'s MPI/month mode, `pipeline.py`),
or one file per chunk (the current `lct_pipeline`'s non-MPI/chunk mode,
`pipeline_chunk.py`) — all three write the same
`tstart`/`uphi`/`utheta`/`latitude`/`longitude` schema
(`lct_pipeline.io.create_output_hdf5`). Converts `tstart` (TAI strings)
to decimal years via `astropy.time.Time(..., scale='tai').decimalyear`.
Raises `ValueError` if a matched file's `latitude`/`longitude` grid
doesn't match the others (a signal you've globbed together files from
two different runs), and `FileNotFoundError` if nothing matches.

- **`data_root`** (optional): directory to search. Defaults to the
  legacy `IterativeLCT/{which_data}/` layout, for exact backward
  compatibility with existing callers. For current `lct_pipeline`
  output, pass the same `rootdir_out` from the `.ini` config used to
  produce it, e.g. `.../local_correlation_tracking/data/magnetic`.
- **`pattern`** (optional): glob pattern, relative to `data_root`,
  selecting which files to read. A single directory can hold more than
  one run's files side by side with different naming/cadence — e.g.
  `IterativeLCT/hmi.m_720s/` has both a `_dt_1h_dspan_6h_dstep_120m`
  run and an unrelated `_ntry_3_..._extent_73_new` run for the same
  years — so `which_data` alone isn't enough to disambiguate. Defaults
  to `LEGACY_GLOB_PATTERNS[which_data]` (`flow_data.py`, currently
  covers `hmi.ic_45s` and `hmi.m_720s`) when `data_root` isn't given;
  **raises `ValueError`** if omitted for any other `which_data`, since
  there's no safe default to guess. For current `lct_pipeline` output:
  `*_gran_dspan*_4k.hdf5` (month mode, granulation, 4k) or the same
  with `_chunk` before `.hdf5` (chunk mode) — swap `gran`→`mag` and
  `4k`→`2k` per the config's `segname`/`downsample`.

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

### `save(suffix=None)`

Delegates to `utils.io_utils.save_flow_array()` — see
[Output data](#output-data) for the exact naming convention.
`suffix` defaults to `LEGACY_OUTPUT_SUFFIX[which_data]`
(`utils/io_utils.py`) for known legacy datasets; required (raises
`ValueError`) otherwise. Keyed by the same `which_data` string as
`getdata()`'s `LEGACY_GLOB_PATTERNS`, so input selection and output
naming can no longer silently drift out of sync for the legacy case —
previously two independent hardcoded/commented-out lines had to be
hand-toggled together.

## How to run it

Flat imports (`from flow_data import FlowData`,
`from utils.fitting import sin_fit`), so this must run with this
folder as the working directory:

```bash
cd flow_processing
# Legacy pre-refactor LCT output (unchanged usage):
python main.py uphi hmi.ic_45s
python main.py utheta hmi.ic_45s

# Current lct_pipeline output (month mode, granulation, 4k) — point at
# the same rootdir_out the .ini config used to produce it:
python main.py uphi hmi.ic_45s \
    --data-root /data/seismo/joshin/pipeline-test/local_correlation_tracking/data/granulation \
    --pattern '*_gran_dspan*_4k.hdf5' \
    --out-suffix _dspan24h_dstep30m_4k

# Current lct_pipeline output (chunk mode, magnetic, 4k):
python main.py uphi hmi.m_720s \
    --data-root /data/seismo/joshin/pipeline-test/local_correlation_tracking/data/magnetic \
    --pattern '*_mag_dspan*_4k_chunk.hdf5' \
    --out-suffix _chunk_dspan6h_dstep2h_4k
```
`--out-suffix` is whatever you'll then use as the trailing part of
`run_pipeline.py`'s `data` argument for
`inertial_mode_pipeline/` — pick something that uniquely identifies
this specific LCT run (dspan/dstep/resolution), since `which_data`
alone doesn't.

Run it **once per flow component** (`uphi` and `utheta` separately) —
`inertial_mode_pipeline/` needs both. `--help` lists all flags; wrong
positional argument count still exits with a usage message.

## Input data

- **Legacy** (default when `--data-root`/`--pattern` are omitted):
  `/scratch/seismo/joshin/pipeline-test/IterativeLCT/{which_data}/20{YY}_....hdf5`
  for `YY` in `10..24` — one file per year.
- **Current `lct_pipeline`**: whatever `--data-root`/`--pattern` you
  pass — one file per month (`pipeline.py`/MPI mode) or one file per
  chunk (`pipeline_chunk.py`/non-MPI mode), each in the same schema
  `lct_pipeline/lct_pipeline/io.py::create_output_hdf5` writes
  (`tstart`, `uphi`, `utheta`, `latitude`, `longitude` datasets) —
  `getdata()` concatenates however many files match, sorted by time,
  regardless of granularity (see [`getdata()`](#getdatadata_rootnone-patternnone)).
- `data/{crln_obs,crlt_obs,rsun_obs}.npy` — loaded once at import time
  as `FlowData` class attributes (shared across all instances). Only
  `crlt_obs` is actually used (as the sinusoid-fit seed).

## Output data

- **Flow array**: `data/processed_data/{which_flow}_{which_data_with_underscores}{suffix}_processed.npy`
  — a plain `np.save` of the `(nt, nlat, nlng)` `float32` array (for
  the legacy 2010–2024 run, `(21436, 73, 73)`, ~457 MB, NaN at
  rejected/gapped points). `{suffix}` is `--out-suffix`
  (`main.py`) / `save(suffix=...)`, defaulting to
  `LEGACY_OUTPUT_SUFFIX[which_data]` (`_granule` for `hmi.ic_45s`,
  `_dt_1h` for `hmi.m_720s`) for known legacy datasets — pick your own
  for current `lct_pipeline` output, see [How to run it](#how-to-run-it).
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

- ~~Two hardcoded, must-match-by-hand switches~~ **Fixed.**
  `getdata(data_root, pattern)` and `save(suffix)` are now explicit
  parameters (`--data-root`/`--pattern`/`--out-suffix` on `main.py`)
  instead of commented-out source lines you had to toggle by hand in
  two different files. The legacy defaults (`LEGACY_GLOB_PATTERNS` in
  `flow_data.py`, `LEGACY_OUTPUT_SUFFIX` in `utils/io_utils.py`) are
  both keyed by the same `which_data` string, so they can't drift out
  of sync for the datasets they cover.
- ~~Input path doesn't match either `.ini`'s configured output~~
  **Fixed for finding it — not for the layout mismatch (below).**
  `--data-root`/`--pattern` (or `getdata(data_root=, pattern=)`
  directly) now point at any directory, including a `rootdir_out` from
  the current `lct_pipeline`'s `.ini` configs.
- ~~Only reads the older per-year, all-months-in-one-file HDF5
  layout~~ **Fixed.** `getdata()` globs `pattern` under `data_root` and
  concatenates *every* matching file, sorted by each file's own
  recorded timestamps rather than by filename — this handles the
  legacy one-file-per-year layout, `pipeline.py`'s one-file-per-month
  layout, and `pipeline_chunk.py`'s one-file-per-chunk layout
  identically, since all three share the same HDF5 schema. See
  [`getdata()`](#getdatadata_rootnone-patternnone).
- **No real current-`lct_pipeline` output exists yet to point this
  at.** As of this writing, neither `granulation.ini`'s nor
  `magnetic.ini`'s `rootdir_out` has ever been populated by an actual
  `main.py`/`main_chunk.py` run — so the new-pipeline code paths above
  are validated against synthetic fixtures
  (`tests/test_getdata_synthetic.py`) rather than real files. Once a
  real run exists, double-check the actual filenames it produced
  against the `--pattern` you pass — `output_filename()`/
  `chunk_output_filename()` in `lct_pipeline/config.py`/
  `pipeline_chunk.py` are the source of truth.
- **The legacy test still reads real data and isn't side-effect-free.**
  `tests/test_flow_data.py`'s `FlowData("uphi", "hmi.m_720s").getdata()`
  now works (previously it raised before the first assertion — see the
  history of this doc), but the test still reads all 15 real HDF5
  files, runs the full 5,329-pixel curve-fit loop, and **overwrites the
  real production `.npy` output** when it calls `.save()`. Not part of
  CI; don't run it casually. `tests/test_getdata_synthetic.py` and
  `tests/test_io_utils.py` cover the same logic with synthetic data and
  no side effects, and are safe to run anytime:
  `cd flow_processing && python -m pytest tests/test_getdata_synthetic.py tests/test_io_utils.py -v`.
- **Hardcoded diagnostic-plot parameters**: histogram range `±2000
  m/s`, time-series pixel `(35, 35)`, sample-frame index `1576` — none
  configurable without editing `utils/plotting.py`.
- **No config file, no CLI flags beyond the two positional
  arguments** — cadence, outlier threshold, minimum-valid-points
  cutoff, and the input/output path templates all require editing
  source directly.
