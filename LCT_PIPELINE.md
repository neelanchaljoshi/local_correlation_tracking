# LCT Pipeline — Detailed Documentation

This document describes `lct_pipeline/` end to end: every module, every
config key, the physics/algorithms used at each step, the two execution
modes (MPI and non-MPI chunk-based), output file formats, and the full
CLI/SLURM reference. See the main [README](README.md) for a one-line
quickstart; this file is the exhaustive reference.

## Contents

- [What the pipeline does](#what-the-pipeline-does)
- [Module map](#module-map)
- [Configuration reference (`.ini` files)](#configuration-reference-ini-files)
- [Processing pipeline, step by step](#processing-pipeline-step-by-step)
- [Algorithms in detail](#algorithms-in-detail)
- [Execution modes](#execution-modes)
- [Output file formats](#output-file-formats)
- [CLI reference](#cli-reference)
- [SLURM reference](#slurm-reference)
- [Testing](#testing)
- [Known limitations / gotchas](#known-limitations--gotchas)

---

## What the pipeline does

Local Correlation Tracking (LCT) measures horizontal plasma flow
velocities on the solar surface by cross-correlating pairs of images
(SDO/HMI continuum for granulation, or magnetograms for magnetic
features) taken a short time apart. For a grid of points on the solar
disk, the pipeline:

1. Remaps a small patch of each image around that point onto a common
   Postel (azimuthal equidistant) projection, correcting for the
   spacecraft's changing view angle and, optionally, for differential
   rotation.
2. Cross-correlates the two remapped patches.
3. Fits the cross-correlation peak to sub-pixel precision and converts
   the peak displacement into a velocity in m/s.
4. Repeats over many image pairs within a time window (`dspan`),
   averages the cross-correlation functions before fitting (not the
   velocities — averaging happens on the CCF itself, which is the more
   statistically robust order of operations), and writes one flow map
   per window.

The same physics runs in two different parallelization strategies —
MPI-across-space (`pipeline.py`) and non-MPI-across-time
(`pipeline_chunk.py`) — described in [Execution modes](#execution-modes).

---

## Module map

| File | Role |
|---|---|
| `lct_pipeline/config.py` | Parses `.ini` files into a typed `Config` dataclass. All other modules take a `Config` and never touch `configparser` directly. |
| `lct_pipeline/geometry.py` | PSF construction, HMI→PMI image simulation, differential rotation rate, B0/P-angle correction, and the Postel remap wrapper. |
| `lct_pipeline/interpolation.py` | Cubic-spline interpolation of a 4-image stack to an arbitrary target time (used only when `interpolate=1`). |
| `lct_pipeline/lct.py` | The actual LCT math: 2D Tukey window, FFT cross-correlation, ellipsoid sub-pixel peak fitting, velocity conversion. Pure functions, no I/O. |
| `lct_pipeline/io.py` | All file I/O: FITS keys table loading, FITS image reading (with retries), HDF5 output creation/writing, CCF `.npy` saving. |
| `lct_pipeline/mpi_utils.py` | Rank-safe logging setup, an MPI-gather replacement that avoids hangs on large communicators, log-level parsing. |
| `lct_pipeline/pipeline.py` | Orchestrates the **MPI** pipeline: one SLURM array task per calendar month, MPI ranks split the spatial patch grid. |
| `lct_pipeline/pipeline_chunk.py` | Orchestrates the **non-MPI** pipeline: one SLURM array task per time chunk, single process walks the whole patch grid. |
| `main.py` | CLI entrypoint for the MPI pipeline. |
| `main_chunk.py` | CLI entrypoint for the non-MPI chunk pipeline (month mode and range mode). |
| `run_slurm.sh` | SLURM submission script for `main.py`. |
| `run_slurm_chunk.sh` | SLURM submission script for `main_chunk.py`. |
| `config/granulation.ini`, `config/magnetic.ini` | The two production configs (continuum/granulation tracking and magnetogram/magnetic-feature tracking). |

`geometry.py`'s `remap_patches` and `pipeline.py`'s MPI-distribution
step both import `zclpy3` (`get_delimiters`, `from_tan_to_postel`) from
a hardcoded path, `/data/seismo/zhichao/codes/pypkg` — an MPS-internal
package not on PyPI. Tests mock this import (see
[Testing](#testing)); a real run needs that path to exist.

---

## Configuration reference (`.ini` files)

Every key below is parsed in `config.py`'s `load_config()`. Keys with
no listed default are **required** — `load_config` raises `ValueError`
immediately if missing.

### `[job]`

| Key | Type | Meaning |
|---|---|---|
| `yr_start`, `yr_stop` | int | Nominal valid year range for this config (`yr_start > yr_stop` raises `ValueError`). Informational / self-documenting — not itself enforced against the year you actually pass on the CLI. |
| `dspan_hours` | int | Length of one **time chunk**, in hours. Every image pair inside a chunk contributes to one averaged CCF → one flow map row. This is `dspan` throughout the code (`timedelta(hours=dspan_hours)`). |
| `dstep_minutes` | int | Step between successive image-pair timestamps *within* a chunk, in minutes. `dspan / dstep` (rounded) is roughly how many image pairs get averaged into one chunk's CCF. |
| `range_start`, `range_end` | ISO datetime, optional | Explicit chunk range for the **non-MPI range mode** only (`pipeline_chunk.run_chunk_range`). Both must be set together or both omitted; `range_start` must be strictly before `range_end`; both must fall in the same calendar year (the keys table is loaded once per year). See [Execution modes](#execution-modes). |

### `[instrument]`

| Key | Type | Meaning |
|---|---|---|
| `segname` | str | FITS filename to read from each timestamp's directory — `continuum.fits` for granulation, `magnetogram.fits` for magnetic tracking. `Config.is_magnetic` is `'magnetogram' in segname`. |
| `cadence_seconds` | int | Target time separation between the two images in one correlation pair. |
| `dataset_cadence_seconds` | int | Actual time spacing between consecutive rows in the FITS keys table. |
| `NX`, `NY` | int | Full-frame image dimensions (4096×4096 for native HMI resolution). Not directly used in remapping (which reads real image shapes), mainly documentation/sanity metadata. |
| `Ntry` | int | Retry count for FITS reads that fail with `IOError` (5s sleep between attempts). |
| `downsample` | bool (`0`/`1`) | If true, HMI images are convolved with a simulated PMI PSF and 2×2-averaged down to PMI-like resolution before correlation (see [PMI simulation](#psf-construction-and-pmi-simulation)). Also selects `patch_size_2k`/`pixel_size_2k`/`infile_fmt_2k` instead of the `_4k` variants. |
| `interpolate` | bool | If true, uses 4 consecutive frames + cubic-spline interpolation to build a synthetic second image at a fixed target cadence (`cadence_interp = 60`) instead of directly using the frame `njump` steps away. See [Interpolation](#interpolation-mode). |

Derived (not read directly from the file):
- `njump = cadence_seconds // dataset_cadence_seconds` — how many keys-table rows apart the two correlated frames are.
- `cadence_interp = 60 if interpolate else dataset_cadence_seconds` — the time baseline (seconds) used to convert pixel displacement to velocity.

### `[psf]`

| Key | Type | Meaning |
|---|---|---|
| `wavelength_nm` | float | Observation wavelength, converted to meters internally (`wavelength_m = wavelength_nm * 1e-9`). 617.3 nm = Fe I line used by both HMI and PMI. |
| `aperture_hmi_m`, `aperture_pmi_m` | float | Telescope aperture diameters [m], used to compute each instrument's diffraction-limited Airy disk radius. |
| `pixel_scale_arcsec` | float | Native pixel scale [arcsec/pixel], assumed the same for HMI and PMI at this stage. |
| `psf_size` | int | Side length [pixels] of the square PSF kernel array. |

Only used when `downsample=1` (PMI simulation needs the relative PSF);
computed unconditionally at startup regardless (`build_psfs` always
runs) but only consumed downstream if downsampling is on.

### `[grid]`

| Key | Type | Meaning |
|---|---|---|
| `clat_start`, `clat_stop`, `clat_step` | float | Latitude sampling grid [degrees], built via `np.arange(start, stop + 1e-9, step)` — the `+1e-9` guards against floating-point step accumulation excluding the endpoint. |
| `clng_start`, `clng_stop`, `clng_step` | float | Same, for longitude. |

`granulation.ini` uses a narrow ±10°/0.1° grid (201×201 = 40,401
points) — fine spatial resolution near disk center for supergranulation
studies. `magnetic.ini` uses a coarser ±60°/2.5° grid (49×49 = 2,401
points) — wider disk coverage, coarser resolution, appropriate for
larger-scale magnetic features.

### `[lct]`

| Key | Type | Meaning |
|---|---|---|
| `patch_size_4k`, `patch_size_2k` | int | Side length [pixels] of each remapped patch, at full and downsampled resolution respectively. Selected by `downsample`. |
| `pixel_size_4k`, `pixel_size_2k` | float | Pixel scale of the remapped patch [deg/pixel]. |
| `alpha` | float | Tukey window taper fraction for CCF apodization (`0`=rectangular, `1`=Hanning). Both production configs use `0.8`. |
| `R_sun_Mm` | float | Solar radius [Mm], used in the pixel-displacement → velocity conversion. |
| `grid_len` | int | Neighbourhood size (must be odd) for the sub-pixel ellipsoid peak fit around the CCF maximum. |
| `ntry_fit` | int | Number of iterative refinement passes in the peak fit (each pass re-centers the CCF on the current best estimate and re-fits). |

### `[tracking]`

| Key | Type | Meaning |
|---|---|---|
| `change_track` | bool | If true, applies a per-patch, per-timestep Carrington-longitude correction for differential rotation (see [Differential rotation tracking](#differential-rotation-tracking)) so the patch tracks the local rotation rate rather than the fixed Carrington rate. |
| `A`, `B`, `C` | float | Differential rotation profile coefficients [deg/day]: `rate(lat) = A + B·sin²(lat) + C·sin⁴(lat)`. `granulation.ini` uses Snodgrass (1984) coefficients; `magnetic.ini` uses Hathaway (2011) coefficients — granulation and magnetic features rotate at measurably different rates. |
| `CRrate` | float | Fixed Carrington rotation rate [deg/day] (both configs: `14.184`), the rate the base coordinate frame itself rotates at. |

### `[b0_correction]`

| Key | Type | Meaning |
|---|---|---|
| `dI` | float | Amplitude [degrees] of the annual B0/P-angle correction oscillation. |
| `t_ref_b0` | ISO datetime | Reference epoch for the correction's phase. |

### `[paths]`

| Key | Type | Meaning |
|---|---|---|
| `infile_fmt_4k`, `infile_fmt_2k` | strftime-format str | Path to the year's FITS keys table, with `%Y` substituted for the requested year. Selected by `downsample`. |
| `rootdir_out` | path | Output directory for HDF5 flow maps. Created (with parents) at `Config.__post_init__` time. |
| `ccf_dir` | path, optional | Output directory for individual CCF `.npy` dumps. Empty string → `None`. Created at `__post_init__` if set. |

### `[output]`

| Key | Type | Meaning |
|---|---|---|
| `save_ccf` | bool | If true, individual CCF arrays get saved as `.npy` files (see [CCF `.npy` files](#ccf-npy-files)). Requires `ccf_dir` to be set — `load_config` raises `ValueError` otherwise. |
| `ccf_lat_threshold`, `ccf_lng_threshold` | float | Only patches within this many degrees of disk center (`abs(clat) <= threshold` and `abs(clng) <= threshold`) get their CCFs saved — avoids writing tens of thousands of files per timestep. |

---

## Processing pipeline, step by step

Both execution modes ([Execution modes](#execution-modes)) run the same
per-patch, per-timestep physics; they differ only in *how work is
split across processes*. This section describes that shared physics —
concretely, `pipeline.py::_process_patch()`, reused directly by both
`pipeline.py::run()` and `pipeline_chunk.py::_run_chunk_body()`.

For one timestep and one grid point `(clng, clat)`:

1. **Astrometry lookup.** Pull `crpix1/2`, `cdelt1/2`, `rsun_obs` for
   the two (or four, if `interpolate=1`) frames being correlated, from
   the FITS keys table.
2. **B0/P-angle correction.** `compute_b0_correction(t_rec, cfg)`
   returns `(dB, dP)`, added to each frame's `crlt_obs`/`crota2`.
3. **Differential-rotation longitude shift.** If `change_track=1`:
   `carrington_longitude_shift(clat, dt_seconds, cfg)` — the local
   patch's target Carrington longitude is nudged by the difference
   between the local differential-rotation rate at that latitude and
   the fixed Carrington rate, over the elapsed time between frames.
   This is what makes the patch "track" the actual moving feature
   instead of a fixed Carrington-frame location.
4. **Postel remap.** `remap_patches()` (wraps
   `zclpy3.remap.from_tan_to_postel`) reprojects each full-frame image
   into a small `patch_size × patch_size` patch centered at
   `(clng, clat)` in the corrected frame, using bilinear interpolation.
5. **(Interpolation mode only)** the four remapped frames get
   cubic-spline-interpolated to a synthetic frame at `cadence_interp`
   seconds after the first — see [Interpolation](#interpolation-mode).
6. **CCF.** `get_ccf(patch1, patch2, kernel)`: both patches are
   mean-subtracted, apodized with the pre-built 2D Tukey `kernel`, and
   cross-correlated via `rfft2`/`irfft2` (real FFT — patches are
   real-valued, so this is exact and ~2× faster than a complex FFT).
7. Any patch whose CCF contains NaNs is dropped (`_process_patch`
   returns `None`); everything else accumulates into a running sum
   `ccfs[patch] += ccf` and count `nums[patch] += 1`.

After all timesteps in the chunk:

8. **Average.** `ccfs_avg = ccfs / nums` (per patch; patches with
   `nums == 0` — e.g. every timestep failed — get `NaN`).
9. **Sub-pixel peak fit → velocity.** `get_flow_velocity()`
   (see [Sub-pixel peak fitting](#sub-pixel-peak-fitting-and-velocity-conversion))
   converts the averaged CCF into `(ux, uy)` in m/s.
10. **Write.** One row (this chunk) of `(nlat, nlng)` `uphi`/`utheta`
    arrays gets written to the output file.

---

## Algorithms in detail

### PSF construction and PMI simulation

`geometry.airy_disk_psf` builds a diffraction-limited Airy pattern
(`(2·J₁(k)/k)²`, `J₁` the first-order Bessel function) from the
instrument's aperture and wavelength via
`compute_airy_radius_pixels` (standard `1.22·λ/D` Airy radius,
converted from radians to pixels via the pixel scale).

`build_psfs(cfg)` computes three kernels: `psf_hmi`, `psf_pmi`, and
`psf_rel = psf_pmi ⊛ flip(psf_hmi)` — the PSF that would turn an HMI
image into a simulated lower-resolution PMI image (deconvolving HMI's
own blur and reconvolving with PMI's, in one combined kernel).

When `downsample=1`, `simulate_pmi_from_hmi()` convolves each raw HMI
image with `psf_rel` (via `scipy.signal.fftconvolve`) and then does a
2×2 box-mean downsample to PMI's coarser pixel scale — used for the
ESA Vigil/PMI synthetic-parameter analysis (see the related
publication in the README).

### Differential rotation tracking

`differential_rotation_rate(lat, A, B, C) = A + B·sin²(lat) + C·sin⁴(lat)`
[deg/day] — the classic sidereal differential-rotation profile,
fit separately for granulation (Snodgrass 1984) and magnetic features
(Hathaway 2011), since the two trace different depths/physics and
rotate at measurably different rates.

`carrington_longitude_shift(lat, dt_seconds, cfg)` returns
`(rate(lat) - CRrate) · dt_days` — the extra longitude drift, relative
to the fixed Carrington frame, that a feature at that latitude
accumulates over `dt_seconds`. This gets folded into the remap's `dL`
(longitude offset) parameter for the second (and later) frame in a
pair, so the patch effectively re-centers on where the feature should
have rotated to, not a fixed Carrington coordinate.

### B0/P-angle correction

`compute_b0_correction(t_rec, cfg)` models a smooth annual oscillation
in two systematic angle corrections:
```
phase = 2π · (t_rec - t_ref_b0) / 1 year
dB    =  dI · sin(phase)
dP    = -dI · cos(phase)
```
`dB` is added to the observed B0 angle (`crlt_obs`), `dP` to the
negated `crota2` (P angle), before remapping.

### Interpolation mode

When `interpolate=1`, instead of directly correlating the frame at
index `ii` against the frame `njump` steps later, the pipeline reads
**four** frames (`ii`, `ii+njump`, `ii+2·njump`, `ii+3·njump`), remaps
all four, then uses `interpolation.interpolate_image_stack` (cubic
spline, `scipy.interpolate.CubicSpline`, fit independently per pixel
across the 4 time points) to synthesize an image at a fixed target
time `cadence_interp = 60` seconds after the first frame. That
synthetic image is what actually gets correlated against the first
remapped frame. This decouples the velocity measurement's effective
time baseline from the dataset's native cadence, letting datasets with
different native cadences (e.g. `cadence_seconds=45` vs `720`) produce
directly comparable velocities.

### Tukey window and CCF

`lct.tukey_2d(width, alpha)` builds a **radially symmetric** 2D Tukey
window (not a separable product of two 1D windows): a 1D Tukey profile
is evaluated once and then indexed by radius from the patch center, so
the taper is circular rather than square. `build_tukey_kernel()`
computes this once at startup (`alpha` from `[lct] alpha`, `granulation.ini`/`magnetic.ini` both use `0.8`) and reuses it for every patch and
timestep — it's identical everywhere since it only depends on
`patch_size`/`alpha`, both config constants.

`get_ccf(patch1, patch2, kernel)`:
```
patch1w = kernel · (patch1 - mean(patch1))
patch2w = kernel · (patch2 - mean(patch2))
ccf     = fftshift(irfft2(conj(rfft2(patch1w)) · rfft2(patch2w)))
```
i.e. the standard FFT cross-correlation theorem, using the real FFT
(`rfft2`/`irfft2`) since both patches are real-valued.

### Sub-pixel peak fitting and velocity conversion

`_fit_ellipsoid_peak(ccf, grid_len)` fits a 2D quadratic
`f(i,j) = a·i² + b·j² + c·i + d·j + e·ij + const` by least squares
(`scipy.linalg.lstsq`) over a `grid_len × grid_len` neighborhood
centered on the CCF's integer-pixel maximum, then solves the
quadratic's stationary point analytically:
```
ypar = (e·c - 2a·d) / (4ab - e²)
xpar = (-e·ypar - c) / (2a)
```
giving the sub-pixel offset from the integer peak.

`get_flow_velocity()` iterates this `ntry_fit` times: fit the sub-pixel
peak, compute the total displacement from patch center, then
**re-center the CCF** on that estimate (`scipy.ndimage.shift`,
reflect-mode boundary) before the next fit — each pass refines the
estimate around a better-centered peak, rather than re-fitting the
same off-center neighborhood repeatedly. The accumulated total
displacement `(dx_tot, dy_tot)` [pixels] converts to velocity via:
```
u = R_sun_Mm · dx_tot · deg2rad(pixel_size_deg) / cadence_interp · 1e6   [m/s]
```
(`R_sun_Mm · Δθ` = arc length in Mm; `/ cadence_interp` = Mm/s;
`· 1e6` = m/s).

---

## Execution modes

### Mode 1 — MPI (`pipeline.py` / `main.py`)

One SLURM array task per **calendar month**; within that task, MPI
ranks split the **spatial patch grid** (`get_delimiters` divides the
`clat_arr × clng_arr` list across ranks). Every rank reads the same
broadcast image pair, processes its own slice of patches, and results
are `Gatherv`'d back to rank 0 for writing. One HDF5 file per
(year, month), with `nt` rows (one per `dspan` chunk within that
month — e.g. ~30 rows for a 24h `dspan`).

```bash
sbatch --array=1-12 run_slurm.sh config/granulation.ini 2019
```

**Known sharp edge:** `main.py`'s month loop calls
`Config.validate_month(year, month)`, which does
`datetime(year, month, 1)` — for `month` outside 1–12 (e.g. submitting
`--array=1-25`), this raises an **uncaught `ValueError`**, crashing
that array task with a traceback. There is no bounds check before it.
Always size `--array` to exactly `1-12`.

### Mode 2 — non-MPI chunk pipeline (`pipeline_chunk.py` / `main_chunk.py`)

One SLURM array task per **time chunk** (one `dspan` window — a day at
`dspan_hours=24`, an hour at `dspan_hours=1`, whatever the config
sets). No MPI: each task is a single plain-Python process that walks
the *entire* patch grid itself, sequentially, and writes its own small
one-row output file. Tasks share no state and need no inter-task
communication — trading the MPI mode's spatial parallelism for
temporal parallelism at the SLURM-array level.

Two sub-modes, chosen by whether `year`/`month` are given on the
command line:

**Month mode** — chunks span one calendar month, same as MPI mode's
month scoping, just chunk-granular instead of whole-month-per-task:
```bash
python main_chunk.py config/granulation.ini 2019 6 --print-nchunks   # -> 30
sbatch --array=1-30 run_slurm_chunk.sh config/granulation.ini 2019 6
```
Chunk index is **relative to the start of the month** — chunk 1 is
always hour/day 1 of the 1st of the month, not of whatever day you
actually care about. To target one specific day within a month at
hourly granularity, you'd need to compute
`offset = (day - 1) * 24` yourself and submit
`--array=(offset+1)-(offset+24)`. Range mode below exists specifically
to avoid that arithmetic.

**Range mode** — chunks span an explicit `range_start`/`range_end`
set in the config's `[job]` section; `year`/`month` are omitted
entirely on the command line:
```bash
python main_chunk.py config/one_day_hourly.ini --print-nchunks   # -> 24
sbatch --array=1-24 run_slurm_chunk.sh config/one_day_hourly.ini
```
This is the way to get "one day of hourly files" cleanly: set
`range_start`/`range_end` in the config to just that one day and
`dspan_hours = 1`; `--array=1-24` then maps **chunk 1 straight to hour
0 of that specific day**, no offset arithmetic. `year` is inferred
internally from `range_start.year` (for loading the right keys table);
`load_config` already rejects a range spanning more than one year.

Both sub-modes resolve their chunk index through a pure function
(`resolve_chunk_bounds`/`resolve_range_chunk_bounds`) that returns
`None`, not an exception, for an out-of-range index — the direct fix
for the MPI mode's crash-on-bad-`--array` sharp edge above. An
oversized `--array` in either mode exits that task cleanly with a
logged warning, exit code 0.

---

## Output file formats

### Monthly HDF5 (MPI pipeline)

`Config.output_filename(year, month)`:
```
{year}_{month:02d}_{gran|mag}_dspan{H}h_dstep{M}m_{4k|2k}.hdf5
```
e.g. `2019_06_gran_dspan24h_dstep45m_4k.hdf5`. Datasets:

| Dataset | Shape | dtype | Meaning |
|---|---|---|---|
| `uphi` | `(nt, nlat, nlng)` | `f4` | Zonal (east-west) velocity [m/s] |
| `utheta` | `(nt, nlat, nlng)` | `f4` | Meridional (north-south) velocity [m/s] |
| `tstart` | `(nt,)` | `S19` | Chunk start timestamp, `%Y.%m.%d_%H:%M:%S` |
| `longitude` | `(nlng,)` | `f8` | `clng_arr` from the config |
| `latitude` | `(nlat,)` | `f8` | `clat_arr` from the config |

`nt` is the number of `dspan` chunks in that month; note the sign
convention in `io.write_chunk_velocities`: `utheta_ds[...] = -uy`,
`uphi_ds[...] = ux` (the CCF's pixel-space `y` axis is flipped
relative to the meridional velocity's physical sign).

### Chunk HDF5 (non-MPI chunk pipeline)

`pipeline_chunk.chunk_output_filename(cfg, dstart_chunk)`:
```
{dstart_chunk:%Y%m%d_%H%M}_{gran|mag}_dspan{H}h_dstep{M}m_{4k|2k}_chunk.hdf5
```
e.g. `20190615_0000_gran_dspan24h_dstep45m_4k_chunk.hdf5`. Same
datasets as above, but with `nt = 1` — one row per file. The `_chunk`
suffix and the full datetime (vs. the monthly file's bare
`year_month`) guarantee these never collide with monthly-pipeline
output filenames landing in the same `rootdir_out`.

### CCF `.npy` files

Only written when `[output] save_ccf = 1`, and only for patches within
`ccf_lat_threshold`/`ccf_lng_threshold` degrees of disk center
(`io.save_ccf`):
```
{timestamp:%Y%m%d_%H%M%S}_ccf_dspan{H}_dstep{M}_{clat}_{clng}_{4k|2k}_{av|no_av}.npy
```
`no_av` = a single raw per-timestep CCF (written inside the timestep
loop, one per surviving patch per timestep); `av` = the time-averaged
CCF for the whole chunk (written once per patch, after averaging).
These are diagnostic dumps, independent of the main HDF5 output.

---

## CLI reference

### `main.py` (MPI pipeline)

```
python main.py <config_file> <year> [--month M] [--loglevel LEVEL]
mpirun -n 500 python main.py <config_file> <year> --month M
```
| Arg | Meaning |
|---|---|
| `config_file` | Path to `.ini` config |
| `year` | Year to process |
| `--month`, `-m` | Month 1–12. Omit to loop all 12 months sequentially in one process (single-node use only — no MPI array benefit if you do this). |
| `--loglevel`, `-l` | `debug`/`info`/`warning`/`error`/`critical` (default `info`) |

### `main_chunk.py` (non-MPI chunk pipeline)

```
# Month mode
python main_chunk.py <config_file> <year> <month> --chunk N [--loglevel LEVEL]
python main_chunk.py <config_file> <year> <month> --print-nchunks

# Range mode (year/month omitted; config must set range_start/range_end)
python main_chunk.py <config_file> --chunk N [--loglevel LEVEL]
python main_chunk.py <config_file> --print-nchunks
```
| Arg | Meaning |
|---|---|
| `config_file` | Path to `.ini` config |
| `year`, `month` | Optional. Both given → month mode. Both omitted → range mode (requires `range_start`/`range_end` in the config). Giving only one is an error. |
| `--chunk`, `-c` | 1-indexed chunk (matches `$SLURM_ARRAY_TASK_ID`). Required unless `--print-nchunks`. |
| `--print-nchunks` | Print the chunk count for this month/range (for sizing `--array=1-N`) and exit. |
| `--loglevel`, `-l` | Same as above |

---

## SLURM reference

### `run_slurm.sh` (MPI)

```bash
sbatch --array=1-12 run_slurm.sh <config_file> <year>
```
Loads the `GCC/12.2.0`/`OpenMPI/4.1.4` modules, activates the `py311`
conda env, sets `PMIX_MCA_psec=^munge` (workaround for a PMIx/munge
interaction on this cluster) and `OMP_NUM_THREADS`, then
`srun --mpi=pmix python main.py ...` with `--month "$SLURM_ARRAY_TASK_ID"`.
`--ntasks` controls the MPI world size (spatial parallelism); tune to
the patch grid size.

### `run_slurm_chunk.sh` (non-MPI)

```bash
# Month mode
sbatch --array=1-N run_slurm_chunk.sh <config_file> <year> <month>
# Range mode (year/month omitted)
sbatch --array=1-N run_slurm_chunk.sh <config_file>
```
No module loads, no `srun`, no MPI — just `conda activate py311` and a
plain `python main_chunk.py ...` with `--chunk "$SLURM_ARRAY_TASK_ID"`.
`--ntasks=1 --cpus-per-task=1` since each task is a single serial
process. `$2`/`$3` (year/month) are optional together — passing only
one is rejected before any Python runs.

---

## Testing

```bash
cd lct_pipeline && python -m pytest tests/ -v
```

| Test file | Covers |
|---|---|
| `test_config.py` | `.ini` parsing/validation, `validate_month`/`validate_range`, `range_start`/`range_end` parsing and cross-validation. |
| `test_geometry.py` | PSF construction, differential rotation, B0 correction, Carrington gap filling. |
| `test_interpolation.py` | Cubic-spline image-stack interpolation. |
| `test_lct.py` | Tukey window, CCF, ellipsoid peak fit, velocity conversion. |
| `test_mpi_utils.py` | Logging setup, log-level parsing (MPI itself is mocked, see `conftest.py`). |
| `test_pipeline_chunk.py` | `resolve_chunk_bounds`/`resolve_range_chunk_bounds` (month and range mode, including the leap-year and non-evenly-divisible-range edge cases), `chunk_output_filename`, and `run_chunk_range`'s validation/warning paths. |
| `test_main_chunk.py` | `main_chunk.py`'s CLI end-to-end via subprocess, both modes, all error paths. |

`conftest.py` injects mock `zclpy3`/`mpi4py` packages
(`tests/mocks/`) so the full suite runs without the MPS-internal
`zclpy3` dependency or a real MPI installation. `pipeline.py::run()`
and `pipeline_chunk.py::run_chunk`/`run_chunk_range`'s actual FITS
I/O + patch loop are **not** unit-tested directly (both are I/O-heavy
orchestration over already-tested physics) — only their pure
bounds/validation logic is.

---

## Known limitations / gotchas

- **`main.py` crashes on an out-of-range `--month`.** No bounds check
  before `datetime(year, month, 1)`; `--array` must be exactly `1-12`.
  (`main_chunk.py` does not have this problem — see
  [Execution modes](#execution-modes).)
- **Chunk pipeline trades spatial parallelism for temporal.** A single
  chunk task walks the *entire* patch grid serially — for
  `granulation.ini`'s 40,401-point grid this may be significantly
  slower per unit of work than the MPI pipeline's per-timestep
  gather/broadcast across many ranks. Worth timing before relying on
  it for a large grid.
- **Range mode is single-year only.** `range_start`/`range_end`
  spanning a year boundary is rejected at config-load time (the keys
  table is loaded once per year).
- **`zclpy3` is a hardcoded local path**, not a package dependency —
  `/data/seismo/zhichao/codes/pypkg`. Real runs need that path to
  exist on the executing machine; tests substitute a mock.
- **CCF averaging order matters.** The pipeline averages CCFs across
  timesteps *before* the sub-pixel peak fit, not velocities after
  independent fits per timestep — this is deliberate (more robust to
  per-timestep noise) but means a single very-bad timestep's CCF
  still contributes to the average unless it was already dropped for
  containing NaNs.
