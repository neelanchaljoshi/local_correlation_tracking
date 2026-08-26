# Inertial Mode Pipeline — Detailed Documentation

This document describes `inertial_mode_pipeline/` end to end: every
module, the SVD/Fourier/Legendre algorithm chain, every config
constant, the CLI tools, output file formats, and known gotchas. See
the main [README](README.md) for how this stage fits into the full
processing chain; this file is the exhaustive reference for this
stage alone.

## Contents

- [What this pipeline does](#what-this-pipeline-does)
- [Module map](#module-map)
- [Input data](#input-data)
- [Algorithm, step by step](#algorithm-step-by-step)
- [Configuration constants](#configuration-constants)
- [CLI reference](#cli-reference)
- [Output file format](#output-file-format)
- [Testing](#testing)
- [Known limitations / gotchas](#known-limitations--gotchas)

---

## What this pipeline does

Solar inertial modes are large-scale oscillatory horizontal flows.
Given LCT-derived surface flow maps (from `lct_pipeline/` +
`flow_processing/`), this pipeline extracts the spatial eigenfunction
of one specific inertial mode — identified by its azimuthal order `m`
and central frequency — by:

1. Fourier-transforming the flow maps in time and (via a Carrington
   de-rotation) longitude.
2. Bandpass-filtering around the mode's frequency and inverse-transforming
   back to the time domain.
3. Using an SVD to extract the mode's dominant spatial pattern
   (eigenfunction) and time dependence simultaneously.
4. Projecting the eigenfunction onto Legendre polynomials, statistically
   distinguishing signal from noise modes via a chi-squared threshold,
   and reconstructing a cleaned eigenfunction from only the
   signal-bearing coefficients.
5. Estimating 1-sigma errors from the discarded (noise) coefficients via
   Monte Carlo or a deterministic sum.

One run produces one eigenfunction, for one `(m, cent_freq, mode,
symmetry, data)` combination.

---

## Module map

| File | Role |
|---|---|
| `inertial_mode_pipeline/config.py` | All pipeline-wide constants: data paths, grid dimensions, disk-masking radii, Legendre/noise-filtering defaults, output filename template. |
| `inertial_mode_pipeline/io.py` | Loading pre-processed flow maps + observational metadata, saving/loading eigenfunction `.npz` results. Pure I/O. |
| `inertial_mode_pipeline/geometry.py` | Disk geometry: longitude/latitude grids, projected-radius array, disk clipping/apodization, equatorial symmetrization, Carrington-longitude gap filling, missing-data amplitude correction factors. |
| `inertial_mode_pipeline/fourier.py` | Time/longitude Fourier transform into `(frequency, latitude, m)` space, Tukey-windowed frequency bandpass filtering, inverse time transform. |
| `inertial_mode_pipeline/eigenfunction.py` | The core SVD step: extracts the dominant spatial eigenfunction + time dependence at a given `(m, cent_freq)`. |
| `inertial_mode_pipeline/legendre.py` | Legendre-polynomial projection, symmetry enforcement, chi-squared noise-mode filtering, reconstruction, phase alignment, and the top-level `project_and_clean()` orchestrator. |
| `inertial_mode_pipeline/errors.py` | Three interchangeable error-estimation methods over the *discarded* (noise-classified) Legendre coefficients. |
| `inertial_mode_pipeline/lorentzian_fit.py` | Maximum-likelihood Lorentzian fit of a power spectrum (`LorentzianMLE`) with parametric Monte Carlo error estimates. Ported from Zhi-Chao's rewrite of `plotting_scripts/table_lor_fit.py` (`/data/seismo/zhichao/codes/Joshi/git_lorentzian_fit_joshi1`). |
| `run_pipeline.py` | CLI entrypoint tying all of the above together for one mode. |
| `check_span.py` | Standalone diagnostic: checks whether a proposed `--span_lower`/`--span_upper`/`--df` combination will actually produce a non-zero eigenfunction, before spending time on a full run. |
| `plot_power_spectrum.py` | Standalone diagnostic/visualization tool: plots the latitude-averaged power spectrum for one `(m, component)` and fits it with a Lorentzian. See [CLI reference](#plot_power_spectrumpy). |

`geometry.py` imports `zclpy3.remap.get_tan_from_lnglat` from the same
hardcoded MPS-internal path used by `lct_pipeline/geometry.py`
(`/data/seismo/zhichao/codes/pypkg`) — real runs need that path; tests
mock it (see [Testing](#testing)).

---

## Input data

`io.load_flow_data(data_name)` reads, from `config.DATA_ROOT` /
`config.PROC_DATA`:

| File | Shape | Meaning |
|---|---|---|
| `processed_data/uphi_{data_name}_processed.npy` | `(nt, nlat, nlng)` | Zonal flow, produced by `flow_processing/` |
| `processed_data/utheta_{data_name}_processed.npy` | `(nt, nlat, nlng)` | Meridional flow, produced by `flow_processing/` |
| `t_rec.npy` | `(nt,)` bytes | Observation timestamps, `%Y.%m.%d_%H:%M:%S_TAI` |
| `crln_obs.npy` | `(nt,)` | Carrington longitude of disk centre [deg] |
| `crlt_obs.npy` | `(nt,)` | Carrington latitude of disk centre [deg] |
| `rsun_obs.npy` | `(nt,)` | Solar radius [arcsec] |

`data_name` is the CLI's `data` argument with `.` replaced by `_`
(e.g. `hmi.m_720s_dt_1h` → `hmi_m_720s_dt_1h`) — this is the naming
convention `flow_processing/` must produce its output under. `t_rec`
is parsed via `parse_time_array` into decimal years (`year_fraction`);
`crln_obs`/`crlt_obs`/`rsun_obs` get linearly interpolated across any
NaN gaps (`pandas.DataFrame.interpolate`) before use.

---

## Algorithm, step by step

### 1. Geometry setup

- `make_lon_lat_grids(LON_OG, LAT_OG)` — 73-point linspace grids from
  −90° to 90° in both longitude and latitude (`config.LON_OG`/`LAT_OG`).
- `fill_carrington_gaps(crln_obs)` — linearly extrapolates any NaN gaps
  in Carrington longitude using the mean *negative* step size between
  consecutive samples (falls back to −0.5°/step if no valid negative
  steps exist), wrapping past 0° back to 360°.
- `apply_symmetry(uphi_all, uthe_all, symmetry)` — folds the flow maps
  about the equator: `'sym'` makes `u_phi` symmetric / `u_theta`
  anti-symmetric, `'anti'` the reverse, `'all'` leaves them untouched.

### 2. Disk masking

- `build_radius_array(crlt_obs, rsun_obs, lon_og, lat_og)` — projects
  every `(lat, lon)` grid point to disk `(x, y)` coordinates via
  `zclpy3.remap.get_tan_from_lnglat` for every timestep, and returns
  the radial distance from disk center in arcsec.
- `--reject_type clip` → `clip_flow_data`: pixels beyond
  `CLIP_RADIUS` (0.99) × R_sun become NaN, then the longitude axis is
  NaN-padded by `(36, 35)` pixels.
- `--reject_type noclip` → `apodize_flow_data`: a half-cosine taper
  between `APOD_R_MIN` (0.96) and `APOD_R_MAX` (0.99) × R_sun instead
  of a hard cutoff (full weight below `r_min`, zero above `r_max`),
  then zero-padded the same way.

### 3. Amplitude correction

`get_correction_factor(arr, nlng_carr)` computes two correction
factors that compensate for missing data before the Fourier transform:
- `cft` (time): `nt / (number of timesteps with any valid longitude data at that latitude)`
- `cfl` (longitude): `nlng_carr / (number of valid longitude bins at that time/latitude)`

Both saturate to `inf` above `1e200`. These get applied inside
`transform_to_fourier` (`cfl` per-timestep before the time FFT,
`sqrt(cft)` after it) to rescale for the fraction of data actually
present, computed separately for `u_phi` and `u_theta`.

### 4. Fourier transform

`transform_to_fourier(arr, crln, cft, cfl, span, dt)`:
1. Real FFT in longitude → azimuthal order `m` (`np.fft.rfft`).
2. Multiply by `cfl` (longitude correction).
3. De-rotate to the Carrington frame: multiply by
   `exp(-i·m·crln[t])` per timestep — this is what makes a fixed `m`
   correspond to a genuine Carrington-frame pattern rather than one
   that drifts with the spacecraft's view.
4. Complex FFT in time (`np.fft.fft`), scaled by `sqrt(cft)`.
5. `fftshift` to center zero frequency; frequency axis in nHz via
   `-fftfreq(N, dt) * 1e9` (note the sign flip).

### 5. Bandpass filter

`bandpass_filter(uphi_m, uthe_m, freq_nHz, cent_freq, df, tukey_alpha=0.0)`:
keeps only bins strictly inside `(cent_freq - df, cent_freq + df)`,
weighted by a Tukey window (`tukeywin`) over just the surviving bins.
**`tukey_alpha` defaults to `0.0`** (rectangular — no edge-bin
zeroing); see
[Known limitations](#known-limitations--gotchas) for why this default
matters a lot for narrow spans. `extract_m_slice` is a thin wrapper
that first indexes out column `m` from the full `(nfreq, nlat, nm)`
array.

### 6. SVD eigenfunction extraction

`extract_eigenfunction()` ([eigenfunction.py](inertial_mode_pipeline/inertial_mode_pipeline/eigenfunction.py)):
1. Bandpass-filters and inverse-FFTs `u_phi`/`u_theta` back to the time
   domain at column `m` (`inverse_time_transform`).
2. Restricts to `|lat| ≤ LAT_SVD_MAX` (75°) and stacks `u_phi`/`u_theta`
   side by side into one `(nt, 2·nlat_restricted)` matrix.
3. SVD's that matrix; the **leading left singular vector** (`U[:, 0]`)
   is taken as the mode's complex time dependence.
4. Projects both `u_phi(t, lat)` and `u_theta(t, lat)` onto the
   conjugate of that time dependence (time-averaged), giving one
   complex spatial value per latitude for each — the raw eigenfunction.
5. Also returns `final_td`, the time-dependence amplitude scaled to a
   reference latitude (`lat_for_scaling`, default 0°=equator) via the
   singular value and right singular vector.

This is the statistically principled way to separate "the mode" from
noise at other latitudes/times: the SVD finds the one time series that
best explains covariance across the whole restricted-latitude flow
field, rather than picking an arbitrary reference latitude's time
series.

### 7. Legendre projection and noise filtering

`project_and_clean()` ([legendre.py](inertial_mode_pipeline/inertial_mode_pipeline/legendre.py)) — the biggest module, orchestrating:

1. **Project**: `project_to_legendre_coefficients` computes
   `f_ℓ = ∫ ef(θ)·P_ℓ(cosθ)·sinθ·√((2ℓ+1)/2) dθ` (Simpson's rule) for
   `ℓ = 0 .. L_ARRAY_MAX-1` (36).
2. **Enforce symmetry**: `enforce_symmetry` zeroes out the
   wrong-parity coefficients based on `symmetryuphi` (`'sym'`/`'anti'`/`'all'`)
   — e.g. for `'anti'`, `u_phi` keeps only odd ℓ and `u_theta` only even ℓ.
3. **Noise threshold**: `compute_keep_mask` estimates a chi-squared
   noise floor from the median power of modes above `l_theory_cutoff`
   (15), sets a confidence-level threshold (`NOISE_CONFIDENCE`=0.90,
   `chi2.ppf` with 2 d.o.f.), and keeps a high-ℓ mode only if its power
   exceeds that threshold **and** it falls within the cumulative 99%
   power mass. Modes at or below `l_theory_cutoff` are always kept
   regardless of power (they're theoretically expected signal).
4. **Reconstruct**: `reconstruct_from_coefficients` sums back only the
   kept modes (up to `l_max`=`L_MAX_RECON`=22) into a cleaned spatial
   eigenfunction; `reconstruct_full` does the same with *all* modes
   (no noise cut) as a smoothed reference (`ef_uphi_sm`/`ef_uthe_sm`).
5. **Error estimation**: `compute_errors` dispatches to one of three
   methods over the *discarded* coefficients (see [errors.py](inertial_mode_pipeline/inertial_mode_pipeline/errors.py)):
   - `monte_carlo` (default): randomizes only the *phase* of each
     discarded coefficient uniformly, reconstructs `num_mc_samples`
     times, takes the std of the real/imaginary parts as the 1-σ error.
   - `monte_carlo_amp`: also randomizes the *amplitude*,
     `N(|f_ℓ|, 0.2·|f_ℓ|)`.
   - `fl_sum`: deterministic — the absolute value of the direct sum of
     all discarded modes (no randomization, no distribution).
6. **Phase alignment**: `align_phase_at_equator` rotates both
   eigenfunctions by a common phase so that `u_theta` is purely real
   at the equator (a normalization convention for comparing
   eigenfunctions across modes/methods).

---

## Configuration constants

All in `config.py`, no `.ini` file for this pipeline (unlike
`lct_pipeline/`) — change constants directly in the module or override
via CLI flags where exposed:

| Constant | Value | Meaning |
|---|---|---|
| `DATA_ROOT` | `.../local_correlation_tracking/data` | Root for input flow data and output eigenfunctions |
| `EF_OUT` | `DATA_ROOT/eigenfunctions` | Output directory |
| `PROC_DATA` | `DATA_ROOT/processed_data` | Input processed-flow-map directory (from `flow_processing/`) |
| `LON_OG`, `LAT_OG` | `(-90, 90, 73)` each | Longitude/latitude grid definition |
| `DT_SEC` | `21600` (6h) | Assumed cadence — **only used as a default arg where callers don't pass their own `dt`**; real cadence comes from the actual data spacing in practice |
| `LAT_SVD_MAX` | `75.0` | Latitude cutoff for the SVD step |
| `CLIP_RADIUS` | `0.99` | Hard-clip disk radius fraction |
| `APOD_R_MIN`, `APOD_R_MAX` | `0.96`, `0.99` | Apodization taper radii |
| `L_ARRAY_MAX` | `36` | Total ℓ modes computed in the projection |
| `L_MAX_RECON` | `22` | Max ℓ kept in reconstruction (CLI: `--l_max`) |
| `L_THEORY_CUTOFF` | `15` | Always-keep ℓ boundary (CLI: `--l_cutoff`) |
| `NOISE_CONFIDENCE` | `0.90` | Chi-squared confidence level for the noise threshold |
| `SPAN_LOWER`, `SPAN_UPPER` | `2010`, `2025` | Default `--span_lower`/`--span_upper` |
| `EF_FILENAME` | `eigenfunction_clean_m{m}_{freq}_{mode}_{symmetry}_{data}.npz` | Output filename template |
| `PS_OUT` | `DATA_ROOT/power_spectra` | `plot_power_spectrum.py`'s default output directory |
| `TILE_SIZE_DEG` | `5.0` | LCT patch size in degrees, used by `plot_power_spectrum.py` to compute the effective number of independent latitude samples (`n_avg`) for the Lorentzian fit's Monte Carlo error estimate |
| `PS_MODE_LAT_BANDS` | `{highlat: (45,75), critlat: (15,45), rossby: (0,30), hfr: (0,30)}` | Default latitude band per mode label, used by `plot_power_spectrum.py` when `--lat_min`/`--lat_max` are omitted |
| `PS_FILENAME` | `power_spectrum_m{m}_{component}_{mode}_{symmetry}_{data}.pdf` | `plot_power_spectrum.py`'s default output filename template |

---

## CLI reference

### `run_pipeline.py`

```
python run_pipeline.py <m> <cent_freq> <mode> <data> <symmetry> [options]
```

| Arg | Meaning |
|---|---|
| `m` | Azimuthal order |
| `cent_freq` | Central frequency [nHz] |
| `mode` | Free-text mode label — filename metadata only, not otherwise interpreted by the pipeline. In practice one of: `rossby` (equatorial Rossby mode), `highlat` (high-latitude mode), `critlat` (critical-latitude mode), `hfr` (high-frequency retrograde mode). |
| `data` | Data product name, `.` gets replaced with `_` to match `flow_processing/`'s output naming. Selects which upstream LCT run's flow maps to use — depends on what `lct_pipeline/` tracked and at what cadence: `hmi.ic_45s` (continuum intensity, 45s cadence, granulation tracking — needs the `_granule` suffix, e.g. `hmi.ic_45s_granule`), `hmi.m_45s` (magnetograms, 45s cadence, magnetic-feature tracking), `hmi.m_720s_dt_1h` (magnetograms, 720s/12min cadence, magnetic-feature tracking, flow maps binned to 1h cadence in `flow_processing/`). Whatever value is passed must exactly match the `data_name` that `flow_processing/` wrote its `.npy` output under (see [Input data](#input-data) and [FLOW_PROCESSING.md](FLOW_PROCESSING.md)). |
| `symmetry` | `sym` \| `anti` \| `all` — equatorial symmetry of `u_phi` |
| `--l_max` | Max ℓ in reconstruction (default `L_MAX_RECON`=22) |
| `--l_cutoff` | Always-keep ℓ boundary (default `L_THEORY_CUTOFF`=15) |
| `--mc_samples` | Monte Carlo trials for error estimation (default 500) |
| `--error_method` | `monte_carlo` \| `monte_carlo_amp` \| `fl_sum` |
| `--span_lower`, `--span_upper` | Decimal-year time window (float — sub-year spans are expressible) |
| `--reject_type` | `clip` \| `noclip` |
| `--df` | Bandpass half-width [nHz]. **If omitted, auto-selected from the span** via `resolve_df()` — see below |
| `--min_bins` | Minimum passband bins required for a safe `--df` (default 2) |

`resolve_df(n_span, dt_seconds, cent_freq, df_arg, min_bins, margin=1.2)`
(defined in `run_pipeline.py`, above `main()`): if `--df` is omitted,
picks `df = min_bins · Δf / 2 · margin` where
`Δf = 1e9 / (n_span · dt_seconds)` is the actual frequency resolution
for the requested span. If `--df` is given explicitly but yields fewer
than `min_bins` surviving passband bins for the chosen span, prints a
warning with a concrete suggested value — but always respects the
user's explicit choice rather than silently overriding it.

### `check_span.py`

```
python check_span.py <cent_freq> <data> [--span_lower Y] [--span_upper Y] [--df D] [--min_bins N]
```
Loads the real `t_array` for `data` via `io.load_flow_data`, applies
the same span mask `run_pipeline.py` would, and reports `OK`/`FAIL`
plus a concrete suggested `--df` or span widening — a fast way to
check a proposed run *won't* come back with a silently all-zero
eigenfunction, without actually running the full SVD/Legendre chain.

### `plot_power_spectrum.py`

```
python plot_power_spectrum.py <m> <component> <mode> <data> <symmetry> \
    --fit_range LOW HIGH [options]
```

| Arg | Meaning |
|---|---|
| `m` | Azimuthal order |
| `component` | `uphi` \| `uthe` — flow component to plot/fit |
| `mode` | Mode label — also used to look up the default latitude band (`highlat`/`critlat`/`rossby`/`hfr`, see `config.PS_MODE_LAT_BANDS`) |
| `data` | Data product name (same convention as `run_pipeline.py`) |
| `symmetry` | `sym` \| `anti` \| `all` — equatorial symmetry of `u_phi`, same convention as `run_pipeline.py`'s `symmetry` argument (`u_theta` gets the opposite parity via `geometry.apply_symmetry`) |
| `--fit_range LOW HIGH` | **Required.** Frequency window [nHz] to fit the Lorentzian over; also the default plot x-limits |
| `--lat_min`, `--lat_max` | Latitude band [deg]. Either or both may be omitted to fall back to `mode`'s default band (`resolve_lat_band()`) — an unrecognized `mode` requires both explicitly |
| `--span_lower`, `--span_upper` | Decimal-year time window (default `SPAN_LOWER`/`SPAN_UPPER`, same as `run_pipeline.py`) |
| `--reject_type` | `clip` \| `noclip` (same as `run_pipeline.py`) |
| `--n_mc` | Monte Carlo realisations for the fit's error bars (default 2000 — table_lor_fit.py-scale runs use 10000, but that's slow for an interactive tool) |
| `--use_differential_evolution` | Run a global optimizer for the initial guess instead of the automatic heuristic |
| `--seed` | Random seed for the Monte Carlo error estimate (default 42) |
| `--xlim LOW HIGH` | Plot x-limits [nHz] (default: `--fit_range`) |
| `--outfile` | Output figure path (default: `PS_OUT / PS_FILENAME`, see `config.py`) |
| `--no-show` | Skip `plt.show()` even on an interactive backend |

Runs the same geometry + Fourier-transform steps as `run_pipeline.py`
(`make_lon_lat_grids` → `apply_symmetry` → `build_radius_array` →
`clip_flow_data`/`apodize_flow_data` → `get_correction_factor` →
`transform_to_fourier`) but **without** the bandpass filter, SVD, or
Legendre steps, since fitting a Lorentzian linewidth needs the full
spectrum around the peak, not just the passband `run_pipeline.py`
would extract the mode from. Averages power over the resolved
latitude band at the given `m`, fits it with
`lorentzian_fit.LorentzianMLE`, and saves a figure plus a `.json` fit
summary (amplitude, frequency, FWHM, background, SNR, all with Monte
Carlo 1-σ errors) next to it.

Validated against Zhi-Chao's reference implementation
(`git_lorentzian_fit_joshi1/table_lor_fit.py` +
`utils.py`): recomputing the `m=1 uphi highlat anti` and `m=8 uthe
rossby anti` power spectra from the raw processed flow data via this
tool's `compute_power_spectrum()` reproduces the reference's
precomputed `uphi_ft_2010_2024_*.npy`/`uthe_ft_2010_2024_*.npy` arrays
and fit parameters (amplitude, frequency, FWHM) to full float
precision.

**Gotcha inherited from the legacy `uthe_ft_2010_2024_*.npy` cache
filenames:** those cached arrays' `_sym_`/`_anti_` filename suffixes
describe `u_theta`'s *own* parity, not the `symmetry` CLI argument
that produced them — and for `u_theta` those two are swapped (`sym`
u_theta was generated by running with `symmetry='anti'`, and vice
versa), because `apply_symmetry` gives `u_theta` the *opposite* parity
from `u_phi`. This tool takes `symmetry` directly (same meaning as
`run_pipeline.py`) and needs no such translation — the caveat only
matters if you're comparing against those specific legacy cache files
by filename.

---

## Output file format

`io.save_eigenfunction()` writes an `.npz` at
`EF_OUT / EF_FILENAME.format(m=m, freq=cent_freq, mode=mode, symmetry=symmetry, data=data_name)`,
e.g. `eigenfunction_clean_m2_-171.0_highlat_sym_hmi_m_720s_dt_1h.npz`.
Contents (all the keys `project_and_clean()` returns, plus what
`run_pipeline.py` adds):

| Key | Meaning |
|---|---|
| `ef_uphi`, `ef_uthe` | Noise-filtered, phase-aligned reconstructed eigenfunctions |
| `ef_uphi_sm`, `ef_uthe_sm` | Fully-smoothed (no noise cut) reconstructions |
| `uphi_err_real`, `uphi_err_imag`, `uthe_err_real`, `uthe_err_imag` | 1-σ errors from the chosen error method |
| `lats` | Latitude grid the eigenfunction is sampled on |
| `final_td` | Time-dependence amplitude from the SVD step, scaled to the reference latitude |

`load_eigenfunction()` reverses the filename template to load one
back as a plain dict.

---

## Testing

```bash
cd inertial_mode_pipeline && python -m pytest tests/ -v
```

| Test file | Covers |
|---|---|
| `test_fourier.py` | `tukeywin`, `bandpass_filter`, `inverse_time_transform` |
| `test_geometry.py` | Grid construction, radius array, clip/apodize masking, symmetrization, Carrington gap filling |
| `test_eigenfunction.py` | SVD extraction against synthetic tones |
| `test_legendre.py` | Projection, symmetry enforcement, noise-mode keep/discard logic, reconstruction, phase alignment |
| `test_errors.py` | All three error-estimation methods |
| `test_span_resolution.py` | The narrow-span zero-eigenfunction regression suite — see [min_df_reference.md](inertial_mode_pipeline/tests/min_df_reference.md) for the generated span→minimum-`--df` lookup table |
| `test_lorentzian_fit.py` | The Lorentzian profile, negative log-likelihood, and `LorentzianMLE` (parameter recovery on synthetic chi-squared-noise data, error estimation, `resolved` flag) |
| `test_plot_power_spectrum.py` | `resolve_lat_band()`'s mode-default/explicit-override logic |

`io.py` has **no dedicated tests** (0% coverage) — it's pure file
loading, same category of gap as `lct_pipeline/io.py`. `run_pipeline.py`,
`check_span.py`, and `plot_power_spectrum.py`'s data-loading/plotting
functions (`compute_power_spectrum`, `fit_and_plot`) are CLI/plotting
code that needs real flow data, so they're exercised manually rather
than via pytest — see the manual validation against Zhi-Chao's
reference implementation noted under
[`plot_power_spectrum.py`](#plot_power_spectrumpy) above.

---

## Known limitations / gotchas

- **A too-narrow `--span_lower`/`--span_upper` window can silently
  return an all-zero eigenfunction.** The frequency resolution
  `Δf = 1e9/(N·dt)` must be fine enough that the `(cent_freq−df,
  cent_freq+df)` passband actually contains a bin. `run_pipeline.py`'s
  `resolve_df()` auto-selects a safe `--df` when omitted, and
  `check_span.py` lets you check any explicit combination before a
  full run. This was a real reported bug — see the generated
  [`tests/min_df_reference.md`](inertial_mode_pipeline/tests/min_df_reference.md)
  table for exact minimum-`--df` values by span length.
- **`bandpass_filter`'s Tukey taper default changed from `0.1` to
  `0.0`** (rectangular). At `0.1`, the taper always zeroed the first
  and last surviving passband bin — compounding the narrow-span
  problem above by requiring `band.sum() >= 3` just to leave one
  usable bin. At the current default, `band.sum() >= 1` is enough.
  Passing `tukey_alpha > 0` explicitly reintroduces the old
  edge-zeroing behavior.
- **`zclpy3` is a hardcoded local path**, same MPS-internal dependency
  as `lct_pipeline/geometry.py` uses — real runs need
  `/data/seismo/zhichao/codes/pypkg` to exist; tests mock it.
- **No `.ini` config** — unlike `lct_pipeline/`, tuning knobs here are
  either CLI flags or require editing `config.py` directly (e.g.
  `L_ARRAY_MAX`, `LAT_SVD_MAX`, disk-masking radii have no CLI
  override).
- **The legacy `uthe_ft_2010_2024_*.npy` cache filenames' `_sym_`/`_anti_`
  suffix is inverted relative to the `symmetry` CLI argument that
  generated them**, because that suffix names `u_theta`'s own parity
  while `apply_symmetry` gives `u_theta` the *opposite* parity from
  `u_phi`. Doesn't affect `plot_power_spectrum.py` (which takes
  `symmetry` directly, same convention as `run_pipeline.py`) — only
  matters when matching those specific cached files by filename. See
  [`plot_power_spectrum.py`](#plot_power_spectrumpy) above.
