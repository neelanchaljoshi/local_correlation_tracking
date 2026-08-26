# get_hmi_keys — Detailed Documentation

Stage 1 of the pipeline. Queries the local NetDRMS catalogue for a
year of HMI observation metadata and writes it to a FITS "keys" table
— the file `lct_pipeline/` reads to know which images exist, where
they're stored, and whether each frame is usable.

Config-driven now: a `.ini` file plus `--year` on the command line,
same convention as `lct_pipeline/`/`inertial_mode_pipeline/`. No more
hand-editing source before each run.

## Contents

- [What it does](#what-it-does)
- [Files](#files)
- [Configuration (`.ini`)](#configuration-ini)
- [How to run it](#how-to-run-it)
- [Output file format](#output-file-format)
- [How this feeds into lct_pipeline](#how-this-feeds-into-lct_pipeline)
- [Testing](#testing)
- [Known rough edges](#known-rough-edges)

---

## What it does

For a given year, HMI records its observation metadata (pointing,
solar radius, quality flags, storage location, ...) in DRMS. This
stage:

1. Builds a DRMS record-set query string for the full year (or from a
   configurable `data_start` date for a year that needs one, e.g. HMI
   science data begins 2010-05-01) at a fixed cadence.
2. Calls the NetDRMS `show_info` command-line tool twice — once for
   the metadata keywords, once for each record's on-disk storage path.
3. Decodes each record's hex quality flag into a boolean `isbad`.
4. Writes everything to one FITS binary table per year.

No image data is read or written here — only metadata and file paths.
The actual FITS images get read later, by `lct_pipeline/`, using the
`path` column this stage produces.

## Files

| File | Role |
|---|---|
| `settings.py` | `Config` dataclass + `load_config()` — parses `.ini` files, same convention as `lct_pipeline/lct_pipeline/config.py`. Also defines `BASE_KEY_LIST`, the fixed DRMS keyword schema downstream code depends on by name. |
| `main.py` | CLI entrypoint: `python main.py <config_file> [--year Y]`. |
| `process.py` | `process_year(yr, cfg)` — builds the DRMS query, calls `fetch_keys.get_info`, validates record counts, decodes quality, writes the FITS table. |
| `fetch_keys.py` | `get_info(ds, keylist)` — the two `show_info` subprocess calls and the raw-text-to-`ndarray` parsing. Unchanged — took no hardcoded values to begin with. |
| `utils/time_helpers.py` | `get_start_stop(yr, data_start=None)` — year boundaries, with an optional configurable start-date override for the first partial year of a series. |
| `config/*.ini` | Per-series config files — see [Configuration](#configuration-ini). |
| `tests/` | Unit tests for `settings.py`/`utils/time_helpers.py` — see [Testing](#testing). |

`config.py` (the module) doesn't exist any more — renamed to
`settings.py` so it doesn't collide with the new `config/` directory
of `.ini` files (same split as `lct_pipeline`, which keeps its
`config.py` *inside* the `lct_pipeline/` package precisely to avoid
this collision with its own top-level `config/` directory of `.ini`
files).

## Configuration (`.ini`)

Three ready-made configs ship in `config/`:

| File | Series | Cadence | Writes to |
|---|---|---|---|
| `hmi_v_45s.ini` | `hmi.v_45s` (Dopplergram) | 45s | `IterativeLCT/hmi.v_45s/keys-%Y.fits` — reproduces the values that were previously hardcoded in `config.py`/`main.py` |
| `hmi_ic_45s.ini` | `hmi.ic_45s` (continuum, granulation tracking) | 45s | `IterativeLCT/hmi.ic_45s/keys_new_swan/keys-%Y.fits` — matches `lct_pipeline/config/granulation.ini`'s `infile_fmt_4k` exactly |
| `hmi_m_720s.ini` | `hmi.m_720s` (magnetogram, magnetic-feature tracking) | 720s | `IterativeLCT/hmi.m_720s/keys_new_swan/keys-%Y.fits` — matches `lct_pipeline/config/magnetic.ini`'s `infile_fmt_4k` exactly |

| Section | Key | Meaning |
|---|---|---|
| `[job]` | `yr_start`, `yr_stop` | Inclusive year range processed when `--year` is omitted on the CLI. |
| `[series]` | `seriesname` | DRMS series to query, e.g. `hmi.v_45s`. |
| | `cadence_seconds` | Seconds between records; used both in the DRMS `@Ns` sampling interval and to compute the expected record count. |
| | `extra_keys` | Comma-separated extra DRMS keywords to fetch beyond `BASE_KEY_LIST` (`settings.py`) — the 14 keywords `lct_pipeline`/downstream code needs by name. Extras are typed `float`; leave empty unless you need something extra. |
| `[data_availability]` | `data_start` | Optional ISO date (`YYYY-MM-DD`). If set and its year matches the year being processed, the query window starts here instead of Jan 1 — e.g. `2010-05-01` for HMI's science-data start. Leave empty for series/years with no such cutoff. |
| `[quality]` | `qbits_pass` | Quality bitmask (`0x...`, `0b...`, or plain decimal — `int(..., 0)` auto-detects the base). A frame passes only if `(quality \| qbits_pass) == qbits_pass`; `0x00000000` is strictest (only `quality == 0` passes). |
| `[paths]` | `outdir` | Output root directory — **created automatically now** (`Config.__post_init__`), unlike before. |
| | `outfile_fmt` | strftime-format filename, relative to `outdir` (default `keys-%Y.fits`). Set this to match whatever subdirectory an `lct_pipeline` `.ini`'s `infile_fmt_4k`/`infile_fmt_2k` expects, e.g. `keys_new_swan/keys-%Y.fits` — this is what closes the path-mismatch rough edge documented below. |

`BASE_KEY_LIST` itself (the 14 required keywords: `t_rec`, `t_obs`,
`obs_vr`, `quality`, `crpix1/2`, `crval1/2`, `cdelt1/2`, `crota2`,
`crln_obs`, `crlt_obs`, `rsun_obs`) stays a Python constant in
`settings.py`, not an `.ini` value — it's a fixed downstream contract
(`lct_pipeline` reads these columns by name), not a per-run tunable,
same reasoning as `lct_pipeline`'s own HDF5 dataset names not being
configurable either.

## How to run it

`main.py`/`process.py`/`fetch_keys.py` use flat imports (`from process
import ...`), so this must run with this folder as the working
directory:

```bash
cd get_hmi_keys
python main.py config/hmi_ic_45s.ini --year 2018
```

Omit `--year` to process every year in `[job] yr_start`–`yr_stop`
from the config:

```bash
python main.py config/hmi_ic_45s.ini
```

Requires the NetDRMS `show_info` binary on `PATH` and a working
DRMS/JSOC site configuration — there is no way to run this against
data that isn't in the local NetDRMS catalogue.

Each year takes a while (`show_info` scans the whole catalogue twice —
once for keywords, once for storage paths); `main.py` prints elapsed
time per year.

To point at a different series or output location, copy one of the
`config/*.ini` files and edit it — no source changes needed.

## Output file format

`{outdir}/{outfile_fmt expanded for the year}` — a FITS binary table,
one row per `cadence_seconds`-second interval in the year. Confirmed
schema (2018, `hmi.v_45s`, 45s cadence — verified against real DRMS
data in this stage's tests/manual runs):

| Column | dtype | Example |
|---|---|---|
| `t_rec` | `S24` | `2018.01.01_00:00:00_TAI` |
| `t_obs` | `S24` | `2017.12.31_23:59:52_TAI` |
| `obs_vr` | `f8` | `2138.685266` |
| `quality` | `S24` | `0x00000000` |
| `crpix1`, `crpix2` | `f8` | `2042.883789`, `2044.963867` |
| `crval1`, `crval2` | `f8` | `0.0`, `0.0` |
| `cdelt1`, `cdelt2` | `f8` | `0.504331` |
| `crota2` | `f8` | `179.9298` |
| `crln_obs`, `crlt_obs` | `f8` | `346.055664`, `-3.005263` |
| `rsun_obs` | `f8` | `976.046936` (arcsec) |
| `isbad` | `bool` | `False` |
| `path` | `S45` | `/pfs/scratch/SUMS/SUM1768/D1005610960/S00008` — some records carry a trailing newline preserved from the raw `show_info` output (a real, previously-reported bug downstream — see `lct_pipeline`'s `io.read_fits_image`, which now strips it) |

## How this feeds into lct_pipeline

`lct_pipeline/lct_pipeline/config.py`'s `[paths] infile_fmt_4k`/
`infile_fmt_2k` (a strftime-format path with `%Y`) is what points at
these files, and `io.load_keys_table(year, cfg)` does
`Table.read(datetime(year, 1, 1).strftime(cfg.infile_fmt))` to load
one. Downstream, `read_fits_pair`/`read_fits_quad` use the `isbad`,
`path`, and `t_rec` columns; `pipeline.py` uses `crln_obs`; the
`.ini`'s `dataset_cadence_seconds` should match `cadence_seconds` used
to generate the table. `config/hmi_ic_45s.ini` and
`config/hmi_m_720s.ini`'s `outfile_fmt` are set to write directly to
the exact paths `granulation.ini`/`magnetic.ini` already expect, so no
manual copying/renaming is needed for those two.

## Testing

```bash
cd get_hmi_keys && python -m pytest tests/ -v
```

| Test file | Covers |
|---|---|
| `test_settings.py` | `load_config()` parsing, `Config.output_path()`'s strftime templating and auto-mkdir, `extra_keys` merging, error paths (missing file, missing required field, `yr_stop < yr_start`) |
| `test_time_helpers.py` | `get_start_stop()`'s `data_start` override logic |

Pure parsing/path-logic tests, no DRMS calls, no real data, side-effect-free
— safe to run anytime. `process.py`/`fetch_keys.py` (the actual DRMS
I/O) have no dedicated tests — they need a live NetDRMS site
connection and take a while per year; validated manually against real
`show_info` output instead (see this stage's commit history for the
exact commands used). Not part of CI.

## Known rough edges

- **`process.py`/`fetch_keys.py` have no automated tests** — DRMS
  access requires a live site connection with no fast local mock, and
  a full-year query is slow. Validated manually instead (small
  time-window queries, and a full `process_year()` run against a
  scratch output directory) — see [Testing](#testing).
- **Strict quality mask by default** (`qbits_pass = 0x00000000`) —
  only perfect-quality frames pass `isbad = False`. Loosen it per
  config if your dataset needs a wider tolerance.
- **No record of which series/years have already been generated**
  other than the files present in `outdir` — there's no manifest or
  state tracking across runs.
