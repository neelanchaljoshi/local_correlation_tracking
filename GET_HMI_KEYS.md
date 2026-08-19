# get_hmi_keys — Detailed Documentation

Stage 1 of the pipeline. Queries the local NetDRMS catalogue for a
year of HMI observation metadata and writes it to a FITS "keys" table
— the file `lct_pipeline/` reads to know which images exist, where
they're stored, and whether each frame is usable.

Unlike `lct_pipeline/` and `inertial_mode_pipeline/`, this folder is a
small, unrefactored script: no CLI arguments, no config file, no
tests, no SLURM script. You edit two Python files directly before each
run. This document describes it exactly as it is, including the rough
edges.

## Contents

- [What it does](#what-it-does)
- [Files](#files)
- [Configuration (edit-the-source)](#configuration-edit-the-source)
- [How to run it](#how-to-run-it)
- [Output file format](#output-file-format)
- [How this feeds into lct_pipeline](#how-this-feeds-into-lct_pipeline)
- [Known rough edges](#known-rough-edges)

---

## What it does

For a given year, HMI records its observation metadata (pointing,
solar radius, quality flags, storage location, ...) in DRMS. This
stage:

1. Builds a DRMS record-set query string for the full year (or May–Dec
   only for 2010, when HMI science data begins) at a fixed cadence.
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
| `config.py` | Module-level constants: cadence, DRMS series name, output directory, quality bitmask, and the list of keywords to fetch. **Edited by hand before each run.** |
| `main.py` | Entrypoint. Hardcodes the year range to process — **also edited by hand**. |
| `process.py` | `process_year(yr)` — builds the DRMS query, calls `fetch_keys.get_info`, validates record counts, decodes quality, writes the FITS table. |
| `fetch_keys.py` | `get_info(ds, keylist)` — the two `show_info` subprocess calls and the raw-text-to-`ndarray` parsing. |
| `utils/time_helpers.py` | `get_start_stop(yr)` — year boundaries, with the 2010 special case. |

No `.ini`/`.yaml`/`.json` config, no `requirements.txt`, no `tests/`,
not run by CI.

## Configuration (edit-the-source)

`config.py`:

| Name | Current value | Meaning |
|---|---|---|
| `cadence` | `45` | Seconds between records; used both in the series name and the DRMS `@Ns` sampling interval. |
| `seriesname` | `f'hmi.v_{cadence}s'` → `hmi.v_45s` | DRMS series to query. (A comment in the source still says "HMI Continuum Intensity" — stale, left over from when this was `hmi.ic_45s`.) |
| `outdir` | `/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.v_45s` | Output directory. **Must already exist** — this code never creates it. |
| `QbitsPass` | `0b0...0` (32 zero bits) | Quality bitmask. A frame passes only if `(quality \| QbitsPass) == QbitsPass` — with an all-zero mask, only `quality == 0x00000000` passes (strictest possible setting; any flag at all rejects the frame). |
| `KeyList` | 14 `(name, dtype)` tuples | DRMS keywords fetched, in order: `t_rec`, `t_obs`, `obs_vr`, `quality`, `crpix1/2`, `crval1/2`, `cdelt1/2`, `crota2`, `crln_obs`, `crlt_obs`, `rsun_obs`. |

`main.py`:
```python
for yr in range(2018, 2019):  # Change range as needed
    process_year(yr)
```
The year range is a literal you edit before every run — there's no
`--year` flag.

`utils/time_helpers.py`:
```python
def get_start_stop(yr):
    if yr == 2010:
        return datetime(2010, 5, 1), datetime(2011, 1, 1)
    else:
        return datetime(yr, 1, 1), datetime(yr + 1, 1, 1)
```
2010 is special-cased because HMI science observations start
2010-05-01 — `keys-2010.fits` correspondingly has fewer rows than a
full year.

## How to run it

Both `main.py` and `process.py` use flat imports (`from process import
...`, not relative imports), so it must run with this folder as the
working directory:

```bash
cd get_hmi_keys
# 1. Edit config.py: cadence, seriesname, outdir
# 2. Edit main.py: the year range
python main.py
```

Requires the NetDRMS `show_info` binary on `PATH` and a working
DRMS/JSOC site configuration — there is no way to run this against
data that isn't in the local NetDRMS catalogue.

Each year takes a while (`show_info` scans the whole catalogue twice —
once for keywords, once for storage paths); `main.py` prints elapsed
time per year.

## Output file format

`{outdir}/keys-{yr}.fits` — a FITS binary table, one row per
`cadence`-second interval in the year. Confirmed schema (2018,
700,800 rows at 45s cadence):

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
| `path` | `S45` | `/pfs/scratch/SUMS/SUM1768/D1005610960/S00008\n` (note the trailing newline, preserved from the raw `show_info` output) |

Naming convention: `keys-<YYYY>.fits`. Some downstream paths this
repo's `.ini` configs point to (`keys_new_swan/`, `keys_2k/`,
`keys_all_bad_excluded/keys_2k_2k/`) are **not** produced by this
code — those subdirectories and the 2k-resolution variants were
created by a manual/separate step not present in this repo. See
[Known rough edges](#known-rough-edges).

## How this feeds into lct_pipeline

`lct_pipeline/lct_pipeline/config.py`'s `[paths] infile_fmt_4k`/
`infile_fmt_2k` (a strftime-format path with `%Y`) is what points at
these files, and `io.load_keys_table(year, cfg)` does
`Table.read(datetime(year, 1, 1).strftime(cfg.infile_fmt))` to load
one. Downstream, `read_fits_pair`/`read_fits_quad` use the `isbad`,
`path`, and `t_rec` columns; `pipeline.py` uses `crln_obs`; the
`dataset_cadence_seconds` config value should match the `cadence` this
stage used to generate the table.

## Known rough edges

- **No CLI, no config file** — every run requires hand-editing
  `config.py` (series/cadence/output dir) and `main.py` (year range).
  There's no record of which series/years have already been generated
  other than the files present in `outdir`.
- **Path mismatch with `lct_pipeline`'s configs.** This code writes
  flat into `{outdir}/keys-{yr}.fits`, but both `granulation.ini` and
  `magnetic.ini` point at subdirectories (e.g. `keys_new_swan/`,
  `keys_2k/`) that this code never creates. On disk, those
  subdirectories and their 2k-resolution variants exist from a prior
  manual reorganization — if you regenerate keys from scratch with
  this code as-is, you'll need to move/rename the output to match
  whatever `.ini` you're using, or update the `.ini`'s `infile_fmt_4k`
  to point at the flat `outdir` path this code actually produces.
- **No validation that `outdir` exists** — the FITS write fails with a
  generic I/O error if you forget to create it first.
- **Strict quality mask by default** (`QbitsPass = 0`) — only
  perfect-quality frames pass `isbad = False`. If your dataset needs a
  looser tolerance, `QbitsPass` needs hand-editing too.
- **No tests, not part of CI.**
