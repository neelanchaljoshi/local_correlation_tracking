"""
config.py
---------
All pipeline-wide constants and file paths.
Change DATA_ROOT here and everything else follows.
"""

import pathlib

# ── Root paths ────────────────────────────────────────────────────────────
DATA_ROOT = pathlib.Path('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data')
EF_OUT    = DATA_ROOT / 'eigenfunctions'
PROC_DATA = DATA_ROOT / 'processed_data'
PS_OUT    = DATA_ROOT / 'power_spectra'

# ── Grid constants ────────────────────────────────────────────────────────
LON_OG     = (-90.0, 90.0, 73)   # (start, stop, n_points)
LAT_OG     = (-90.0, 90.0, 73)
DT_SEC     = 6 * 3600             # cadence in seconds
LAT_SVD_MAX = 75.0                # latitude cutoff for SVD mask

# ── Disk masking defaults ─────────────────────────────────────────────────
CLIP_RADIUS      = 0.99    # fraction of R_sun for hard clip
APOD_R_MIN       = 0.96
APOD_R_MAX       = 0.99

# ── Legendre projection defaults ──────────────────────────────────────────
L_ARRAY_MAX      = 36      # total ell modes computed
L_MAX_RECON      = 22      # maximum ell kept in reconstruction
L_THEORY_CUTOFF  = 15      # modes at or below this are always kept
NOISE_CONFIDENCE = 0.90    # confidence level for noise threshold

# ── Time span defaults ────────────────────────────────────────────────────
SPAN_LOWER = 2010
SPAN_UPPER = 2025

# ── Output filename template ──────────────────────────────────────────────
EF_FILENAME = 'eigenfunction_clean_m{m}_{freq}_{mode}_{symmetry}_{data}.npz'

# ── Power spectrum / Lorentzian-fit defaults (plot_power_spectrum.py) ─────
TILE_SIZE_DEG = 5.0   # LCT patch size in degrees, used for the effective
                       # number of independent latitude samples (n_avg) in
                       # the Monte Carlo error estimate
PS_MODE_LAT_BANDS = {
    # mode label -> default (lat_min, lat_max) [degrees], matching the
    # published highlat/critlat/rossby/hfr mode conventions
    'highlat': (45.0, 75.0),
    'critlat': (15.0, 45.0),
    'rossby':  (0.0, 30.0),
    'hfr':     (0.0, 30.0),
}
PS_FILENAME = 'power_spectrum_m{m}_{component}_{mode}_{symmetry}_{data}.pdf'
