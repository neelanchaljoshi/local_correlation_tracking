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

# Ensure output directory exists at import time
EF_OUT.mkdir(parents=True, exist_ok=True)

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
