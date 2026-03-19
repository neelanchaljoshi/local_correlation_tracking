# %%
"""
Supergranulation LCT Cadence Comparison — HMI vlos vs LCT vlos
===============================================================
Metrics and publication-quality plots comparing the line-of-sight
velocity from HMI Dopplergrams (reference) against LCT-derived vlos
at multiple cadences, with matched-cadence HMI averages as a baseline.

QUICK START
-----------
    from sg_lct_metrics import run_all

    # vlos_hmi_ref  : 2D array [m/s] — full-resolution HMI reference
    #                 (e.g. 45s cadence averaged over the full 8h window)
    #
    # vlos_hmi_cad  : dict {cadence_label: 2D array [m/s]}
    #                 HMI averaged to the same cadence as LCT
    #                 e.g. {"90s": arr1, "360s": arr2, "900s": arr3}
    #
    # vlos_lct      : dict {cadence_label: 2D array [m/s]}
    #                 LCT-derived vlos; keys must match vlos_hmi_cad exactly

    m = run_all(
        vlos_hmi_ref,
        vlos_hmi_cad,
        vlos_lct,
        lon=lon,             # 2D longitude array [deg] — Postel grid
        lat=lat,             # 2D latitude array [deg]  — Postel grid
        save_prefix="sg_run",
    )

COMPARISON DESIGN
-----------------
  Three-way comparison for each cadence C:

    HMI_ref  →  HMI_C   : effect of temporal averaging alone
    HMI_ref  →  LCT_C   : total degradation (averaging + LCT tracking noise)
    HMI_C    →  LCT_C   : LCT tracking noise only, averaging removed

  This decomposition is the main scientific contribution of having matched
  HMI cadences: it isolates what LCT adds beyond simple time-averaging.

NORMALISATION PHILOSOPHY
------------------------
  RAW fields      → amplitude metrics (RMSE, bias, ratio per cadence)
  NORMALISED      → structural metrics (Pearson r, NCC, skill score, PSD shape)

COORDINATE CONVENTION
---------------------
  vlos > 0  : flow away from observer (redshift)
  Supergranules show ~300-500 m/s vlos amplitude near the limb,
  weaker near disk centre.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.fft import fft2, fftshift
from scipy.odr import ODR, Model, RealData
import warnings
warnings.filterwarnings("ignore")


# ── publication style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":    "white",   "axes.facecolor":      "white",
    "axes.edgecolor":      "#333333", "axes.labelcolor":     "#222222",
    "axes.titlecolor":     "#111111", "axes.linewidth":      1.0,
    "axes.grid":           True,      "grid.color":          "#dddddd",
    "grid.linewidth":      0.5,       "grid.linestyle":      "--",
    "xtick.color":         "#333333", "ytick.color":         "#333333",
    "xtick.direction":     "in",      "ytick.direction":     "in",
    "xtick.major.size":    4.0,       "ytick.major.size":    4.0,
    "xtick.minor.size":    2.5,       "ytick.minor.size":    2.5,
    "xtick.major.width":   0.8,       "ytick.major.width":   0.8,
    "text.color":          "#222222", "font.family":         "sans-serif",
    "font.size":           12,        "axes.labelsize":      13,
    "axes.titlesize":      13,        "legend.fontsize":     11,
    "legend.framealpha":   0.9,       "legend.edgecolor":    "#cccccc",
    "legend.borderpad":    0.5,
    "figure.dpi":          150,       "savefig.dpi":         300,
    "savefig.facecolor":   "white",   "savefig.bbox":        "tight",
    "image.origin":        "lower",   "image.interpolation": "nearest",
})

# colours
CHMI   = "#1f77b4"   # blue       — HMI reference
CGOOD  = "#2ca02c"   # green      — reference lines / HMI cadence-averaged
CGREY  = "#7f7f7f"   # grey       — neutral lines
# per-cadence colour cycle for LCT curves
CADENCE_COLORS = ["#d62728", "#ff7f0e", "#9467bd"]  # one per cadence: 90s, 360s, 900s

R_SUN_MM    = 695.7   # Mm


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PLUG IN YOUR DATA HERE
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Load HMI full-resolution reference, HMI cadence-averaged fields,
    and LCT cadence fields.

    Returns
    -------
    vlos_hmi_ref  : 2D array [m/s] — HMI 90s reference (averaged over 8h window)
    vlos_hmi_cad  : dict {label: 2D array [m/s]} — HMI averaged to 360s and 900s
                    Keys: {"360s": ..., "900s": ...}
    vlos_lct      : dict {label: 2D array [m/s]} — LCT vlos at all three cadences
                    Keys: {"90s": ..., "360s": ..., "900s": ...}
    lon           : 2D array [deg] — Postel-projected longitude, shape = vlos shape
    lat           : 2D array [deg] — Postel-projected latitude,  shape = vlos shape

    Notes
    -----
    - All vlos arrays must be on the same Postel spatial grid.
    - Subtract background (differential rotation, meridional flow) from all
      fields before passing in.
    - The 8h averaging window should be the same for HMI_ref, HMI_cad, and LCT.
    - HMI 90s is used as the reference (highest cadence available).
      The avg comparison only covers 360s and 900s; LCT 90s is compared
      directly against HMI 90s ref in the lct block.
    """

    # ── REPLACE BELOW ─────────────────────────────────────────────────────
    # raise NotImplementedError("Fill in load_data() with your file paths.")

    coords   = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/supergran/data/detrended_vlos/detrended_vlos_from_lct_mean_flows_dspan_90s.npz')
    lon, lat = coords['longitude'], coords['latitude']   # 2D Postel arrays

    hmi_ref = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/supergran/data/detrended_vlos/vlos_cleaned_langfellner_90s.npy')    # HMI 90s = reference

    hmi_cad = {
        "360s": np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/supergran/data/detrended_vlos/vlos_cleaned_langfellner_360s.npy'),
        "900s": np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/supergran/data/detrended_vlos/vlos_cleaned_langfellner_900s.npy'),
    }
    lct_cad = {
        "90s":  np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/supergran/data/detrended_vlos/detrended_vlos_from_lct_mean_flows_dspan_90s.npz')['v_los_detrended'],
        "360s": np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/supergran/data/detrended_vlos/detrended_vlos_from_lct_mean_flows_dspan_360s.npz')['v_los_detrended'],
        "900s": np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/supergran/data/detrended_vlos/detrended_vlos_from_lct_mean_flows_dspan_900s.npz')['v_los_detrended'],
    }
    return hmi_ref, hmi_cad, lct_cad, lon, lat
    # ── END REPLACE ───────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# No pixel_scale_deg needed — derived from Postel grid coordinates automatically.


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def postel_pixel_scale_Mm(lon, lat):
    """
    Compute the uniform physical pixel scale [Mm] from a Postel
    (azimuthal equidistant) projection grid.

    On a Postel grid distances from the projection centre are preserved,
    so the physical pixel spacing is constant across the patch even though
    degree spacing in lon/lat is not uniform.

    Parameters
    ----------
    lon, lat : 2D arrays [degrees] — heliographic coordinates on Postel grid

    Returns
    -------
    pxMm : float — physical pixel spacing in Mm (same in x and y)
    """
    ny, nx = lon.shape
    cy, cx = ny // 2, nx // 2

    def angular_sep(lon1, lat1, lon2, lat2):
        """Great-circle angular separation [deg]."""
        lo1, la1, lo2, la2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
        return np.rad2deg(np.arccos(np.clip(
            np.sin(la1)*np.sin(la2) + np.cos(la1)*np.cos(la2)*np.cos(lo2-lo1),
            -1, 1)))

    # sample several adjacent pairs near the centre in both directions
    seps_x = [angular_sep(lon[cy, i], lat[cy, i],
                           lon[cy, i+1], lat[cy, i+1])
               for i in range(cx-3, cx+3)]
    seps_y = [angular_sep(lon[i, cx], lat[i, cx],
                           lon[i+1, cx], lat[i+1, cx])
               for i in range(cy-3, cy+3)]

    step_x_deg = np.mean(seps_x)
    step_y_deg = np.mean(seps_y)
    pxMm_x = np.deg2rad(step_x_deg) * R_SUN_MM
    pxMm_y = np.deg2rad(step_y_deg) * R_SUN_MM

    if abs(pxMm_x - pxMm_y) / np.mean([pxMm_x, pxMm_y]) > 0.01:
        print(f"  WARNING: x/y pixel scales differ by "
              f"{100*abs(pxMm_x-pxMm_y)/np.mean([pxMm_x,pxMm_y]):.1f}% "
              f"— grid may not be square Postel")

    pxMm = float(np.mean([pxMm_x, pxMm_y]))
    print(f"  Postel pixel scale: {pxMm:.4f} Mm/px  "
          f"(x={pxMm_x:.4f}, y={pxMm_y:.4f})")
    return pxMm

def normalise(field):
    """Zero-mean, unit-std normalisation."""
    return (field - np.mean(field)) / np.std(field)

def cadence_color(i):
    return CADENCE_COLORS[i % len(CADENCE_COLORS)]


# ══════════════════════════════════════════════════════════════════════════════
# SCALAR METRICS
# ══════════════════════════════════════════════════════════════════════════════

def rmse(ref, test):
    return float(np.sqrt(np.nanmean((ref - test)**2)))

def bias(ref, test):
    return float(np.nanmean(test - ref))

def odr_fit(ref, test, sx=None, sy=None):
    """
    Orthogonal Distance Regression of test on ref.
    Treats both variables as having measurement errors (errors-in-variables).
    Appropriate when neither HMI nor LCT is a perfect reference.

    Parameters
    ----------
    ref, test : 2D arrays (will be flattened)
    sx        : scalar or array — std of errors in ref.
                If None, estimated as std(ref) * 0.05  (5% noise floor).
    sy        : scalar or array — std of errors in test.
                If None, estimated as std(test) * 0.10 (10% noise floor).
                LCT noise is assumed larger than direct HMI measurement noise.

    Returns
    -------
    dict with keys:
        slope     : ODR slope  (= amplitude ratio corrected for errors in both)
        intercept : ODR intercept
        slope_err : 1-sigma uncertainty on slope
        int_err   : 1-sigma uncertainty on intercept
    """
    x = ref.ravel().astype(float)
    y = test.ravel().astype(float)

    # error estimates — use sensible defaults if not provided
    if sx is None:
        sx = np.std(x) * 0.05
    if sy is None:
        sy = np.std(y) * 0.10

    linear = Model(lambda B, x: B[0]*x + B[1])
    data   = RealData(x, y, sx=sx, sy=sy)
    # initialise with OLS slope as starting guess
    beta0  = [np.cov(x, y)[0,1] / np.var(x), np.mean(y) - np.mean(x)]
    result = ODR(data, linear, beta0=beta0).run()

    slope, intercept = result.beta
    slope_err, int_err = result.sd_beta

    return {
        "slope":     float(slope),
        "intercept": float(intercept),
        "slope_err": float(slope_err),
        "int_err":   float(int_err),
    }

def pearson_r(ref, test):
    """Standard Pearson r — kept for NCC and quick scalar use."""
    r, _ = pearsonr(ref.ravel(), test.ravel())
    return float(r)

def amplitude_ratio(ref, test):
    """RMS ratio test/ref — quick amplitude check, not ODR-corrected."""
    return float(np.sqrt(np.nanmean(test**2)) / np.sqrt(np.nanmean(ref**2)))

def normalised_cross_correlation(ref, test):
    """
    Spatial NCC at zero lag — scalar pattern correlation, range [-1, 1].
    Equivalent to Pearson r on zero-mean fields.
    """
    ref_n = ref - np.mean(ref)
    tst_n = test - np.mean(test)
    num   = np.nansum(ref_n * tst_n)
    den   = np.sqrt(np.nansum(ref_n**2) * np.nansum(tst_n**2))
    return float(num / den) if den > 0 else np.nan

def spatial_skill(ref_n, test_n):
    """
    Murphy & Epstein (1989) normalised spatial skill score.
    = 1 - MSE(normalised) / 2
    Range (-inf, 1]; 1 = perfect. Penalises pattern AND amplitude errors.
    Applied to normalised fields so amplitude bias does not dominate.
    """
    return float(1.0 - np.nanmean((ref_n - test_n)**2) / 2.0)


# ══════════════════════════════════════════════════════════════════════════════
# PSD
# ══════════════════════════════════════════════════════════════════════════════

def _radial_avg(power):
    """Radially averaged (isotropic) 1D PSD from a 2D power spectrum."""
    N     = power.shape[0]
    cy, cx = N // 2, N // 2
    y, x  = np.ogrid[:N, :N]
    r     = np.hypot(x - cx, y - cy).astype(int).ravel()
    p     = power.ravel()
    k_max = min(cx, cy)
    bins  = np.zeros(k_max); cnts = np.zeros(k_max)
    for ri, pi in zip(r, p):
        if ri < k_max:
            bins[ri] += pi; cnts[ri] += 1
    cnts[cnts == 0] = np.nan
    return bins / cnts

def compute_psd(field, pixel_scale_Mm):
    """
    Radially averaged 1D PSD.
    Returns (spatial_scale_Mm, power). k=0 excluded (nan at index 0).
    """
    psd      = np.abs(fftshift(fft2(field)))**2
    avg      = _radial_avg(psd)
    k_cpp    = np.arange(len(avg)) / field.shape[0]
    k_Mm     = k_cpp / pixel_scale_Mm
    k_Mm[0]  = np.nan
    return 1.0 / k_Mm, avg

def psd_peak_scale(scale, psd, scale_min=5.0):
    sel = (scale > scale_min) & np.isfinite(scale)
    return float(scale[sel][np.nanargmax(psd[sel])])


# ══════════════════════════════════════════════════════════════════════════════
# METRIC BLOCK HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _metric_block(ref, test, pxMm, scale):
    """
    Compute all metrics comparing test field against ref field.
    ref and test are raw [m/s].

    Pearson r and ODR are computed on raw (physical) fields — no normalisation —
    so that amplitude differences are captured in the correlation metrics.
    Normalised fields are still computed for PSD shape comparison and map plots.
    """
    ref_n  = normalise(ref)
    test_n = normalise(test)
    _, psd_ref_norm  = compute_psd(ref_n,  pxMm)
    _, psd_test_raw  = compute_psd(test,   pxMm)
    _, psd_test_norm = compute_psd(test_n, pxMm)
    _, psd_ref_raw   = compute_psd(ref,    pxMm)

    # ODR and Pearson on raw fields — preserves amplitude information
    odr_raw = odr_fit(ref, test)

    return {
        # amplitude (raw)
        "rmse_raw":          rmse(ref, test),
        "bias_raw":          bias(ref, test),
        "amplitude_ratio":   amplitude_ratio(ref, test),
        # ODR (raw) — slope and correlation on physical [m/s] fields
        "odr_slope_raw":     odr_raw["slope"],
        "odr_slope_raw_err": odr_raw["slope_err"],
        "odr_intercept_raw": odr_raw["intercept"],
        # Pearson r on raw fields
        "r":                 pearson_r(ref, test),
        # structural (normalised) — pattern-only metrics
        "ncc":               normalised_cross_correlation(ref_n, test_n),
        "skill":             spatial_skill(ref_n, test_n),
        "rmse_norm":         rmse(ref_n, test_n),
        # PSD
        "psd_raw":           psd_test_raw,
        "psd_norm":          psd_test_norm,
        "psd_ratio_raw":     np.where(psd_ref_raw  > 0,
                                      psd_test_raw  / psd_ref_raw,  np.nan),
        "psd_ratio_norm":    np.where(psd_ref_norm > 0,
                                      psd_test_norm / psd_ref_norm, np.nan),
        "psd_peak_raw":      psd_peak_scale(scale, psd_test_raw),
        "psd_peak_norm":     psd_peak_scale(scale, psd_test_norm),
        # raw fields for scatter and map plots
        "test_raw":          test,
        "residual_raw":      test - ref,
        # normalised fields for pattern-only plots (PSD, skill, NCC)
        "test_n":            test_n,
        "residual_n":        test_n - ref_n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MASTER COMPUTE
# ══════════════════════════════════════════════════════════════════════════════

def compute_all(vlos_hmi_ref, vlos_hmi_cad, vlos_lct, lon, lat):
    """
    Compute all metrics for three comparisons per cadence:
      'avg'   : HMI_ref  → HMI_cad   (averaging effect only)
      'total' : HMI_ref  → LCT_cad   (total degradation)
      'lct'   : HMI_cad  → LCT_cad   (LCT tracking noise only)

    Parameters
    ----------
    vlos_hmi_ref  : 2D array [m/s] — HMI 90s reference
    vlos_hmi_cad  : dict {label: 2D array [m/s]} — HMI at 360s and 900s
    vlos_lct      : dict {label: 2D array [m/s]} — LCT at 90s, 360s, 900s
    lon           : 2D array [deg] — Postel longitude grid
    lat           : 2D array [deg] — Postel latitude grid

    Returns
    -------
    m : dict with all metrics, arrays, and PSD data
    """
    assert set(vlos_hmi_cad.keys()).issubset(set(vlos_lct.keys())), \
        "Every HMI cadence key must also exist in vlos_lct."

    pxMm   = postel_pixel_scale_Mm(lon, lat)
    labels = list(vlos_lct.keys())
    m      = {"labels": labels, "pxMm": pxMm, "lon": lon, "lat": lat}

    # ── HMI reference PSD ─────────────────────────────────────────────────
    ref_n = normalise(vlos_hmi_ref)
    scale, psd_hmi_ref_raw  = compute_psd(vlos_hmi_ref, pxMm)
    _,     psd_hmi_ref_norm = compute_psd(ref_n,        pxMm)

    m["vlos_hmi_ref"]      = vlos_hmi_ref
    m["vlos_hmi_ref_n"]    = ref_n
    m["psd_scale"]         = scale
    m["psd_hmi_ref_raw"]   = psd_hmi_ref_raw
    m["psd_hmi_ref_norm"]  = psd_hmi_ref_norm
    m["psd_hmi_ref_peak"]  = psd_peak_scale(scale, psd_hmi_ref_raw)

    m["vlos_hmi_cad"]  = vlos_hmi_cad
    m["vlos_lct"]      = vlos_lct

    # ── per-cadence metrics ────────────────────────────────────────────────
    # Three comparison types stored as nested dicts: m["avg"][label], etc.
    m["avg"]   = {}   # HMI_ref → HMI_cad  : averaging effect
    m["avg_hmi_raw"] = {}   # raw HMI field at each cadence (ref for 90s)
    m["total"] = {}   # HMI_ref → LCT_cad  : total degradation
    m["lct"]   = {}   # HMI_cad → LCT_cad  : LCT tracking noise only

    for label in labels:
        lct_c = vlos_lct[label]

        # total: LCT vs HMI ref — always available
        m["total"][label] = _metric_block(vlos_hmi_ref, lct_c, pxMm, scale)

        if label in vlos_hmi_cad:
            # matched HMI cadence exists (360s, 900s)
            hmi_c = vlos_hmi_cad[label]
            m["avg"][label] = _metric_block(vlos_hmi_ref, hmi_c, pxMm, scale)
            m["avg"][label]["hmi_cad_n"]   = normalise(hmi_c)
            m["avg"][label]["hmi_cad_raw"] = hmi_c
            m["avg_hmi_raw"][label] = hmi_c
            m["lct"][label] = _metric_block(hmi_c, lct_c, pxMm, scale)
        else:
            # no matched HMI cadence (90s) — avg undefined, lct uses ref directly
            m["avg"][label] = None
            m["avg_hmi_raw"][label] = vlos_hmi_ref   # ref itself is the best HMI at this cadence
            m["lct"][label] = _metric_block(vlos_hmi_ref, lct_c, pxMm, scale)

    return m


# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _imshow(ax, data, cmap, title, unit="", sym=True):
    if sym:
        vmax = np.nanpercentile(np.abs(data), 99); vmin = -vmax
    else:
        vmin, vmax = np.nanpercentile(data, [1, 99])
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=4, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.6)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(unit, fontsize=7)
    cb.ax.tick_params(labelsize=6, direction="in")
    cb.outline.set_linewidth(0.6)
    return im

def _psd_fmt_ax(ax, ylabel, title, scale_min, scale_max):
    scale_ticks = np.array([3, 5, 10, 20, 30, 50, 100, 200])
    scale_ticks = scale_ticks[(scale_ticks >= scale_min) & (scale_ticks <= scale_max)]
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlim(scale_max * 1.05, scale_min * 0.95)
    ax.set_xticks(scale_ticks)
    ax.set_xticklabels([str(t) for t in scale_ticks], fontsize=8)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("Spatial scale [Mm]", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, pad=6)



# ══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_vlos_maps(m, save=None):
    """
    2D vlos maps in physical units [m/s] using pcolormesh on the Postel lon/lat grid.
    Rows: HMI (matched cadence) | LCT | Residual (LCT - HMI).
    Columns: one per LCT cadence. vmin/vmax fixed at +/-500 m/s.
    """
    labels = m["labels"]
    lon    = m["lon"]
    lat    = m["lat"]
    ncols  = len(labels)
    nrows  = 3
    VMAX   = 500.0

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.2 * ncols, 4.8 * nrows),
        constrained_layout=True,
    )
    if ncols == 1:
        axes = axes[:, np.newaxis]

    def _pcolor(ax, data, vmax, title=""):
        im = ax.pcolormesh(lon, lat, data, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto",
                           rasterized=True)
        if title:
            ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
        ax.set_xlabel("Longitude [deg]", fontsize=12)
        ax.set_ylabel("Latitude [deg]",  fontsize=12)
        ax.tick_params(labelsize=11)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("m/s", fontsize=12)
        cb.ax.tick_params(labelsize=10, direction="in")
        cb.outline.set_linewidth(0.8)
        return im

    row_labels = ["HMI", "LCT", "Residual"]

    for i, rl in enumerate(row_labels):
        for j, lbl in enumerate(labels):
            hmi_cad = m["avg_hmi_raw"][lbl]
            lct_raw = m["total"][lbl]["test_raw"]
            res_hmi = m["total"][lbl]["residual_raw"]
            data    = [hmi_cad, lct_raw, res_hmi][i]
            hmi_tag = f"HMI {lbl}" + (" (= ref)" if m["avg"][lbl] is None else "")
            # title: row label + cadence on first row; row label only on first col otherwise
            if i == 0:
                title = f"{lbl}"
            elif j == 0:
                title = ""
            else:
                title = ""
            if i == 2:
                _pcolor(axes[i, j], data, 100, title)
            else:
                _pcolor(axes[i, j], data, VMAX, title)
        # row label on left of each row via first-column ylabel
        axes[i, 0].set_ylabel(f"{rl}\nLatitude [deg]", fontsize=12)

    if save: plt.savefig(save)
    plt.show()


def plot_psd(m, save=None):
    """
    Raw PSD for HMI ref and each LCT cadence, single panel.
    """
    scale     = m["psd_scale"]
    sel       = (scale > 3.0) & np.isfinite(scale)
    scale_min = scale[sel].min()
    scale_max = scale[sel].max()
    labels    = m["labels"]

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    ax.semilogy(scale[sel], m["psd_hmi_ref_raw"][sel],
                color=CHMI, lw=2.5, ls="-", label="HMI ref (90s)", zorder=5)

    for i, lbl in enumerate(labels):
        blk = m["total"][lbl]
        if blk is None:
            continue
        ax.semilogy(scale[sel], blk["psd_raw"][sel],
                    color=cadence_color(i), lw=2.0, ls="--", label=f"LCT  {lbl}")
        pk = blk["psd_peak_raw"]
        if np.isfinite(pk) and scale_min < pk < scale_max:
            ax.axvline(pk, color=cadence_color(i), lw=1.0, ls=":", alpha=0.8)
            ax.text(pk, ax.get_ylim()[0] * 1.5, f"{pk:.1f} Mm", color=cadence_color(i),
                    fontsize=9, ha="center", va="bottom", rotation=90)

    scale_ticks = np.array([3, 5, 10, 20, 30, 50, 100, 200])
    scale_ticks = scale_ticks[(scale_ticks >= scale_min) & (scale_ticks <= scale_max)]
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlim(scale_max * 1.05, scale_min * 0.95)
    ax.set_xticks(scale_ticks)
    ax.set_xticklabels([str(t) for t in scale_ticks], fontsize=12)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("Spatial scale [Mm]", fontsize=13)
    ax.set_ylabel(r"PSD  [(m/s)$^2$ Mm]", fontsize=13)
    ax.legend(fontsize=12, framealpha=0.9)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)

    if save: plt.savefig(save)
    plt.show()


def plot_metric_vs_cadence(m, save=None):
    """
    Three-group bar chart per metric showing all three comparisons side by side.
    For each cadence: avg (green) | total (blue) | lct (red).
    Immediately shows how much of total degradation is averaging vs LCT noise.
    """
    labels  = m["labels"]
    x       = np.arange(len(labels))
    width   = 0.25

    metrics = [
        ("r",               "Pearson $r$  (raw)",                  True,  1.0),
        ("skill",           "Spatial skill  (norm. pattern)",      True,  1.0),
        ("odr_slope_raw",   "ODR slope  (raw)",                    False, 1.0),
        ("rmse_raw",        "RMSE  [m s$^{-1}$]  (raw)",          False, None),
        ("amplitude_ratio", "Amplitude ratio  RMS  (raw)",         False, 1.0),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8),
                             gridspec_kw={"hspace": 0.5, "wspace": 0.4})
    fig.suptitle("Metric vs cadence — decomposition into averaging vs LCT noise",
                 fontsize=11)

    # legend patches
    import matplotlib.patches as mpatches
    leg_handles = [
        mpatches.Patch(color=CGOOD,  label="Averaging only  (HMI-cad vs HMI-ref)"),
        mpatches.Patch(color=CHMI,   label="Total  (LCT vs HMI-ref)"),
        mpatches.Patch(color="#d62728", label="LCT noise only  (LCT vs HMI-cad)"),
    ]

    for ax, (key, ylabel, add_ref, ref_val) in zip(axes.ravel(), metrics):
        vals_avg   = [m["avg"][lbl][key] if m["avg"][lbl] is not None else np.nan
                      for lbl in labels]
        vals_total = [m["total"][lbl][key] for lbl in labels]
        vals_lct   = [m["lct"][lbl][key]   for lbl in labels]

        bars_avg   = ax.bar(x - width, vals_avg,   width, color=CGOOD,     zorder=3, label="Averaging")
        bars_total = ax.bar(x,         vals_total, width, color=CHMI,      zorder=3, label="Total")
        bars_lct   = ax.bar(x + width, vals_lct,   width, color="#d62728", zorder=3, label="LCT noise")

        ax.axhline(0, color=CGREY, lw=0.5, ls="--")
        if add_ref and ref_val is not None:
            ax.axhline(ref_val, color=CGOOD, lw=0.7, ls=":", label=f"Perfect = {ref_val}")
            ax.set_ylim(-0.15, 1.15)
        if key == "amplitude_ratio":
            ax.axhline(1, color=CGOOD, lw=0.7, ls=":")
            ax.set_ylim(0, 1.4)

        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("LCT cadence")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(ylabel, fontsize=9)

        for bars in [bars_avg, bars_total, bars_lct]:
            for bar in bars:
                v = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + abs(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
                        f"{v:.2g}", ha="center", va="bottom", fontsize=6.5)

    # shared legend at top
    fig.legend(handles=leg_handles, loc="upper center", ncol=3,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save: plt.savefig(save)
    plt.show()


def plot_degradation_decomposition(m, save=None):
    """
    The key result plot.
    For each metric, shows stacked decomposition:
      total degradation = averaging component + LCT noise component
    Uses (1 - metric) as the "error" so that 0 = perfect, 1 = worst.
    Only applicable to r, ncc, skill where perfect = 1.
    """
    labels  = m["labels"]
    x       = np.arange(len(labels))
    width   = 0.5

    metrics = [
        ("r",     "Pearson $r$"),
        ("skill", "Spatial skill"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5),
                             gridspec_kw={"wspace": 0.4})
    fig.suptitle("Degradation decomposition: averaging vs LCT noise\n"
                 "(bar height = 1 − metric, so 0 = perfect)",
                 fontsize=11)

    import matplotlib.patches as mpatches
    leg_handles = [
        mpatches.Patch(color=CGOOD,     label="Averaging component  (1 − r_avg)"),
        mpatches.Patch(color="#d62728", label="LCT noise component  (r_avg − r_lct)"),
    ]

    for ax, (key, title) in zip(axes, metrics):
        err_avg   = np.array([1 - m["avg"][lbl][key] if m["avg"][lbl] is not None else np.nan
                               for lbl in labels])
        err_lct   = np.array([1 - m["lct"][lbl][key]   for lbl in labels])
        err_total = np.array([1 - m["total"][lbl][key] for lbl in labels])

        # averaging component = err_avg
        # LCT noise on top = err_lct (relative to HMI-cad reference)
        # Note: err_total is shown as a line for verification
        ax.bar(x, err_avg, width, color=CGOOD,     zorder=3, label="Averaging")
        ax.bar(x, err_lct, width, bottom=err_avg,  color="#d62728",
               zorder=3, alpha=0.85, label="LCT noise")
        ax.plot(x, err_total, "ko--", ms=5, lw=1.2, zorder=5,
                label="Total (verification)")

        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("LCT cadence")
        ax.set_ylabel(f"1 − {title}")
        ax.set_title(title)
        ax.set_ylim(0, max(err_total.max() * 1.3, 0.2))

        for i, (ea, el, et) in enumerate(zip(err_avg, err_lct, err_total)):
            ax.text(i, ea / 2, f"{ea:.2f}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
            ax.text(i, ea + el / 2, f"{el:.2f}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

    fig.legend(handles=leg_handles + [
        plt.Line2D([0], [0], color="k", ls="--", marker="o", ms=5,
                   label="Total (verification)")],
        loc="lower center", ncol=3, fontsize=8,
        bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save: plt.savefig(save)
    plt.show()


def plot_scatter(m, save=None):
    """
    Scatter: 2 rows x N cadence columns.
    Row 0: LCT vs HMI ref (90s).  Row 1: LCT vs matched HMI cadence.
    ODR fit line and slope per panel.
    """
    labels = m["labels"]
    ncols  = len(labels)
    nrows  = 2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.2 * ncols, 5.2 * nrows),
        constrained_layout=True,
    )
    if ncols == 1:
        axes = axes[:, np.newaxis]

    def _scatter_panel(ax, ref_f, test_f, blk, color, xlabel, title=""):
        idx = np.random.choice(len(ref_f), min(5000, len(ref_f)), replace=False)
        ax.scatter(ref_f[idx], test_f[idx], s=2.0, alpha=0.25,
                   color=color, rasterized=True)
        lo = min(ref_f[idx].min(), test_f[idx].min())
        hi = max(ref_f[idx].max(), test_f[idx].max())
        ax.plot([lo, hi], [lo, hi], color=CGREY, lw=1.2, ls="--", label="1:1")
        sl     = blk["odr_slope_raw"]
        ic     = blk["odr_intercept_raw"]
        sl_err = blk["odr_slope_raw_err"]
        x_line = np.array([lo, hi])
        ax.plot(x_line, sl * x_line + ic, color=color, lw=2.0, ls="-",
                label=f"ODR: slope = {sl:.2f} +/- {sl_err:.2f}")
        ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("LCT  [m/s]", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.text(0.97, 0.05, f"r = {blk['r']:.3f}",
                transform=ax.transAxes, fontsize=11,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
        if title:
            ax.set_title(title, fontsize=13, fontweight="bold", pad=6)

    row_xlabels = ["HMI ref (90s)  [m/s]", "HMI cad  [m/s]"]

    for j, lbl in enumerate(labels):
        color     = cadence_color(j)
        blk_total = m["total"][lbl]
        blk_lct   = m["lct"][lbl]
        hmi_cad   = m["avg_hmi_raw"][lbl]
        hmi_tag   = f"HMI {lbl}" + (" (= ref)" if m["avg"][lbl] is None else "")

        _scatter_panel(axes[0, j],
                       m["vlos_hmi_ref"].ravel(),
                       blk_total["test_raw"].ravel(),
                       blk_total, color,
                       xlabel="HMI ref (90s)  [m/s]",
                       title=f"LCT {lbl}")

        _scatter_panel(axes[1, j],
                       hmi_cad.ravel(),
                       blk_lct["test_raw"].ravel(),
                       blk_lct, color,
                       xlabel=f"{hmi_tag}  [m/s]",
                       title="")

    # row labels via first-column ylabels
    axes[0, 0].set_ylabel("vs HMI ref\nLCT  [m/s]", fontsize=12)
    axes[1, 0].set_ylabel("vs HMI cad\nLCT  [m/s]", fontsize=12)

    if save: plt.savefig(save)
    plt.show()


def plot_psd_peak_vs_cadence(m, save=None):
    """
    PSD peak scale vs cadence for all three comparisons.
    X-axis uses actual cadence in seconds on a log scale so the gap between
    90s→360s (4x) and 360s→900s (2.5x) is physically proportional.
    """
    def _cadence_to_seconds(label):
        """Parse label like '90s', '360s', '900s' to integer seconds."""
        return int(label.rstrip("s"))

    labels = m["labels"]
    x_sec  = np.array([_cadence_to_seconds(lbl) for lbl in labels], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(m["psd_hmi_ref_peak"], color=CHMI, lw=1.2, ls=":",
               label=f"HMI ref peak  ({m['psd_hmi_ref_peak']:.0f} Mm)", zorder=5)

    for ckey, color, marker, ls, lbl_str in [
        ("avg",   CGOOD,     "s", "-",  "HMI-cad  (averaging only)"),
        ("total", CHMI,      "o", "--", "LCT vs HMI-ref  (total)"),
        ("lct",   "#d62728", "^", "-.", "LCT vs HMI-cad  (LCT noise)"),
    ]:
        peaks = [m[ckey][lbl]["psd_peak_norm"] if m[ckey][lbl] is not None else np.nan
                 for lbl in labels]
        # split into valid/nan segments so log axis handles gaps cleanly
        x_v = [xv for xv, p in zip(x_sec, peaks) if np.isfinite(p)]
        p_v = [p  for p       in peaks            if np.isfinite(p)]
        if x_v:
            ax.plot(x_v, p_v, marker=marker, ls=ls, color=color,
                    lw=1.5, ms=6, label=lbl_str)
        # open marker for any missing cadence (None block)
        for xv, p in zip(x_sec, peaks):
            if not np.isfinite(p):
                ax.plot(xv, m["psd_hmi_ref_peak"], marker=marker, color=color,
                        ms=7, mfc="none", mew=1.5, zorder=4)

    ax.set_xscale("log")
    ax.set_xticks(x_sec)
    ax.set_xticklabels([f"{int(s)}s" for s in x_sec], fontsize=9)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("Cadence  [s]  —  log scale")
    ax.set_ylabel("PSD peak spatial scale [Mm]")
    ax.set_title("Dominant spatial scale vs cadence\n"
                 "(log x-axis: spacing proportional to cadence ratio)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save: plt.savefig(save)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MASTER ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_all(vlos_hmi_ref, vlos_hmi_cad, vlos_lct, lon, lat,
            save_prefix=None):
    """
    Compute everything and produce all plots.

    Parameters
    ----------
    vlos_hmi_ref : 2D array [m/s] — HMI 90s reference
    vlos_hmi_cad : dict {label: 2D array [m/s]} — HMI at 360s and 900s
    vlos_lct     : dict {label: 2D array [m/s]} — LCT at 90s, 360s, 900s
    lon          : 2D array [deg] — Postel longitude grid
    lat          : 2D array [deg] — Postel latitude grid
    save_prefix  : filename prefix for saved plots, or None to display only

    Returns
    -------
    m : dict with all metrics and arrays
    """
    m = compute_all(vlos_hmi_ref, vlos_hmi_cad, vlos_lct, lon, lat)

    def _s(tag):
        return f"{save_prefix}_{tag}.pdf" if save_prefix else None

    labels = m["labels"]

    def _get(ckey, lbl, key, fmt=""):
        blk = m[ckey][lbl]
        if blk is None:
            return f"{'—':>{len(fmt) if fmt else 7}}"
        v = blk[key]
        return f"{v:{fmt}}" if fmt else str(v)

    print("\n── AMPLITUDE  (raw) ──────────────────────────────────────────────")
    print(f"  {'Cadence':<12}  {'':>18}  {'Ratio':>7}  {'RMSE m/s':>9}  {'Bias m/s':>9}")
    print(f"  {'-'*12}  {'-'*18}  {'-'*7}  {'-'*9}  {'-'*9}")
    for lbl in labels:
        for ckey, cname in [("avg","HMI-cad/ref"), ("total","LCT/ref"), ("lct","LCT/HMI-cad")]:
            blk = m[ckey][lbl]
            if blk is None:
                print(f"  {lbl:<12}  {cname:>18}  {'—':>7}  {'—':>9}  {'—':>9}")
            else:
                print(f"  {lbl:<12}  {cname:>18}  "
                      f"{blk['amplitude_ratio']:>7.3f}  "
                      f"{blk['rmse_raw']:>9.2f}  "
                      f"{blk['bias_raw']:>9.2f}")
        print()

    print("\n── STRUCTURE  (raw fields) ───────────────────────────────────────")
    print(f"  {'Cadence':<12}  {'':>18}  {'r':>7}  {'ODR slope':>12}  {'NCC':>7}  {'Skill':>7}")
    print(f"  {'-'*12}  {'-'*18}  {'-'*7}  {'-'*12}  {'-'*7}  {'-'*7}")
    for lbl in labels:
        for ckey, cname in [("avg","HMI-cad/ref"), ("total","LCT/ref"), ("lct","LCT/HMI-cad")]:
            blk = m[ckey][lbl]
            if blk is None:
                print(f"  {lbl:<12}  {cname:>18}  {'—':>7}  {'—':>12}  {'—':>7}  {'—':>7}")
            else:
                print(f"  {lbl:<12}  {cname:>18}  "
                      f"{blk['r']:>7.3f}  "
                      f"{blk['odr_slope_raw']:>7.3f}±{blk['odr_slope_raw_err']:.3f}  "
                      f"{blk['ncc']:>7.3f}  "
                      f"{blk['skill']:>7.3f}")
        print()

    print("\n── PSD PEAKS  [Mm] ───────────────────────────────────────────────")
    print(f"  HMI ref peak : {m['psd_hmi_ref_peak']:.1f} Mm  "
)
    print(f"  {'Cadence':<12}  {'':>18}  {'Raw peak':>9}  {'Norm peak':>10}")
    print(f"  {'-'*12}  {'-'*18}  {'-'*9}  {'-'*10}")
    for lbl in labels:
        for ckey, cname in [("avg","HMI-cad/ref"), ("total","LCT/ref"), ("lct","LCT/HMI-cad")]:
            blk = m[ckey][lbl]
            if blk is None:
                print(f"  {lbl:<12}  {cname:>18}  {'—':>9}  {'—':>10}")
            else:
                print(f"  {lbl:<12}  {cname:>18}  "
                      f"{blk['psd_peak_raw']:>9.1f}  "
                      f"{blk['psd_peak_norm']:>10.1f}")
        print()

    plot_vlos_maps(m,              save=_s("vlos_maps"))
    plot_scatter(m,                save=_s("scatter"))
    plot_psd(m,                    save=_s("psd"))
    plot_psd_peak_vs_cadence(m,    save=_s("psd_peak_vs_cadence"))

    return m


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    vlos_hmi_ref, vlos_hmi_cad, vlos_lct, lon, lat = load_data()

    m = run_all(
        vlos_hmi_ref,
        vlos_hmi_cad,
        vlos_lct,
        lon=lon,
        lat=lat,
        save_prefix="sg_lct",
    )
# %%
