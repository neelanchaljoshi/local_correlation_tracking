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
    "axes.titlecolor":     "#111111", "axes.linewidth":      0.8,
    "axes.grid":           True,      "grid.color":          "#dddddd",
    "grid.linewidth":      0.5,       "grid.linestyle":      "--",
    "xtick.color":         "#333333", "ytick.color":         "#333333",
    "xtick.direction":     "in",      "ytick.direction":     "in",
    "xtick.major.size":    3.5,       "ytick.major.size":    3.5,
    "xtick.minor.size":    2.0,       "ytick.minor.size":    2.0,
    "text.color":          "#222222", "font.family":         "sans-serif",
    "font.size":           9,         "axes.labelsize":      9,
    "axes.titlesize":      10,        "legend.fontsize":     8,
    "legend.framealpha":   0.9,       "legend.edgecolor":    "#cccccc",
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
SG_SCALE_MM = 30.0    # expected supergranulation dominant scale [Mm]


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
        r_odr     : ODR correlation coefficient (from residual variance)
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

    # ODR r: 1 - residual_var / total_var (analogous to R² but for ODR)
    res  = y - (slope * x + intercept)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_odr  = float(np.sqrt(max(0, 1 - ss_res / ss_tot)))
    # preserve sign from Pearson
    r_sign, _ = pearsonr(x, y)
    if r_sign < 0:
        r_odr = -r_odr

    return {
        "slope":     float(slope),
        "intercept": float(intercept),
        "r_odr":     r_odr,
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
    Returns a flat dict of scalars and arrays.
    """
    ref_n  = normalise(ref)
    test_n = normalise(test)
    _, psd_ref_norm  = compute_psd(ref_n,  pxMm)
    _, psd_test_raw  = compute_psd(test,   pxMm)
    _, psd_test_norm = compute_psd(test_n, pxMm)
    _, psd_ref_raw   = compute_psd(ref,    pxMm)

    # ODR fits — raw and normalised
    odr_raw  = odr_fit(ref,   test)
    odr_norm = odr_fit(ref_n, test_n)

    return {
        # amplitude (raw)
        "rmse_raw":        rmse(ref, test),
        "bias_raw":        bias(ref, test),
        "amplitude_ratio": amplitude_ratio(ref, test),
        # ODR (raw) — slope = amplitude ratio corrected for errors in both
        "odr_slope_raw":   odr_raw["slope"],
        "odr_slope_raw_err": odr_raw["slope_err"],
        "odr_intercept_raw": odr_raw["intercept"],
        # structural (normalised)
        "r":               pearson_r(ref_n, test_n),
        "r_odr":           odr_norm["r_odr"],
        "odr_slope_norm":  odr_norm["slope"],
        "odr_slope_norm_err": odr_norm["slope_err"],
        "ncc":             normalised_cross_correlation(ref_n, test_n),
        "skill":           spatial_skill(ref_n, test_n),
        "rmse_norm":       rmse(ref_n, test_n),
        # PSD
        "psd_raw":         psd_test_raw,
        "psd_norm":        psd_test_norm,
        "psd_ratio_raw":   np.where(psd_ref_raw  > 0,
                                    psd_test_raw  / psd_ref_raw,  np.nan),
        "psd_ratio_norm":  np.where(psd_ref_norm > 0,
                                    psd_test_norm / psd_ref_norm, np.nan),
        "psd_peak_raw":    psd_peak_scale(scale, psd_test_raw),
        "psd_peak_norm":   psd_peak_scale(scale, psd_test_norm),
        # normalised fields for plotting
        "test_n":          test_n,
        "residual_n":      test_n - ref_n,
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
    m      = {"labels": labels, "pxMm": pxMm}

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
            m["avg"][label]["hmi_cad_n"] = normalise(hmi_c)
            m["lct"][label] = _metric_block(hmi_c, lct_c, pxMm, scale)
        else:
            # no matched HMI cadence (90s) — avg undefined, lct uses ref directly
            m["avg"][label] = None
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
    if scale_min < SG_SCALE_MM < scale_max:
        ax.axvline(SG_SCALE_MM, color=CGREY, lw=0.7, ls=":", zorder=0)
        ax.text(SG_SCALE_MM, 1.01, f"SG ~{SG_SCALE_MM:.0f} Mm",
                fontsize=6, ha="center", va="bottom", color=CGREY,
                transform=ax.get_xaxis_transform())


# ══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_vlos_maps(m, save=None):
    """
    Grid of vlos maps (normalised).
    Rows: HMI ref | HMI cadence-avg | LCT | residual HMI-cad vs ref |
          residual LCT vs ref | residual LCT vs HMI-cad
    Columns: one per cadence.
    """
    labels = m["labels"]
    ncols  = len(labels)
    nrows  = 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.2 * nrows),
                             gridspec_kw={"hspace": 0.3, "wspace": 0.1})
    if ncols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("$v_{\\rm LOS}$ maps  (normalised)  |  columns = cadences",
                 fontsize=11)

    row_labels = [
        "HMI ref",
        "HMI cad-avg",
        "LCT",
        "Residual: HMI-cad − HMI-ref\n(averaging effect)",
        "Residual: LCT − HMI-ref\n(total degradation)",
        "Residual: LCT − HMI-cad\n(LCT noise only)",
    ]

    for j, lbl in enumerate(labels):
        hmi_ref_n  = m["vlos_hmi_ref_n"]
        has_hmi_cad = m["avg"][lbl] is not None
        hmi_cad_n  = m["avg"][lbl]["hmi_cad_n"] if has_hmi_cad else np.full_like(hmi_ref_n, np.nan)
        lct_n      = m["total"][lbl]["test_n"]

        rows_data = [
            hmi_ref_n,
            hmi_cad_n if has_hmi_cad else np.full_like(hmi_ref_n, np.nan),
            lct_n,
            m["avg"][lbl]["residual_n"] if has_hmi_cad else np.full_like(hmi_ref_n, np.nan),
            m["total"][lbl]["residual_n"],
            m["lct"][lbl]["residual_n"],
        ]

        for i, (data, row_lbl) in enumerate(zip(rows_data, row_labels)):
            ax = axes[i, j]
            title = (f"{lbl}" if i == 0 else "") + (f"\n{row_lbl}" if j == 0 else "")
            # first column: show row label; other columns: show cadence on top row only
            col_title = lbl if i == 0 else ""
            row_title = row_lbl if j == 0 else ""
            full_title = col_title + ("\n" if col_title and row_title else "") + row_title
            _imshow(ax, data, "RdBu_r", full_title, unit="arb.")

    plt.tight_layout()
    if save: plt.savefig(save)
    plt.show()


def plot_psd(m, save=None):
    """
    2x3 PSD figure.
    Row 0: raw PSD for all three comparisons.
    Row 1: PSD ratio (test/ref) for all three comparisons.
    Columns: (avg) HMI-cad/HMI-ref | (total) LCT/HMI-ref | (lct) LCT/HMI-cad
    """
    scale     = m["psd_scale"]
    sel       = (scale > 3.0) & np.isfinite(scale)
    scale_min = scale[sel].min()
    scale_max = scale[sel].max()
    labels    = m["labels"]

    comp_keys   = ["avg",       "total",        "lct"]
    comp_titles = ["Averaging effect\n(HMI-cad / HMI-ref)",
                   "Total degradation\n(LCT / HMI-ref)",
                   "LCT noise only\n(LCT / HMI-cad)"]
    ref_psds    = {
        "avg":   m["psd_hmi_ref_norm"],
        "total": m["psd_hmi_ref_norm"],
        "lct":   None,   # reference is HMI-cad, varies per label — ratio already stored
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                             gridspec_kw={"hspace": 0.5, "wspace": 0.38})
    fig.suptitle("Power Spectral Density (radial average) — three-way decomposition",
                 fontsize=11, y=1.01)

    for col, (ckey, ctitle) in enumerate(zip(comp_keys, comp_titles)):
        # row 0 — normalised PSD
        ax = axes[0, col]
        ax.semilogy(scale[sel], m["psd_hmi_ref_norm"][sel],
                    color=CHMI, lw=2.0, ls="-", label="HMI ref", zorder=5)
        for i, lbl in enumerate(labels):
            blk = m[ckey][lbl]
            if blk is None:
                continue
            psd = blk["psd_norm"]
            ax.semilogy(scale[sel], psd[sel],
                        color=cadence_color(i), lw=1.3, ls="--",
                        label=f"{lbl}")
            pk = blk["psd_peak_norm"]
            if np.isfinite(pk) and scale_min < pk < scale_max:
                ax.axvline(pk, color=cadence_color(i), lw=0.7, ls=":", alpha=0.7)
                ax.text(pk, 0.03, f"{pk:.0f}", fontsize=6,
                        ha="center", va="bottom", color=cadence_color(i),
                        transform=ax.get_xaxis_transform())
        ax.legend(fontsize=7)
        _psd_fmt_ax(ax, "PSD [arb.]", f"Norm. PSD — {ctitle}", scale_min, scale_max)

        # row 1 — normalised PSD ratio
        ax = axes[1, col]
        ax.axhline(1, color=CGREY, lw=0.8, ls="--", label="Ratio = 1")
        for i, lbl in enumerate(labels):
            blk = m[ckey][lbl]
            if blk is None:
                continue
            ratio = blk["psd_ratio_norm"]
            ax.plot(scale[sel], ratio[sel],
                    color=cadence_color(i), lw=1.3, label=f"{lbl}")
        ax.set_ylim(0, 2.0)
        ax.legend(fontsize=7)
        _psd_fmt_ax(ax, "PSD ratio",
                    f"Norm. PSD ratio — {ctitle}", scale_min, scale_max)

    plt.tight_layout()
    if save: plt.savefig(save, bbox_inches="tight")
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
        ("r_odr",           "ODR $r$  (normalised)",               True,  1.0),
        ("r",               "Pearson $r$  (normalised)",           True,  1.0),
        ("skill",           "Spatial skill  (normalised)",         True,  1.0),
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
        ("r_odr", "ODR $r$"),
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
    Scatter plots for each cadence, each comparison type (3 rows x N cols).
    Row 0: HMI-cad vs HMI-ref.
    Row 1: LCT vs HMI-ref.
    Row 2: LCT vs HMI-cad.
    """
    labels = m["labels"]
    ncols  = len(labels)
    fig, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 11),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.3})
    if ncols == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle("Scatter: normalised $v_{\\rm LOS}$  (3 rows = 3 comparisons)",
                 fontsize=11)

    row_info = [
        ("avg",   "HMI-ref", "HMI-cad"),
        ("total", "HMI-ref", "LCT"),
        ("lct",   "HMI-cad", "LCT"),
    ]

    for j, lbl in enumerate(labels):
        for row, (ckey, ref_lbl, test_lbl) in enumerate(row_info):
            ax = axes[row, j]
            if ckey == "avg":
                ref_f  = m["vlos_hmi_ref_n"].ravel()
                if m["avg"][lbl] is None:
                    ax.set_title(f"{lbl}  — no matched HMI cadence", fontsize=8)
                    ax.axis("off")
                    continue
                test_f = m["avg"][lbl]["hmi_cad_n"].ravel()
            elif ckey == "total":
                ref_f  = m["vlos_hmi_ref_n"].ravel()
                test_f = m["total"][lbl]["test_n"].ravel()
            else:
                ref_f  = (m["avg"][lbl]["hmi_cad_n"].ravel()
                          if m["avg"][lbl] is not None
                          else m["vlos_hmi_ref_n"].ravel())
                test_f = m["lct"][lbl]["test_n"].ravel()

            idx = np.random.choice(len(ref_f), min(4000, len(ref_f)), replace=False)
            ax.scatter(ref_f[idx], test_f[idx], s=1.5, alpha=0.3,
                       color=cadence_color(j) if ckey == "lct" else
                       (CGOOD if ckey == "avg" else CHMI),
                       rasterized=True)
            lo = min(ref_f[idx].min(), test_f[idx].min())
            hi = max(ref_f[idx].max(), test_f[idx].max())
            ax.plot([lo, hi], [lo, hi], color=CGREY, lw=1, ls="--")
            blk = m[ckey][lbl]
            r = blk["r"] if blk is not None else np.nan
            ax.set_xlabel(f"{ref_lbl}  (norm.)", fontsize=8)
            ax.set_ylabel(f"{test_lbl}  (norm.)", fontsize=8)
            ax.set_title(f"{lbl}  —  {test_lbl} vs {ref_lbl}\n$r$ = {r:.3f}",
                         fontsize=8)

    plt.tight_layout()
    if save: plt.savefig(save)
    plt.show()


def _cadence_to_seconds(label):
    """Parse cadence label like '90s', '360s', '900s' to integer seconds."""
    return int(label.rstrip("s"))

def plot_psd_peak_vs_cadence(m, save=None):
    """
    PSD peak scale vs cadence for all three comparisons.
    X-axis uses actual cadence in seconds on a log scale so the gap between
    90s→360s (4x) and 360s→900s (2.5x) is physically proportional.
    """
    labels = m["labels"]
    x_sec  = np.array([_cadence_to_seconds(lbl) for lbl in labels], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(m["psd_hmi_ref_peak"], color=CHMI, lw=1.2, ls=":",
               label=f"HMI ref peak  ({m['psd_hmi_ref_peak']:.0f} Mm)", zorder=5)
    ax.axhline(SG_SCALE_MM, color=CGREY, lw=0.7, ls="--",
               label=f"Expected SG  ({SG_SCALE_MM:.0f} Mm)")

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
        return f"{save_prefix}_{tag}.png" if save_prefix else None

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

    print("\n── STRUCTURE  (normalised) ───────────────────────────────────────")
    print(f"  {'Cadence':<12}  {'':>18}  {'r_ODR':>7}  {'r':>7}  {'NCC':>7}  {'Skill':>7}  {'ODR slope':>10}")
    print(f"  {'-'*12}  {'-'*18}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}")
    for lbl in labels:
        for ckey, cname in [("avg","HMI-cad/ref"), ("total","LCT/ref"), ("lct","LCT/HMI-cad")]:
            blk = m[ckey][lbl]
            if blk is None:
                print(f"  {lbl:<12}  {cname:>18}  {'—':>7}  {'—':>7}  {'—':>7}  {'—':>7}  {'—':>10}")
            else:
                print(f"  {lbl:<12}  {cname:>18}  "
                      f"{blk['r_odr']:>7.3f}  "
                      f"{blk['r']:>7.3f}  "
                      f"{blk['ncc']:>7.3f}  "
                      f"{blk['skill']:>7.3f}  "
                      f"{blk['odr_slope_norm']:>7.3f}±{blk['odr_slope_norm_err']:.3f}")
        print()

    print("\n── PSD PEAKS  [Mm] ───────────────────────────────────────────────")
    print(f"  HMI ref peak : {m['psd_hmi_ref_peak']:.1f} Mm  "
          f"(expected SG ~{SG_SCALE_MM:.0f} Mm)")
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

    plot_vlos_maps(m,                    save=_s("vlos_maps"))
    plot_psd(m,                          save=_s("psd"))
    plot_scatter(m,                      save=_s("scatter"))
    plot_metric_vs_cadence(m,            save=_s("metric_vs_cadence"))
    plot_degradation_decomposition(m,    save=_s("degradation"))
    plot_psd_peak_vs_cadence(m,          save=_s("psd_peak_vs_cadence"))

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
