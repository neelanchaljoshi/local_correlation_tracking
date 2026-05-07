# %%
"""
AR Inflow — Complete Analysis Script
=====================================
Single file combining:
  - Data loading and compute_all (original LCT metrics)
  - Original paper plots (flow maps, divergence maps, radial profiles, metric bars)
  - Robust diagnostics:
      METHOD 2 — Integrated flux through circular apertures (bootstrap errors)
      METHOD 3 — 2D cross-correlation in fixed AR box (bootstrap errors)

QUICK START
-----------
    m15 = compute_all(*load_data("15min"))
    m30 = compute_all(*load_data("30min"))

    # original plots
    run_paper_plots(m15, m30, save_prefix="ar_inflow")

    # robust plots (flux + XC)
    run_robust_plots(m15, m30, save_prefix="ar_inflow")

NORMALISATION PHILOSOPHY
------------------------
  RAW velocities  → amplitude metrics (RMSE, bias, ratio), physical divergence [s⁻¹]
  NORMALISED      → structural metrics (Pearson r, radial profile shape)

The ~2.7x amplitude bias in PMI 2K is a known artefact of PSF-broadened CCF
peaks; normalisation removes it so structural fidelity can be assessed cleanly.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import matplotlib.patheffects as mpe
import matplotlib.patches as mpatches
from scipy.ndimage import (gaussian_filter, uniform_filter1d,
                            center_of_mass, zoom, binary_dilation)
from scipy.stats import pearsonr
from scipy.signal import fftconvolve
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLE — large fonts, clean axes, publication-ready
# ══════════════════════════════════════════════════════════════════════════════

BASE_FS = 14   # base font size — change this one number to rescale everything

plt.rcParams.update({
    "figure.facecolor":    "white",       "axes.facecolor":      "white",
    "axes.edgecolor":      "#222222",     "axes.labelcolor":     "#111111",
    "axes.titlecolor":     "#111111",     "axes.linewidth":      1.2,
    "axes.grid":           False,
    "xtick.color":         "#222222",     "ytick.color":         "#222222",
    "xtick.direction":     "in",          "ytick.direction":     "in",
    "xtick.major.size":    6.0,           "ytick.major.size":    6.0,
    "xtick.minor.size":    3.5,           "ytick.minor.size":    3.5,
    "xtick.major.width":   1.0,           "ytick.major.width":   1.0,
    "xtick.labelsize":     BASE_FS,       "ytick.labelsize":     BASE_FS,
    "text.color":          "#111111",     "font.family":         "sans-serif",
    "font.size":           BASE_FS,       "axes.labelsize":      BASE_FS + 1,
    "axes.titlesize":      BASE_FS + 3,   "axes.titlepad":       10,
    "legend.fontsize":     BASE_FS,       "legend.framealpha":   0.92,
    "legend.edgecolor":    "#bbbbbb",     "legend.borderpad":    0.6,
    "figure.dpi":          150,           "savefig.dpi":         300,
    "savefig.facecolor":   "white",       "savefig.bbox":        "tight",
    "image.origin":        "lower",       "image.interpolation": "nearest",
    "pdf.fonttype":        42,            "ps.fonttype":         42,
})

# ── colour palette ─────────────────────────────────────────────────────────
C4K   = "#1f77b4"   # blue   — HMI 4K
C2K   = "#d62728"   # red    — PMI 2K
CGOOD = "#2ca02c"   # green  — 15-min cadence
C30   = "#ff7f0e"   # orange — 30-min cadence
CGREY = "#7f7f7f"   # grey   — neutral lines

# ── constants ──────────────────────────────────────────────────────────────
R_SUN_MM        = 695.7   # Mm
PIXEL_SCALE_DEG = 0.5     # degrees per pixel (both grids)
MAG_THRESHOLD_G = 50.0    # |Bz| threshold for AR core [Gauss]
DILATION_PX     = 5       # dilation radius [pixels]
INFLOW_PERCENTILE = 20    # bottom N% of 4K divergence = inflow zone
QUIVER_STRIDE   = 2

# robust-analysis constants
AR_PAD_DEG  = 1.0    # padding around Bz bbox for XC box [deg]
AR_SIZE_DEG = 19.0   # approximate AR diameter [deg] — sets R_max


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(cadence="15min"):
    """
    Load LCT output and the LOS magnetogram for a given cadence.

    Returns
    -------
    vx_4k, vy_4k, vx_2k, vy_2k : 2D arrays [m/s]
    bz                           : 2D array [Gauss] on flow-map grid
    longitude, latitude          : 1D arrays [degrees]
    """
    BASE = ('/data/seismo/joshin/pipeline-test/local_correlation_tracking/'
            'pmi/ar_inflow/data/data_cleaned/')

    if cadence == "15min":
        f1 = np.load(BASE + 'smooth_data_2k_15.npz')
        f2 = np.load(BASE + 'smooth_data_4k_15.npz')
    elif cadence == "30min":
        f1 = np.load(BASE + 'smooth_data_2k_30.npz')
        f2 = np.load(BASE + 'smooth_data_4k_30.npz')
    else:
        raise ValueError(f"Unknown cadence '{cadence}'. Use '15min' or '30min'.")

    vx_2k     = f1['smooth_zx_corrected']
    vy_2k     = -f1['smooth_zy_corrected']
    vx_4k     = f2['smooth_zx_corrected']
    vy_4k     = -f2['smooth_zy_corrected']
    longitude = f2['longitude']
    latitude  = f2['latitude']

    bz_file    = np.load(BASE + 'magnetogram_cropped.npz')
    bz_highres = bz_file['img_cropped']
    bz         = _downsample_bz(bz_highres, target_shape=vx_4k.shape)

    return vx_4k, vy_4k, vx_2k, vy_2k, bz, longitude, latitude


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _px_scale_m(pixel_scale_deg=PIXEL_SCALE_DEG):
    return np.deg2rad(pixel_scale_deg) * R_SUN_MM * 1e6

def _px_scale_Mm(pixel_scale_deg=PIXEL_SCALE_DEG):
    return _px_scale_m(pixel_scale_deg) / 1e6

# alias used by robust code
def _px_Mm(pixel_scale_deg=PIXEL_SCALE_DEG):
    return np.deg2rad(pixel_scale_deg) * R_SUN_MM

def _downsample_bz(bz_highres, target_shape):
    factor = (target_shape[0] / bz_highres.shape[0],
              target_shape[1] / bz_highres.shape[1])
    return zoom(np.nan_to_num(bz_highres), factor, order=1)

def _normalise_vector(vx, vy):
    scale = np.std(np.hypot(vx, vy))
    return (vx - np.mean(vx)) / scale, (vy - np.mean(vy)) / scale

def _divergence_physical(vx, vy, pixel_scale_deg=PIXEL_SCALE_DEG):
    px = _px_scale_m(pixel_scale_deg)
    return np.gradient(vx, axis=1) / px + np.gradient(vy, axis=0) / px

def _divergence_normalised(vx_n, vy_n):
    return np.gradient(vx_n, axis=1) + np.gradient(vy_n, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# SCALAR METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _flat(a, mask=None):
    return a[mask].ravel() if mask is not None else a.ravel()

def rmse(ref, test, mask=None):
    d = (ref - test)[mask] if mask is not None else ref - test
    return float(np.sqrt(np.nanmean(d**2)))

def bias(ref, test, mask=None):
    d = (test - ref)[mask] if mask is not None else test - ref
    return float(np.nanmean(d))

def pearson_r(ref, test, mask=None):
    r, _ = pearsonr(_flat(ref, mask), _flat(test, mask))
    return float(r)

def vector_skill(vx_ref, vy_ref, vx_test, vy_test, mask=None):
    axr = _flat(vx_ref, mask); ayr = _flat(vy_ref, mask)
    axt = _flat(vx_test, mask); ayt = _flat(vy_test, mask)
    num = np.nansum(axr * axt + ayr * ayt)
    den = np.sqrt(np.nansum(axr**2 + ayr**2) * np.nansum(axt**2 + ayt**2))
    return float(num / den) if den > 0 else np.nan

def amplitude_ratio(vx_4k, vy_4k, vx_2k, vy_2k):
    return float(np.mean(np.hypot(vx_2k, vy_2k)) /
                 np.mean(np.hypot(vx_4k, vy_4k)))


# ══════════════════════════════════════════════════════════════════════════════
# MASKING
# ══════════════════════════════════════════════════════════════════════════════

def make_inflow_mask(div_4k_phys, percentile=INFLOW_PERCENTILE):
    return div_4k_phys < np.percentile(div_4k_phys, percentile)

def make_ar_mask_from_bz(bz, mag_threshold_G=MAG_THRESHOLD_G,
                          dilation_px=DILATION_PX):
    core = np.abs(bz) > mag_threshold_G
    return binary_dilation(core, iterations=dilation_px)

def make_field_strength_masks(bz):
    bz_abs = np.abs(bz)
    return {
        "quiet":  bz_abs <  50,
        "medium": (bz_abs >= 50) & (bz_abs < 300),
        "strong": bz_abs >= 300,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RADIAL PROFILE
# ══════════════════════════════════════════════════════════════════════════════

def inflow_centre(div_4k_phys, smooth_sigma=3):
    smoothed = gaussian_filter(div_4k_phys, sigma=smooth_sigma)
    inflow   = np.where(smoothed < 0, -smoothed, 0)
    return center_of_mass(inflow)

def radial_profile_stats(field, centre):
    cy, cx  = centre
    y, x    = np.indices(field.shape)
    r       = np.hypot(x - cx, y - cy).astype(int)
    r_flat  = r.ravel(); f_flat = field.ravel()
    counts  = np.bincount(r_flat)
    mean    = np.bincount(r_flat, weights=f_flat) / counts
    mean_sq = np.bincount(r_flat, weights=f_flat**2) / counts
    var     = np.maximum(mean_sq - mean**2, 0)
    return mean, np.sqrt(var), counts

def _zero_crossing_after(profile, start_idx, r_Mm):
    zc = np.where(np.diff(np.sign(profile[start_idx:])))[0]
    return float(r_Mm[start_idx + zc[0]]) if len(zc) > 0 else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# MASTER COMPUTE — original metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_all(vx_4k, vy_4k, vx_2k, vy_2k,
                bz=None, longitude=None, latitude=None,
                pixel_scale_deg=PIXEL_SCALE_DEG):
    """Compute all LCT metrics and intermediate arrays."""
    m    = {}
    pxMm = _px_scale_Mm(pixel_scale_deg)

    vx_4n, vy_4n = _normalise_vector(vx_4k, vy_4k)
    vx_2n, vy_2n = _normalise_vector(vx_2k, vy_2k)

    div_4k_phys = _divergence_physical(vx_4k, vy_4k, pixel_scale_deg)
    div_2k_phys = _divergence_physical(vx_2k, vy_2k, pixel_scale_deg)
    div_4k_norm = _divergence_normalised(vx_4n, vy_4n)
    div_2k_norm = _divergence_normalised(vx_2n, vy_2n)

    speed_4k = np.hypot(vx_4k, vy_4k)
    speed_2k = np.hypot(vx_2k, vy_2k)

    imask    = make_inflow_mask(div_4k_phys)
    ar_mask  = make_ar_mask_from_bz(bz) if bz is not None else None
    bz_masks = make_field_strength_masks(bz) if bz is not None else None

    com    = inflow_centre(div_4k_phys)
    cy, cx = com
    shape  = div_4k_phys.shape
    max_r  = int(min(cy, shape[0]-cy, cx, shape[1]-cx)) - 2
    y_idx, x_idx = np.indices(shape)
    r_map  = np.hypot(x_idx - cx, y_idx - cy)

    m["amplitude_ratio"] = amplitude_ratio(vx_4k, vy_4k, vx_2k, vy_2k)
    m["rmse_speed_raw"]  = rmse(speed_4k, speed_2k)
    m["bias_speed_raw"]  = bias(speed_4k, speed_2k)
    m["rmse_div_raw"]    = rmse(div_4k_phys, div_2k_phys)
    m["bias_div_raw"]    = bias(div_4k_phys, div_2k_phys)

    m["r_vx"]         = pearson_r(vx_4n, vx_2n)
    m["r_vy"]         = pearson_r(vy_4n, vy_2n)
    m["r_speed"]      = pearson_r(np.hypot(vx_4n, vy_4n), np.hypot(vx_2n, vy_2n))
    m["r_div"]        = pearson_r(div_4k_norm, div_2k_norm)
    m["vector_skill"] = vector_skill(vx_4n, vy_4n, vx_2n, vy_2n)

    m["r_div_inflow"]    = pearson_r(div_4k_norm, div_2k_norm, mask=imask)
    m["rmse_div_inflow"] = rmse(div_4k_norm, div_2k_norm, mask=imask)
    m["bias_div_inflow"] = bias(div_4k_norm, div_2k_norm, mask=imask)

    if ar_mask is not None:
        m["r_div_ar"]    = pearson_r(div_4k_norm, div_2k_norm, mask=ar_mask)
        m["rmse_div_ar"] = rmse(div_4k_norm, div_2k_norm, mask=ar_mask)
        m["bias_div_ar"] = bias(div_4k_norm, div_2k_norm, mask=ar_mask)
        for label, fmask in bz_masks.items():
            m[f"r_div_{label}"] = (pearson_r(div_4k_norm, div_2k_norm, mask=fmask)
                                   if fmask.any() else np.nan)

    p4_raw, p4_std, p4_counts = radial_profile_stats(div_4k_norm, com)
    p2_raw, p2_std, p2_counts = radial_profile_stats(div_2k_norm, com)
    p4 = uniform_filter1d(p4_raw[:max_r], size=3)
    p2 = uniform_filter1d(p2_raw[:max_r], size=3)
    p4_err = uniform_filter1d(p4_std[:max_r] / np.sqrt(p4_counts[:max_r]), size=3)
    p2_err = uniform_filter1d(p2_std[:max_r] / np.sqrt(p2_counts[:max_r]), size=3)
    r_Mm   = np.arange(max_r) * pxMm

    trough_px_4k = int(np.argmin(p4))
    trough_px_2k = int(np.argmin(p2))
    inner_4k = r_map <  trough_px_4k
    outer_4k = (r_map >= trough_px_4k) & (r_map < max_r)
    inner_2k = r_map <  trough_px_2k
    outer_2k = (r_map >= trough_px_2k) & (r_map < max_r)

    m.update({
        "r_div_inner_4k":  pearson_r(div_4k_norm, div_2k_norm, mask=inner_4k),
        "r_div_outer_4k":  pearson_r(div_4k_norm, div_2k_norm, mask=outer_4k),
        "r_div_inner_2k":  pearson_r(div_4k_norm, div_2k_norm, mask=inner_2k),
        "r_div_outer_2k":  pearson_r(div_4k_norm, div_2k_norm, mask=outer_2k),
        "trough_px_4k":    trough_px_4k,
        "trough_px_2k":    trough_px_2k,
        "trough_Mm_4k":    trough_px_4k * pxMm,
        "trough_Mm_2k":    trough_px_2k * pxMm,
        "trough_shift_Mm": (trough_px_2k - trough_px_4k) * pxMm,
        "trough_Mm":       trough_px_4k * pxMm,
        "extent_Mm_4k":    _zero_crossing_after(p4, trough_px_4k, r_Mm),
        "extent_Mm_2k":    _zero_crossing_after(p2, trough_px_2k, r_Mm),
        "radial_r_Mm":     r_Mm,
        "radial_p4":       p4,
        "radial_p2":       p2,
        "radial_p4_err":   p4_err,
        "radial_p2_err":   p2_err,
        "vx_4k": vx_4k,  "vy_4k": vy_4k,  "vx_2k": vx_2k,  "vy_2k": vy_2k,
        "vx_4n": vx_4n,  "vy_4n": vy_4n,  "vx_2n": vx_2n,  "vy_2n": vy_2n,
        "div_4k_phys": div_4k_phys, "div_2k_phys": div_2k_phys,
        "div_4k_norm": div_4k_norm, "div_2k_norm": div_2k_norm,
        "speed_4k": speed_4k,       "speed_2k": speed_2k,
        "inflow_mask": imask,       "ar_mask": ar_mask,
        "bz": bz,                   "pxMm": pxMm,
        "longitude": longitude,     "latitude": latitude,
    })
    return m


# ══════════════════════════════════════════════════════════════════════════════
# SHARED PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _axis_kw(longitude, latitude):
    if longitude is not None and latitude is not None:
        return {"extent": [longitude[0], longitude[-1],
                            latitude[0],  latitude[-1]], "origin": "lower"}
    return {}

def _set_latlon_ticks(ax, longitude, latitude):
    if longitude is not None and latitude is not None:
        ax.set_xlabel("Longitude [°]", fontsize=BASE_FS + 1)
        ax.set_ylabel("Latitude [°]",  fontsize=BASE_FS + 1)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4, integer=True))
    else:
        ax.set_xticks([]); ax.set_yticks([])

def _overlay_bz(ax, bz, longitude=None, latitude=None):
    kw = _axis_kw(longitude, latitude)
    ax.contour(bz, levels=[ 100, 300, 500], colors="white",
               linewidths=1.2, linestyles="-",  **kw)
    ax.contour(bz, levels=[-500,-300,-100], colors="#aaaaaa",
               linewidths=1.2, linestyles="--", **kw)

def _overlay_ar_mask(ax, mask, longitude=None, latitude=None):
    kw = _axis_kw(longitude, latitude)
    ax.contour(mask.astype(float), levels=[0.5],
               colors="#eeeeee", linewidths=1.0, linestyles="--", **kw)

def _streamplot(ax, vx, vy, longitude=None, latitude=None,
                color="white", density=1.0, lw=0.9, alpha=0.50):
    """Streamlines — clean direction indicator, no arrow clutter."""
    ny, nx = vx.shape
    x = longitude if longitude is not None else np.arange(nx, dtype=float)
    y = latitude  if latitude  is not None else np.arange(ny, dtype=float)
    ax.streamplot(x, y, vx, vy, color=color, density=density,
                  linewidth=lw, arrowsize=1.2, arrowstyle="->",
                  broken_streamlines=True)

def _style_ax(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
    ax.xaxis.set_major_locator(mticker.AutoLocator())
    ax.yaxis.set_major_locator(mticker.AutoLocator())
    ax.grid(True, ls="--", alpha=0.35, lw=0.7)

def _nice_colorbar(im, ax, label):
    cb = plt.colorbar(im, ax=ax, fraction=0.044, pad=0.03)
    cb.set_label(label, fontsize=BASE_FS + 1)
    cb.ax.tick_params(direction="in", labelsize=BASE_FS)
    cb.outline.set_linewidth(0.9)
    return cb


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 0 — FLOW MAPS  (inferno colourmap + streamlines)
# ══════════════════════════════════════════════════════════════════════════════

def plot_flow_maps(m15, m30, save=None):
    """
    2×2: LCT speed [m/s] with streamlines.
    Colourmap: inferno (deep black → vivid yellow, high contrast).
    Streamlines replace quiver — direction without arrow clutter.
    """
    lon = m15.get("longitude"); lat = m15.get("latitude")

    vmax_spd = max(np.nanpercentile(m15["speed_2k"], 98),
                   np.nanpercentile(m30["speed_2k"], 98))

    if lon is not None and lat is not None:
        asp = abs(lat[-1]-lat[0]) / abs(lon[-1]-lon[0])
    else:
        asp = "equal"

    kw_img = _axis_kw(lon, lat)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.10})

    row_data = [(m15, "15 min"), (m30, "30 min")]
    col_data = [("speed_4k","vx_4k","vy_4k","HMI 4K"),
                ("speed_2k","vx_2k","vy_2k","PMI 2K")]
    ims = []

    for row_i, (m, row_label) in enumerate(row_data):
        for col_i, (spd_key, vx_key, vy_key, col_label) in enumerate(col_data):
            ax = axes[row_i, col_i]
            im = ax.imshow(m[spd_key], cmap="plasma",
                           vmin=0, vmax=vmax_spd, aspect=asp, **kw_img)
            _streamplot(ax, m[vx_key], m[vy_key], lon, lat)
            _set_latlon_ticks(ax, lon, lat)
            for sp in ax.spines.values(): sp.set_linewidth(1.0)
            if row_i == 0:
                ax.set_title(col_label, fontsize=BASE_FS+4, pad=8,
                             fontweight="semibold")
            if col_i == 0:
                ax.set_ylabel(f"{row_label}\n\nLatitude [°]",
                              fontsize=BASE_FS+1)
            else:
                ax.set_ylabel("")
            if col_i == 1:
                ax.text(0.97, 0.97, f"2K/4K = {m['amplitude_ratio']:.2f}×",
                        transform=ax.transAxes, fontsize=BASE_FS,
                        color="white", va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                                  alpha=0.45, edgecolor="none"))
            ims.append(im)

    fig.subplots_adjust(right=0.87)
    cax = fig.add_axes([0.89,
                        axes[1,1].get_position().y0,
                        0.020,
                        axes[0,1].get_position().y1 - axes[1,1].get_position().y0])
    cb = fig.colorbar(ims[0], cax=cax)
    cb.set_label("Speed  [m s$^{-1}$]", fontsize=BASE_FS+1)
    cb.ax.tick_params(direction="in", labelsize=BASE_FS)
    cb.outline.set_linewidth(0.9)

    if save: plt.savefig(save, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — DIVERGENCE MAPS  (RdBu_r, saturated & familiar)
# ══════════════════════════════════════════════════════════════════════════════

def plot_divergence_maps(m15, m30, save=None):
    """2×3: physical divergence [s⁻¹]. RdBu_r colourmap."""
    lon = m15.get("longitude"); lat = m15.get("latitude")
    asp = (abs(lat[-1]-lat[0]) / abs(lon[-1]-lon[0])
           if lon is not None and lat is not None else "equal")

    all_divs = np.concatenate([m15["div_4k_phys"].ravel(),
                                m15["div_2k_phys"].ravel(),
                                m30["div_4k_phys"].ravel(),
                                m30["div_2k_phys"].ravel()])
    vmax_div = np.nanpercentile(np.abs(all_divs), 99)
    kw_img   = _axis_kw(lon, lat)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8),
                             gridspec_kw={"hspace": 0.02, "wspace": 0.10})
    fig.subplots_adjust(right=0.87)
    row_data = [(m15, "15 min"), (m30, "30 min")]
    last_ims = {}

    for row_i, (m, row_label) in enumerate(row_data):
        d4, d2 = m["div_4k_phys"], m["div_2k_phys"]
        panels = [(d4, "HMI 4K"), (d2, "PMI 2K"),
                  (d2-d4, "Residual  2K$-$4K")]
        for col_i, (arr, col_title) in enumerate(panels):
            ax = axes[row_i, col_i]
            im = ax.imshow(arr, cmap="RdBu_r",
                           vmin=-vmax_div, vmax=vmax_div,
                           aspect=asp, **kw_img)
            _set_latlon_ticks(ax, lon, lat)
            for sp in ax.spines.values(): sp.set_linewidth(1.0)
            if m["ar_mask"] is not None:
                _overlay_ar_mask(ax, m["ar_mask"], lon, lat)
            if m["bz"] is not None:
                _overlay_bz(ax, m["bz"], lon, lat)
            if row_i == 0:
                ax.set_title(col_title, fontsize=BASE_FS+3, pad=8,
                             fontweight="semibold")
            if col_i == 0:
                ax.set_ylabel(f"{row_label}\n\nLatitude [°]",
                              fontsize=BASE_FS+1)
            else:
                ax.set_ylabel("")
            if col_i == 2:
                last_ims[row_i] = im

    for row_i in range(2):
        pos = axes[row_i, 2].get_position()
        cax = fig.add_axes([0.895, pos.y0, 0.016, pos.height])
        cb  = fig.colorbar(last_ims[row_i], cax=cax)
        cb.set_label("s$^{-1}$", fontsize=BASE_FS+1)
        cb.ax.tick_params(direction="in", labelsize=BASE_FS)
        cb.outline.set_linewidth(0.9)

    legend_handles = []
    if m15["ar_mask"] is not None:
        legend_handles.append(
            mlines.Line2D([], [], color="#eeeeee", ls="--", lw=1.4,
                          label=f"AR mask  $|B_z| > {MAG_THRESHOLD_G:.0f}$ G"))
    if m15["bz"] is not None:
        legend_handles += [
            mlines.Line2D([], [], color="white", ls="-", lw=2.0,
                          label="$B_+$ contours",
                          path_effects=[mpe.withStroke(linewidth=3,
                                                        foreground="#888888")]),
            mlines.Line2D([], [], color="#aaaaaa", ls="--", lw=1.4,
                          label="$B_-$ contours"),
        ]
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=len(legend_handles), framealpha=0.92,
                   fontsize=BASE_FS, bbox_to_anchor=(0.5, 0.0))

    if save: plt.savefig(save, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — RADIAL DIVERGENCE PROFILES
# ══════════════════════════════════════════════════════════════════════════════

def plot_radial_profiles_comparison(m15, m30, save=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True,
                             gridspec_kw={"wspace": 0.05})
    for ax, (m, title) in zip(axes, [(m15, "15 min cadence"),
                                      (m30, "30 min cadence")]):
        r  = m["radial_r_Mm"]
        p4 = m["radial_p4"];    p2 = m["radial_p2"]
        e4 = m["radial_p4_err"]; e2 = m["radial_p2_err"]

        ax.fill_between(r, p4-e4, p4+e4, color=C4K, alpha=0.22)
        ax.fill_between(r, p2-e2, p2+e2, color=C2K, alpha=0.22)
        ax.plot(r, p4, color=C4K, lw=2.5, label="HMI 4K")
        ax.plot(r, p2, color=C2K, lw=2.5, ls="--", label="PMI 2K")
        ax.axhline(0, color=CGREY, lw=1.0)
        ax.set_title(title, fontsize=BASE_FS+3, pad=8)
        ax.set_xlabel("Radius [Mm]", fontsize=BASE_FS+1)
        ax.grid(True, ls="--", alpha=0.35)
        ax.legend(fontsize=BASE_FS)
        ax.tick_params(labelsize=BASE_FS)

    axes[0].set_ylabel("Normalised divergence", fontsize=BASE_FS+1)
    plt.tight_layout()
    if save: plt.savefig(save, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — STRUCTURAL METRIC BAR CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_metric_comparison(m15, m30, save=None):
    metric_keys   = ["r_speed","r_div","r_div_quiet","r_div_medium","r_div_strong"]
    metric_labels = [
        "$r_{\\mathrm{speed}}$\n(full domain)",
        "$r_{\\nabla v}$\n(full domain)",
        "$r_{\\nabla v}$\nquiet  $|B|<50$ G",
        "$r_{\\nabla v}$\nmedium  $50$–$300$ G",
        "$r_{\\nabla v}$\nstrong  $|B|>300$ G",
    ]
    vals_15 = [m15.get(k, np.nan) for k in metric_keys]
    vals_30 = [m30.get(k, np.nan) for k in metric_keys]
    x       = np.arange(len(metric_keys))
    width   = 0.28

    fig, ax = plt.subplots(figsize=(14, 8))
    bars_15 = ax.bar(x-width/2, vals_15, width, color=CGOOD, label="15 min",
                     zorder=3, edgecolor="white", linewidth=0.8)
    bars_30 = ax.bar(x+width/2, vals_30, width, color=C30, label="30 min",
                     zorder=3, edgecolor="white", linewidth=0.8)

    ax.axhline(1, color=CGREY, lw=1.2, ls=":", zorder=2, label="Perfect = 1")
    ax.axhline(0, color=CGREY, lw=0.8, ls="--", zorder=2)

    for bar, v in zip(bars_15, vals_15):
        if np.isfinite(v):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=BASE_FS, color="#1a5c1a", fontweight="bold")
    for bar, v in zip(bars_30, vals_30):
        if np.isfinite(v):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=BASE_FS, color="#8b3a00", fontweight="bold")

    for i, (v15, v30) in enumerate(zip(vals_15, vals_30)):
        if not (np.isfinite(v15) and np.isfinite(v30)): continue
        delta = v30-v15; sign = "+" if delta >= 0 else ""
        ax.text(x[i], max(v15,v30)+0.08, f"$\\Delta${sign}{delta:.3f}",
                ha="center", va="bottom", fontsize=BASE_FS,
                color="#444444", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=BASE_FS+4)
    ax.tick_params(axis="y", labelsize=BASE_FS)
    ax.set_ylim(-0.05, 1.38)
    ax.set_ylabel("Pearson $r$  [normalised fields]", fontsize=BASE_FS+2)
    ax.set_title("Structural fidelity: PMI 2K vs HMI 4K\n"
                 "Grouped by cadence  (15 min vs 30 min)",
                 fontsize=BASE_FS+4, pad=12)
    ax.legend(loc="lower right", fontsize=BASE_FS+1)
    ax.grid(True, axis="y", color="#e0e0e0", lw=0.6, ls="--", zorder=0)
    plt.tight_layout()
    if save: plt.savefig(save, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ROBUST ANALYSIS — SHARED GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

def bz_centroid(bz, threshold_G=MAG_THRESHOLD_G):
    """Return |Bz|-weighted centroid (cy, cx) in pixel coordinates."""
    mask  = np.abs(bz) > threshold_G
    w     = np.abs(bz) * mask
    total = w.sum()
    if total == 0:
        return np.array(bz.shape) / 2
    iy, ix = np.indices(bz.shape)
    return float((w * iy).sum() / total), float((w * ix).sum() / total)


def ar_bounding_box(bz, threshold_G=MAG_THRESHOLD_G, pad_deg=AR_PAD_DEG,
                    pixel_scale_deg=PIXEL_SCALE_DEG):
    """Axis-aligned bounding box of AR pixels, padded by pad_deg."""
    pad_px = int(np.round(pad_deg / pixel_scale_deg))
    mask   = np.abs(bz) > threshold_G
    rows   = np.where(mask.any(axis=1))[0]
    cols   = np.where(mask.any(axis=0))[0]
    r0 = max(0,              rows[0]  - pad_px)
    r1 = min(bz.shape[0]-1, rows[-1] + pad_px)
    c0 = max(0,              cols[0]  - pad_px)
    c1 = min(bz.shape[1]-1, cols[-1] + pad_px)
    return slice(r0, r1+1), slice(c0, c1+1)


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — INTEGRATED FLUX THROUGH CIRCULAR APERTURES
# ══════════════════════════════════════════════════════════════════════════════

def _perimeter_pixels(cy, cx, R_px, shape):
    iy, ix = np.indices(shape)
    dy = iy - cy; dx = ix - cx
    r  = np.hypot(dy, dx)
    mask   = (r >= R_px - 0.5) & (r < R_px + 0.5)
    r_vals = r[mask]
    ny = dy[mask] / np.where(r_vals > 0, r_vals, 1)
    nx = dx[mask] / np.where(r_vals > 0, r_vals, 1)
    return iy[mask], ix[mask], ny, nx

def integrated_flux(vx, vy, cy, cx, R_px):
    rows, cols, ny, nx = _perimeter_pixels(cy, cx, R_px, vx.shape)
    return float((vx[rows, cols] * nx + vy[rows, cols] * ny).sum())

def bootstrap_flux_curve(vx, vy, cy, cx, R_px_arr, pxMm,
                          n_boot=300, seed=42):
    """Bootstrap 1-sigma uncertainty on flux at each radius."""
    rng      = np.random.default_rng(seed)
    flux_err = np.zeros(len(R_px_arr))
    for i, R_px in enumerate(R_px_arr):
        rows, cols, ny, nx = _perimeter_pixels(cy, cx, R_px, vx.shape)
        N  = len(rows)
        if N == 0: continue
        vn = vx[rows, cols] * nx + vy[rows, cols] * ny
        boot = np.array([vn[rng.integers(0, N, size=N)].sum()
                         for _ in range(n_boot)])
        flux_err[i] = boot.std() * pxMm
    return flux_err


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3 — 2D CROSS-CORRELATION IN FIXED AR BOX
# ══════════════════════════════════════════════════════════════════════════════

def normalised_xcorr_2d(ref, test):
    ref_z  = ref  - ref.mean()
    test_z = test - test.mean()
    norm   = np.linalg.norm(ref_z) * np.linalg.norm(test_z)
    if norm == 0:
        return np.zeros((2*ref.shape[0]-1, 2*ref.shape[1]-1))
    return fftconvolve(ref_z, test_z[::-1, ::-1], mode="full") / norm

def xcorr_peak(xc):
    peak_val = xc.max()
    peak_idx = np.unravel_index(xc.argmax(), xc.shape)
    centre   = ((xc.shape[0]-1)//2, (xc.shape[1]-1)//2)
    return float(peak_val), int(peak_idx[0]-centre[0]), int(peak_idx[1]-centre[1])

def xcorr_in_box(m, rslice, cslice):
    pxMm = _px_Mm()
    out  = {}
    for key, ref_key, test_key in [
        ("div", "div_4k_norm", "div_2k_norm"),
        ("vx",  "vx_4n",      "vx_2n"),
        ("vy",  "vy_4n",      "vy_2n"),
    ]:
        ref  = m[ref_key][rslice, cslice]
        test = m[test_key][rslice, cslice]
        xc   = normalised_xcorr_2d(ref, test)
        pv, dy, dx = xcorr_peak(xc)
        out[key] = {"xc": xc, "peak_val": pv,
                    "dy_px": dy, "dx_px": dx,
                    "dy_Mm": dy * pxMm, "dx_Mm": dx * pxMm}
    return out

def bootstrap_xcorr_peak(ref, test, n_boot=300, seed=42):
    """Bootstrap 1-sigma uncertainty on the XC peak value."""
    rng   = np.random.default_rng(seed)
    nr, nc = ref.shape; N = nr * nc
    ref_f  = ref.ravel(); test_f = test.ravel()
    peaks  = np.zeros(n_boot)
    for b in range(n_boot):
        idx    = rng.integers(0, N, size=N)
        xc_b   = normalised_xcorr_2d(ref_f[idx].reshape(nr, nc),
                                      test_f[idx].reshape(nr, nc))
        peaks[b] = xc_b.max()
    return float(peaks.mean()), float(peaks.std())


# ══════════════════════════════════════════════════════════════════════════════
# MASTER COMPUTE — robust diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def compute_robust(m, pixel_scale_deg=PIXEL_SCALE_DEG, n_boot=300):
    """Run flux + XC diagnostics with bootstrap uncertainties."""
    bz   = m["bz"]; r = {}
    cy, cx = bz_centroid(bz)
    rslice, cslice = ar_bounding_box(bz)
    pxMm = _px_Mm(pixel_scale_deg)

    r.update({"cy": cy, "cx": cx, "rslice": rslice, "cslice": cslice,
               "bz": bz, "longitude": m.get("longitude"),
               "latitude": m.get("latitude")})

    # flux curves
    R_min_px = max(1, int(np.round(2.0 / pxMm)))
    R_max_px = int(np.round(np.deg2rad(AR_SIZE_DEG/2) * R_SUN_MM / pxMm))
    R_px_arr = np.arange(R_min_px, R_max_px + 1)
    R_Mm     = R_px_arr * pxMm

    flux_4k = np.array([integrated_flux(m["vx_4k"], m["vy_4k"], cy, cx, R)
                         for R in R_px_arr]) * pxMm
    flux_2k = np.array([integrated_flux(m["vx_2k"], m["vy_2k"], cy, cx, R)
                         for R in R_px_arr]) * pxMm

    print(f"  Bootstrapping flux ({n_boot} iterations)...")
    err_4k = bootstrap_flux_curve(m["vx_4k"], m["vy_4k"], cy, cx,
                                   R_px_arr, pxMm, n_boot=n_boot)
    err_2k = bootstrap_flux_curve(m["vx_2k"], m["vy_2k"], cy, cx,
                                   R_px_arr, pxMm, n_boot=n_boot)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid     = np.abs(flux_4k) > 0.1 * np.abs(flux_4k).max()
        ratio     = np.where(valid, flux_2k / flux_4k, np.nan)
        ratio_err = np.where(
            valid,
            np.abs(ratio) * np.sqrt(
                (err_2k / np.where(np.abs(flux_2k)>0, np.abs(flux_2k), np.nan))**2 +
                (err_4k / np.where(np.abs(flux_4k)>0, np.abs(flux_4k), np.nan))**2),
            np.nan)

    r.update({"R_Mm": R_Mm, "flux_4k": flux_4k, "flux_2k": flux_2k,
               "flux_err_4k": err_4k, "flux_err_2k": err_2k,
               "flux_ratio": ratio, "flux_ratio_err": ratio_err,
               "div_4k_norm": m["div_4k_norm"],
               "div_2k_norm": m["div_2k_norm"]})

    # XC
    r["xcorr"] = xcorr_in_box(m, rslice, cslice)
    print(f"  Bootstrapping XC ({n_boot} iterations)...")
    for key, ref_key, test_key in [
        ("div", "div_4k_norm", "div_2k_norm"),
        ("vx",  "vx_4n",      "vx_2n"),
        ("vy",  "vy_4n",      "vy_2n"),
    ]:
        _, pstd = bootstrap_xcorr_peak(m[ref_key][rslice, cslice],
                                        m[test_key][rslice, cslice],
                                        n_boot=n_boot)
        r["xcorr"][key]["peak_std"] = pstd

    return r


# ══════════════════════════════════════════════════════════════════════════════
# PLOT A — INTEGRATED FLUX VS APERTURE RADIUS  (with bootstrap error bands)
# ══════════════════════════════════════════════════════════════════════════════

def plot_flux(r15, r30, save=None):
    """
    3-panel: flux(R) for 15 min, 30 min, and 2K/4K ratio.
    Shaded bands = 1-sigma bootstrap uncertainty.
    """
    AR_half_Mm = np.deg2rad(AR_SIZE_DEG/2) * R_SUN_MM

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5),
                             gridspec_kw={"wspace": 0.38})

    all_flux = np.concatenate([r15["flux_4k"], r15["flux_2k"],
                                r30["flux_4k"], r30["flux_2k"]])
    ylo   = np.nanmin(all_flux) * 1.15
    yhi   = np.nanmax(all_flux) * 1.15
    x_max = max(r15["R_Mm"].max()+10, r30["R_Mm"].max()+10)

    for ax, r, title in zip(axes[:2], [r15, r30],
                             ["15 min cadence", "30 min cadence"]):
        R = r["R_Mm"]
        ax.fill_between(R, r["flux_4k"]-r["flux_err_4k"],
                           r["flux_4k"]+r["flux_err_4k"],
                        color=C4K, alpha=0.22, zorder=2)
        ax.fill_between(R, r["flux_2k"]-r["flux_err_2k"],
                           r["flux_2k"]+r["flux_err_2k"],
                        color=C2K, alpha=0.22, zorder=2)
        ax.plot(R, r["flux_4k"], color=C4K, lw=2.5, label="HMI 4K", zorder=3)
        ax.plot(R, r["flux_2k"], color=C2K, lw=2.5, ls="--",
                label="PMI 2K", zorder=3)
        ax.axhline(0, color=CGREY, lw=1.0, ls=":")
        ax.axvline(AR_half_Mm, color=CGREY, lw=1.0, ls="--", alpha=0.6)
        ax.text(AR_half_Mm - 1.5, yhi * 0.90, "AR\nedge",
                fontsize=BASE_FS-1, color=CGREY, va="top", ha="right",
                linespacing=1.3)
        ax.set_xlabel("Aperture radius  [Mm]", fontsize=BASE_FS+1)
        ax.set_ylabel(r"$\oint \mathbf{v}\cdot\hat{n}\,\mathrm{d}l$"
                      r"  [m s$^{-1}$ Mm]", fontsize=BASE_FS+1)
        ax.set_title(title, fontsize=BASE_FS+3, pad=8)
        ax.set_ylim(ylo, yhi); ax.set_xlim(0, x_max*1.02)
        ax.legend(fontsize=BASE_FS)
        ax.tick_params(labelsize=BASE_FS)
        _style_ax(ax)

    ax = axes[2]
    for r, lc, ls, lab in [(r15, C4K, "-", "15 min"),
                            (r30, C2K, "--","30 min")]:
        ratio = r["flux_ratio"]; ratio_err = r["flux_ratio_err"]
        valid = np.isfinite(ratio) & np.isfinite(ratio_err)
        if valid.any():
            ax.fill_between(r["R_Mm"][valid],
                            (ratio-ratio_err)[valid],
                            (ratio+ratio_err)[valid],
                            color=lc, alpha=0.22, zorder=2)
            ax.plot(r["R_Mm"][valid], ratio[valid], color=lc, lw=2.5,
                    ls=ls, label=lab, zorder=3)

    ax.axhline(1.0, color=CGREY, lw=1.2, ls=":", label="Ideal = 1", zorder=1)
    ax.axhline(0.0, color=CGREY, lw=0.8, ls="--", zorder=1)
    ax.axvline(AR_half_Mm, color=CGREY, lw=1.0, ls="--", alpha=0.6)
    ax.text(AR_half_Mm - 1.5, 1.85, "AR\nedge",
            fontsize=BASE_FS-1, color=CGREY, va="top", ha="right",
            linespacing=1.3)
    ax.set_xlabel("Aperture radius  [Mm]", fontsize=BASE_FS+1)
    ax.set_ylabel("PMI 2K / HMI 4K  flux ratio", fontsize=BASE_FS+1)
    ax.set_title("Amplitude fidelity", fontsize=BASE_FS+3, pad=8)
    ax.legend(fontsize=BASE_FS)
    ax.set_ylim(-1.2, 2.0); ax.set_xlim(0, x_max*1.02)
    ax.text(0.97, 0.05, "Ratio < 0: opposite sign to 4K",
            transform=ax.transAxes, fontsize=BASE_FS-1, color=C2K,
            va="bottom", ha="right", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor="#dddddd"))
    ax.tick_params(labelsize=BASE_FS)
    _style_ax(ax)

    plt.tight_layout()
    if save: plt.savefig(save, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT B — 2D CROSS-CORRELATION MAPS  (with ± uncertainty in annotation)
# ══════════════════════════════════════════════════════════════════════════════

def plot_xcorr(r15, r30, save=None):
    """2×3: normalised 2D XC maps. Peak ± bootstrap std in annotation."""
    pxMm       = _px_Mm()
    col_keys   = ["div", "vx", "vy"]
    col_titles = [r"$\nabla\cdot\mathbf{v}$  (divergence)",
                  r"$v_x$  (longitudinal)",
                  r"$v_y$  (latitudinal)"]
    row_data   = [(r15, "15 min"), (r30, "30 min")]
    AR_half_Mm = np.deg2rad(AR_SIZE_DEG/2) * R_SUN_MM
    print(AR_half_Mm)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.55})

    for row_i, (r, row_label) in enumerate(row_data):
        for col_i, (key, col_title) in enumerate(zip(col_keys, col_titles)):
            ax  = axes[row_i, col_i]
            xc  = r["xcorr"][key]["xc"]
            pv  = r["xcorr"][key]["peak_val"]
            pstd = r["xcorr"][key].get("peak_std", 0.0)
            dy  = r["xcorr"][key]["dy_Mm"]
            dx  = r["xcorr"][key]["dx_Mm"]

            nr, nc = xc.shape
            r_axis = (np.arange(nr) - (nr-1)/2) * pxMm
            c_axis = (np.arange(nc) - (nc-1)/2) * pxMm

            ri = np.abs(r_axis) <= AR_half_Mm
            ci = np.abs(c_axis) <= AR_half_Mm
            xc_clip = xc[np.ix_(ri, ci)]
            r_clip  = r_axis[ri]; c_clip = c_axis[ci]
            vmax    = min(1.0, np.nanpercentile(np.abs(xc_clip), 99.5))

            im = ax.imshow(xc_clip,
                           extent=[c_clip[0], c_clip[-1],
                                   r_clip[0], r_clip[-1]],
                           origin="lower", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, aspect="equal")

            if abs(dy) > 0 or abs(dx) > 0:
                ax.axhline(dy, color="white", lw=1.0, ls="--", alpha=0.8)
                ax.axvline(dx, color="white", lw=1.0, ls="--", alpha=0.8)
            ax.plot(dx, dy, "w+", ms=16, mew=2.5, zorder=5)
            ax.axhline(0, color="white", lw=0.6, ls=":", alpha=0.45)
            ax.axvline(0, color="white", lw=0.6, ls=":", alpha=0.45)

            sgn_x = "E" if dx >= 0 else "W"
            sgn_y = "N" if dy >= 0 else "S"
            shift_str = (f"({abs(dx):.1f} {sgn_x}, {abs(dy):.1f} {sgn_y}) Mm"
                         if (abs(dx)+abs(dy)) > 0 else "no shift")
            ann = f"peak = {pv:.3f} ± {pstd:.3f}   shift = {shift_str}"

            ax.set_title(col_title if row_i == 0 else "",
                         fontsize=BASE_FS+2, pad=6)
            ax.text(0.5, -0.20, ann,
                    transform=ax.transAxes, fontsize=BASE_FS-1,
                    va="top", ha="center", color="#333333",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#f5f5f5",
                              alpha=0.9, edgecolor="#cccccc", lw=0.7))

            ax.set_xlabel("Longitude offset  [Mm]", fontsize=BASE_FS)
            ax.tick_params(labelsize=BASE_FS-1)
            if col_i == 0:
                ax.set_ylabel(f"{row_label}\n\nLatitude offset  [Mm]",
                              fontsize=BASE_FS+1)
            else:
                ax.set_ylabel("Latitude offset  [Mm]", fontsize=BASE_FS)
            for sp in ax.spines.values(): sp.set_linewidth(0.9)

            cb = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
            cb.set_label("Normalised XC", fontsize=BASE_FS)
            cb.ax.tick_params(direction="in", labelsize=BASE_FS-1)
            cb.outline.set_linewidth(0.8)

    plt.tight_layout()
    if save: plt.savefig(save, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════

def _print_table(m15, m30):
    """Print original amplitude + structural metrics."""
    print("\n" + "═"*65)
    print(f"  {'Metric':<35}  {'15 min':>8}  {'30 min':>8}")
    print("─"*65)
    rows = [
        ("Amplitude ratio 2K/4K",          "amplitude_ratio",  ".3f"),
        ("Speed RMSE [m/s]",               "rmse_speed_raw",   ".1f"),
        ("Speed bias 2K−4K [m/s]",         "bias_speed_raw",   ".1f"),
        ("Div RMSE [s⁻¹]",                 "rmse_div_raw",     ".3g"),
        ("Div bias 2K−4K [s⁻¹]",           "bias_div_raw",     ".3g"),
        ("",                               None,               ""),
        ("Trough radius HMI 4K [Mm]",      "trough_Mm_4k",     ".1f"),
        ("Trough radius PMI 2K [Mm]",      "trough_Mm_2k",     ".1f"),
        ("Trough shift 2K−4K [Mm]",        "trough_shift_Mm",  ".1f"),
        ("Inflow extent HMI 4K [Mm]",      "extent_Mm_4k",     ".1f"),
        ("Inflow extent PMI 2K [Mm]",      "extent_Mm_2k",     ".1f"),
        ("",                               None,               ""),
        ("Pearson r  vx",                  "r_vx",             ".3f"),
        ("Pearson r  vy",                  "r_vy",             ".3f"),
        ("Pearson r  speed",               "r_speed",          ".3f"),
        ("Pearson r  divergence (global)", "r_div",            ".3f"),
        ("Pearson r  divergence (inflow)", "r_div_inflow",     ".3f"),
        ("Pearson r  divergence (quiet)",  "r_div_quiet",      ".3f"),
        ("Pearson r  divergence (medium)", "r_div_medium",     ".3f"),
        ("Pearson r  divergence (strong)", "r_div_strong",     ".3f"),
        ("Vector skill score",             "vector_skill",     ".3f"),
    ]
    for label, key, fmt in rows:
        if key is None: print(); continue
        v15 = m15.get(key, np.nan); v30 = m30.get(key, np.nan)
        v15_s = f"{v15:{fmt}}" if np.isfinite(float(v15)) else "—"
        v30_s = f"{v30:{fmt}}" if np.isfinite(float(v30)) else "—"
        print(f"  {label:<35}  {v15_s:>8}  {v30_s:>8}")
    print("═"*65 + "\n")


def print_robust_table(r15, r30):
    """Print robust diagnostics with bootstrap uncertainties."""
    AR_half_Mm = np.deg2rad(AR_SIZE_DEG/2) * R_SUN_MM
    print("\n" + "═"*85)
    print(f"  {'Metric':<40}  {'15 min':>18}  {'30 min':>18}")
    print("─"*85)

    rows_15 = _flux_summary(r15, AR_half_Mm)
    rows_30 = _flux_summary(r30, AR_half_Mm)
    for (label, v15, e15), (_, v30, e30) in zip(rows_15, rows_30):
        s15 = f"{v15:.3g} ± {e15:.2g}" if np.isfinite(e15) else f"{v15:.3g}"
        s30 = f"{v30:.3g} ± {e30:.2g}" if np.isfinite(e30) else f"{v30:.3g}"
        print(f"  {label:<40}  {s15:>18}  {s30:>18}")

    print()
    for key, field in [("div","divergence"), ("vx","vx"), ("vy","vy")]:
        pv15 = r15["xcorr"][key]["peak_val"]
        pv30 = r30["xcorr"][key]["peak_val"]
        ps15 = r15["xcorr"][key].get("peak_std", np.nan)
        ps30 = r30["xcorr"][key].get("peak_std", np.nan)
        s15  = f"{pv15:.3f} ± {ps15:.3f}" if np.isfinite(ps15) else f"{pv15:.3f}"
        s30  = f"{pv30:.3f} ± {ps30:.3f}" if np.isfinite(ps30) else f"{pv30:.3f}"
        print(f"  {'XC peak  ' + field:<40}  {s15:>18}  {s30:>18}")
        print(f"  {'XC shift lon  ' + field + ' [Mm]':<40}"
              f"  {r15['xcorr'][key]['dx_Mm']:>18.1f}"
              f"  {r30['xcorr'][key]['dx_Mm']:>18.1f}")
        print(f"  {'XC shift lat  ' + field + ' [Mm]':<40}"
              f"  {r15['xcorr'][key]['dy_Mm']:>18.1f}"
              f"  {r30['xcorr'][key]['dy_Mm']:>18.1f}")
        print()
    print("═"*85 + "\n")


def _flux_summary(r, AR_half_Mm):
    idx  = np.argmin(np.abs(r["R_Mm"] - AR_half_Mm))
    f4   = r["flux_4k"][idx];  e4  = r["flux_err_4k"][idx]
    f2   = r["flux_2k"][idx];  e2  = r["flux_err_2k"][idx]
    rat  = r["flux_ratio"][idx]
    erat = r["flux_ratio_err"][idx] if np.isfinite(r["flux_ratio_err"][idx]) else np.nan
    return [
        (f"Flux 4K at R=AR edge ({AR_half_Mm:.0f} Mm)  [m/s·Mm]", f4,  e4),
        (f"Flux 2K at R=AR edge ({AR_half_Mm:.0f} Mm)  [m/s·Mm]", f2,  e2),
        (f"Flux ratio 2K/4K at AR edge",                           rat, erat),
        (f"Min flux 4K  [m/s·Mm]",
         r["flux_4k"].min(), r["flux_err_4k"][np.argmin(r["flux_4k"])]),
        (f"Min flux 2K  [m/s·Mm]",
         r["flux_2k"].min(), r["flux_err_2k"][np.argmin(r["flux_2k"])]),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# MASTER ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════════

def run_paper_plots(m15, m30, save_prefix=None):
    """Original four plots + summary table."""
    def _s(tag):
        return f"{save_prefix}_{tag}.pdf" if save_prefix else None
    plot_flow_maps(m15, m30,                  save=_s("flowmaps"))
    plot_divergence_maps(m15, m30,            save=_s("divmaps"))
    plot_radial_profiles_comparison(m15, m30, save=_s("radial"))
    plot_metric_comparison(m15, m30,          save=_s("metrics"))
    _print_table(m15, m30)


def run_robust_plots(m15, m30, save_prefix=None, n_boot=300):
    """Robust flux + XC plots + bootstrap table."""
    def _s(tag):
        return f"{save_prefix}_{tag}.pdf" if save_prefix else None
    r15 = compute_robust(m15, n_boot=n_boot)
    r30 = compute_robust(m30, n_boot=n_boot)
    plot_flux(r15, r30,  save=_s("flux"))
    plot_xcorr(r15, r30, save=_s("xcorr"))
    print_robust_table(r15, r30)
    return r15, r30


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    m15 = compute_all(*load_data("15min"))
    m30 = compute_all(*load_data("30min"))

    run_paper_plots(m15, m30, save_prefix="ar_inflow_new")
    run_robust_plots(m15, m30, save_prefix="ar_inflow_new", n_boot=300)
# %%
