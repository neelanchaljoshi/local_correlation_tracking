# %%
"""
LCT Resolution Comparison — HMI 4K vs PMI 2K
=============================================
Metrics and publication-quality plots for comparing LCT-derived
active region inflow maps at two resolutions.

QUICK START
-----------
    from lct_metrics import run_all, downsample_bz

    m = run_all(
        vx_4k, vy_4k,        # HMI 4K LCT velocities  [m/s]
        vx_2k, vy_2k,        # PMI 2K LCT velocities  [m/s]
        bz=bz_downsampled,   # LOS magnetogram on flow-map grid [Gauss] — or None
        pixel_scale_deg=0.5,
        save_prefix="ar_run",
    )

NORMALISATION PHILOSOPHY
------------------------
  RAW velocities  → amplitude metrics (RMSE, bias, ratio), physical divergence [s⁻¹]
  NORMALISED      → structural metrics (Pearson r, PSD shape, radial profile shape)

The ~2.7x amplitude bias in PMI 2K is a known artefact of PSF-broadened CCF
peaks; normalisation removes it so structural fidelity can be assessed cleanly.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.ndimage import gaussian_filter, uniform_filter1d, center_of_mass, zoom, binary_dilation
from scipy.stats import pearsonr
from scipy.fft import fft2, fftshift
import warnings
warnings.filterwarnings("ignore")


# ── publication style ──────────────────────────────────────────────────────
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

C4K   = "#1f77b4"   # blue  — HMI 4K
C2K   = "#d62728"   # red   — PMI 2K
CGOOD = "#2ca02c"   # green — reference lines
CGREY = "#7f7f7f"   # grey  — neutral lines
R_SUN_MM = 695.7    # Mm


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PLUG IN YOUR DATA HERE
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Load your LCT output and (optionally) the LOS magnetogram.

    Returns
    -------
    vx_4k, vy_4k : 2D arrays — HMI 4K LCT velocities [m/s], shape (nlat, nlng)
    vx_2k, vy_2k : 2D arrays — PMI 2K LCT velocities [m/s], same shape
    bz           : 2D array or None — LOS magnetogram [Gauss] on the flow-map
                   grid (nlat, nlng). If your magnetogram is higher resolution,
                   use downsample_bz(bz_highres, target_shape=vx_4k.shape) first.
                   Pass None to skip magnetogram-based AR masking.

    Notes
    -----
    - vx = longitudinal (phi) component;  positive = eastward
    - vy = latitudinal (theta) component; positive = northward
    - In your HDF5 output: vx = uphi, vy = -utheta
    - Both fields must be time-averaged before passing in (or pass a single
      time step if you want per-snapshot metrics)
    """

    # ── REPLACE BELOW ─────────────────────────────────────────────────────
    f1 = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/ar_inflow/data/data_cleaned/smooth_data_2k_15.npz')
    vx_2k = f1['smooth_zx_corrected']
    vy_2k = -f1['smooth_zy_corrected']

    f2 = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/ar_inflow/data/data_cleaned/smooth_data_4k_15.npz')
    vx_4k = f2['smooth_zx_corrected']
    vy_4k = -f2['smooth_zy_corrected']

    bz_file    = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/ar_inflow/data/data_cleaned/magnetogram_cropped.npz')
    bz_highres = bz_file['img_cropped']
    bz         = downsample_bz(bz_highres, target_shape=vx_4k.shape)

    return vx_4k, vy_4k, vx_2k, vy_2k, bz
    # ── END REPLACE ───────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PIXEL_SCALE_DEG   = 0.5    # degrees per pixel in the flow map
MAG_THRESHOLD_G   = 50.0   # |Bz| threshold for AR core definition [Gauss]
DILATION_PX       = 5      # dilation radius around AR core [pixels] (~30 Mm)
INFLOW_PERCENTILE = 20     # bottom N% of 4K divergence = inflow zone


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def px_scale_m(pixel_scale_deg=PIXEL_SCALE_DEG):
    """Physical size of one flow-map pixel in metres."""
    return np.deg2rad(pixel_scale_deg) * R_SUN_MM * 1e6

def px_scale_Mm(pixel_scale_deg=PIXEL_SCALE_DEG):
    return px_scale_m(pixel_scale_deg) / 1e6

def downsample_bz(bz_highres, target_shape):
    """
    Downsample a high-resolution magnetogram to the flow-map grid.
    Uses order=1 (bilinear) to avoid ringing on sharp field concentrations.
    """
    factor = (target_shape[0] / bz_highres.shape[0],
              target_shape[1] / bz_highres.shape[1])
    return zoom(np.nan_to_num(bz_highres), factor, order=1)

def normalise_vector(vx, vy):
    """
    Remove mean and divide both components by std(speed).
    Single scale factor preserves relative vx/vy strength.
    """
    scale = np.std(np.hypot(vx, vy))
    return (vx - np.mean(vx)) / scale, (vy - np.mean(vy)) / scale

def divergence_physical(vx, vy, pixel_scale_deg=PIXEL_SCALE_DEG):
    """Divergence in s⁻¹.  vx, vy must be in m/s."""
    px = px_scale_m(pixel_scale_deg)
    return np.gradient(vx, axis=1) / px + np.gradient(vy, axis=0) / px

def divergence_normalised(vx_n, vy_n):
    """Divergence of normalised (unitless) fields — for structure comparison."""
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
    """Lhermitte & Lemaitre (1984) vector correlation, range [0, 1]."""
    axr = _flat(vx_ref, mask);  ayr = _flat(vy_ref, mask)
    axt = _flat(vx_test, mask); ayt = _flat(vy_test, mask)
    num = np.nansum(axr * axt + ayr * ayt)
    den = np.sqrt(np.nansum(axr**2 + ayr**2) * np.nansum(axt**2 + ayt**2))
    return float(num / den) if den > 0 else np.nan

def amplitude_ratio(vx_4k, vy_4k, vx_2k, vy_2k):
    """Mean speed ratio 2K / 4K — quantifies systematic amplitude bias."""
    return float(np.mean(np.hypot(vx_2k, vy_2k)) /
                 np.mean(np.hypot(vx_4k, vy_4k)))


# ══════════════════════════════════════════════════════════════════════════════
# MASKING
# ══════════════════════════════════════════════════════════════════════════════

def make_inflow_mask(div_4k_phys, percentile=INFLOW_PERCENTILE):
    """Bottom-percentile of 4K divergence — the inflow zone."""
    return div_4k_phys < np.percentile(div_4k_phys, percentile)

def make_ar_mask_from_bz(bz, mag_threshold_G=MAG_THRESHOLD_G,
                          dilation_px=DILATION_PX):
    """AR mask: |Bz| > threshold, dilated by dilation_px pixels."""
    core = np.abs(bz) > mag_threshold_G
    return binary_dilation(core, iterations=dilation_px)

def make_field_strength_masks(bz):
    """Split domain into quiet / medium / strong field for stratified analysis."""
    bz_abs = np.abs(bz)
    return {
        "quiet":  bz_abs <  50,
        "medium": (bz_abs >= 50) & (bz_abs < 300),
        "strong": bz_abs >= 300,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PSD
# ══════════════════════════════════════════════════════════════════════════════

def _radial_avg(power):
    """Radially averaged (isotropic) 1D PSD from a 2D power spectrum."""
    N = power.shape[0]
    cy, cx = N // 2, N // 2
    y, x = np.ogrid[:N, :N]
    r = np.hypot(x - cx, y - cy).astype(int).ravel()
    p = power.ravel()
    k_max = min(cx, cy)
    bins = np.zeros(k_max); cnts = np.zeros(k_max)
    for ri, pi in zip(r, p):
        if ri < k_max:
            bins[ri] += pi; cnts[ri] += 1
    cnts[cnts == 0] = np.nan
    return bins / cnts

def compute_psd(field, pixel_scale_Mm):
    """
    Radially averaged 1D PSD.
    Returns spatial scale in Mm (1/k) and power.
    k=0 (the mean) is set to nan to avoid division by zero.
    """
    psd      = np.abs(fftshift(fft2(field)))**2
    avg      = _radial_avg(psd)
    k_cpp    = np.arange(len(avg)) / field.shape[0]   # cycles per pixel
    k_Mm     = k_cpp / pixel_scale_Mm                  # cycles per Mm
    k_Mm[0]  = np.nan                                  # exclude DC component
    scale_Mm = 1.0 / k_Mm                              # spatial scale in Mm
    return scale_Mm, avg


# ══════════════════════════════════════════════════════════════════════════════
# RADIAL PROFILE
# ══════════════════════════════════════════════════════════════════════════════

def inflow_centre(div_4k_phys, smooth_sigma=3):
    smoothed = gaussian_filter(div_4k_phys, sigma=smooth_sigma)
    inflow   = np.where(smoothed < 0, -smoothed, 0)
    return center_of_mass(inflow)

def radial_profile(field, centre):
    cy, cx = centre
    y, x = np.indices(field.shape)
    r = np.hypot(x - cx, y - cy).astype(int)
    return np.bincount(r.ravel(), weights=field.ravel()) / np.bincount(r.ravel())

def _zero_crossing_after(profile, start_idx, r_Mm):
    """Return Mm position of first zero crossing in profile after start_idx."""
    zc = np.where(np.diff(np.sign(profile[start_idx:])))[0]
    if len(zc) > 0:
        return float(r_Mm[start_idx + zc[0]])
    return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# MASTER COMPUTE
# ══════════════════════════════════════════════════════════════════════════════

def compute_all(vx_4k, vy_4k, vx_2k, vy_2k,
                bz=None,
                pixel_scale_deg=PIXEL_SCALE_DEG):
    """
    Compute all metrics and intermediate arrays.

    Parameters
    ----------
    vx_4k, vy_4k    : HMI 4K LCT velocities [m/s], shape (nlat, nlng)
    vx_2k, vy_2k    : PMI 2K LCT velocities [m/s], same shape
    bz               : LOS magnetogram [Gauss] on flow-map grid, or None
    pixel_scale_deg  : degrees per pixel in the flow map

    Returns
    -------
    m : dict — all metrics and arrays needed for plotting
    """
    m    = {}
    pxMm = px_scale_Mm(pixel_scale_deg)

    # ── normalised fields ──────────────────────────────────────────────────
    vx_4n, vy_4n = normalise_vector(vx_4k, vy_4k)
    vx_2n, vy_2n = normalise_vector(vx_2k, vy_2k)

    # ── divergence ────────────────────────────────────────────────────────
    div_4k_phys = divergence_physical(vx_4k, vy_4k, pixel_scale_deg)   # s⁻¹
    div_2k_phys = divergence_physical(vx_2k, vy_2k, pixel_scale_deg)   # s⁻¹
    div_4k_norm = divergence_normalised(vx_4n, vy_4n)                  # arb.
    div_2k_norm = divergence_normalised(vx_2n, vy_2n)                  # arb.

    # ── speed ─────────────────────────────────────────────────────────────
    speed_4k = np.hypot(vx_4k, vy_4k)
    speed_2k = np.hypot(vx_2k, vy_2k)
    speed_4n = np.hypot(vx_4n, vy_4n)
    speed_2n = np.hypot(vx_2n, vy_2n)

    # ── masks ─────────────────────────────────────────────────────────────
    imask    = make_inflow_mask(div_4k_phys)
    ar_mask  = make_ar_mask_from_bz(bz) if bz is not None else None
    bz_masks = make_field_strength_masks(bz) if bz is not None else None

    # ── inflow centre and radial geometry ─────────────────────────────────
    com    = inflow_centre(div_4k_phys)
    cy, cx = com
    shape  = div_4k_phys.shape
    max_r  = int(min(cy, shape[0]-cy, cx, shape[1]-cx)) - 2
    y_idx, x_idx = np.indices(shape)
    r_map  = np.hypot(x_idx - cx, y_idx - cy)

    # ── AMPLITUDE METRICS — raw, physical ─────────────────────────────────
    m["amplitude_ratio"] = amplitude_ratio(vx_4k, vy_4k, vx_2k, vy_2k)
    m["rmse_speed_raw"]  = rmse(speed_4k, speed_2k)
    m["bias_speed_raw"]  = bias(speed_4k, speed_2k)
    m["rmse_div_raw"]    = rmse(div_4k_phys, div_2k_phys)
    m["bias_div_raw"]    = bias(div_4k_phys, div_2k_phys)

    # ── STRUCTURAL METRICS — normalised ───────────────────────────────────
    m["r_vx"]         = pearson_r(vx_4n,       vx_2n)
    m["r_vy"]         = pearson_r(vy_4n,       vy_2n)
    m["r_speed"]      = pearson_r(speed_4n,    speed_2n)
    m["r_div"]        = pearson_r(div_4k_norm, div_2k_norm)
    m["vector_skill"] = vector_skill(vx_4n, vy_4n, vx_2n, vy_2n)

    # ── INFLOW ZONE — divergence-threshold mask ────────────────────────────
    m["r_div_inflow"]    = pearson_r(div_4k_norm, div_2k_norm, mask=imask)
    m["rmse_div_inflow"] = rmse(div_4k_norm, div_2k_norm, mask=imask)
    m["bias_div_inflow"] = bias(div_4k_norm, div_2k_norm, mask=imask)

    # ── AR MASK METRICS — magnetogram-based ───────────────────────────────
    if ar_mask is not None:
        m["r_div_ar"]    = pearson_r(div_4k_norm, div_2k_norm, mask=ar_mask)
        m["rmse_div_ar"] = rmse(div_4k_norm, div_2k_norm, mask=ar_mask)
        m["bias_div_ar"] = bias(div_4k_norm, div_2k_norm, mask=ar_mask)
        for label, fmask in bz_masks.items():
            m[f"r_div_{label}"] = (pearson_r(div_4k_norm, div_2k_norm, mask=fmask)
                                   if fmask.any() else np.nan)

    # ── PSD ───────────────────────────────────────────────────────────────
    k, psd_4k_raw  = compute_psd(speed_4k, pxMm)
    _, psd_2k_raw  = compute_psd(speed_2k, pxMm)
    _, psd_4k_norm = compute_psd(speed_4n, pxMm)
    _, psd_2k_norm = compute_psd(speed_2n, pxMm)
    m["psd_k"]          = k                # spatial scale in Mm
    m["psd_4k_raw"]     = psd_4k_raw
    m["psd_2k_raw"]     = psd_2k_raw
    m["psd_ratio_raw"]  = np.where(psd_4k_raw  > 0, psd_2k_raw  / psd_4k_raw,  np.nan)
    m["psd_4k_norm"]    = psd_4k_norm
    m["psd_2k_norm"]    = psd_2k_norm
    m["psd_ratio_norm"] = np.where(psd_4k_norm > 0, psd_2k_norm / psd_4k_norm, np.nan)

    # PSD peaks (spatial scale at maximum power, excluding sub-granulation noise)
    sel_psd = (k > 2.0) & np.isfinite(k)
    for tag, psd in [("4k_raw", psd_4k_raw), ("2k_raw", psd_2k_raw),
                     ("4k_norm", psd_4k_norm), ("2k_norm", psd_2k_norm)]:
        idx = np.nanargmax(psd[sel_psd])
        m[f"psd_peak_scale_{tag}"] = float(k[sel_psd][idx])

    # ── RADIAL PROFILE — normalised divergence, Mm axis ───────────────────
    p4_raw = radial_profile(div_4k_norm, com)
    p2_raw = radial_profile(div_2k_norm, com)
    p4     = uniform_filter1d(p4_raw[:max_r], size=3)
    p2     = uniform_filter1d(p2_raw[:max_r], size=3)
    r_Mm   = np.arange(max_r) * pxMm

    # trough = actual profile minimum for each instrument independently
    trough_px_4k = int(np.argmin(p4))
    trough_px_2k = int(np.argmin(p2))

    # zero crossing after each trough = inflow spatial extent
    extent_Mm_4k = _zero_crossing_after(p4, trough_px_4k, r_Mm)
    extent_Mm_2k = _zero_crossing_after(p2, trough_px_2k, r_Mm)

    # inner/outer masks using each instrument's own trough boundary
    inner_4k = r_map <  trough_px_4k
    outer_4k = (r_map >= trough_px_4k) & (r_map < max_r)
    inner_2k = r_map <  trough_px_2k
    outer_2k = (r_map >= trough_px_2k) & (r_map < max_r)

    m["r_div_inner_4k"]  = pearson_r(div_4k_norm, div_2k_norm, mask=inner_4k)
    m["r_div_outer_4k"]  = pearson_r(div_4k_norm, div_2k_norm, mask=outer_4k)
    m["r_div_inner_2k"]  = pearson_r(div_4k_norm, div_2k_norm, mask=inner_2k)
    m["r_div_outer_2k"]  = pearson_r(div_4k_norm, div_2k_norm, mask=outer_2k)

    m["trough_px_4k"]    = trough_px_4k
    m["trough_px_2k"]    = trough_px_2k
    m["trough_Mm_4k"]    = trough_px_4k * pxMm
    m["trough_Mm_2k"]    = trough_px_2k * pxMm
    m["trough_shift_Mm"] = (trough_px_2k - trough_px_4k) * pxMm  # + = outward shift
    m["trough_Mm"]       = m["trough_Mm_4k"]   # backward compat
    m["extent_Mm_4k"]    = extent_Mm_4k
    m["extent_Mm_2k"]    = extent_Mm_2k

    m["radial_r_Mm"] = r_Mm
    m["radial_p4"]   = p4
    m["radial_p2"]   = p2

    # ── store arrays ──────────────────────────────────────────────────────
    m.update({
        "vx_4k": vx_4k,  "vy_4k": vy_4k,  "vx_2k": vx_2k,  "vy_2k": vy_2k,
        "vx_4n": vx_4n,  "vy_4n": vy_4n,  "vx_2n": vx_2n,  "vy_2n": vy_2n,
        "div_4k_phys": div_4k_phys, "div_2k_phys": div_2k_phys,
        "div_4k_norm": div_4k_norm, "div_2k_norm": div_2k_norm,
        "speed_4k": speed_4k,       "speed_2k": speed_2k,
        "inflow_mask": imask,       "ar_mask": ar_mask,
        "bz": bz,                   "pxMm": pxMm,
    })
    return m


# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _imshow(ax, data, cmap, title, unit="", sym=False):
    if sym:
        vmax = np.nanpercentile(np.abs(data), 99); vmin = -vmax
    else:
        vmin, vmax = np.nanpercentile(data, [1, 99])
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.6)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(unit, fontsize=8)
    cb.ax.tick_params(labelsize=7, direction="in")
    cb.outline.set_linewidth(0.6)
    return im

def _overlay_bz(ax, bz):
    """Overplot magnetogram polarity contours."""
    ax.contour(bz, levels=[ 100,  300,  500], colors=C2K,
               linewidths=0.6, linestyles="-")
    ax.contour(bz, levels=[-500, -300, -100], colors=C4K,
               linewidths=0.6, linestyles="--")

def _overlay_inflow_mask(ax, mask):
    ax.contour(mask.astype(float), levels=[0.5],
               colors="#333333", linewidths=0.7, linestyles=":")

def _overlay_ar_mask(ax, mask):
    ax.contour(mask.astype(float), levels=[0.5],
               colors="#555555", linewidths=0.8, linestyles="--")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_velocity_maps(m, save=None):
    """Raw velocity maps — illustrates amplitude difference."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 10),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.15})
    fig.suptitle("Velocity fields (raw)  [m s$^{-1}$]  |  "
                 "HMI 4K · PMI 2K · Residual", fontsize=11)
    rows = [
        (m["vx_4k"],    m["vx_2k"],    m["vx_2k"]    - m["vx_4k"],    "$v_x$",  "m s$^{-1}$"),
        (m["vy_4k"],    m["vy_2k"],    m["vy_2k"]    - m["vy_4k"],    "$v_y$",  "m s$^{-1}$"),
        (m["speed_4k"], m["speed_2k"], m["speed_2k"] - m["speed_4k"], "Speed",  "m s$^{-1}$"),
    ]
    for i, (r, t, res, label, unit) in enumerate(rows):
        _imshow(axes[i,0], r,   "RdBu_r", f"{label}  HMI 4K",           unit, sym=(i<2))
        _imshow(axes[i,1], t,   "RdBu_r", f"{label}  PMI 2K",           unit, sym=(i<2))
        _imshow(axes[i,2], res, "RdBu_r", f"{label}  Residual (2K−4K)", unit, sym=True)
    plt.tight_layout()
    if save: plt.savefig(save)
    plt.show()


def plot_divergence_maps(m, save=None):
    """
    Two rows: physical [s⁻¹] and normalised [arb.] divergence.
    Overlays: inflow mask (dotted), AR mask if bz provided (dashed),
              magnetogram polarity contours if bz provided.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 8),
                             gridspec_kw={"hspace": 0.4, "wspace": 0.18})
    fig.suptitle("Divergence $\\nabla\\cdot\\mathbf{v}$  |  Negative = Inflow",
                 fontsize=11)

    rows = [
        (m["div_4k_phys"], m["div_2k_phys"], "s$^{-1}$", "Physical"),
        (m["div_4k_norm"], m["div_2k_norm"], "arb.",      "Normalised"),
    ]
    for row_i, (d4, d2, unit, label) in enumerate(rows):
        for col_i, (arr, title) in enumerate(zip(
            [d4, d2, d2 - d4],
            [f"{label}  HMI 4K", f"{label}  PMI 2K",
             f"{label}  Residual (2K−4K)"]
        )):
            ax = axes[row_i, col_i]
            _imshow(ax, arr, "RdBu_r", title, unit=unit, sym=True)
            _overlay_inflow_mask(ax, m["inflow_mask"])
            if m["ar_mask"] is not None:
                _overlay_ar_mask(ax, m["ar_mask"])
            if m["bz"] is not None:
                _overlay_bz(ax, m["bz"])

    # legend
    legend_handles = [
        mlines.Line2D([], [], color="#333333", ls=":", lw=0.8,
                      label=f"Inflow zone (bot. {INFLOW_PERCENTILE}% div.)"),
    ]
    if m["ar_mask"] is not None:
        legend_handles.append(
            mlines.Line2D([], [], color="#555555", ls="--", lw=0.8,
                          label=f"AR mask  |Bz|>{MAG_THRESHOLD_G} G"))
    if m["bz"] is not None:
        legend_handles += [
            mlines.Line2D([], [], color=C2K, ls="-",  lw=0.7, label="$B_+$"),
            mlines.Line2D([], [], color=C4K, ls="--", lw=0.7, label="$B_-$"),
        ]
    axes[0, 0].legend(handles=legend_handles, fontsize=6.5,
                      loc="lower left", framealpha=0.85)
    plt.tight_layout()
    if save: plt.savefig(save)
    plt.show()


def plot_psd(m, save=None):
    """
    Raw and normalised PSD + ratios (4 panels).
    X-axis: spatial scale in Mm (log, inverted — large scales left).
    Vertical lines mark the PSD peak scale for each curve.
    """
    scale = m["psd_k"]                              # spatial scale in Mm
    sel   = (scale > 2.0) & np.isfinite(scale)     # exclude sub-granulation noise

    # tick positions at physically meaningful scales
    scale_ticks = np.array([3, 5, 10, 20, 30, 50, 100, 200])
    scale_min   = scale[sel].min()
    scale_max   = scale[sel].max()
    scale_ticks = scale_ticks[(scale_ticks >= scale_min) & (scale_ticks <= scale_max)]

    # reference lines at known solar scales
    ref_lines = {"Granule\n~2 Mm": 2, "Supergranule\n~30 Mm": 30, "Inflow\n~50 Mm": 50}

    def _fmt_ax(ax, ylabel, title):
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlim(scale_max * 1.05, scale_min * 0.95)
        ax.set_xticks(scale_ticks)
        ax.set_xticklabels([str(t) for t in scale_ticks], fontsize=8)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlabel("Spatial scale [Mm]", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9, pad=6)

    def _add_refs(ax):
        for label, s in ref_lines.items():
            if scale_min < s < scale_max:
                ax.axvline(s, color=CGREY, lw=0.6, ls=":", zorder=0)
                ax.text(s, 1.01, label, fontsize=6, ha="center", va="bottom",
                        color=CGREY, transform=ax.get_xaxis_transform())

    def _add_peak_line(ax, scale_val, color, label):
        """Vertical line at the PSD peak scale with a small label."""
        if np.isfinite(scale_val) and scale_min < scale_val < scale_max:
            ax.axvline(scale_val, color=color, lw=1.0, ls="--", alpha=0.7, zorder=2)
            ax.text(scale_val, 0.97, f"{scale_val:.0f} Mm",
                    fontsize=6, ha="center", va="top", color=color,
                    transform=ax.get_xaxis_transform())

    # taller figure, extra top margin so suptitle clears panel titles
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.2),
                             gridspec_kw={"wspace": 0.42})
    fig.suptitle("Power Spectral Density (radial average)",
                 fontsize=11, y=1.03)

    # panel 0 — raw PSD
    axes[0].semilogy(scale[sel], m["psd_4k_raw"][sel],
                     color=C4K, lw=1.5, label="HMI 4K")
    axes[0].semilogy(scale[sel], m["psd_2k_raw"][sel],
                     color=C2K, lw=1.5, ls="--", label="PMI 2K")
    axes[0].legend(fontsize=8)
    _fmt_ax(axes[0], "PSD [arb.]", "Raw PSD\n(amplitude bias visible)")
    _add_refs(axes[0])
    _add_peak_line(axes[0], m["psd_peak_scale_4k_raw"], C4K, "HMI 4K")
    _add_peak_line(axes[0], m["psd_peak_scale_2k_raw"], C2K, "PMI 2K")

    # panel 1 — raw ratio
    axes[1].axhline(1, color=CGREY, lw=0.8, ls="--", label="Ratio = 1")
    axes[1].plot(scale[sel], m["psd_ratio_raw"][sel],
                 color=CGOOD, lw=1.5, label="PMI / HMI")
    axes[1].set_ylim(0, max(3, np.nanpercentile(m["psd_ratio_raw"][sel], 95) * 1.2))
    axes[1].legend(fontsize=8)
    _fmt_ax(axes[1], "PSD ratio", "Raw PSD ratio\n>1 = amplitude overestimation")
    _add_refs(axes[1])

    # panel 2 — normalised PSD
    axes[2].semilogy(scale[sel], m["psd_4k_norm"][sel],
                     color=C4K, lw=1.5, label="HMI 4K")
    axes[2].semilogy(scale[sel], m["psd_2k_norm"][sel],
                     color=C2K, lw=1.5, ls="--", label="PMI 2K")
    axes[2].legend(fontsize=8)
    _fmt_ax(axes[2], "PSD [arb.]", "Normalised PSD\n(structure only)")
    _add_refs(axes[2])
    _add_peak_line(axes[2], m["psd_peak_scale_4k_norm"], C4K, "HMI 4K")
    _add_peak_line(axes[2], m["psd_peak_scale_2k_norm"], C2K, "PMI 2K")

    # panel 3 — normalised ratio
    axes[3].axhline(1, color=CGREY, lw=0.8, ls="--", label="Ratio = 1")
    axes[3].plot(scale[sel], m["psd_ratio_norm"][sel],
                 color=CGOOD, lw=1.5, label="PMI / HMI")
    axes[3].set_ylim(0, 2)
    axes[3].legend(fontsize=8)
    _fmt_ax(axes[3], "PSD ratio",
            "Normalised PSD ratio\n$\\approx$1 = structure preserved")
    _add_refs(axes[3])

    plt.tight_layout()
    if save: plt.savefig(save, bbox_inches="tight")
    plt.show()


def plot_scatter(m, save=None):
    """Scatter on normalised fields — structure only, no amplitude."""
    pairs = [
        (m["vx_4n"],       m["vx_2n"],       "$v_x$ (norm.)",             m["r_vx"]),
        (m["vy_4n"],       m["vy_2n"],       "$v_y$ (norm.)",             m["r_vy"]),
        (m["div_4k_norm"], m["div_2k_norm"], "$\\nabla\\cdot v$ (norm.)", m["r_div"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6),
                             gridspec_kw={"wspace": 0.35})
    fig.suptitle("Scatter: HMI 4K vs PMI 2K  (normalised fields)", fontsize=11)
    for ax, (ref, tst, label, r) in zip(axes, pairs):
        ref_f, tst_f = ref.ravel(), tst.ravel()
        idx = np.random.choice(len(ref_f), min(5000, len(ref_f)), replace=False)
        ax.scatter(ref_f[idx], tst_f[idx], s=1.5, alpha=0.3,
                   color=C4K, rasterized=True)
        lo = min(ref_f[idx].min(), tst_f[idx].min())
        hi = max(ref_f[idx].max(), tst_f[idx].max())
        ax.plot([lo, hi], [lo, hi], color=C2K, lw=1, ls="--", label="1:1")
        ax.set_xlabel(f"HMI 4K  {label}")
        ax.set_ylabel(f"PMI 2K  {label}")
        ax.set_title(f"{label}\n$r$ = {r:.3f}")
        ax.legend()
    plt.tight_layout()
    if save: plt.savefig(save)
    plt.show()


def plot_radial_profile(m, save=None):
    """
    Radial divergence profile — shape comparison in physical distance.
    Annotates:
      - trough location for 4K (solid grey) and 2K (dashed grey)
      - zero crossing (inflow extent) for 4K (solid) and 2K (dashed)
    """
    r   = m["radial_r_Mm"]
    p4  = m["radial_p4"]
    p2  = m["radial_p2"]
    t4  = m["trough_Mm_4k"]
    t2  = m["trough_Mm_2k"]
    e4  = m["extent_Mm_4k"]
    e2  = m["extent_Mm_2k"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r, p4, color=C4K, lw=1.5, label="HMI 4K")
    ax.plot(r, p2, color=C2K, lw=1.5, ls="--", label="PMI 2K")
    ax.axhline(0, color=CGREY, lw=0.7)

    # trough lines
    ax.axvline(t4, color=C4K, lw=0.9, ls=":",
               label=f"Trough 4K  {t4:.0f} Mm")
    ax.axvline(t2, color=C2K, lw=0.9, ls=":",
               label=f"Trough 2K  {t2:.0f} Mm")

    # zero crossing / inflow extent lines
    if np.isfinite(e4):
        ax.axvline(e4, color=C4K, lw=0.9, ls="-.",
                   label=f"Extent 4K  {e4:.0f} Mm")
    if np.isfinite(e2):
        ax.axvline(e2, color=C2K, lw=0.9, ls="-.",
                   label=f"Extent 2K  {e2:.0f} Mm")

    ax.set_xlabel("Radius from inflow centre [Mm]")
    ax.set_ylabel("Mean normalised divergence [arb.]")
    ax.set_title("Radial divergence profile  (normalised — shape only)\n"
                 "Dotted = trough   Dash-dot = inflow extent (zero crossing)")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    if save: plt.savefig(save)
    plt.show()


def plot_field_strength_correlation(m, save=None):
    """
    Bar chart of divergence Pearson r by magnetic field strength regime.
    Only produced when bz is provided.
    """
    if m["bz"] is None:
        print("[INFO] No magnetogram provided — skipping field strength plot.")
        return

    labels = ["Quiet\n|B| < 50 G", "Medium\n50–300 G", "Strong\n|B| > 300 G"]
    keys   = ["r_div_quiet", "r_div_medium", "r_div_strong"]
    vals   = [m.get(k, np.nan) for k in keys]
    colors = [CGOOD, C4K, C2K]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(np.arange(3), vals, color=colors, width=0.5, zorder=3)
    ax.axhline(0, color=CGREY, lw=0.7, ls="--")
    ax.axhline(1, color=CGOOD, lw=0.6, ls=":", label="Perfect = 1")
    ax.set_ylim(-0.1, 1.15)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Pearson $r$  ($\\nabla\\cdot v$, normalised)")
    ax.set_title("Divergence correlation vs field strength\n"
                 "(tests whether magnetic suppression of granulation limits accuracy)")
    ax.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    if save: plt.savefig(save)
    plt.show()


def plot_metric_summary(m, save=None):
    """Three-panel summary: amplitude, structural, inflow-zone breakdown."""
    has_bz = m["bz"] is not None
    fig, axes = plt.subplots(1, 3, figsize=(14, 5),
                             gridspec_kw={"wspace": 0.4})
    fig.suptitle("Scalar Metric Summary", fontsize=11)

    # panel 1 — amplitude (raw, physical)
    ax = axes[0]
    labels = ["Speed\nRMSE", "Speed\nBias", "Div\nRMSE", "Div\nBias", "Speed\nRatio"]
    vals   = [m["rmse_speed_raw"], m["bias_speed_raw"],
              m["rmse_div_raw"],   m["bias_div_raw"],
              m["amplitude_ratio"]]
    colors = [C4K, C2K, C4K, C2K, CGOOD]
    bars   = ax.bar(np.arange(5), vals, color=colors, width=0.55, zorder=3)
    ax.axhline(0, color=CGREY, lw=0.7, ls="--")
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Amplitude metrics\n(raw — physical units)", fontsize=9)
    ax.set_ylabel("m s$^{-1}$ / s$^{-1}$ / ratio")
    for bar, v in zip(bars, vals):
        ypos = bar.get_height() if v >= 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{v:.3g}", ha="center", va="bottom", fontsize=7.5)

    # panel 2 — structural (normalised)
    ax = axes[1]
    labels2 = ["$r_{v_x}$", "$r_{v_y}$", "$r_{speed}$",
               "$r_{\\nabla v}$", "Vector\nskill"]
    vals2   = [m["r_vx"], m["r_vy"], m["r_speed"],
               m["r_div"], m["vector_skill"]]
    bars2   = ax.bar(np.arange(5), vals2, color=C4K, width=0.55, zorder=3)
    ax.axhline(0, color=CGREY, lw=0.7, ls="--")
    ax.axhline(1, color=CGOOD, lw=0.6, ls=":", label="Perfect = 1")
    ax.set_ylim(-0.1, 1.15)
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(labels2, fontsize=8)
    ax.set_title("Structural metrics\n(normalised — amplitude-independent)", fontsize=9)
    ax.set_ylabel("Pearson $r$ / vector skill")
    ax.legend(fontsize=7)
    for bar, v in zip(bars2, vals2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    # panel 3 — inflow zone breakdown (uses 4K trough boundary for display)
    ax = axes[2]
    t4 = m["trough_Mm_4k"]
    t2 = m["trough_Mm_2k"]
    if has_bz:
        labels3 = ["Global", "Inflow\nzone",
                   f"Inner 4K\n(<{t4:.0f} Mm)",
                   f"Outer 4K\n(≥{t4:.0f} Mm)",
                   f"Inner 2K\n(<{t2:.0f} Mm)",
                   f"Outer 2K\n(≥{t2:.0f} Mm)",
                   "AR mask\n(Bz)"]
        vals3   = [m["r_div"], m["r_div_inflow"],
                   m["r_div_inner_4k"], m["r_div_outer_4k"],
                   m["r_div_inner_2k"], m["r_div_outer_2k"],
                   m["r_div_ar"]]
        colors3 = [C4K, C2K, C4K, C4K, C2K, C2K, "#9467bd"]
    else:
        labels3 = ["Global", "Inflow\nzone",
                   f"Inner 4K\n(<{t4:.0f} Mm)",
                   f"Outer 4K\n(≥{t4:.0f} Mm)",
                   f"Inner 2K\n(<{t2:.0f} Mm)",
                   f"Outer 2K\n(≥{t2:.0f} Mm)"]
        vals3   = [m["r_div"], m["r_div_inflow"],
                   m["r_div_inner_4k"], m["r_div_outer_4k"],
                   m["r_div_inner_2k"], m["r_div_outer_2k"]]
        colors3 = [C4K, C2K, C4K, C4K, C2K, C2K]

    bars3 = ax.bar(np.arange(len(labels3)), vals3,
                   color=colors3, width=0.55, zorder=3)
    ax.axhline(0, color=CGREY, lw=0.7, ls="--")
    ax.axhline(1, color=CGOOD, lw=0.6, ls=":")
    ax.set_ylim(-0.1, 1.15)
    ax.set_xticks(np.arange(len(labels3)))
    ax.set_xticklabels(labels3, fontsize=7)
    ax.set_title("Divergence $r$ by region\n(normalised)", fontsize=9)
    ax.set_ylabel("Pearson $r$")
    for bar, v in zip(bars3, vals3):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    if save: plt.savefig(save)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MASTER ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_all(vx_4k, vy_4k, vx_2k, vy_2k,
            bz=None,
            pixel_scale_deg=PIXEL_SCALE_DEG,
            save_prefix=None):
    """
    Compute everything and produce all plots.

    Parameters
    ----------
    vx_4k, vy_4k    : HMI 4K LCT velocities [m/s]
    vx_2k, vy_2k    : PMI 2K LCT velocities [m/s]
    bz               : LOS magnetogram [Gauss] on flow-map grid, or None.
                       Use downsample_bz(bz_highres, vx_4k.shape) if needed.
    pixel_scale_deg  : degrees per pixel (default 0.5)
    save_prefix      : e.g. "my_ar" saves my_ar_velocity_maps.png etc.
                       None = display only

    Returns
    -------
    m : dict with all metrics and arrays
    """
    m = compute_all(vx_4k, vy_4k, vx_2k, vy_2k,
                    bz=bz, pixel_scale_deg=pixel_scale_deg)

    def _s(tag):
        return f"{save_prefix}_{tag}.png" if save_prefix else None

    t4 = m["trough_Mm_4k"]
    t2 = m["trough_Mm_2k"]
    sh = m["trough_shift_Mm"]

    print("\n── AMPLITUDE  (raw, physical units) ─────────────────────────────")
    print(f"  Speed amplitude ratio 2K/4K : {m['amplitude_ratio']:.3f}x")
    print(f"  Speed RMSE                  : {m['rmse_speed_raw']:.2f} m/s")
    print(f"  Speed bias (2K−4K)          : {m['bias_speed_raw']:.2f} m/s")
    print(f"  Divergence RMSE             : {m['rmse_div_raw']:.3g} s⁻¹")
    print(f"  Divergence bias (2K−4K)     : {m['bias_div_raw']:.3g} s⁻¹")
    print("\n── STRUCTURE  (normalised, amplitude-independent) ────────────────")
    print(f"  Pearson r  vx               : {m['r_vx']:.3f}")
    print(f"  Pearson r  vy               : {m['r_vy']:.3f}")
    print(f"  Pearson r  speed            : {m['r_speed']:.3f}")
    print(f"  Pearson r  divergence       : {m['r_div']:.3f}")
    print(f"  Vector skill score          : {m['vector_skill']:.3f}")
    print("\n── INFLOW ZONE  (normalised divergence) ──────────────────────────")
    print(f"  r  global                        : {m['r_div']:.3f}")
    print(f"  r  inflow zone (bot. {INFLOW_PERCENTILE}%)        : {m['r_div_inflow']:.3f}")
    print(f"\n  Trough radius  HMI 4K            : {t4:.1f} Mm")
    print(f"  Trough radius  PMI 2K            : {t2:.1f} Mm")
    direction = 'outward' if sh > 0 else 'inward'
    print(f"  Trough shift   2K vs 4K          : {abs(sh):.1f} Mm {direction}")
    print(f"\n  r  inner  HMI boundary (r < {t4:.0f} Mm) : {m['r_div_inner_4k']:.3f}")
    print(f"  r  outer  HMI boundary (r >= {t4:.0f} Mm): {m['r_div_outer_4k']:.3f}")
    print(f"  r  inner  PMI boundary (r < {t2:.0f} Mm) : {m['r_div_inner_2k']:.3f}")
    print(f"  r  outer  PMI boundary (r >= {t2:.0f} Mm): {m['r_div_outer_2k']:.3f}")
    if m["bz"] is not None:
        print(f"  r  AR mask (Bz-based)            : {m['r_div_ar']:.3f}")
        print("\n── BY FIELD STRENGTH  (normalised divergence) ────────────────────")
        for label in ["quiet", "medium", "strong"]:
            print(f"  r  {label:7s}                       : "
                  f"{m.get(f'r_div_{label}', np.nan):.3f}")

    plot_velocity_maps(m,              save=_s("velocity_maps"))
    plot_divergence_maps(m,            save=_s("divergence_maps"))
    plot_psd(m,                        save=_s("psd"))
    plot_scatter(m,                    save=_s("scatter"))
    plot_radial_profile(m,             save=_s("radial_profile"))
    plot_field_strength_correlation(m, save=_s("field_strength"))
    plot_metric_summary(m,             save=_s("metric_summary"))

    return m


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION & DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def print_physical_validation(m, pixel_scale_deg=PIXEL_SCALE_DEG):
    """
    Print physically meaningful inflow diagnostics vs literature.
    Expected from Loptien et al. (2017): inflow ~20-50 m/s, up to 10 deg from AR.
    """
    div_4k   = m["div_4k_phys"]
    div_2k   = m["div_2k_phys"]
    speed_4k = m["speed_4k"]
    speed_2k = m["speed_2k"]
    imask    = m["inflow_mask"]
    p4       = m["radial_p4"]
    p2       = m["radial_p2"]
    r        = m["radial_r_Mm"]
    t4       = m["trough_Mm_4k"]
    t2       = m["trough_Mm_2k"]
    sh       = m["trough_shift_Mm"]

    def _to_deg(Mm):
        return Mm / R_SUN_MM * (180 / np.pi)

    print("\n── PHYSICAL VALIDATION  (4K vs literature) ──────────────────────")

    print(f"\n  Inflow zone velocities (bottom {INFLOW_PERCENTILE}% divergence mask):")
    print(f"    Mean speed   4K : {np.mean(speed_4k[imask]):.1f} m/s")
    print(f"    Mean speed   2K : {np.mean(speed_2k[imask]):.1f} m/s")
    print(f"    Median speed 4K : {np.median(speed_4k[imask]):.1f} m/s")
    print(f"    Max speed    4K : {np.max(speed_4k[imask]):.1f} m/s")
    print(f"    Literature      : 20-50 m/s  (Loptien et al. 2017)")

    print(f"\n  Inflow zone divergence (bottom {INFLOW_PERCENTILE}% mask):")
    print(f"    Mean  div 4K : {np.mean(div_4k[imask]):.3e} s⁻¹")
    print(f"    Min   div 4K : {np.min(div_4k[imask]):.3e} s⁻¹  (peak inflow)")
    print(f"    Mean  div 2K : {np.mean(div_2k[imask]):.3e} s⁻¹")
    print(f"    Literature   : ~1e-6 to 1e-5 s⁻¹")

    direction = "outward" if sh > 0 else "inward"
    print(f"\n  Inflow trough radius:")
    print(f"    HMI 4K      : {t4:.1f} Mm  =  {_to_deg(t4):.2f} deg")
    print(f"    PMI 2K      : {t2:.1f} Mm  =  {_to_deg(t2):.2f} deg")
    print(f"    Shift 2K-4K : {abs(sh):.1f} Mm {direction}  "
          f"(PSF smearing displaces apparent inflow annulus)")
    print(f"    Literature  : 30-60 Mm  (~5-10 deg)  (Loptien et al. 2017)")

    print(f"\n  Inflow spatial extent (zero crossing after trough):")
    for label, extent_Mm in [("HMI 4K", m["extent_Mm_4k"]),
                               ("PMI 2K", m["extent_Mm_2k"])]:
        if np.isfinite(extent_Mm):
            print(f"    {label}  : {extent_Mm:.1f} Mm  =  {_to_deg(extent_Mm):.2f} deg")
        else:
            print(f"    {label}  : no zero crossing within patch — "
                  f"inflow extends beyond boundary")
    print(f"    Literature  : up to ~10 deg  (Loptien et al. 2017)")

    moat_speed = (np.mean(speed_4k[m["ar_mask"]])
                  if m["ar_mask"] is not None else np.nan)
    print(f"\n  AR core (moat) outflow:")
    print(f"    Mean speed inside AR mask 4K : {moat_speed:.1f} m/s")
    print(f"    Peak norm. div at r=0  4K    : {p4[0]:.3f}  (positive = outflow)")
    print(f"    Peak norm. div at r=0  2K    : {p2[0]:.3f}")
    print(f"    Literature moat extent       : ~30-40 Mm beyond penumbra")

    print(f"\n  Divergence r by region (each instrument's own trough boundary):")
    print(f"    Inner 4K  (r < {t4:.0f} Mm)  : {m['r_div_inner_4k']:.3f}")
    print(f"    Outer 4K  (r >= {t4:.0f} Mm) : {m['r_div_outer_4k']:.3f}")
    print(f"    Inner 2K  (r < {t2:.0f} Mm)  : {m['r_div_inner_2k']:.3f}")
    print(f"    Outer 2K  (r >= {t2:.0f} Mm) : {m['r_div_outer_2k']:.3f}")


def print_psd_peaks(m):
    """Print the spatial scale at peak power for each PSD curve."""
    print("\n── PSD PEAKS  (spatial scale at maximum power) ───────────────────")
    for label, tag in [
        ("HMI 4K  raw",        "4k_raw"),
        ("PMI 2K  raw",        "2k_raw"),
        ("HMI 4K  normalised", "4k_norm"),
        ("PMI 2K  normalised", "2k_norm"),
    ]:
        print(f"  {label:25s} : peak at {m[f'psd_peak_scale_{tag}']:.1f} Mm")

    scale = m["psd_k"]
    sel   = (scale > 2.0) & np.isfinite(scale)
    ratio_raw  = m["psd_ratio_raw"][sel]
    ratio_norm = m["psd_ratio_norm"][sel]
    scale_sel  = scale[sel]
    print(f"\n  Raw ratio peak (max overestimation)  : "
          f"{scale_sel[np.nanargmax(ratio_raw)]:.1f} Mm  "
          f"(ratio = {np.nanmax(ratio_raw):.2f}x)")
    print(f"  Norm ratio worst deviation from 1    : "
          f"{scale_sel[np.nanargmax(np.abs(ratio_norm - 1))]:.1f} Mm  "
          f"(ratio = {ratio_norm[np.nanargmax(np.abs(ratio_norm - 1))]:.2f}x)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    vx_4k, vy_4k, vx_2k, vy_2k, bz = load_data()

    m = run_all(
        vx_4k, vy_4k, vx_2k, vy_2k,
        bz=bz,
        pixel_scale_deg=PIXEL_SCALE_DEG,
        save_prefix="ar_inflow",   # set None to display only
    )

    print_physical_validation(m)
    print_psd_peaks(m)
# %%
