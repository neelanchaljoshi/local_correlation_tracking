# %%
"""
LCT Resolution Comparison — HMI 4K vs PMI 2K
=============================================
Publication plots for comparing LCT-derived active region inflow maps at two
resolutions (HMI 4K vs PMI-like 2K) and two cadences (15 min vs 30 min).

QUICK START
-----------
    from lct_metrics import run_paper_plots, load_data, compute_all

    m15 = compute_all(*load_data(cadence="15min"))
    m30 = compute_all(*load_data(cadence="30min"))
    run_paper_plots(m15, m30, save_prefix="ar_inflow")

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
from scipy.ndimage import gaussian_filter, uniform_filter1d, center_of_mass, zoom, binary_dilation
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")


# ── publication style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":    "white",      "axes.facecolor":      "white",
    "axes.edgecolor":      "#333333",    "axes.labelcolor":     "#222222",
    "axes.titlecolor":     "#111111",    "axes.linewidth":      1.0,
    "axes.grid":           False,
    "xtick.color":         "#333333",    "ytick.color":         "#333333",
    "xtick.direction":     "in",         "ytick.direction":     "in",
    "xtick.major.size":    5.0,          "ytick.major.size":    5.0,
    "xtick.minor.size":    3.0,          "ytick.minor.size":    3.0,
    "xtick.major.width":   0.8,          "ytick.major.width":   0.8,
    "xtick.labelsize":     11,           "ytick.labelsize":     11,
    "text.color":          "#222222",    "font.family":         "sans-serif",
    "font.size":           12,           "axes.labelsize":      12,
    "axes.titlesize":      13,           "axes.titlepad":       8,
    "legend.fontsize":     10,           "legend.framealpha":   0.92,
    "legend.edgecolor":    "#bbbbbb",    "legend.borderpad":    0.5,
    "figure.dpi":          150,          "savefig.dpi":         300,
    "savefig.facecolor":   "white",      "savefig.bbox":        "tight",
    "image.origin":        "lower",      "image.interpolation": "nearest",
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})

C4K   = "#1f77b4"   # blue   — HMI 4K
C2K   = "#d62728"   # red    — PMI 2K
CGOOD = "#2ca02c"   # green  — 15-min cadence
C30   = "#ff7f0e"   # orange — 30-min cadence
CGREY = "#7f7f7f"   # grey   — neutral lines
R_SUN_MM = 695.7    # Mm


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(cadence="15min"):
    """
    Load LCT output and the LOS magnetogram for a given cadence.

    Parameters
    ----------
    cadence : "15min" or "30min"

    Returns
    -------
    vx_4k, vy_4k : 2D arrays — HMI 4K LCT velocities [m/s], shape (nlat, nlng)
    vx_2k, vy_2k : 2D arrays — PMI 2K LCT velocities [m/s], same shape
    bz           : 2D array or None — LOS magnetogram [Gauss] on flow-map grid
    longitude    : 1D array — longitude axis [degrees], length nlng
    latitude     : 1D array — latitude axis [degrees], length nlat

    Notes
    -----
    - vx = longitudinal (phi) component;  positive = eastward
    - vy = latitudinal (theta) component; positive = northward
    - In your HDF5 output: vx = uphi, vy = -utheta
    - Both fields must be time-averaged before passing in
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
    bz         = downsample_bz(bz_highres, target_shape=vx_4k.shape)

    return vx_4k, vy_4k, vx_2k, vy_2k, bz, longitude, latitude


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PIXEL_SCALE_DEG   = 0.5    # degrees per pixel in the flow map
MAG_THRESHOLD_G   = 50.0   # |Bz| threshold for AR core definition [Gauss]
DILATION_PX       = 5      # dilation radius around AR core [pixels] (~30 Mm)
INFLOW_PERCENTILE = 20     # bottom N% of 4K divergence = inflow zone
QUIVER_STRIDE     = 2      # plot every Nth vector in flow maps (tune to taste)


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def px_scale_m(pixel_scale_deg=PIXEL_SCALE_DEG):
    """Physical size of one flow-map pixel in metres."""
    return np.deg2rad(pixel_scale_deg) * R_SUN_MM * 1e6

def px_scale_Mm(pixel_scale_deg=PIXEL_SCALE_DEG):
    return px_scale_m(pixel_scale_deg) / 1e6

def downsample_bz(bz_highres, target_shape):
    """Downsample a high-resolution magnetogram to the flow-map grid (bilinear)."""
    factor = (target_shape[0] / bz_highres.shape[0],
              target_shape[1] / bz_highres.shape[1])
    return zoom(np.nan_to_num(bz_highres), factor, order=1)

def normalise_vector(vx, vy):
    """Remove mean; divide both components by std(speed) to preserve vx/vy ratio."""
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
    """Split domain into quiet / medium / strong field."""
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
                longitude=None,
                latitude=None,
                pixel_scale_deg=PIXEL_SCALE_DEG):
    """
    Compute all metrics and intermediate arrays.

    Parameters
    ----------
    vx_4k, vy_4k    : HMI 4K LCT velocities [m/s], shape (nlat, nlng)
    vx_2k, vy_2k    : PMI 2K LCT velocities [m/s], same shape
    bz               : LOS magnetogram [Gauss] on flow-map grid, or None
    longitude        : 1D array of longitudes [deg], length nlng, or None
    latitude         : 1D array of latitudes [deg], length nlat, or None
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
    div_4k_phys = divergence_physical(vx_4k, vy_4k, pixel_scale_deg)
    div_2k_phys = divergence_physical(vx_2k, vy_2k, pixel_scale_deg)
    div_4k_norm = divergence_normalised(vx_4n, vy_4n)
    div_2k_norm = divergence_normalised(vx_2n, vy_2n)

    # ── speed ─────────────────────────────────────────────────────────────
    speed_4k = np.hypot(vx_4k, vy_4k)
    speed_2k = np.hypot(vx_2k, vy_2k)

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
    m["r_speed"]      = pearson_r(np.hypot(vx_4n, vy_4n), np.hypot(vx_2n, vy_2n))
    m["r_div"]        = pearson_r(div_4k_norm, div_2k_norm)
    m["vector_skill"] = vector_skill(vx_4n, vy_4n, vx_2n, vy_2n)

    # ── INFLOW ZONE — divergence-threshold mask ────────────────────────────
    m["r_div_inflow"]    = pearson_r(div_4k_norm, div_2k_norm, mask=imask)
    m["rmse_div_inflow"] = rmse(div_4k_norm, div_2k_norm, mask=imask)
    m["bias_div_inflow"] = bias(div_4k_norm, div_2k_norm, mask=imask)

    # ── AR MASK METRICS ───────────────────────────────────────────────────
    if ar_mask is not None:
        m["r_div_ar"]    = pearson_r(div_4k_norm, div_2k_norm, mask=ar_mask)
        m["rmse_div_ar"] = rmse(div_4k_norm, div_2k_norm, mask=ar_mask)
        m["bias_div_ar"] = bias(div_4k_norm, div_2k_norm, mask=ar_mask)
        for label, fmask in bz_masks.items():
            m[f"r_div_{label}"] = (pearson_r(div_4k_norm, div_2k_norm, mask=fmask)
                                   if fmask.any() else np.nan)

    # ── RADIAL PROFILE — normalised divergence, Mm axis ───────────────────
    p4_raw = radial_profile(div_4k_norm, com)
    p2_raw = radial_profile(div_2k_norm, com)
    p4     = uniform_filter1d(p4_raw[:max_r], size=3)
    p2     = uniform_filter1d(p2_raw[:max_r], size=3)
    r_Mm   = np.arange(max_r) * pxMm

    trough_px_4k = int(np.argmin(p4))
    trough_px_2k = int(np.argmin(p2))

    extent_Mm_4k = _zero_crossing_after(p4, trough_px_4k, r_Mm)
    extent_Mm_2k = _zero_crossing_after(p2, trough_px_2k, r_Mm)

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
    m["trough_shift_Mm"] = (trough_px_2k - trough_px_4k) * pxMm
    m["trough_Mm"]       = m["trough_Mm_4k"]
    m["extent_Mm_4k"]    = extent_Mm_4k
    m["extent_Mm_2k"]    = extent_Mm_2k
    m["radial_r_Mm"]     = r_Mm
    m["radial_p4"]       = p4
    m["radial_p2"]       = p2

    # ── store arrays ──────────────────────────────────────────────────────
    m.update({
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
    """Return extent dict for imshow/contour when coordinates are available."""
    if longitude is not None and latitude is not None:
        return {"extent": [longitude[0], longitude[-1],
                            latitude[0],  latitude[-1]],
                "origin": "lower"}
    return {}

def _set_latlon_ticks(ax, longitude, latitude):
    """Apply degree-labelled ticks; no-op if coordinates are absent."""
    if longitude is not None and latitude is not None:
        ax.set_xlabel("Longitude [°]")
        ax.set_ylabel("Latitude [°]")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4, integer=True))
    else:
        ax.set_xticks([]); ax.set_yticks([])

def _imshow_latlon(ax, data, cmap, title, unit="", sym=False,
                   longitude=None, latitude=None):
    """imshow with optional lat/lon axes and a clean colourbar."""
    if sym:
        vmax = np.nanpercentile(np.abs(data), 99); vmin = -vmax
    else:
        vmin, vmax = np.nanpercentile(data, [1, 99])

    if longitude is not None and latitude is not None:
        lon_span = abs(longitude[-1] - longitude[0])
        lat_span = abs(latitude[-1]  - latitude[0])
        asp = lat_span / lon_span if lon_span > 0 else "auto"
    else:
        asp = "equal"

    kw = _axis_kw(longitude, latitude)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect=asp, **kw)
    _set_latlon_ticks(ax, longitude, latitude)
    ax.set_title(title)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(unit)
    cb.ax.tick_params(direction="in")
    cb.outline.set_linewidth(0.8)
    return im

def _overlay_bz(ax, bz, longitude=None, latitude=None):
    kw = _axis_kw(longitude, latitude)
    # white solid for B+, black dashed for B- — visible against any divergence cmap
    ax.contour(bz, levels=[ 100,  300,  500], colors="white",
               linewidths=1.0, linestyles="-",  **kw)
    ax.contour(bz, levels=[-500, -300, -100], colors="black",
               linewidths=1.0, linestyles="--", **kw)

def _overlay_ar_mask(ax, mask, longitude=None, latitude=None):
    kw = _axis_kw(longitude, latitude)
    ax.contour(mask.astype(float), levels=[0.5],
               colors="#555555", linewidths=0.8, linestyles="--", **kw)

def _quiver(ax, vx, vy, longitude=None, latitude=None,
            color="k", stride=QUIVER_STRIDE, scale=None, alpha=0.85):
    """
    Overplot subsampled velocity vectors.

    Positions are in degree coordinates when longitude/latitude are given,
    otherwise in pixel indices.  A common `scale` value keeps arrow lengths
    comparable across panels.
    """
    s    = stride
    vx_s = vx[::s, ::s]
    vy_s = vy[::s, ::s]

    if longitude is not None and latitude is not None:
        X, Y = np.meshgrid(longitude[::s], latitude[::s])
    else:
        ny, nx = vx.shape
        X, Y   = np.meshgrid(np.arange(0, nx, s), np.arange(0, ny, s))

    X = X[:vx_s.shape[0], :vx_s.shape[1]]
    Y = Y[:vx_s.shape[0], :vx_s.shape[1]]

    ax.quiver(X, Y, vx_s, vy_s,
              color=color, scale=scale, alpha=alpha,
              width=0.003, headwidth=3, headlength=4,
              headaxislength=3.5, pivot="mid")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 0 — FLOW MAPS  (2 cadences × 2 instruments, single figure)
# ══════════════════════════════════════════════════════════════════════════════

def plot_flow_maps(m15, m30, save=None):
    """
    2 rows × 2 columns showing raw LCT speed [m/s] + quiver arrows.

      Row 0 — 15 min cadence:  HMI 4K  |  PMI 2K
      Row 1 — 30 min cadence:  HMI 4K  |  PMI 2K

    A single shared colourbar (right) and shared quiver scale make the
    amplitude difference visible across both instruments and both cadences.
    Row labels are placed as y-axis titles on the left column.
    """
    lon = m15.get("longitude")
    lat = m15.get("latitude")

    vmax_spd = max(np.nanpercentile(m15["speed_2k"], 98),
                   np.nanpercentile(m30["speed_2k"], 98))

    spd_ref      = np.nanpercentile(m15["speed_4k"], 95)
    nx_grid      = m15["speed_4k"].shape[1] / QUIVER_STRIDE
    quiver_scale = spd_ref * nx_grid * 0.3

    if lon is not None and lat is not None:
        lon_span = abs(lon[-1] - lon[0])
        lat_span = abs(lat[-1] - lat[0])
        asp = lat_span / lon_span if lon_span > 0 else "auto"
    else:
        asp = "equal"

    kw_img = _axis_kw(lon, lat)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.08},
                             layout=None)
    fig.suptitle(
        "LCT flow speed  [m s$^{-1}$]\n"
        "Amplitude difference driven by PSF broadening",
        fontsize=13, y = 0.92
    )

    row_data = [(m15, "15 min"), (m30, "30 min")]
    col_data = [
        ("speed_4k", "vx_4k", "vy_4k", "HMI 4K"),
        ("speed_2k", "vx_2k", "vy_2k", "PMI 2K"),
    ]

    ims = []
    for row_i, (m, row_label) in enumerate(row_data):
        for col_i, (spd_key, vx_key, vy_key, col_label) in enumerate(col_data):
            ax = axes[row_i, col_i]
            im = ax.imshow(m[spd_key], cmap="magma", vmin=0, vmax=vmax_spd,
                           aspect=asp, **kw_img)
            _set_latlon_ticks(ax, lon, lat)
            _quiver(ax, m[vx_key], m[vy_key], lon, lat,
                    color="white", scale=quiver_scale)
            for sp in ax.spines.values():
                sp.set_linewidth(0.8)
            if row_i == 0:
                ax.set_title(col_label, fontsize=13, pad=6)
            if col_i == 0:
                ax.set_ylabel(f"{row_label}\n\nLatitude [°]", fontsize=12)
            else:
                ax.set_ylabel("")
            ims.append(im)

        ratio = m["amplitude_ratio"]
        axes[row_i, 1].text(
            1.02, 0.5, f"2K/4K = {ratio:.2f}×",
            transform=axes[row_i, 1].transAxes,
            fontsize=10, color=CGREY, style="italic",
            va="center", ha="left", rotation=90
        )

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, axes[1, 1].get_position().y0,
                        0.018,
                        axes[0, 1].get_position().y1 - axes[1, 1].get_position().y0])
    cb = fig.colorbar(ims[0], cax=cax)
    cb.set_label("Speed  [m s$^{-1}$]", fontsize=12)
    cb.ax.tick_params(direction="in")
    cb.outline.set_linewidth(0.8)

    if save:
        plt.savefig(save)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — DIVERGENCE MAPS  (2 cadences × 3 cols, single figure)
# ══════════════════════════════════════════════════════════════════════════════

def plot_divergence_maps(m15, m30, save=None):
    """
    2 rows × 3 columns — physical divergence [s⁻¹]:

      Row 0 — 15 min:  HMI 4K  |  PMI 2K  |  Residual 2K−4K
      Row 1 — 30 min:  HMI 4K  |  PMI 2K  |  Residual 2K−4K

    Shared symmetric colour scale across all six panels.
    Overlays: AR mask (dashed), Bz polarity contours.
    Legend placed below the figure, outside all panels.
    """
    lon = m15.get("longitude")
    lat = m15.get("latitude")

    if lon is not None and lat is not None:
        lon_span = abs(lon[-1] - lon[0])
        lat_span = abs(lat[-1] - lat[0])
        asp = lat_span / lon_span if lon_span > 0 else "auto"
    else:
        asp = "equal"

    all_divs = np.concatenate([
        m15["div_4k_phys"].ravel(), m15["div_2k_phys"].ravel(),
        m30["div_4k_phys"].ravel(), m30["div_2k_phys"].ravel(),
    ])
    vmax_div = np.nanpercentile(np.abs(all_divs), 99)

    kw_img = _axis_kw(lon, lat)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.08},
                             layout=None)
    fig.suptitle(
        r"Divergence  $\nabla\cdot\mathbf{v}$  [s$^{-1}$]",
        fontsize=18, y=0.9
    )
    fig.subplots_adjust(right=0.88)

    row_data = [(m15, "15 min"), (m30, "30 min")]

    last_ims = {}
    for row_i, (m, row_label) in enumerate(row_data):
        d4, d2 = m["div_4k_phys"], m["div_2k_phys"]
        panels = [
            (d4,      "HMI 4K"),
            (d2,      "PMI 2K"),
            (d2 - d4, "Residual  2K$-$4K"),
        ]
        for col_i, (arr, col_title) in enumerate(panels):
            ax = axes[row_i, col_i]
            im = ax.imshow(arr, cmap="PuOr_r",
                           vmin=-vmax_div, vmax=vmax_div,
                           aspect=asp, **kw_img)
            _set_latlon_ticks(ax, lon, lat)
            for sp in ax.spines.values():
                sp.set_linewidth(0.8)
            if m["ar_mask"] is not None:
                _overlay_ar_mask(ax, m["ar_mask"], lon, lat)
            if m["bz"] is not None:
                _overlay_bz(ax, m["bz"], lon, lat)

            if row_i == 0:
                ax.set_title(col_title, fontsize=13, pad=6)
            if col_i == 0:
                ax.set_ylabel(f"{row_label}\n\nLatitude [°]", fontsize=12)
            else:
                ax.set_ylabel("")

            if col_i == 2:
                last_ims[row_i] = im

    for row_i in range(2):
        ax_ref = axes[row_i, 2]
        pos = ax_ref.get_position()
        cax = fig.add_axes([0.895, pos.y0, 0.015, pos.height])
        cb = fig.colorbar(last_ims[row_i], cax=cax)
        cb.set_label("s$^{-1}$", fontsize=11)
        cb.ax.tick_params(direction="in", labelsize=10)
        cb.outline.set_linewidth(0.8)

    # legend — no inflow mask entry
    legend_handles = []
    if m15["ar_mask"] is not None:
        legend_handles.append(
            mlines.Line2D([], [], color="#555555", ls="--", lw=1.2,
                          label=f"AR mask  $|B_z| > {MAG_THRESHOLD_G:.0f}$ G"))
    if m15["bz"] is not None:
        legend_handles += [
            mlines.Line2D([], [], color="white", ls="-", lw=2.0,
                          label="$B_+$ contours",
                          path_effects=[
                              mpe.withStroke(linewidth=3, foreground="#888888")]),
            mlines.Line2D([], [], color="black", ls="--", lw=1.2,
                          label="$B_-$ contours"),
        ]
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=len(legend_handles), framealpha=0.92,
                   fontsize=11, bbox_to_anchor=(0.5, 0.0))

    if save:
        plt.savefig(save, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — RADIAL DIVERGENCE PROFILES  (side-by-side cadence comparison)
# ══════════════════════════════════════════════════════════════════════════════

def plot_radial_profiles_comparison(m15, m30, save=None):
    """
    Two-panel figure (shared y-axis) comparing radial divergence profiles.

    Left  — 15-min cadence
    Right — 30-min cadence

    Each panel overlays HMI 4K (blue solid) and PMI 2K (red dashed).
    Dotted verticals mark trough positions; dash-dot verticals mark the
    inflow extent (first zero crossing after the trough).
    A text box annotates the trough shift between instruments.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True,
                             gridspec_kw={"wspace": 0.04})

    for ax, (m, title) in zip(axes, [(m15, "15 min cadence"),
                                      (m30, "30 min cadence")]):
        r  = m["radial_r_Mm"]
        p4 = m["radial_p4"]
        p2 = m["radial_p2"]
        t4 = m["trough_Mm_4k"]
        t2 = m["trough_Mm_2k"]
        e4 = m["extent_Mm_4k"]
        e2 = m["extent_Mm_2k"]

        ax.plot(r, p4, color=C4K, lw=2.2, label="HMI 4K")
        ax.plot(r, p2, color=C2K, lw=2.2, ls="--", label="PMI 2K")
        ax.axhline(0, color=CGREY, lw=0.9)

        ax.axvline(t4, color=C4K, lw=1.1, ls=":",
                   label=f"Trough 4K  {t4:.0f} Mm")
        ax.axvline(t2, color=C2K, lw=1.1, ls=":",
                   label=f"Trough 2K  {t2:.0f} Mm")
        if np.isfinite(e4):
            ax.axvline(e4, color=C4K, lw=1.1, ls="-.",
                       label=f"Extent 4K  {e4:.0f} Mm")
        if np.isfinite(e2):
            ax.axvline(e2, color=C2K, lw=1.1, ls="-.",
                       label=f"Extent 2K  {e2:.0f} Mm")

        ax.set_title(title, pad=8)
        ax.set_xlabel("Radius from inflow centre [Mm]")
        ax.grid(True, color="#dddddd", lw=0.6, ls="--")
        ax.legend(loc="upper right")

        shift     = m["trough_shift_Mm"]
        direction = "outward" if shift > 0 else "inward"
        ax.text(0.03, 0.97,
                f"Trough shift: {abs(shift):.1f} Mm {direction}",
                transform=ax.transAxes, fontsize=10, va="top", color=CGREY,
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#bbbbbb", lw=0.8))

    axes[0].set_ylabel("Mean normalised divergence [arb.]")
    fig.suptitle("Azimuthally averaged normalised divergence",
                 y=1.01)
    fig.text(0.5, -0.02,
             "Dotted verticals = trough location   "
             "Dash-dot verticals = inflow extent (zero crossing)",
             ha="center", fontsize=10, color=CGREY, style="italic")
    if save:
        plt.savefig(save)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — STRUCTURAL METRIC BAR CHART  (grouped by cadence)
# ══════════════════════════════════════════════════════════════════════════════

def plot_metric_comparison(m15, m30, save=None):
    """
    Grouped bar chart of five structural metrics (normalised, amplitude-free):

      r_speed       — speed Pearson r, full domain
      r_div         — divergence Pearson r, full domain
      r_div_quiet   — divergence Pearson r, quiet-field pixels (|B| < 50 G)
      r_div_medium  — divergence Pearson r, medium-field pixels (50–300 G)
      r_div_strong  — divergence Pearson r, strong-field pixels (|B| > 300 G)

    Splitting by field strength tests whether the cadence degradation is
    uniform or concentrated in magnetically active regions (where granulation
    is suppressed and LCT has lower SNR at longer cadence).

    Requires bz to have been passed to compute_all(); bars will show nan
    if the magnetogram is unavailable.

    Green = 15 min,  orange = 30 min.
    Δ annotations above each group show the cadence-induced change.
    """
    metric_keys   = ["r_speed", "r_div", "r_div_quiet", "r_div_medium", "r_div_strong"]
    metric_labels = [
        "$r_{\\mathrm{speed}}$\n(full domain)",
        "$r_{\\nabla v}$\n(full domain)",
        "$r_{\\nabla v}$\nquiet  $|B|<50$ G",
        "$r_{\\nabla v}$\nmedium  $50$–$300$ G",
        "$r_{\\nabla v}$\nstrong  $|B|>300$ G",
    ]

    vals_15 = [m15.get(k, np.nan) for k in metric_keys]
    vals_30 = [m30.get(k, np.nan) for k in metric_keys]

    x     = np.arange(len(metric_keys))
    width = 0.28

    fig, ax = plt.subplots(figsize=(13, 11))

    bars_15 = ax.bar(x - width/2, vals_15, width, color=CGOOD, label="15 min",
                     zorder=3, edgecolor="white", linewidth=0.8)
    bars_30 = ax.bar(x + width/2, vals_30, width, color=C30,  label="30 min",
                     zorder=3, edgecolor="white", linewidth=0.8)

    ax.axhline(1, color=CGREY, lw=1.2, ls=":",  zorder=2, label="Perfect = 1")
    ax.axhline(0, color=CGREY, lw=0.8, ls="--", zorder=2)

    for bar, v in zip(bars_15, vals_15):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=15, color="#1a5c1a", fontweight="bold")
    for bar, v in zip(bars_30, vals_30):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=15, color="#8b3a00", fontweight="bold")

    for i, (v15, v30) in enumerate(zip(vals_15, vals_30)):
        if not (np.isfinite(v15) and np.isfinite(v30)):
            continue
        delta = v30 - v15
        sign  = "+" if delta >= 0 else ""
        ax.text(x[i], max(v15, v30) + 0.08,
                f"$\\Delta${sign}{delta:.3f}",
                ha="center", va="bottom", fontsize=13,
                color="#444444", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(-0.05, 1.38)
    ax.set_ylabel("Pearson $r$  [normalised fields]", fontsize=18)
    ax.set_title(
        "Structural fidelity: PMI 2K vs HMI 4K\n"
        "Grouped by cadence  (15 min vs 30 min)",
        fontsize=20, pad=12
    )
    ax.legend(loc="lower right", fontsize=14)
    ax.grid(True, axis="y", color="#e0e0e0", lw=0.6, ls="--", zorder=0)

    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# MASTER ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_paper_plots(m15, m30, save_prefix=None):
    """
    Produce all paper plots and print the summary table.

    Parameters
    ----------
    m15, m30     : dicts from compute_all() for 15-min and 30-min datasets
    save_prefix  : e.g. "ar_inflow" → saves ar_inflow_flowmaps.pdf etc.
                   None = display only

    Files produced  (all PDF, vector graphics)
    -------------------------------------------
    <prefix>_flowmaps.pdf  — raw speed + quiver, 15 & 30 min combined (2×2)
    <prefix>_divmaps.pdf   — physical divergence maps, 15 & 30 min combined (2×3)
    <prefix>_radial.pdf    — radial profile side-by-side comparison
    <prefix>_metrics.pdf   — structural metric grouped bar chart
    """
    def _s(tag):
        return f"{save_prefix}_{tag}.pdf" if save_prefix else None

    plot_flow_maps(m15, m30,                   save=_s("flowmaps"))
    plot_divergence_maps(m15, m30,             save=_s("divmaps"))
    plot_radial_profiles_comparison(m15, m30,  save=_s("radial"))
    plot_metric_comparison(m15, m30,           save=_s("metrics"))

    _print_table(m15, m30)


def _print_table(m15, m30):
    """Print amplitude + spatial numbers ready to copy into a paper table."""
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
        if key is None:
            print(); continue
        v15 = m15.get(key, np.nan)
        v30 = m30.get(key, np.nan)
        v15_s = f"{v15:{fmt}}" if np.isfinite(float(v15)) else "—"
        v30_s = f"{v30:{fmt}}" if np.isfinite(float(v30)) else "—"
        print(f"  {label:<35}  {v15_s:>8}  {v30_s:>8}")
    print("═"*65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    vx_4k_15, vy_4k_15, vx_2k_15, vy_2k_15, bz_15, lon_15, lat_15 = load_data("15min")
    m15 = compute_all(vx_4k_15, vy_4k_15, vx_2k_15, vy_2k_15,
                      bz=bz_15, longitude=lon_15, latitude=lat_15)

    vx_4k_30, vy_4k_30, vx_2k_30, vy_2k_30, bz_30, lon_30, lat_30 = load_data("30min")
    m30 = compute_all(vx_4k_30, vy_4k_30, vx_2k_30, vy_2k_30,
                      bz=bz_30, longitude=lon_30, latitude=lat_30)

    run_paper_plots(m15, m30, save_prefix="ar_inflow")
# %%
