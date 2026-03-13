# %%
"""
LCT Resolution Comparison — Metrics & Plots
============================================
Plug in your own vx_4k, vy_4k, vx_2k, vy_2k (all on the same grid,
i.e. 2K fields already upsampled to 4K resolution) and optionally bz_4k.

Quick start:
    from lct_metrics import *
    metrics = compute_all_metrics(vx_4k, vy_4k, vx_2k, vy_2k, bz_4k)
    plot_all(vx_4k, vy_4k, vx_2k, vy_2k, metrics)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom, binary_dilation, gaussian_filter
from scipy.stats import pearsonr
from scipy.fft import fft2, fftshift
import warnings
warnings.filterwarnings("ignore")

# ── style — clean light mode for publication ───────────────────────────────
plt.rcParams.update({
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.edgecolor":      "#333333",
    "axes.labelcolor":     "#222222",
    "axes.titlecolor":     "#111111",
    "axes.linewidth":      0.8,
    "axes.grid":           True,
    "grid.color":          "#dddddd",
    "grid.linewidth":      0.5,
    "grid.linestyle":      "--",
    "xtick.color":         "#333333",
    "ytick.color":         "#333333",
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.major.size":    3.5,
    "ytick.major.size":    3.5,
    "xtick.minor.size":    2.0,
    "ytick.minor.size":    2.0,
    "text.color":          "#222222",
    "font.family":         "serif",
    "font.size":           9,
    "axes.labelsize":      9,
    "axes.titlesize":      10,
    "legend.fontsize":     8,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#cccccc",
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.facecolor":   "white",
    "savefig.bbox":        "tight",
    "image.origin":        "lower",
    "image.interpolation": "nearest",
})

# colour palette — accessible, print-safe
C4K   = "#1f77b4"   # blue  — 4K / reference
C2K   = "#d62728"   # red   — 2K
CGOOD = "#2ca02c"   # green — ratio=1 reference
CGREY = "#7f7f7f"   # grey  — neutral reference lines


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def upsample_2k_to_4k(vx_2k, vy_2k, target_shape):
    """Bicubic upsample 2K velocity fields to match the 4K grid."""
    factor = np.array(target_shape) / np.array(vx_2k.shape)
    vx_up = zoom(vx_2k, factor, order=3)
    vy_up = zoom(vy_2k, factor, order=3)
    return vx_up, vy_up


def divergence(vx, vy):
    """2-D divergence via central differences (∇·V)."""
    dvx_dx = np.gradient(vx, axis=1)
    dvy_dy = np.gradient(vy, axis=0)
    return dvx_dx + dvy_dy


def ar_mask(bz, mag_threshold_G=50, dilation_px=50):
    """
    Boolean mask of the inflow zone around active regions.

    Parameters
    ----------
    bz              : 2D array  – LOS magnetogram (Gauss)
    mag_threshold_G : float     – |Bz| cutoff for AR core
    dilation_px     : int       – dilation radius in pixels (~inflow extent)
    """
    core = np.abs(bz) > mag_threshold_G
    return binary_dilation(core, iterations=dilation_px)


def _flat(a, mask=None):
    return a[mask].ravel() if mask is not None else a.ravel()


# ══════════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def rmse(ref, test, mask=None):
    """Root-mean-square error of test w.r.t. ref."""
    d = ref - test
    if mask is not None:
        d = d[mask]
    return float(np.sqrt(np.nanmean(d**2)))


def bias(ref, test, mask=None):
    """Mean signed error (test − ref)."""
    d = test - ref
    if mask is not None:
        d = d[mask]
    return float(np.nanmean(d))


def spatial_correlation(ref, test, mask=None):
    """Pearson r between ref and test fields."""
    r_flat = _flat(ref, mask) if mask is not None else ref.ravel()
    t_flat = _flat(test, mask) if mask is not None else test.ravel()
    r, _ = pearsonr(r_flat, t_flat)
    return float(r)


def vector_skill_score(vx_ref, vy_ref, vx_test, vy_test, mask=None):
    """
    Lhermitte & Lemaitre (1984) vector correlation coefficient.
    Returns a scalar in [0, 1]; 1 = perfect agreement.
    """
    if mask is not None:
        vx_ref, vy_ref   = _flat(vx_ref, mask), _flat(vy_ref, mask)
        vx_test, vy_test = _flat(vx_test, mask), _flat(vy_test, mask)
    num = np.nansum(vx_ref * vx_test + vy_ref * vy_test)
    den = np.sqrt(np.nansum(vx_ref**2 + vy_ref**2) *
                  np.nansum(vx_test**2 + vy_test**2))
    return float(num / den) if den > 0 else np.nan


def psd_ratio(ref, test):
    """
    Azimuthally averaged power spectral density ratio (test / ref)
    as a function of spatial frequency k (cycles / pixel).

    Returns
    -------
    k_bins : 1D array   – spatial frequency bin centres
    ratio  : 1D array   – PSD(test) / PSD(ref) per bin
    psd_r  : 1D array   – PSD of ref  (for reference plot)
    psd_t  : 1D array   – PSD of test
    """
    def azimuthal_avg(power):
        N = power.shape[0]
        cy, cx = N // 2, N // 2
        yy, xx = np.ogrid[:N, :N]
        r = np.hypot(xx - cx, yy - cy).astype(int)
        r_flat, p_flat = r.ravel(), power.ravel()
        k_max = min(cx, cy)
        bins  = np.zeros(k_max)
        cnts  = np.zeros(k_max)
        for ri, pi in zip(r_flat, p_flat):
            if ri < k_max:
                bins[ri] += pi
                cnts[ri] += 1
        cnts[cnts == 0] = np.nan
        return bins / cnts

    psd_r = azimuthal_avg(np.abs(fftshift(fft2(ref)))**2)
    psd_t = azimuthal_avg(np.abs(fftshift(fft2(test)))**2)
    N     = ref.shape[0]
    k_bins = np.arange(len(psd_r)) / N      # cycles / pixel
    ratio  = np.where(psd_r > 0, psd_t / psd_r, np.nan)
    return k_bins, ratio, psd_r, psd_t


def compute_all_metrics(vx_4k, vy_4k, vx_2k_up, vy_2k_up,
                        bz_4k=None, mag_threshold_G=50, dilation_px=50):
    """
    Master metric function.

    Parameters
    ----------
    vx_4k, vy_4k       : ground-truth velocity components (4K grid)
    vx_2k_up, vy_2k_up : 2K velocities upsampled to 4K grid
    bz_4k              : magnetogram at 4K; if None, AR metrics are skipped
    mag_threshold_G     : AR core |Bz| threshold (Gauss)
    dilation_px         : inflow-zone dilation radius (pixels)

    Returns
    -------
    dict with all scalar metrics and arrays needed for plotting
    """
    speed_4k  = np.hypot(vx_4k,    vy_4k)
    speed_2k  = np.hypot(vx_2k_up, vy_2k_up)
    div_4k    = divergence(vx_4k,    vy_4k)
    div_2k    = divergence(vx_2k_up, vy_2k_up)

    m = {}

    # ── global metrics (full patch) ───────────────────────────────────────
    m["rmse_vx"]      = rmse(vx_4k,    vx_2k_up)
    m["rmse_vy"]      = rmse(vy_4k,    vy_2k_up)
    m["rmse_speed"]   = rmse(speed_4k, speed_2k)
    m["rmse_div"]     = rmse(div_4k,   div_2k)
    m["bias_vx"]      = bias(vx_4k,    vx_2k_up)
    m["bias_vy"]      = bias(vy_4k,    vy_2k_up)
    m["bias_speed"]   = bias(speed_4k, speed_2k)
    m["bias_div"]     = bias(div_4k,   div_2k)
    m["r_vx"]         = spatial_correlation(vx_4k,    vx_2k_up)
    m["r_vy"]         = spatial_correlation(vy_4k,    vy_2k_up)
    m["r_speed"]      = spatial_correlation(speed_4k, speed_2k)
    m["r_div"]        = spatial_correlation(div_4k,   div_2k)
    m["vector_skill"] = vector_skill_score(vx_4k, vy_4k, vx_2k_up, vy_2k_up)

    # ── AR-localised metrics ──────────────────────────────────────────────
    if bz_4k is not None:
        amask = ar_mask(bz_4k, mag_threshold_G, dilation_px)
        m["ar_rmse_vx"]    = rmse(vx_4k,    vx_2k_up,  amask)
        m["ar_rmse_speed"] = rmse(speed_4k, speed_2k,  amask)
        m["ar_rmse_div"]   = rmse(div_4k,   div_2k,    amask)
        m["ar_r_div"]      = spatial_correlation(div_4k, div_2k, amask)
        m["ar_bias_div"]   = bias(div_4k, div_2k, amask)
        m["ar_mask"]       = amask
    else:
        m["ar_mask"] = None

    # ── arrays for plotting ───────────────────────────────────────────────
    m["speed_4k"]  = speed_4k
    m["speed_2k"]  = speed_2k
    m["div_4k"]    = div_4k
    m["div_2k"]    = div_2k
    m["res_vx"]    = vx_2k_up - vx_4k
    m["res_vy"]    = vy_2k_up - vy_4k
    m["res_speed"] = speed_2k - speed_4k
    m["res_div"]   = div_2k   - div_4k

    # PSD on speed maps
    m["psd_k"], m["psd_ratio"], m["psd_4k"], m["psd_2k"] = psd_ratio(speed_4k, speed_2k)

    return m


# ══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _imshow(ax, data, cmap, title, unit="", sym=False, mask=None):
    d = np.where(mask, data, np.nan) if mask is not None else data
    if sym:
        vmax = np.nanpercentile(np.abs(d), 99)
        vmin = -vmax
    else:
        vmin, vmax = np.nanpercentile(d, [1, 99])
    im = ax.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(unit, fontsize=8)
    cb.ax.tick_params(labelsize=7, direction="in")
    cb.outline.set_linewidth(0.6)
    return im


def plot_velocity_maps(vx_4k, vy_4k, vx_2k_up, vy_2k_up, metrics, save=None):
    """Side-by-side speed maps + residual for Vx, Vy, speed."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 10),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.15})
    fig.suptitle("Velocity Field Comparison  |  HMI 4K (GT)  ·  PMI 2K  ·  Residual",
                 fontsize=11, y=1.01)

    rows = [
        (vx_4k,              vx_2k_up,             metrics["res_vx"],    "Vx",    "km/s"),
        (vy_4k,              vy_2k_up,             metrics["res_vy"],    "Vy",    "km/s"),
        (metrics["speed_4k"], metrics["speed_2k"], metrics["res_speed"], "Speed", "km/s"),
    ]
    cmaps = ["RdBu_r", "RdBu_r", "plasma"]

    for i, (ref, test, res, label, unit) in enumerate(rows):
        _imshow(axes[i, 0], ref,  cmaps[i], f"{label}  4K (GT)",  unit, sym=(i<2))
        _imshow(axes[i, 1], test, cmaps[i], f"{label}  2K",       unit, sym=(i<2))
        _imshow(axes[i, 2], res,  "RdBu_r", f"{label}  Residual (2K−4K)", unit, sym=True)

    plt.tight_layout()
    if save: plt.savefig(save, bbox_inches="tight")
    plt.show()


def plot_divergence_maps(metrics, save=None):
    """Divergence (inflow proxy) maps + residual + AR mask overlay."""
    amask = metrics["ar_mask"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                             gridspec_kw={"wspace": 0.18})
    fig.suptitle("Divergence  $\\nabla \\cdot \\mathbf{v}$  |  Negative = Inflow", fontsize=11)

    titles = ["4K  (GT)", "2K  (upsampled)", "Residual  (2K − 4K)"]
    arrays = [metrics["div_4k"], metrics["div_2k"], metrics["res_div"]]

    for ax, arr, title in zip(axes, arrays, titles):
        _imshow(ax, arr, "RdBu_r", title, unit="s⁻¹", sym=True)
        if amask is not None:
            contour_data = amask.astype(float)
            ax.contour(contour_data, levels=[0.5], colors="#333333",
                       linewidths=0.8, linestyles="--", alpha=0.9)

    if amask is not None:
        axes[0].text(0.02, 0.96, "- - AR inflow zone", transform=axes[0].transAxes,
                     fontsize=7, color="#333333", va="top")

    plt.tight_layout()
    if save: plt.savefig(save, bbox_inches="tight")
    plt.show()


def plot_psd(metrics, save=None):
    """Power spectral density of 4K vs 2K speed maps + their ratio."""
    k     = metrics["psd_k"]
    psd4  = metrics["psd_4k"]
    psd2  = metrics["psd_2k"]
    ratio = metrics["psd_ratio"]

    # clip Nyquist artefact
    k_cut = 0.45
    sel   = k < k_cut

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   gridspec_kw={"wspace": 0.3})
    fig.suptitle("Power Spectral Density  (azimuthal average)", fontsize=11)

    ax1.semilogy(k[sel], psd4[sel], color=C4K, lw=1.5, label="HMI 4K (GT)")
    ax1.semilogy(k[sel], psd2[sel], color=C2K, lw=1.5, ls="--", label="PMI 2K")
    ax1.set_xlabel("Spatial frequency  [cycles / pixel]")
    ax1.set_ylabel("PSD  [arb.]")
    ax1.set_title("PSD  4K vs 2K")
    ax1.legend(fontsize=8)

    ax2.axhline(1.0, color=CGREY, lw=0.8, ls="--", label="Ratio = 1")
    ax2.plot(k[sel], ratio[sel], color=CGOOD, lw=1.5, label="PSD ratio  PMI / HMI")
    ax2.set_xlabel("Spatial frequency  [cycles / pixel]")
    ax2.set_ylabel("PSD ratio")
    ax2.set_title("PSD ratio  (2K / 4K)  — where it drops below 1\nmarks the effective resolution loss")
    ax2.set_ylim(0, 2)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    if save: plt.savefig(save, bbox_inches="tight")
    plt.show()


def plot_scatter(vx_4k, vy_4k, vx_2k_up, vy_2k_up, metrics, save=None):
    """Scatter plots: 4K vs 2K for Vx, Vy, speed, divergence."""
    pairs = [
        (_flat(vx_4k),              _flat(vx_2k_up),             "Vx",      "km/s", metrics["r_vx"]),
        (_flat(vy_4k),              _flat(vy_2k_up),             "Vy",      "km/s", metrics["r_vy"]),
        (_flat(metrics["speed_4k"]),_flat(metrics["speed_2k"]),  "Speed",   "km/s", metrics["r_speed"]),
        (_flat(metrics["div_4k"]),  _flat(metrics["div_2k"]),    "Div ∇·V", "s⁻¹",  metrics["r_div"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4),
                             gridspec_kw={"wspace": 0.35})
    fig.suptitle("Scatter: HMI 4K (GT) vs PMI 2K", fontsize=11)

    subsample = 5000   # keep plot fast; increase if you want denser

    for ax, (ref, tst, label, unit, r) in zip(axes, pairs):
        idx = np.random.choice(len(ref), min(subsample, len(ref)), replace=False)
        ax.scatter(ref[idx], tst[idx], s=1.5, alpha=0.3, color=C4K, rasterized=True)
        lo = min(ref[idx].min(), tst[idx].min())
        hi = max(ref[idx].max(), tst[idx].max())
        ax.plot([lo, hi], [lo, hi], color=C2K, lw=1, ls="--", label="1:1")
        ax.set_xlabel(f"4K  [{unit}]", fontsize=8)
        ax.set_ylabel(f"2K  [{unit}]", fontsize=8)
        ax.set_title(f"{label}\nr = {r:.3f}", fontsize=9)
        ax.legend(fontsize=7)

    plt.tight_layout()
    if save: plt.savefig(save, bbox_inches="tight")
    plt.show()


def plot_metric_summary(metrics, save=None):
    """
    Bar chart summary of RMSE, bias, and Pearson r for each quantity,
    globally and (if available) inside the AR inflow zone.
    """
    labels  = ["Vx", "Vy", "Speed", "Divergence"]
    rmses   = [metrics["rmse_vx"], metrics["rmse_vy"],
               metrics["rmse_speed"], metrics["rmse_div"]]
    biases  = [metrics["bias_vx"], metrics["bias_vy"],
               metrics["bias_speed"], metrics["bias_div"]]
    rs      = [metrics["r_vx"], metrics["r_vy"],
               metrics["r_speed"], metrics["r_div"]]

    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                             gridspec_kw={"wspace": 0.35})
    fig.suptitle("Scalar Metric Summary", fontsize=11)

    colors = [C4K, CGOOD, C2K, "#9467bd"]

    for ax, vals, title, unit in zip(
        axes,
        [rmses,  biases,  rs],
        ["RMSE", "Bias (2K − 4K)", "Pearson r"],
        ["(mixed units)", "(mixed units)", ""],
    ):
        bars = ax.bar(x, vals, color=colors, width=0.55, zorder=3)
        ax.axhline(0, color=CGREY, lw=0.7, ls="--")
        if title == "Pearson r":
            ax.set_ylim(-1.1, 1.1)
            ax.axhline(1, color=CGOOD, lw=0.6, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"{title}  {unit}", fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3g}", ha="center", va="bottom", fontsize=7.5)

    # AR inset if available
    if metrics.get("ar_rmse_div") is not None:
        txt = (
            f"AR inflow zone\n"
            f"RMSE div : {metrics['ar_rmse_div']:.3g} s⁻¹\n"
            f"Bias div : {metrics['ar_bias_div']:.3g} s⁻¹\n"
            f"r   div  : {metrics['ar_r_div']:.3f}\n"
            f"Vector skill : {metrics['vector_skill']:.3f}"
        )
        fig.text(0.98, 0.5, txt, va="center", ha="right", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#aaaaaa"))

    plt.tight_layout()
    if save: plt.savefig(save, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MASTER CALL
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(vx_4k, vy_4k, vx_2k_up, vy_2k_up, metrics, save_prefix=None):
    """
    Run all four plot functions in one go.

    Parameters
    ----------
    vx_4k, vy_4k       : 4K ground-truth velocity components
    vx_2k_up, vy_2k_up : 2K velocities already upsampled to 4K grid
    metrics             : dict returned by compute_all_metrics()
    save_prefix         : if given, saves each figure as <prefix>_<name>.png
    """
    def _save(name):
        return f"{save_prefix}_{name}.png" if save_prefix else None

    plot_velocity_maps(vx_4k, vy_4k, vx_2k_up, vy_2k_up, metrics,
                       save=_save("velocity_maps"))
    plot_divergence_maps(metrics, save=_save("divergence_maps"))
    plot_scatter(vx_4k, vy_4k, vx_2k_up, vy_2k_up, metrics,
                 save=_save("scatter"))
    plot_psd(metrics, save=_save("psd"))
    plot_metric_summary(metrics, save=_save("summary"))


# %% Load the data
f1 = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/ar_inflow/data/data_cleaned/smooth_data_2k_15.npz')
vx_2k_up = f1['smooth_zx_corrected']
vy_2k_up = -f1['smooth_zy_corrected']
hdiv_2k = f1['smooth_hdiv']
latitude = f1['latitude']
longitude = f1['longitude']

f2 = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/ar_inflow/data/data_cleaned/smooth_data_4k_15.npz')
vx_4k = f2['smooth_zx_corrected']
vy_4k = -f2['smooth_zy_corrected']
hdiv_4k = f2['smooth_hdiv']

def normalise_vector(vx, vy):
    scale = np.std(np.hypot(vx, vy))
    mean_vx, mean_vy = np.mean(vx), np.mean(vy)
    return (vx - mean_vx) / scale, (vy - mean_vy) / scale

vx_4k, vy_4k = normalise_vector(vx_4k, vy_4k)
vx_2k_up, vy_2k_up = normalise_vector(vx_2k_up, vy_2k_up)

# 2. compute all metrics
metrics = compute_all_metrics(vx_4k, vy_4k, vx_2k_up, vy_2k_up, bz_4k=None)

# 3. plot everything
plot_all(vx_4k, vy_4k, vx_2k_up, vy_2k_up, metrics)

# %%
from scipy.ndimage import gaussian_filter

# bandpass: isolate large-scale vs small-scale power
low  = gaussian_filter(metrics["speed_2k"], sigma=20)
high = metrics["speed_2k"] - low

low_4k  = gaussian_filter(metrics["speed_4k"], sigma=20)
high_4k = metrics["speed_4k"] - low_4k

print("Large-scale power 2K:", np.var(low))
print("Large-scale power 4K:", np.var(low_4k))
print("Small-scale power 2K:", np.var(high))
print("Small-scale power 4K:", np.var(high_4k))
# %%
print("Mean speed 4K:", np.mean(metrics["speed_4k"]))
print("Mean speed 2K:", np.mean(metrics["speed_2k"]))
print("Ratio:", np.mean(metrics["speed_2k"]) / np.mean(metrics["speed_4k"]))

# %%
div_threshold = np.percentile(metrics["div_4k"], 20)
inflow_zone   = metrics["div_4k"] < div_threshold


r    = spatial_correlation(metrics["div_4k"], metrics["div_2k"], mask=inflow_zone)
rmse_ = rmse(metrics["div_4k"], metrics["div_2k"], mask=inflow_zone)
bias_ = bias(metrics["div_4k"], metrics["div_2k"], mask=inflow_zone)

print(f"Inside inflow zone (bottom 20% divergence):")
print(f"  Pearson r : {r:.3f}")
print(f"  RMSE      : {rmse_:.3g} s⁻¹")
print(f"  Bias      : {bias_:.3g} s⁻¹")
# %%
from scipy.ndimage import gaussian_filter, center_of_mass

# smooth before finding centre
div_smooth = gaussian_filter(metrics["div_4k"], sigma=3)
inflow_smooth = np.where(div_smooth < 0, -div_smooth, 0)
com = center_of_mass(inflow_smooth)

# clip radius to stay within patch bounds from the centre
cy, cx = com
max_r = int(min(cy, metrics["div_4k"].shape[0]-cy,
                cx, metrics["div_4k"].shape[1]-cx)) - 2

p4k = radial_profile(metrics["div_4k"], com)
p2k = radial_profile(metrics["div_2k"], com)

# also smooth the profiles themselves to reduce per-annulus noise
from scipy.ndimage import uniform_filter1d
p4k_s = uniform_filter1d(p4k[:max_r], size=3)
p2k_s = uniform_filter1d(p2k[:max_r], size=3)

r_ax = np.arange(max_r)  # multiply by pixel_scale_Mm to get physical units

plt.plot(r_ax, p4k_s, color=C4K, label="HMI 4K")
plt.plot(r_ax, p2k_s, color=C2K, ls="--", label="PMI 2K")
plt.axhline(0, color=CGREY, lw=0.7)
plt.xlabel("Radius from inflow centre [pixels]")
plt.ylabel("Mean divergence [s$^{-1}$]")
plt.legend()
plt.show()
# %%
# compare r values inside vs outside the inflow trough radius
trough_px = 10  # approximate from your plot
y, x = np.indices(metrics["div_4k"].shape)
cy, cx = com
r_map = np.hypot(x - cx, y - cy)

inner_mask = r_map < trough_px
outer_mask = (r_map >= trough_px) & (r_map < max_r)

r_inner = spatial_correlation(metrics["div_4k"], metrics["div_2k"], mask=inner_mask)
r_outer = spatial_correlation(metrics["div_4k"], metrics["div_2k"], mask=outer_mask)

print(f"r inside  trough (r < {trough_px} px): {r_inner:.3f}")
print(f"r outside trough (r >= {trough_px} px): {r_outer:.3f}")
# %%
print(latitude, longitude)
# %%
pixel_scale_deg = 0.5
R_sun_Mm = 695.7
pixel_scale_Mm = np.deg2rad(pixel_scale_deg) * R_sun_Mm  # ~6.07 Mm/pixel

max_r = int(min(cy, metrics["div_4k"].shape[0]-cy,
                cx, metrics["div_4k"].shape[1]-cx)) - 2

r_ax_Mm = np.arange(max_r) * pixel_scale_Mm

plt.plot(r_ax_Mm, p4k_s[:max_r], color=C4K, label="HMI 4K")
plt.plot(r_ax_Mm, p2k_s[:max_r], color=C2K, ls="--", label="PMI 2K")
plt.axhline(0, color=CGREY, lw=0.7)
plt.xlabel("Radius from inflow centre [Mm]")
plt.ylabel("Mean divergence [s$^{-1}$]")
plt.legend()
plt.show()
# %%
