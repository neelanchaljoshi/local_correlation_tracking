# %%
"""
Differential Rotation Comparison — HMI 4K vs PMI 2K
=====================================================
Cell structure:
  Cell 1  — imports and publication style
  Cell 2  — paths, keys, analysis config
  Cell 3  — file I/O and data accumulation functions
  Cell 4  — differential rotation fit functions
  Cell 5  — convergence test function
  Cell 6  — low-level panel drawing helpers
  Cell 7  — DATA LOADING  ← run once, stores: data
  Cell 8  — FIT PROFILES  ← run once, stores: fits
  Cell 9  — CONVERGENCE   ← run once (slow), stores: conv
  Cell 10 — PLOT: systematics maps
  Cell 11 — PLOT: systematics profiles
  Cell 12 — PLOT: differential rotation profiles
  Cell 13 — PLOT: convergence
  Cell 14 — print summary table

Tweak any plot cell and rerun it without touching the data cells.
"""


# %%  ── CELL 1: imports and publication style ─────────────────────────────────

import os
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor":  "white",   "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333", "axes.labelcolor":   "#222222",
    "axes.titlecolor":   "#111111", "axes.linewidth":    1.0,
    "axes.grid":         True,      "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.6,       "grid.linestyle":    "--",
    "xtick.color":       "#333333", "ytick.color":       "#333333",
    "xtick.direction":   "out",      "ytick.direction":   "out",
    "xtick.major.size":  5.0,       "ytick.major.size":  5.0,
    "xtick.labelsize":   12,        "ytick.labelsize":   12,
    "text.color":        "#222222", "font.family":       "sans-serif",
    "font.size":         12,        "axes.labelsize":    13,
    "axes.titlesize":    14,        "axes.titlepad":     8,
    "legend.fontsize":   11,        "legend.framealpha": 0.92,
    "legend.edgecolor":  "#bbbbbb",
    "figure.dpi":        150,       "savefig.dpi":       300,
    "savefig.facecolor": "white",   "savefig.bbox":      "tight",
    "pdf.fonttype":      42,        "ps.fonttype":       42,
})


# %%  ── CELL 2: paths, keys, analysis config ───────────────────────────────────

BASE = ('/data/seismo/joshin/pipeline-test/local_correlation_tracking/'
        'pmi/diff_rot/data/')

DIRS = {
    "gran_4k": BASE + "data_4k_2017/",
    "gran_2k": BASE + "data_2k_2017/",
    "mag_4k":  BASE + "data_for_im/data_mag_4k_2017/",
    "mag_2k":  BASE + "data_for_im/data_mag_2k_2017/",
}

FILE_PATTERN = "*.hdf5"
VPHI_KEY     = "uphi"       # HDF5 key for vφ — shape (4, n_lat, n_lon)
VTHETA_KEY   = "utheta"     # HDF5 key for vθ — same shape
LAT_KEY      = "latitude"
LON_KEY      = "longitude"

# ── clipping ─────────────────────────────────────────────────────────────────
LAT_CLIP = 60.0   # load only latitudes within ±LAT_CLIP
LON_CLIP = 60.0   # load only longitudes within ±LON_CLIP

# ── fit range ─────────────────────────────────────────────────────────────────
LAT_FIT_MIN = -60.0
LAT_FIT_MAX =  60.0

# ── central meridian selection ────────────────────────────────────────────────
# "central" → single bin closest to 0°
# "strip"   → all bins within ±LON_STRIP_WIDTH
LON_MODE        = "central"
LON_STRIP_WIDTH = 10.0   # degrees, only used when LON_MODE = "strip"

# ── smoothing / bootstrap / convergence ───────────────────────────────────────
SMOOTH_LAT           = 1
N_BOOTSTRAP          = 300
N_CONVERGENCE_POINTS = 50

# ── output prefix (set None to display only) ──────────────────────────────────
SAVE_PREFIX = "diffrot"


# %%  ── CELL 3: file I/O and data accumulation functions ─────────────────────

def discover_files(dataset_key):
    pattern = os.path.join(DIRS[dataset_key], FILE_PATTERN)
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files for '{dataset_key}' matching {pattern}")
    return files


def read_axes(filepath):
    with h5py.File(filepath, "r") as f:
        lat = f[LAT_KEY][:].astype(float)
        lon = f[LON_KEY][:].astype(float)
    return lat, lon


def lon_indices(lon):
    if LON_MODE == "central":
        return np.array([int(np.argmin(np.abs(lon)))])
    elif LON_MODE == "strip":
        idx = np.where(np.abs(lon) <= LON_STRIP_WIDTH)[0]
        if not len(idx):
            raise ValueError(
                f"No lon columns within ±{LON_STRIP_WIDTH}° of meridian "
                f"(range [{lon.min():.1f}, {lon.max():.1f}])")
        return idx
    raise ValueError("LON_MODE must be 'central' or 'strip'")


def lon_label():
    if LON_MODE == "central":
        return "central meridian"
    return f"$|\\ell| \\leq {LON_STRIP_WIDTH:.0f}°$"


def accumulate_dataset(dataset_key, n_days=None, verbose=False):
    """
    Load daily HDF5 files one at a time.
    Per day: median over 4 frames.  Final: median across days.
    Returns lat, lon, vphi_profile (1D), vphi_map (2D), vtheta_map (2D), n_frames.
    """
    files = discover_files(dataset_key)
    if n_days is not None:
        files = files[:n_days]

    lat_full, lon_full = read_axes(files[0])
    lat_mask = np.abs(lat_full) <= LAT_CLIP
    lon_mask = np.abs(lon_full) <= LON_CLIP
    lat      = lat_full[lat_mask]
    lon      = lon_full[lon_mask]
    lidx     = lon_indices(lon)

    daily_profile = []
    daily_vphi    = []
    daily_vtheta  = []
    n_frames      = 0

    for i, fpath in enumerate(files):
        if verbose and i % 50 == 0:
            print(f"    {dataset_key}  day {i+1}/{len(files)}")
        try:
            with h5py.File(fpath, "r") as f:
                vp = f[VPHI_KEY][:].astype(float)[:, lat_mask, :][:, :, lon_mask]
                vt = f[VTHETA_KEY][:].astype(float)[:, lat_mask, :][:, :, lon_mask]
            daily_profile.append(np.nanmedian(vp[:, :, lidx], axis=(0, 2)))
            daily_vphi.append(np.nanmedian(vp, axis=0))
            daily_vtheta.append(np.nanmedian(vt, axis=0))
            n_frames += vp.shape[0]
        except Exception as e:
            if verbose:
                print(f"    WARNING: skipping {fpath}: {e}")

    if not daily_profile:
        raise RuntimeError(f"No valid frames for '{dataset_key}'")

    return (lat, lon,
            np.nanmedian(np.stack(daily_profile), axis=0),
            np.nanmedian(np.stack(daily_vphi),    axis=0),
            np.nanmedian(np.stack(daily_vtheta),  axis=0),
            n_frames)


def load_all(n_days=None, verbose=True):
    """
    Load all four datasets.
    Returns dict keyed by gran_4k / gran_2k / mag_4k / mag_2k, each with:
        lat, lon, profile, vphi_map, vtheta_map, n_frames
    """
    out = {}
    for key in ["gran_4k", "gran_2k", "mag_4k", "mag_2k"]:
        if verbose:
            print(f"  {key} ...")
        lat, lon, profile, vphi_map, vtheta_map, nf = \
            accumulate_dataset(key, n_days=n_days, verbose=verbose)
        out[key] = dict(lat=lat, lon=lon, profile=profile,
                        vphi_map=vphi_map, vtheta_map=vtheta_map,
                        n_frames=nf)
        if verbose:
            print(f"    lat [{lat[0]:.0f}, {lat[-1]:.0f}]  "
                  f"lon [{lon[0]:.0f}, {lon[-1]:.0f}]  n_frames={nf}")
    return out


# %%  ── CELL 4: differential rotation fit functions ───────────────────────────

def diffrot_model(lat_deg, A, B, C):
    """Standard model: A + B sin²θ + C sin⁴θ"""
    s2 = np.sin(np.deg2rad(lat_deg)) ** 2
    return A + B * s2 + C * s2 ** 2


def fit_one(lat, profile):
    """Smooth and fit.  Returns a dict with all fit products."""
    mask        = (lat >= LAT_FIT_MIN) & (lat <= LAT_FIT_MAX)
    lat_sel     = lat[mask]
    prof_raw    = profile[mask]
    prof_smooth = uniform_filter1d(prof_raw, size=SMOOTH_LAT, mode="nearest")
    valid       = np.isfinite(prof_smooth)
    popt, _     = curve_fit(diffrot_model,
                            lat_sel[valid], prof_smooth[valid],
                            p0=(2000., -500., -300.), maxfev=10000)
    lat_fine  = np.linspace(lat_sel.min(), lat_sel.max(), 400)
    fit_curve = diffrot_model(lat_fine, *popt)
    return dict(lat_sel=lat_sel, prof_raw=prof_raw, smooth=prof_smooth,
                popt=popt, lat_fine=lat_fine, fit_curve=fit_curve)


def bootstrap_one(lat, profile):
    """Bootstrap std on (A, B, C) by resampling latitude bins."""
    mask   = (lat >= LAT_FIT_MIN) & (lat <= LAT_FIT_MAX)
    lat_s  = lat[mask]
    prof_s = uniform_filter1d(profile[mask], size=SMOOTH_LAT, mode="nearest")
    valid  = np.where(np.isfinite(prof_s))[0]
    coeffs = np.full((N_BOOTSTRAP, 3), np.nan)
    for i in range(N_BOOTSTRAP):
        idx = np.sort(np.random.choice(valid, size=len(valid), replace=True))
        try:
            popt, _ = curve_fit(diffrot_model, lat_s[idx], prof_s[idx],
                                p0=(2000., -500., -300.), maxfev=5000)
            coeffs[i] = popt
        except RuntimeError:
            pass
    return np.nanstd(coeffs, axis=0)


def compute_fits(data):
    """Run fit_one + bootstrap_one for all four datasets."""
    fits = {}
    for key, d in data.items():
        f     = fit_one(d["lat"], d["profile"])
        bserr = bootstrap_one(d["lat"], d["profile"])
        fits[key] = {**f, "bserr": bserr, "n_frames": d["n_frames"]}
        A, B, C = f["popt"]
        eA, eB, eC = bserr
        print(f"  {key:10s}  A={A:.1f}±{eA:.1f}  "
              f"B={B:.1f}±{eB:.1f}  C={C:.1f}±{eC:.1f}")
    return fits


# %%  ── CELL 5: convergence test function ─────────────────────────────────────

def compute_convergence(verbose=False):
    """
    RMS(2K−4K) and Pearson r as a function of days averaged (log-spaced).
    Returns (days, rms_gran, rms_mag, pearson_gran, pearson_mag).
    """
    n_max     = len(discover_files("gran_4k"))
    day_steps = np.unique(np.concatenate([
        [1],
        np.round(np.logspace(0, np.log10(n_max),
                             N_CONVERGENCE_POINTS)).astype(int),
        [n_max],
    ]))
    day_steps = day_steps[(day_steps >= 1) & (day_steps <= n_max)]

    rms_gran = []; rms_mag = []
    pr_gran  = []; pr_mag  = []

    print(f"  {len(day_steps)} day counts  (1 → {n_max}) ...")

    for nd in day_steps:
        nd   = int(nd)
        d    = load_all(n_days=nd, verbose=False)
        for key4, key2, rms_lst, pr_lst in [
            ("gran_4k", "gran_2k", rms_gran, pr_gran),
            ("mag_4k",  "mag_2k",  rms_mag,  pr_mag),
        ]:
            mask = ((d[key4]["lat"] >= LAT_FIT_MIN) &
                    (d[key4]["lat"] <= LAT_FIT_MAX))
            p4 = uniform_filter1d(d[key4]["profile"][mask],
                                  size=SMOOTH_LAT, mode="nearest")
            p2 = uniform_filter1d(d[key2]["profile"][mask],
                                  size=SMOOTH_LAT, mode="nearest")
            rms_lst.append(float(np.sqrt(np.nanmean((p2 - p4) ** 2))))
            v = np.isfinite(p4) & np.isfinite(p2)
            pr_lst.append(float(pearsonr(p4[v], p2[v])[0])
                          if v.sum() > 2 else np.nan)
        if verbose:
            print(f"    {nd:4d} d  RMS_gran={rms_gran[-1]:.1f}  "
                  f"r_gran={pr_gran[-1]:.4f}")

    return (day_steps,
            np.array(rms_gran), np.array(rms_mag),
            np.array(pr_gran),  np.array(pr_mag))


# %%  ── CELL 6: low-level panel drawing helpers ───────────────────────────────
# Each helper draws into one Axes.  All parameters are explicit arguments.
# Changing one panel call cannot affect any other panel.

def _draw_map_panel(ax, arr, lat, lon, vmax, cmap,
                    title, xlabel, ylabel,
                    show_meridian, meridian_color, meridian_lw, meridian_ls,
                    n_xticks, n_yticks,
                    spine_lw, title_fs, label_fs, tick_fs):
    """Draw one pcolormesh map panel. Returns the QuadMesh for colorbar use."""

    # Create meshgrid (important for correct mapping)
    Lon, Lat = np.meshgrid(lon, lat)
    ax.set_aspect('equal', adjustable='box')

    im = ax.pcolormesh(
        Lon, Lat, arr,
        cmap=cmap,
        vmin=-vmax, vmax=vmax,
        shading="bilinear"
    )

    if show_meridian:
        ax.axvline(0, color=meridian_color, lw=meridian_lw,
                   ls=meridian_ls, alpha=0.7)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(n_xticks, integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(n_yticks, integer=True))
    ax.tick_params(labelsize=tick_fs)

    ax.set_title(title, fontsize=title_fs, pad=8)
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)

    for sp in ax.spines.values():
        sp.set_linewidth(spine_lw)

    return im


def _draw_profile_panel(ax, lat, prof4, prof2,
                        color4, color2, lw4, lw2, ls4, ls2,
                        label4, label2, ylim,
                        title, xlabel, ylabel,
                        zero_color, zero_lw,
                        legend_loc, legend_fs,
                        title_fs, label_fs, tick_fs, spine_lw):
    """Draw one 1D latitude profile panel with two curves."""
    ax.plot(lat, prof4, '-o', color=color4, lw=lw4, ls=ls4, label=label4)
    ax.plot(lat, prof2, '-o', color=color2, lw=lw2, ls=ls2, label=label2)
    ax.axhline(0, color=zero_color, lw=zero_lw, ls="--")
    ax.axvline(0, color=zero_color, lw=zero_lw * 0.7, ls="--", alpha=0.4)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_title(title, fontsize=title_fs, pad=8)
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.legend(loc=legend_loc, fontsize=legend_fs)
    for sp in ax.spines.values():
        sp.set_linewidth(spine_lw)


def _draw_diffrot_panel(ax, fit4, fit2,
                        color4, color2,
                        data_lw, ls4, ls2,
                        fit_lw, fit_ls, fit_alpha,
                        show_scatter, scatter_s, scatter_alpha,
                        label4, label2,
                        zero_color, zero_lw,
                        title, xlabel, ylabel,
                        legend_loc, legend_fs,
                        title_fs, label_fs, tick_fs, spine_lw):
    """Draw one differential rotation panel (smoothed data + fitted curve)."""
    if show_scatter:
        ax.scatter(fit4["lat_sel"], fit4["prof_raw"],
                   s=scatter_s, color=color4, alpha=scatter_alpha, zorder=2)
        ax.scatter(fit2["lat_sel"], fit2["prof_raw"],
                   s=scatter_s, color=color2, alpha=scatter_alpha, zorder=2)
    ax.plot(fit4["lat_sel"], fit4["smooth"],
            color=color4, lw=data_lw, ls=ls4, label=label4)
    ax.plot(fit2["lat_sel"], fit2["smooth"],
            color=color2, lw=data_lw, ls=ls2, label=label2)
    ax.plot(fit4["lat_fine"], fit4["fit_curve"],
            color=color4, lw=fit_lw, ls=fit_ls, alpha=fit_alpha,
            label=f"4K fit  A={fit4['popt'][0]:.0f} m/s")
    ax.plot(fit2["lat_fine"], fit2["fit_curve"],
            color=color2, lw=fit_lw, ls=fit_ls, alpha=fit_alpha,
            label=f"2K fit  A={fit2['popt'][0]:.0f} m/s")
    ax.axhline(0, color=zero_color, lw=zero_lw)
    ax.axvline(0, color=zero_color, lw=zero_lw * 0.7, ls="--", alpha=0.4)
    ax.set_title(title, fontsize=title_fs, pad=8)
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.legend(loc=legend_loc, fontsize=legend_fs)
    for sp in ax.spines.values():
        sp.set_linewidth(spine_lw)


def _add_colorbar(fig, im, ax, label, x_offset, width, label_fs, tick_fs, lw):
    """Place a colourbar flush with the right edge of ax."""
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + x_offset, pos.y0, width, pos.height])
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=label_fs)
    cb.ax.tick_params(direction="in", labelsize=tick_fs)
    cb.outline.set_linewidth(lw)
    return cb


# %%  ── CELL 7: DATA LOADING ───────────────────────────────────────────────────
# Run this cell once.  `data` is reused by all plot and fit cells.

print("Loading data ...")
data = load_all(verbose=True)
print("Done.")


# %%  ── CELL 8: FIT PROFILES ──────────────────────────────────────────────────
# Run after Cell 7.  `fits` is reused by plot cells 12 and 14.

print("Fitting differential rotation profiles ...")
fits = compute_fits(data)
print("Done.")


# %%  ── CELL 9: CONVERGENCE TEST (slow) ───────────────────────────────────────
# Run after Cell 7.  Stores `conv` for plot cell 13.
# This reruns load_all() ~30 times — expected to take several minutes.

print("Running convergence test ...")
conv = compute_convergence(verbose=True)
print("Done.")


# %%  ── CELL 10: PLOT — systematics maps ──────────────────────────────────────
# Requires: data (Cell 7)
# Tweak SETTINGS and rerun this cell only.

def plot_systematics_maps(data, save=None):
    # ── SETTINGS ─────────────────────────────────────────────────────────────
    S = {
        "figsize":    (10, 8),
        "hspace":     0.20,
        "wspace":     0.26,
        "suptitle":   "Time-averaged flow maps  (full-year, all longitudes)",
        "suptitle_y": 1.01,
        "suptitle_fs": 14,
        "cmap":       "bwr",
        "spine_lw":   0.8,
        "title_fs":   13,
        "label_fs":   11,
        "tick_fs":    10,
        "n_xticks":   5,
        "n_yticks":   5,
        "show_meridian":   True,
        "meridian_color":  "black",
        "meridian_lw":     0.9,
        "meridian_ls":     "--",
        "cb_x_offset": 0.012,
        "cb_width":    0.014,
        "cb_label_fs": 10,
        "cb_tick_fs":  9,
        "cb_lw":       0.7,
        # [vphi_vmax, vtheta_vmax] per dataset
        "vmax": {
            "gran_4k": [1000, 1000],
            "gran_2k": [1000, 1000],
            "mag_4k":  [ 100, 100],
            "mag_2k":  [ 100, 100],
        },
        "col_titles":   [r"$v_\phi$  [m s$^{-1}$]",
                         r"$v_\theta$  [m s$^{-1}$]"],
        "row_labels":   ["Gran 4K", "Gran 2K", "Mag 4K", "Mag 2K"],
        "dataset_keys": ["gran_4k", "gran_2k", "mag_4k", "mag_2k"],
    }
    # ── END SETTINGS ─────────────────────────────────────────────────────────


    fig_phi = plt.figure(figsize=S["figsize"])
    # fig_phi.suptitle("vφ", fontsize=S["suptitle_fs"], y=S["suptitle_y"])

    gs = gridspec.GridSpec(2, 2, figure=fig_phi,
                        hspace=S["hspace"], wspace=S["wspace"])

    # Mapping: (row, col)
    layout = {
        (0, 0): "gran_2k",
        (0, 1): "gran_4k",
        (1, 0): "mag_2k",
        (1, 1): "mag_4k",
    }

    for (i, j), key in layout.items():
        d   = data[key]
        lat = d["lat"]
        lon = d["lon"]

        ax = fig_phi.add_subplot(gs[i, j])

        im = _draw_map_panel(
            ax=ax, arr=d["vphi_map"], lat=lat, lon=lon,
            vmax=S["vmax"][key][0], cmap=S["cmap"],
            title="2K" if i == 0 and j == 0 else "4K" if i == 0 and j == 1 else "",
            xlabel="Longitude [°]",
            ylabel=("Granulation\n\nLatitude [°]" if (i == 0 and j == 0)
                    else "Magnetic\n\nLatitude [°]" if (i == 1 and j == 0)
                    else ""),
            show_meridian=S["show_meridian"],
            meridian_color=S["meridian_color"],
            meridian_lw=S["meridian_lw"],
            meridian_ls=S["meridian_ls"],
            n_xticks=S["n_xticks"], n_yticks=S["n_yticks"],
            spine_lw=S["spine_lw"], title_fs=S["title_fs"],
            label_fs=S["label_fs"], tick_fs=S["tick_fs"],
        )

        # if i != 1:
        #     ax.set_xticklabels([])
        # if j != 0:
        #     ax.set_yticklabels([])

        _add_colorbar(fig_phi, im, ax, label="m s$^{-1}$",
                    x_offset=S["cb_x_offset"], width=S["cb_width"],
                    label_fs=S["cb_label_fs"], tick_fs=S["cb_tick_fs"],
                    lw=S["cb_lw"])

    # if save:
    #     plt.savefig(save, bbox_inches="tight")
    plt.savefig(f"{SAVE_PREFIX}_vphi_maps.pdf", bbox_inches="tight")
    plt.show()

    fig_theta = plt.figure(figsize=S["figsize"])
    # fig_theta.suptitle("vθ", fontsize=S["suptitle_fs"], y=S["suptitle_y"])

    gs = gridspec.GridSpec(2, 2, figure=fig_theta,
                        hspace=S["hspace"], wspace=S["wspace"])

    for (i, j), key in layout.items():
        d   = data[key]
        lat = d["lat"]
        lon = d["lon"]

        ax = fig_theta.add_subplot(gs[i, j])

        im = _draw_map_panel(
            ax=ax, arr=d["vtheta_map"], lat=lat, lon=lon,
            vmax=S["vmax"][key][1], cmap=S["cmap"],
            title="2K" if i == 0 and j == 0 else "4K" if i == 0 and j == 1 else "",
            xlabel="Longitude [°]",
            ylabel=("Granulation\n\nLatitude [°]" if (i == 0 and j == 0)
                    else "Magnetic\n\nLatitude [°]" if (i == 1 and j == 0)
                    else ""),
            show_meridian=S["show_meridian"],
            meridian_color=S["meridian_color"],
            meridian_lw=S["meridian_lw"],
            meridian_ls=S["meridian_ls"],
            n_xticks=S["n_xticks"], n_yticks=S["n_yticks"],
            spine_lw=S["spine_lw"], title_fs=S["title_fs"],
            label_fs=S["label_fs"], tick_fs=S["tick_fs"],
        )

        # if i != 1:
        #     ax.set_xticklabels([])
        # if j != 0:
        #     ax.set_yticklabels([])

        _add_colorbar(fig_theta, im, ax, label="m s$^{-1}$",
                    x_offset=S["cb_x_offset"], width=S["cb_width"],
                    label_fs=S["cb_label_fs"], tick_fs=S["cb_tick_fs"],
                    lw=S["cb_lw"])

    plt.savefig(f"{SAVE_PREFIX}_vtheta_maps.pdf", bbox_inches="tight")
    plt.show()


plot_systematics_maps(data,
                      save=f"{SAVE_PREFIX}_systematics_maps.pdf"
                      if SAVE_PREFIX else None)


# %%  ── CELL 11: PLOT — systematics profiles ──────────────────────────────────
# Requires: data (Cell 7)
# Tweak SETTINGS and rerun this cell only.

def plot_systematics_profiles(data, save=None):
    # ── SETTINGS ─────────────────────────────────────────────────────────────
    S = {
        "figsize":    (16, 13),
        "hspace":     0.12,
        "wspace":     0.15,
        "suptitle":   (r"Longitude-median flow profiles  (full-year)"
                       "\n"
                       r"$v_\phi$ (left)  —  $v_\theta$ (right)"),
        "suptitle_y":  1.02,
        "suptitle_fs": 20,
        "lw":          2.0,
        "ls_4k":       "-",
        "ls_2k":       "--",
        "zero_color":  "#7f7f7f",
        "zero_lw":     0.8,
        "spine_lw":    0.8,
        "title_fs":    20,
        "label_fs":    18,
        "tick_fs":     16,
        "legend_loc":  "lower center",
        "legend_fs":   14,
        # [vphi_ylim, vtheta_ylim] per tracer
        "ylim": {
                    "gran": {
                        "vphi":   (-250, 50),
                        "vtheta": (-1500, 1500),
                    },
                    "mag": {
                        "vphi":   (-200, 50),
                        "vtheta": (-100, 100),
                    },
                },
        "color_gran_4k": "#1f77b4",
        "color_gran_2k": "#6baed6",
        "color_mag_4k":  "#d62728",
        "color_mag_2k":  "#fc8d59",
    }
    # ── END SETTINGS ─────────────────────────────────────────────────────────

    fig = plt.figure(figsize=S["figsize"])
    # fig.suptitle(S["suptitle"], fontsize=S["suptitle_fs"], y=S["suptitle_y"])
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=S["hspace"], wspace=S["wspace"])
    lat_length = len(data["gran_4k"]["lat"])
    # ── Row 0, Col 0 : Granulation vφ ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _draw_profile_panel(
        ax=ax,
        lat=data["gran_4k"]["lat"],
        prof4=data['gran_4k']['vphi_map'][:, lat_length//2],  # central longitude only
        prof2=data['gran_2k']['vphi_map'][:, lat_length//2],  # central longitude only
        color4=S["color_gran_4k"], color2=S["color_gran_2k"],
        lw4=S["lw"], lw2=S["lw"], ls4=S["ls_4k"], ls2=S["ls_2k"],
        label4="HMI 4K", label2="PMI 2K",
        ylim=S["ylim"]["gran"]["vphi"],
        title=r"$v_\phi$  [m s$^{-1}$]",
        xlabel="", ylabel="Granulation\n\n[m s$^{-1}$]",
        zero_color=S["zero_color"], zero_lw=S["zero_lw"],
        legend_loc=S["legend_loc"], legend_fs=S["legend_fs"],
        title_fs=S["title_fs"], label_fs=S["label_fs"],
        tick_fs=S["tick_fs"], spine_lw=S["spine_lw"],
    )
    ax.set_xticklabels([])

    # ── Row 0, Col 1 : Granulation vθ ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    _draw_profile_panel(
        ax=ax,
        lat=data["gran_4k"]["lat"],
        prof4=data['gran_4k']['vtheta_map'][:, lat_length//2],  # central longitude only
        prof2=data['gran_2k']['vtheta_map'][:, lat_length//2],  # central longitude only
        color4=S["color_gran_4k"], color2=S["color_gran_2k"],
        lw4=S["lw"], lw2=S["lw"], ls4=S["ls_4k"], ls2=S["ls_2k"],
        label4="HMI 4K", label2="PMI 2K",
        ylim=S["ylim"]["gran"]["vtheta"],
        title=r"$v_\theta$  [m s$^{-1}$]",
        xlabel="", ylabel="",
        zero_color=S["zero_color"], zero_lw=S["zero_lw"],
        legend_loc=S["legend_loc"], legend_fs=S["legend_fs"],
        title_fs=S["title_fs"], label_fs=S["label_fs"],
        tick_fs=S["tick_fs"], spine_lw=S["spine_lw"],
    )
    ax.set_xticklabels([])
    lat_length = len(data["mag_4k"]["lat"])
    # ── Row 1, Col 0 : Magnetic vφ ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    _draw_profile_panel(
        ax=ax,
        lat=data["mag_4k"]["lat"],
        prof4=data["mag_4k"]["vphi_map"][:, lat_length//2],  # central longitude only
        prof2=data["mag_2k"]["vphi_map"][:, lat_length//2],  # central longitude only
        color4=S["color_mag_4k"], color2=S["color_mag_2k"],
        lw4=S["lw"], lw2=S["lw"], ls4=S["ls_4k"], ls2=S["ls_2k"],
        label4="HMI 4K", label2="PMI 2K",
        ylim=S["ylim"]["mag"]["vphi"],
        title="",
        xlabel="Latitude [°]", ylabel="Magnetic\n\n[m s$^{-1}$]",
        zero_color=S["zero_color"], zero_lw=S["zero_lw"],
        legend_loc=S["legend_loc"], legend_fs=S["legend_fs"],
        title_fs=S["title_fs"], label_fs=S["label_fs"],
        tick_fs=S["tick_fs"], spine_lw=S["spine_lw"],
    )

    # ── Row 1, Col 1 : Magnetic vθ ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    _draw_profile_panel(
        ax=ax,
        lat=data["mag_4k"]["lat"],
        prof4=data["mag_4k"]["vtheta_map"][:, lat_length//2],  # central longitude only
        prof2=data["mag_2k"]["vtheta_map"][:, lat_length//2],  # central longitude only
        color4=S["color_mag_4k"], color2=S["color_mag_2k"],
        lw4=S["lw"], lw2=S["lw"], ls4=S["ls_4k"], ls2=S["ls_2k"],
        label4="HMI 4K", label2="PMI 2K",
        ylim=S["ylim"]["mag"]["vtheta"],
        title="",
        xlabel="Latitude [°]", ylabel="",
        zero_color=S["zero_color"], zero_lw=S["zero_lw"],
        legend_loc=S["legend_loc"], legend_fs=S["legend_fs"],
        title_fs=S["title_fs"], label_fs=S["label_fs"],
        tick_fs=S["tick_fs"], spine_lw=S["spine_lw"],
    )

    if save:
        plt.savefig(save, bbox_inches="tight")
    plt.show()


plot_systematics_profiles(data,
                          save=f"{SAVE_PREFIX}_systematics_profiles.pdf"
                          if SAVE_PREFIX else None)


# %%  ── CELL 12: PLOT — differential rotation profiles ───────────────────────
# Requires: fits (Cell 8)
# Tweak SETTINGS and rerun this cell only.

def plot_profiles(fits, save=None):
    # ── SETTINGS ─────────────────────────────────────────────────────────────
    S = {
        "figsize":    (16, 6),
        "wspace":     0.15,
        "suptitle":   (f"Differential rotation profile  —  "
                       f"{lon_label()}  (full-year median)"),
        "suptitle_y":  1.01,
        "suptitle_fs": 20,
        "data_lw":     2.2,
        "ls_4k":       "-",
        "ls_2k":       "--",
        "fit_lw":      1.3,
        "fit_ls":      ":",
        "fit_alpha":   0.85,
        "show_scatter":    True,
        "scatter_s":       10,
        "scatter_alpha":   0.3,
        "zero_color":  "#7f7f7f",
        "zero_lw":     0.9,
        "legend_loc":  "lower center",
        "legend_fs":   14,
        "title_fs":    18,
        "label_fs":    18,
        "tick_fs":     16,
        "spine_lw":    0.8,
        "color_gran_4k": "#1f77b4",
        "color_gran_2k": "#6baed6",
        "color_mag_4k":  "#d62728",
        "color_mag_2k":  "#fc8d59",
    }
    # ── END SETTINGS ─────────────────────────────────────────────────────────

    fig = plt.figure(figsize=S["figsize"])
    # fig.suptitle(S["suptitle"], fontsize=S["suptitle_fs"], y=S["suptitle_y"])
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=S["wspace"])

    # ── Left panel : Granulation ──────────────────────────────────────────
    ax_gran = fig.add_subplot(gs[0, 0])
    _draw_diffrot_panel(
        ax=ax_gran,
        fit4=fits["gran_4k"], fit2=fits["gran_2k"],
        color4=S["color_gran_4k"], color2=S["color_gran_2k"],
        data_lw=S["data_lw"], ls4=S["ls_4k"], ls2=S["ls_2k"],
        fit_lw=S["fit_lw"], fit_ls=S["fit_ls"], fit_alpha=S["fit_alpha"],
        show_scatter=S["show_scatter"],
        scatter_s=S["scatter_s"], scatter_alpha=S["scatter_alpha"],
        label4="HMI 4K", label2="PMI 2K",
        zero_color=S["zero_color"], zero_lw=S["zero_lw"],
        title="Granulation tracking",
        xlabel="Latitude [°]", ylabel="$v_\\phi$  [m s$^{-1}$]",
        legend_loc=S["legend_loc"], legend_fs=S["legend_fs"],
        title_fs=S["title_fs"], label_fs=S["label_fs"],
        tick_fs=S["tick_fs"], spine_lw=S["spine_lw"],
    )

    # ── Right panel : Magnetic ────────────────────────────────────────────
    ax_mag = fig.add_subplot(gs[0, 1], sharey=ax_gran)
    _draw_diffrot_panel(
        ax=ax_mag,
        fit4=fits["mag_4k"], fit2=fits["mag_2k"],
        color4=S["color_mag_4k"], color2=S["color_mag_2k"],
        data_lw=S["data_lw"], ls4=S["ls_4k"], ls2=S["ls_2k"],
        fit_lw=S["fit_lw"], fit_ls=S["fit_ls"], fit_alpha=S["fit_alpha"],
        show_scatter=S["show_scatter"],
        scatter_s=S["scatter_s"], scatter_alpha=S["scatter_alpha"],
        label4="HMI 4K", label2="PMI 2K",
        zero_color=S["zero_color"], zero_lw=S["zero_lw"],
        title="Magnetic feature tracking",
        xlabel="Latitude [°]", ylabel="",
        legend_loc=S["legend_loc"], legend_fs=S["legend_fs"],
        title_fs=S["title_fs"], label_fs=S["label_fs"],
        tick_fs=S["tick_fs"], spine_lw=S["spine_lw"],
    )
    # ax_mag.set_yticklabels([])

    if save:
        plt.savefig(save, bbox_inches="tight")
    plt.show()


plot_profiles(fits,
              save=f"{SAVE_PREFIX}_profiles.pdf" if SAVE_PREFIX else None)


# %%  ── CELL 13: PLOT — convergence ───────────────────────────────────────────
# Requires: conv (Cell 9)
# Tweak SETTINGS and rerun this cell only.

def plot_convergence(conv, save=None):
    days, rms_gran, rms_mag, pr_gran, pr_mag = conv

    # ── SETTINGS ─────────────────────────────────────────────────────────────
    S = {
        "figsize":    (6, 8),
        "hspace":     0.2,
        "suptitle":   (f"Convergence of 2K vs 4K  —  {lon_label()}\n"
                       f"fit range [{LAT_FIT_MIN:.0f}°, {LAT_FIT_MAX:.0f}°]"),
        "suptitle_y":  1.01,
        "suptitle_fs": 14,
        "lw":          2.2,
        "ls_gran":     "-",
        "ls_mag":      "--",
        "color_gran":  "#1f77b4",
        "color_mag":   "#d62728",
        "perfect_color": "#7f7f7f",
        "perfect_lw":    0.9,
        "perfect_ls":    ":",
        "xlabel_fs":   13,
        "ylabel_fs":   13,
        "title_fs":    12,
        "tick_fs":     11,
        "legend_fs":   11,
        "spine_lw":    0.8,
        "tick_days":   [1, 3, 7, 14, 30, 90, 180, 365],
    }
    # ── END SETTINGS ─────────────────────────────────────────────────────────

    fig = plt.figure(figsize=S["figsize"])
    # fig.suptitle(S["suptitle"], fontsize=S["suptitle_fs"], y=S["suptitle_y"])
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=S["hspace"])

    # ── Top panel : RMS ───────────────────────────────────────────────────
    ax_rms = fig.add_subplot(gs[0, 0])
    ax_rms.plot(days, rms_gran, color=S["color_gran"], lw=S["lw"],
                ls=S["ls_gran"], label="Granulation")
    ax_rms.plot(days, rms_mag,  color=S["color_mag"],  lw=S["lw"],
                ls=S["ls_mag"],  label="Magnetic features")
    ax_rms.set_ylabel(
        r"RMS$(v_{\phi,2K} - v_{\phi,4K})$  [m s$^{-1}$]",
        fontsize=S["ylabel_fs"])
    ax_rms.set_title("Profile RMS difference  (lower = better)",
                     fontsize=S["title_fs"], pad=6)
    ax_rms.tick_params(labelsize=S["tick_fs"])
    ax_rms.legend(fontsize=S["legend_fs"])
    ax_rms.set_xscale("log")
    for sp in ax_rms.spines.values():
        sp.set_linewidth(S["spine_lw"])

    # ── Bottom panel : Pearson r ──────────────────────────────────────────
    ax_r = fig.add_subplot(gs[1, 0], sharex=ax_rms)
    ax_r.plot(days, pr_gran, color=S["color_gran"], lw=S["lw"],
              ls=S["ls_gran"], label="Granulation")
    ax_r.plot(days, pr_mag,  color=S["color_mag"],  lw=S["lw"],
              ls=S["ls_mag"],  label="Magnetic features")
    ax_r.axhline(1.0, color=S["perfect_color"], lw=S["perfect_lw"],
                 ls=S["perfect_ls"], label="Perfect = 1")
    ax_r.set_ylim(None, 1.05)
    ax_r.set_ylabel(r"Pearson $r$  ($v_{\phi,2K}$ vs $v_{\phi,4K}$)",
                    fontsize=S["ylabel_fs"])
    ax_r.set_xlabel("Number of days averaged", fontsize=S["xlabel_fs"])
    ax_r.set_title("Profile correlation  (higher = better)",
                   fontsize=S["title_fs"], pad=6)
    ax_r.tick_params(labelsize=S["tick_fs"])
    ax_r.legend(fontsize=S["legend_fs"])
    ax_r.set_xscale("log")
    for sp in ax_r.spines.values():
        sp.set_linewidth(S["spine_lw"])

    # shared x ticks
    ticks = [d for d in S["tick_days"] if d <= int(days.max())]
    for ax in [ax_rms, ax_r]:
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(d) for d in ticks], fontsize=S["tick_fs"])
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    if save:
        plt.savefig(save, bbox_inches="tight")
    plt.show()


plot_convergence(conv,
                 save=f"{SAVE_PREFIX}_convergence.pdf" if SAVE_PREFIX else None)


# %%  ── CELL 14: SUMMARY TABLE ────────────────────────────────────────────────
# Requires: fits (Cell 8)

def print_table(fits):
    print("\n" + "═" * 76)
    print(f"  {'Dataset':<22}  {'A [m/s]':>12}  {'B [m/s]':>12}  "
          f"{'C [m/s]':>12}")
    print("─" * 76)
    for tracer_label, key4, key2 in [
        ("Granulation", "gran_4k", "gran_2k"),
        ("Magnetic",    "mag_4k",  "mag_2k"),
    ]:
        for res_label, key in [("HMI 4K", key4), ("PMI 2K", key2)]:
            f = fits[key]
            A, B, C = f["popt"]
            eA, eB, eC = f["bserr"]
            print(f"  {tracer_label+' '+res_label:<22}  "
                  f"{A:>8.1f}±{eA:<5.1f}  "
                  f"{B:>8.1f}±{eB:<5.1f}  "
                  f"{C:>8.1f}±{eC:<5.1f}")
        print()
    print("═" * 76 + "\n")


print_table(fits)
# %%
