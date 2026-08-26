# %% imports
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vendor', 'vorticity'))
from vorticity_func import calculate_vorticity_and_divergence

# %% Load power spectra data for m=3
ps_m3 = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/ps_1d_m3_rossby_hl_lctmag.npz')

power_uphi_m3_rossby = ps_m3['power_uphi_m3_rossby']
power_uthe_m3_rossby = ps_m3['power_uthe_m3_rossby']
power_uphi_m3_hl = ps_m3['power_uphi_m3_hl']
power_uthe_m3_hl = ps_m3['power_uthe_m3_hl']
freqs = ps_m3['freqs']
# %% Load the eigenfunction data

f_rossby = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/plotting_scripts/lct_eigenfunction_m3_rossby_anti_hmi_mag_sm.npz')
uphi_m3_rossby_sm = f_rossby['uphi']
uthe_m3_rossby_sm = f_rossby['uthe']
uphi_m3_rossby_err_r = f_rossby['uphi_err_r']
uthe_m3_rossby_err_r = f_rossby['uthe_err_r']
uphi_m3_rossby_err_i = f_rossby['uphi_err_i']
uthe_m3_rossby_err_i = f_rossby['uthe_err_i']

f_hl = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/plotting_scripts/lct_eigenfunction_m3_hl_anti_hmi_mag_sm.npz')
uphi_m3_hl_sm = f_hl['uphi_sm']
uthe_m3_hl_sm = f_hl['uthe_sm']
uphi_m3_hl_err_r = f_hl['uphi_err_r']
uthe_m3_hl_err_r = f_hl['uthe_err_r']
uphi_m3_hl_err_i = f_hl['uphi_err_i']
uthe_m3_hl_err_i = f_hl['uthe_err_i']

lats = np.linspace(-90, 90, 73)

uphi_ft_anti_mag = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uphi_ft_2010_2024_anti_hmi_m_720s_dt_1h.npy')
M_arr = np.arange(uphi_ft_anti_mag.shape[2])
freqs = np.fft.fftfreq(len(uphi_ft_anti_mag), d=6*3600)
freqs = -np.fft.fftshift(freqs)*1e9

nt = len(uphi_ft_anti_mag)
dt = 6.*3600
print('Number of time steps: {}'.format(nt))
conv_factor = 2/nt*1e-9*dt/144/144

power_m3_uphi_2d_lat_freq = abs(uphi_ft_anti_mag[:, :, 3])**2 * conv_factor

# %% plotting parameters
plt.rcParams.update({
    'axes.labelsize': 18,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
})

# %% Color scheme
c_uphi      = '#4059AD'
c_uphi_fill = '#B0BCF0'
c_uthe      = '#D95F02'
c_uthe_fill = '#F5B88A'

# %% Frequency axis for 2D map (fftshifted to match freqs)
# Clip to the same xlim as the 1D plots
freq_xlim  = [-350, -100]
freq_mask  = (freqs >= freq_xlim[0]) & (freqs <= freq_xlim[1])
freqs_plot = freqs[freq_mask]
power_2d_plot = power_m3_uphi_2d_lat_freq[freq_mask, :]   # (nfreq_clip, nlat)

max_mask = (freqs >= -200) & (freqs <= -180)
# Max over frequency axis for each latitude: shape (nlat,)
power_max_per_lat = np.max(power_m3_uphi_2d_lat_freq[max_mask, :], axis=0)

# Normalise each latitude column by its own peak
power_2d_normalised = power_2d_plot / power_max_per_lat[np.newaxis, :]
print('Frequency bin size: {:.1f} nHz'.format(freqs[1] - freqs[0]))

# -------------------------------------------------------------------
# Running average along frequency axis
# -------------------------------------------------------------------
freq_smooth_bin = 3  # variable bin size

def running_average_2d(arr, bin_size, axis=0):
    """Apply running average along specified axis using cumsum for efficiency."""
    if bin_size <= 1:
        return arr
    kernel = np.ones(bin_size) / bin_size
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode='same'),
        axis=axis,
        arr=arr
    )

power_2d_smoothed = running_average_2d(power_2d_normalised, freq_smooth_bin, axis=0)


def add_mode_arrow(ax, freq, label, color, power_at_freq=None, side='left',
                   offset_freq=25, offset_power=0, is_2d=False, lat_arrow=0):
    sign = -1 if side == 'left' else 1

    if is_2d:
        x_tip  = freq
        x_text = freq + sign * offset_freq
        y_tip  = lat_arrow
        y_text = lat_arrow + offset_power
        ax.annotate(
            label,
            xy=(x_tip, y_tip),
            xytext=(x_text, y_text),
            color=color,
            fontsize=15,
            ha='right' if side == 'left' else 'left',
            va='center',
            arrowprops=dict(arrowstyle='->', color=color, lw=2,
                            connectionstyle='arc3,rad=0.0'),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.9),
            annotation_clip=True
        )
    else:
        ax.annotate(
            label,
            xy=(freq, power_at_freq + offset_power),
            xytext=(freq + sign * offset_freq, power_at_freq + offset_power),
            color=color,
            fontsize=15,
            ha='right' if side == 'left' else 'left',
            va='center',
            arrowprops=dict(arrowstyle='->', color=color, lw=2,
                            connectionstyle='arc3,rad=0.0'),
            annotation_clip=True
        )

fig = plt.figure(figsize=(20, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[1.4, 1], wspace=0.2, hspace=0.35)

ax_2d  = fig.add_subplot(gs[:, 0])
ax_top = fig.add_subplot(gs[0, 1])
ax_bot = fig.add_subplot(gs[1, 1])

# -------------------------------------------------------------------
# === Left: 2D power map ===
# -------------------------------------------------------------------
pcm = ax_2d.pcolormesh(
    freqs_plot, lats,
    power_2d_smoothed.T,
    cmap='gray_r',
    vmin=0, vmax=0.5,
    shading='nearest', rasterized=True
)
ax_2d.set_xlim(freq_xlim)
ax_2d.set_ylim(-85, 85)
ax_2d.set_xlabel('Frequency [nHz]')
ax_2d.set_ylabel('Latitude [deg]')
ax_2d.set_title(r'$m=3$  $u_\phi^-$  power')
cbar = fig.colorbar(pcm, ax=ax_2d, pad=0.02)
cbar.set_label(r'Normalised power')

rossby_lat_idx = np.argmax(power_2d_normalised[
    np.argmin(np.abs(freqs_plot - (-265))), :])
hl_lat_idx = np.argmax(power_2d_normalised[
    np.argmin(np.abs(freqs_plot - (-192))), :])

add_mode_arrow(ax_2d, -265, 'Rossby\n$-$265 nHz', color='red',
               is_2d=True, lat_arrow=50,
               side='left', offset_freq=20)
add_mode_arrow(ax_2d, -192, 'New mode\n$-$192 nHz', color='magenta',
               is_2d=True, lat_arrow=50,
               side='right', offset_freq=20)

# -------------------------------------------------------------------
# === Top-right: Rossby 1D power spectrum ===
# -------------------------------------------------------------------
ax_top.plot(freqs, power_uphi_m3_rossby, color=c_uphi, lw=2.5, label=r'$u_\phi^-$')
ax_top.plot(freqs, power_uthe_m3_rossby, color=c_uthe, lw=2.5, label=r'$u_\theta^+$')
ax_top.set_title('$m=3$ $[30\degree$S-$30\degree$N]')
ax_top.set_ylabel(r'Power [$\mathrm{m^2\,s^{-2}\,nHz^{-1}}$]')
ax_top.set_xlim(freq_xlim)
ax_top.legend(frameon=False, loc = 'upper left')
ax_top.grid(alpha=0.3)

rossby_mask = (freqs >= -290) & (freqs <= -240)
rossby_peak_power = np.max(power_uphi_m3_rossby[rossby_mask])
add_mode_arrow(ax_top, -265, '$-$265 nHz', color='red',
               power_at_freq=rossby_peak_power,
               side='right', offset_freq=40, offset_power=0.03)

# -------------------------------------------------------------------
# === Bottom-right: High-latitude 1D power spectrum ===
# -------------------------------------------------------------------
ax_bot.plot(freqs, power_uphi_m3_hl, color=c_uphi, lw=2.5, label=r'$u_\phi^-$')
ax_bot.plot(freqs, power_uthe_m3_hl, color=c_uthe, lw=2.5, label=r'$u_\theta^+$')
ax_bot.set_title('$m=3$ $[45\degree-75\degree$ (N & S)]')
ax_bot.set_xlabel('Frequency [nHz]')
ax_bot.set_ylabel(r'Power [$\mathrm{m^2\,s^{-2}\,nHz^{-1}}$]')
ax_bot.set_xlim(freq_xlim)
ax_bot.set_ylim([0, 0.2])
ax_bot.legend(frameon=False, loc = 'upper left')
ax_bot.grid(alpha=0.3)

hl_mask = (freqs >= -220) & (freqs <= -160)
hl_peak_power = np.max(power_uphi_m3_hl[hl_mask])
add_mode_arrow(ax_bot, -192, '$-$192 nHz', color='magenta',
               power_at_freq=hl_peak_power,
               side='right', offset_freq=40, offset_power=0.005)

# -------------------------------------------------------------------
# Panel labels
# -------------------------------------------------------------------
# for label, ax in zip(['(a)', '(b)', '(c)'], [ax_2d, ax_top, ax_bot]):
#     ax.text(-0.10, 1.03, label, transform=ax.transAxes,
#             fontsize=18, fontweight='bold', va='top', ha='right')

fig.tight_layout()
# plt.savefig('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pdfs/figure_compare_m3_modes_2d_lat.pdf', bbox_inches='tight')
plt.show()

# %% plotting parameters
plt.rcParams.update({
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
})
# %% Figure setup
# === Define a color scheme ===
c_uphi = '#009E73'    # teal
c_uphi_fill = '#A9E4D7'
c_uthe = '#8E44AD'    # purple
c_uthe_fill = '#D8B8E6'

# === Figure setup ===
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 3, width_ratios=[1.1, 1, 1], wspace=0.3, hspace=0.35)

# -------------------------------------------------------------------
# === Row 1: Rossby mode ===
# -------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# --- Power spectrum (Rossby) ---
ax1.plot(freqs, power_uphi_m3_rossby, color=c_uphi, lw=2.5, label=r'$u_\phi$')
ax1.plot(freqs, power_uthe_m3_rossby, color=c_uthe, lw=2.5, label=r'$u_\theta$')
ax1.axvline(-265, color='gray', ls='--', lw=2, label=r'$\endash$265 nHz')
ax1.set_title('$m=3$ Rossby')
ax1.set_ylabel(r'Power [$m^2/s^2/nHz$]')
ax1.set_xlim([-350, -100])
ax1.legend(frameon=False)
ax1.grid(alpha=0.3)

# --- u_phi Rossby ---
ax2.plot(lats, np.real(uphi_m3_rossby_sm), color=c_uphi, lw=2.5, label='Re')
ax2.plot(lats, np.imag(uphi_m3_rossby_sm), color=c_uphi, lw=2.5, ls='--', label='Im')
ax2.fill_between(
    lats,
    np.real(uphi_m3_rossby_sm) - uphi_m3_rossby_err_r,
    np.real(uphi_m3_rossby_sm) + uphi_m3_rossby_err_r,
    color=c_uphi_fill, alpha=0.5
)
ax2.fill_between(
    lats,
    np.imag(uphi_m3_rossby_sm) - uphi_m3_rossby_err_i,
    np.imag(uphi_m3_rossby_sm) + uphi_m3_rossby_err_i,
    color=c_uphi_fill, alpha=0.5
)
ax2.set_title(r'$u_\phi$ Rossby')
ax2.set_ylabel('Amplitude [m/s]')
ax2.grid(alpha=0.3)

# --- u_theta Rossby ---
ax3.plot(lats, np.real(uthe_m3_rossby_sm), color=c_uthe, lw=2.5, label='Re')
ax3.plot(lats, np.imag(uthe_m3_rossby_sm), color=c_uthe, lw=2.5, ls='--', label='Im')
ax3.fill_between(
    lats,
    np.real(uthe_m3_rossby_sm) - uthe_m3_rossby_err_r,
    np.real(uthe_m3_rossby_sm) + uthe_m3_rossby_err_r,
    color=c_uthe_fill, alpha=0.5
)
ax3.fill_between(
    lats,
    np.imag(uthe_m3_rossby_sm) - uthe_m3_rossby_err_i,
    np.imag(uthe_m3_rossby_sm) + uthe_m3_rossby_err_i,
    color=c_uthe_fill, alpha=0.5
)
ax3.set_title(r'$u_\theta$ Rossby')
ax3.grid(alpha=0.3)

# -------------------------------------------------------------------
# === Row 2: High-lat mode ===
# -------------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

# --- Power spectrum (High-ℓ) ---
ax4.plot(freqs, power_uphi_m3_hl, color=c_uphi, lw=2.5, label=r'$u_\phi$')
ax4.plot(freqs, power_uthe_m3_hl, color=c_uthe, lw=2.5, label=r'$u_\theta$')
ax4.axvline(-192, color='gray', ls='--', lw=2, label='-192 nHz')
ax4.set_title('$m=3$ High-latitude')
ax4.set_xlabel('Frequency [nHz]')
ax4.set_xlim([-350, -100])
ax4.set_ylim([0, 0.2])
ax4.set_ylabel(r'Power [$m^2/s^2/nHz$]')
ax4.legend(frameon=False)
ax4.grid(alpha=0.3)

# --- u_phi High-ℓ ---
ax5.plot(lats, np.real(uphi_m3_hl_sm), color=c_uphi, lw=2.5, label='Re')
ax5.plot(lats, np.imag(uphi_m3_hl_sm), color=c_uphi, lw=2.5, ls='--', label='Im')
ax5.fill_between(
    lats,
    np.real(uphi_m3_hl_sm) - uphi_m3_hl_err_r,
    np.real(uphi_m3_hl_sm) + uphi_m3_hl_err_r,
    color=c_uphi_fill, alpha=0.5
)
ax5.fill_between(
    lats,
    np.imag(uphi_m3_hl_sm) - uphi_m3_hl_err_i,
    np.imag(uphi_m3_hl_sm) + uphi_m3_hl_err_i,
    color=c_uphi_fill, alpha=0.5
)
ax5.set_title(r'$u_\phi$ High-latitude')
ax5.set_xlabel('Latitude [deg]')
ax5.set_ylabel('Amplitude [m/s]')
ax5.grid(alpha=0.3)

# --- u_theta High-ℓ ---
ax6.plot(lats, np.real(uthe_m3_hl_sm), color=c_uthe, lw=2.5, label='Re')
ax6.plot(lats, np.imag(uthe_m3_hl_sm), color=c_uthe, lw=2.5, ls='--', label='Im')
ax6.fill_between(
    lats,
    np.real(uthe_m3_hl_sm) - uthe_m3_hl_err_r,
    np.real(uthe_m3_hl_sm) + uthe_m3_hl_err_r,
    color=c_uthe_fill, alpha=0.5
)
ax6.fill_between(
    lats,
    np.imag(uthe_m3_hl_sm) - uthe_m3_hl_err_i,
    np.imag(uthe_m3_hl_sm) + uthe_m3_hl_err_i,
    color=c_uthe_fill, alpha=0.5
)
ax6.set_title(r'$u_\theta$ High-latitude')
ax6.set_xlabel('Latitude [deg]')
ax6.grid(alpha=0.3)

# -------------------------------------------------------------------
# Shared formatting
# -------------------------------------------------------------------
for ax in [ax2, ax3, ax5, ax6]:
    ax.legend(loc = 'lower center')
    ax.set_xlim(-95, 95)
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_ylim(-1.8, 1.8)

# panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
# for label, ax in zip(panel_labels, [ax1, ax2, ax3, ax4, ax5, ax6]):
#     ax.text(-0.08, 1.05, label, transform=ax.transAxes,
#             fontsize=20, fontweight='bold', va='top', ha='right')

fig.tight_layout()
# plt.savefig('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pdfs/figure_compare_m3_modes.pdf', bbox_inches='tight')
plt.show()


# %%
"""
Rossby mode eigenfunction visualization (n=1, m=3)
Produces a 3-row figure:
  Row 1 — Observed surface eigenfunctions (u_phi, u_theta, zeta_r, div_h)
  Row 2 — Model surface eigenfunctions (normalized)
  Row 3 — Model meridional cross-sections
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from scipy import interpolate
from scipy.ndimage import zoom, gaussian_filter

# =============================================================================
# Constants
# =============================================================================
OMEGA0   = 456          # Carrington rotation rate (nHz)
M        = 3            # Azimuthal order
NLON     = 144          # Longitude grid size for 2D eigenfunctions
NLAT     = 73           # Latitude grid size
LMAX     = 35           # Max spherical harmonic degree
MMAX     = 35
RSUN     = 6.96e8       # Solar radius (m)
LON0     = 120          # Orthographic projection centre longitude
LAT0     = 30           # Orthographic projection centre latitude
ROT_DEG  = 70           # Longitude rotation applied to model surface maps
SIGMA_SMOOTH = 0        # Gaussian smoothing sigma (pixels) for model row

DATA_DIR = "/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/sup_rossby_n1_m3"

# =============================================================================
# Load data
# =============================================================================
data     = np.load(f"{DATA_DIR}/Rossby_n1-0_3_new_new.npz")
diff_rot = np.load(f"{DATA_DIR}/diff_rot.npz")

# Differential rotation interpolator (used for contour overlays)
f_rot = interpolate.RectBivariateSpline(
    diff_rot['r'], diff_rot['theta'], diff_rot['ome'].T
)

# =============================================================================
# Observational 2D eigenfunctions
# (uphi_m3_hl_sm, uthe_m3_hl_sm, calculate_vorticity_and_divergence defined
#  externally and assumed to be in scope)
# =============================================================================
lats = np.linspace(-90,  90,  NLAT)
lons = np.linspace(-180, 180, NLON, endpoint=False)
longitudes_rad = np.deg2rad(lons)

def compute_2d_ef(ef_uphi, ef_uthe, m):
    """Project 1D latitudinal eigenfunctions onto a 2D lon-lat grid."""
    vphi   = np.outer(ef_uphi, np.exp(1j * m * longitudes_rad)).real
    vtheta = np.outer(ef_uthe, np.exp(1j * m * longitudes_rad)).real
    return vphi, vtheta

vphi_obs, vtheta_obs = compute_2d_ef(uphi_m3_hl_sm, uthe_m3_hl_sm, m=M)
_, _, rvort_obs, hdiv_obs = calculate_vorticity_and_divergence(
    vphi_obs, vtheta_obs, NLAT, NLON, LMAX, MMAX, RSUN
)

# =============================================================================
# Model surface eigenfunctions (normalized)
# =============================================================================

dim_factor_vort = 2.87e-6
dim_factor_vel = RSUN * dim_factor_vort
utsurf    = data['utsurf'].T * dim_factor_vel
upsurf    = data['upsurf'].T * dim_factor_vel
vortrsurf = data['vortrsurf'].T * dim_factor_vort
divh_surf = data['divh_surf'].T * dim_factor_vort


norm_vel  = np.nanmax(np.real(vtheta_obs))/np.nanmax(np.real(utsurf))
print(utsurf.shape, vtheta_obs.shape)


utsurf_norm    = np.real(utsurf)    * norm_vel
upsurf_norm    = np.real(upsurf)    * norm_vel
vortrsurf_norm = np.real(vortrsurf) * norm_vel
divh_surf_norm = np.real(divh_surf) * norm_vel

print(np.nanmax(vortrsurf_norm), np.nanmax(divh_surf_norm))

# =============================================================================
# Model meridional cross-sections
# =============================================================================
def get_meridional_data(data, f_rot, norm=1.0, ph=None):
    """
    Interpolate meridional eigenfunctions onto a fine (r, theta) grid.

    Returns a dict with Cartesian coords, flow components, differential
    rotation contrast, and boundary curves for plotting.
    """

    dim_factor_vort = 2.87e-6
    dim_factor_vel = RSUN * dim_factor_vort

    utsurf = data['utsurf'].T * dim_factor_vel
    norm_vel = np.nanmax(np.real(vtheta_obs))/np.nanmax(np.real(utsurf))

    r     = data['r'].reshape(-1)
    theta = data['theta'].reshape(-1)

    # Phase alignment: set by equatorial u_theta at the surface
    if ph is None:
        eq_idx = len(theta) // 2
        ph = np.angle(data['utmid'][eq_idx, -1])

    phase = np.exp(-1j * ph)
    upmid    = (data['upmid']    * phase).imag * norm_vel * dim_factor_vel
    utmid    = (data['utmid']    * phase).real * norm_vel * dim_factor_vel
    vortrmid = (data['vortrmid'] * phase).imag * norm_vel * dim_factor_vort
    divh_mid = (data['divh_mid'] * phase).real * norm_vel * dim_factor_vort

    # Fine grid for smooth rendering
    theta_rev = theta[::-1]
    rfine     = np.linspace(r.min(),     r.max(),     4 * len(r))
    tfine     = np.linspace(theta_rev.min(), theta_rev.max(), 4 * len(theta))

    def _interp(arr):
        return interpolate.RectBivariateSpline(theta_rev, r, arr[::-1, :])(tfine, rfine)

    # Differential rotation contrast relative to Carrington rate
    diff_rott = (f_rot(rfine, tfine).T - OMEGA0) / OMEGA0

    # Boundary curves (outer/inner shell edges + pole lines)
    outerx = r.max() * np.sin(theta)
    outery = r.max() * np.cos(theta)
    innerx = r.min() * np.sin(theta)
    innery = r.min() * np.cos(theta)
    polex  = np.min(np.outer(r, np.sin(theta))) * np.ones_like(r)

    x_fine = np.outer(rfine, np.sin(tfine))
    y_fine = np.outer(rfine, np.cos(tfine))

    return {
        'x':        x_fine,
        'y':        y_fine,
        'utmid':    _interp(utmid),
        'upmid':    _interp(upmid),
        'vortrmid': _interp(vortrmid),
        'divh_mid': _interp(divh_mid),
        'diff_rott': diff_rott,
        'outerx': outerx, 'outery': outery,
        'innerx': innerx, 'innery': innery,
        'polex': polex,   'r': r,
    }


def draw_meridional(ax, pdata, z, vmax, title='', diff_rot_level=None, cmap_label = ''):
    """Pcolormesh a meridional cross-section with optional diff-rot contour."""
    im = ax.pcolormesh(
        pdata['x'], pdata['y'], z.T,
        cmap='seismic', vmin=-vmax, vmax=vmax, shading='nearest', rasterized=True
    )
    # Shell boundaries
    for (bx, by) in [(pdata['outerx'], pdata['outery']),
                     (pdata['innerx'], pdata['innery']),
                     (pdata['polex'],  pdata['r']),
                     (pdata['polex'], -pdata['r'])]:
        ax.plot(bx, by, 'k', lw=0.7)

    if diff_rot_level is not None:
        ax.contour(pdata['x'], pdata['y'], pdata['diff_rott'].T,
                   levels=[diff_rot_level], colors='k',
                   linestyles='-', linewidths=0.6)

    ax.set_aspect('equal')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-1.02, 1.02)
    ax.axis('off')
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8, label = cmap_label)
    return im


pdata = get_meridional_data(data, f_rot, norm=1.0)

# Co-rotation level for differential rotation contour
corot_level = data['omega'].real / M

# =============================================================================
# Prepare model surface data: smooth → resample to observation grid
# =============================================================================
def smooth_and_resample(arr, target_shape, sigma=0):
    """Gaussian smooth then zoom to target_shape (nlat, nlon)."""
    arr_smooth = gaussian_filter(arr, sigma=sigma)
    zy = target_shape[0] / arr_smooth.shape[0]
    zx = target_shape[1] / arr_smooth.shape[1]
    return zoom(arr_smooth, (zy, zx), order=1)

target_shape = (NLAT, NLON)
model_surf_resampled = [
    smooth_and_resample(arr, target_shape, sigma=SIGMA_SMOOTH)
    for arr in [upsurf_norm, utsurf_norm, vortrsurf_norm / 1e-8, divh_surf_norm / 1e-8]
]

def rotate_lon(arr, delta_deg):
    """Roll a (nlat, nlon) array eastward by delta_deg."""
    deg_per_px = (lons[-1] - lons[0]) / (NLON - 1)
    n_shift = int(np.round(delta_deg / deg_per_px))
    return np.roll(arr, n_shift, axis=1)

model_surf_resampled = [rotate_lon(arr, ROT_DEG) for arr in model_surf_resampled]

# =============================================================================
# Figure layout
# =============================================================================
VORT_LIM = 2.0   # color limit for vorticity/divergence in observed row (×10⁻⁸ s⁻¹)

col_titles  = [r'$u_\phi$', r'$u_\theta$', r'$\zeta_\mathrm{r}$', r'$\nabla\cdot\mathbf{u}_h$']
row_labels  = ['Observation\n$(-192$ nHz$)$', 'Model\n$(-186$ nHz$)$', 'Meridional\ncross-section\n(Model)']

obs_data    = [vphi_obs, vtheta_obs, rvort_obs / 1e-8, hdiv_obs / 1e-8]
merid_data  = [pdata['upmid'], pdata['utmid'], pdata['vortrmid']/1e-8, pdata['divh_mid']/1e-8]

# Per-column vmax for each row
vmax_obs   = [1.0, 1.0, VORT_LIM, VORT_LIM]
vmax_model = [1.0, 1.0, VORT_LIM, VORT_LIM]
vmax_merid = [1.0, 1.0, VORT_LIM, VORT_LIM]

proj = ccrs.Orthographic(central_longitude=LON0, central_latitude=LAT0)

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(
    4, 5,
    width_ratios=[0.22, 1, 1, 1, 1],
    height_ratios=[0.15, 1, 1, 1],
    wspace=0.2, hspace=0.55
)

# ---- Column titles (row 0) ----
for j, title in enumerate(col_titles):
    ax = fig.add_subplot(gs[0, j + 1])
    ax.text(0.5, -1.5, title, va='bottom', ha='center',
            fontsize=36, transform=ax.transAxes)
    ax.axis('off')

# ---- Rows 1 & 2: globe maps ----
cmap_label = [r'ms$^{-1*}$', r'ms$^{-1*}$', r'10$^{-8}$s$^{-1*}$', r'10$^{-8}$s$^{-1*}$']
for i, (row_data, vmaxs, cbar_units) in enumerate(zip(
    [obs_data, model_surf_resampled],
    [vmax_obs, vmax_model],
    [
        [r'ms$^{-1}$', r'ms$^{-1}$', r'10$^{-8}$s$^{-1}$', r'10$^{-8}$s$^{-1}$'],
        [r'ms$^{-1*}$', r'ms$^{-1*}$', r'10$^{-8}$s$^{-1*}$', r'10$^{-8}$s$^{-1*}$'],
    ]
)):
    # Row label
    ax_lbl = fig.add_subplot(gs[i + 1, 0])
    ax_lbl.text(-1.5, 0.5, row_labels[i], va='center', ha='center',
                fontsize=22, transform=ax_lbl.transAxes)
    ax_lbl.axis('off')

    for j in range(4):
        ax = fig.add_subplot(gs[i + 1, j + 1], projection=proj)
        im = ax.pcolormesh(
            lons, lats, row_data[j],
            transform=ccrs.PlateCarree(),
            cmap='bwr',
            vmin=-vmaxs[j], vmax=vmaxs[j],
            rasterized=True, shading='nearest'
        )
        gl = ax.gridlines(draw_labels=False, linewidth=0.8,
                          alpha=0.8, color='k', linestyle='--')
        gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

        cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=15, pad=0.05,
                            location='bottom', ticks=[-vmaxs[j], 0, vmaxs[j]])
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(cbar_units[j], fontsize=14)

# ---- Row 3: meridional cross-sections ----
# The meridional plots are tall and narrow, so they sit left of centre within
# each column cell. MERID_LEFT_OFFSET shifts the whole row leftward to align
# their visual centres with the globes above. Tune this value (in figure
# fraction units) if you adjust figure size or wspace.
MERID_LEFT_OFFSET = 0.02

ax_lbl = fig.add_subplot(gs[3, 0])
ax_lbl.text(-1.5, 0.5, row_labels[2], va='center', ha='center',
            fontsize=22, transform=ax_lbl.transAxes)
ax_lbl.axis('off')

# Get the bounding box of each gs[3, j+1] cell and shift left by the offset
for j in range(4):
    ss   = gs[3, j + 1].get_position(fig)   # SubplotSpec → Bbox in fig coords
    ax   = fig.add_axes([
        ss.x0 - MERID_LEFT_OFFSET,   # shift left
        ss.y0,
        ss.width,
        ss.height,
    ])
    draw_meridional(ax, pdata, merid_data[j], vmax=vmax_merid[j],
                    diff_rot_level=corot_level, cmap_label=cmap_label[j])

# =============================================================================
# Save / show
# =============================================================================
fig.savefig(
    '/data/seismo/joshin/pipeline-test/local_correlation_tracking/pdfs/m3_eigenfunctions_obs_vs_model.pdf',
    bbox_inches='tight'
)
plt.show()

# %%
