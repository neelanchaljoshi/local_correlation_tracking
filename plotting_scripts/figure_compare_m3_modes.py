# %% imports
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import sys
sys.path.append('/data/seismo/joshin/pipeline-test/paper_lct/vorticity')
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

# %% Get the eigenfunction from simulation loaded
data=np.load("/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/sup_rossby_n1_m3/Rossby_n1-0_3.npz")
print(data['omega']*456)#frequency
print(data.files)

#surface behaviour
utsurf=data['utsurf'].T
upsurf=data['upsurf'].T
print(np.max(utsurf))
utsurf_norm=utsurf/np.max(utsurf)
upsurf_norm=upsurf/np.max(upsurf)
vortrsurf=data['vortrsurf'].T
vortrsurf_norm=vortrsurf/np.max(np.real(vortrsurf))
divh_surf=data['divh_surf'].T
divh_surf_norm=divh_surf/np.max(np.real(divh_surf))

#meridional cross-section
theta=data['theta']
phi = data['phi']

print(np.max(np.real(upsurf_norm)), np.min(np.real(upsurf_norm)))
print(np.max(np.real(utsurf_norm)), np.min(np.real(utsurf_norm)))
print(np.max(np.real(vortrsurf_norm)), np.min(np.real(vortrsurf_norm)))
print(np.max(np.real(divh_surf_norm)), np.min(np.real(divh_surf_norm)))

# upmid=(data['upmid']*np.exp(-1j*phi)).imag*v_fac
# utmid=(data['utmid']*np.exp(-1j*phi)).real*v_fac
# urmid=(data['urmid']*np.exp(-1j*phi)).real*v_fac
# vortrmid= (data['vortrmid']*np.exp(-1j*phi)).imag*vor_fac
# vortrmid= (data['vortrmid']*np.exp(-1j*phi)).imag*vor_fac
# pmid = (data['pmid']*np.exp(-1j*phi)).imag*p_fac

# %% Convert 1d to 2d eigenfunction for observations
def compute_2d_ef(ef_uphi, ef_uthe, m):
    nlon = 144
    longitudes = np.deg2rad(np.linspace(-180,180, nlon, endpoint=False))
    vphi = np.outer(ef_uphi, np.exp(1j * m * longitudes)).real
    vtheta = np.outer(ef_uthe, np.exp(1j * m * longitudes)).real
    return vphi, vtheta

vphi_m3_hl, vtheta_m3_hl = compute_2d_ef(uphi_m3_hl_sm, uthe_m3_hl_sm, m=3)

# %% Compute vorticity and divergence
lmax = 35
mmax = 35
rsun = 6.96e8
nlon = 144
nlat = 73
lats = np.linspace(-90, 90, nlat)
lons = np.linspace(-180, 180, nlon, endpoint=False)
_, _, rvort_m3_hl, hdiv_m3_hl = calculate_vorticity_and_divergence(vphi_m3_hl, vtheta_m3_hl, nlat, nlon, lmax, mmax, rsun)

# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

lon0 = 120
lat0 = 30
limit = 1e-8
proj_ortho = ccrs.Orthographic(central_longitude=lon0, central_latitude=lat0)

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 5, width_ratios=[0.2, 1, 1, 1, 1], height_ratios=[0.2, 1, 1], wspace=0.2, hspace=0.6)

flow_types = [r'$u_\phi$', r'$u_\theta$', r'$\zeta_\mathrm{r}$', r'$\nabla_h \cdot \mathbf{u}$']
row_labels = ['m = 3 HL \n (obs. at -192 nHz)', 'm = 3, n = 1 Rossby \n (sim. at -182 nHz)']

data_arrays = [
    [vphi_m3_hl, vtheta_m3_hl, rvort_m3_hl/limit, hdiv_m3_hl/limit],
    [np.real(upsurf_norm), np.real(utsurf_norm), np.real(vortrsurf_norm), np.real(divh_surf_norm)]
]

limit = 2

vmax_vals = [
    [1.0, 1.0, limit, limit],
    [1.0, 1.0, 1.0, 1.0]
]
vmin_vals = [
    [-1.0, -1.0, -limit, -limit],
    [-1.0, -1.0, -1.0, -1.0]
]

axs = []
for i in range(2):
    row_axes = []
    for j in range(4):
        ax = fig.add_subplot(gs[i+1, j+1], projection=proj_ortho)
        if i == 0:
            im = ax.pcolormesh(lons, lats, data_arrays[i][j], transform=ccrs.PlateCarree(), cmap='bwr', vmax=vmax_vals[i][j], vmin=vmin_vals[i][j], rasterized=True)
        else:
            im = ax.imshow(data_arrays[i][j]/np.max(data_arrays[i][j]), transform=ccrs.PlateCarree(), cmap='bwr', vmax=vmax_vals[i][j], vmin=vmin_vals[i][j], rasterized=True)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.8, alpha=0.8, color='k', linestyle='--')
        gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
        cbar = fig.colorbar(im, ax=ax, label=r'$s^{-1}$' if j > 1 else r'$ms^{-1}$', shrink=0.6, aspect=15, pad=0.05, location='bottom', ticks=[vmin_vals[i][j], 0, vmax_vals[i][j]])
        if j > 1:
            cbar.ax.yaxis.get_offset_text().set_fontsize(18)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(label=r'$10^{-8} s^{-1}$' if j > 1 else r'$ms^{-1}$', fontsize=16)
        if i == 1:
            cbar.set_label(label=r'Normalized (arb. units)', fontsize=16)
        row_axes.append(ax)
    axs.append(row_axes)

# Add row labels
for i in range(2):
    ax_label = fig.add_subplot(gs[i+1, 0])
    ax_label.text(-1.5, 0.5, row_labels[i], va='center', ha='center', fontsize=24, transform=ax_label.transAxes)
    ax_label.set_frame_on(False)
    ax_label.set_xticks([])
    ax_label.set_yticks([])

# Add column headings
for j in range(4):
    ax_label = fig.add_subplot(gs[0, j+1])
    ax_label.text(0.5, -1.5, flow_types[j], va='bottom', ha='center', fontsize=36, transform=ax_label.transAxes)
    ax_label.set_frame_on(False)
    ax_label.set_xticks([])
    ax_label.set_yticks([])
fig.tight_layout()
plt.savefig('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pdfs/new_m3_2d_eigenfunctions.pdf', bbox_inches='tight')
plt.show()

# %%
lon0=120.0
lat0=30.0
nlon = 248
nlat = 124
limit = 1
longitudes = np.deg2rad(np.linspace(-180,180, nlon, endpoint=False))
colatitudes = np.deg2rad(np.linspace(180, 0, nlat))

m = 3
nr = 2
# vphi = np.outer(upsurf/np.max(upsurf), np.exp(1j*m*longitudes)).real
# vtheta = np.outer(utsurf, np.exp(1j*m*longitudes)).real
proj_ortho = ccrs.Orthographic(central_longitude=lon0,central_latitude=lat0)
fig, ax = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': proj_ortho}, dpi = 300)
ax[0].set_title(r'$u_\phi$')
im = ax[0].imshow(upsurf/np.max(upsurf), origin='lower', extent=[-180, 180, -85, 85], cmap = 'seismic', vmax=limit, vmin=-limit, transform=ccrs.PlateCarree())
gl=ax[0].gridlines(crs=ccrs.PlateCarree(),draw_labels=False,linewidth=0.8, alpha =0.8,color='k',linestyle='--')
fig.colorbar(im, ax=ax[0], label=r'ms$^{-1}$', shrink=0.3, aspect=10)
ax[1].set_title(r'$u_\theta$')
im = ax[1].imshow(utsurf/np.max(utsurf), origin='lower', extent=[-180, 180, -85, 85], cmap = 'seismic', vmax=limit, vmin=-limit, transform=ccrs.PlateCarree())
gl=ax[1].gridlines(crs=ccrs.PlateCarree(),draw_labels=False,linewidth=0.8, alpha =0.8,color='k',linestyle='--')
fig.colorbar(im, ax=ax[1], label=r'ms$^{-1}$', shrink=0.3, aspect=10)
plt.show()
# %%
