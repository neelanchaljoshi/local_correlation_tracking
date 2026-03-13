# %%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
import sys
sys.path.append('/data/seismo/joshin/pipeline-test/paper_lct/vorticity')
from vorticity_func import calculate_vorticity_and_divergence

# === Font size control ===
font_sizes = {
    "title": 28,
    "label": 24,
    "tick": 20,
    "legend": 24
}

# %%
# Compute 2d eigenfunctions
def compute_2d_ef(ef_uphi, ef_uthe, m):
    nlon = 144
    longitudes = np.deg2rad(np.linspace(-180,180, nlon, endpoint=False))
    vphi = np.outer(ef_uphi, np.exp(1j * m * longitudes)).real
    vtheta = np.outer(ef_uthe, np.exp(1j * m * longitudes)).real
    return vphi, vtheta

# %% load data
f = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/eigenfunctions/cleaned_rotated/lct_eigenfunction_m1_hmi_mag_rotated.npz')
uphi_m1_hmi_sm = f['uphi']
uthe_m1_hmi_sm = f['uthe']

f = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/eigenfunctions/cleaned_rotated/lct_eigenfunction_m2_hmi_mag_rotated.npz')
uphi_m2_hmi_sm = f['uphi']
uthe_m2_hmi_sm = f['uthe']

f = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/eigenfunctions/cleaned_rotated/lct_eigenfunction_m8_hmi_mag_rotated.npz')
uphi_m8_hmi_sm = f['uphi']
uthe_m8_hmi_sm = f['uthe']

f = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/eigenfunctions/cleaned_rotated/lct_eigenfunction_m13_hmi_mag_rotated.npz')
uphi_m13_hmi_sm = f['uphi']
uthe_m13_hmi_sm = f['uthe']

# %% make 2d efs
vphi_m1, vtheta_m1 = compute_2d_ef(uphi_m1_hmi_sm, uthe_m1_hmi_sm, m=1)
vphi_m2, vtheta_m2 = compute_2d_ef(uphi_m2_hmi_sm, uthe_m2_hmi_sm, m=2)
vphi_m8, vtheta_m8 = compute_2d_ef(uphi_m8_hmi_sm, uthe_m8_hmi_sm, m=8)
vphi_m13, vtheta_m13 = compute_2d_ef(uphi_m13_hmi_sm, uthe_m13_hmi_sm, m=13)


# %% Compute vorticity and divergence
lmax = 20
mmax = 20
rsun = 6.96e8
nlon = 144
nlat = 73
lats = np.linspace(-90, 90, nlat)
lons = np.linspace(-180, 180, nlon, endpoint=False)
_, _, rvort_m1, hdiv_m1 = calculate_vorticity_and_divergence(vphi_m1, vtheta_m1, nlat, nlon, lmax, mmax, rsun)
_, _, rvort_m2, hdiv_m2 = calculate_vorticity_and_divergence(vphi_m2, vtheta_m2, nlat, nlon, lmax, mmax, rsun)
_, _, rvort_m8, hdiv_m8 = calculate_vorticity_and_divergence(vphi_m8, vtheta_m8, nlat, nlon, lmax, mmax, rsun)
_, _, rvort_m13, hdiv_m13 = calculate_vorticity_and_divergence(vphi_m13, vtheta_m13, nlat, nlon, lmax, mmax, rsun)

# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

lon0 = 120
lat0 = 30
limit = 2e-8
proj_ortho = ccrs.Orthographic(central_longitude=lon0, central_latitude=lat0)

fig = plt.figure(figsize=(24, 21))
gs = gridspec.GridSpec(5, 5, width_ratios=[0.2, 1, 1, 1, 1], height_ratios=[0.2, 1, 1, 1, 1], wspace=0.2, hspace=0.6)

flow_types = [r'$u_\phi$', r'$u_\theta$', r'$\zeta_\mathrm{r}$', r'$(\nabla \cdot \mathbf{u})_\mathrm{h}$']
row_labels = ['m = 1 HL (+)', 'm = 2 CL (+)', 'm = 8 Eq.R (+)', 'm=13 HFR (\N{MINUS SIGN})']

data_arrays = [
    [vphi_m1, vtheta_m1, rvort_m1/limit, hdiv_m1/limit],
    [vphi_m2, vtheta_m2, rvort_m2/limit, hdiv_m2/limit],
    [vphi_m8, vtheta_m8, rvort_m8/limit, hdiv_m8/limit],
    [vphi_m13, vtheta_m13, rvort_m13/limit, hdiv_m13/limit]
]

limit = 2

# lats = np.linspace(90, -90, 73)

vmax_vals = [[10, 5, limit * 5, limit * 5], [2, 2, limit, limit], [2, 2, limit, limit], [1, 1, limit/4, limit/4]]
vmin_vals = [[-10, -5, -limit * 5, -limit * 5], [-2, -2, -limit, -limit], [-2, -2, -limit, -limit], [-1, -1, -limit/4, -limit/4]]

axs = []
for i in range(4):
    row_axes = []
    for j in range(4):
        # data_arrays[i][j] = np.flip(data_arrays[i][j], axis=0)
        # lats = np.flip(lats)
        ax = fig.add_subplot(gs[i+1, j+1], projection=proj_ortho)
        im = ax.pcolormesh(lons, lats, data_arrays[i][j], transform=ccrs.PlateCarree(), cmap='bwr', vmax=vmax_vals[i][j], vmin=vmin_vals[i][j], rasterized=True)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.8, alpha=0.8, color='k', linestyle='--')
        gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
        cbar = fig.colorbar(im, ax=ax, label=r'$s^{-1}$' if j > 1 else r'$ms^{-1}$', shrink=0.6, aspect=15, pad = 0.05,location='bottom', ticks=[vmin_vals[i][j],0, vmax_vals[i][j]])
        if j > 1:
            # cbar.ax.ticklabel_format(style='sci', scilimits=(0,0))
            cbar.ax.yaxis.get_offset_text().set_fontsize(18)
            # cbar.ax.yaxis.offsetText.set_text(cbar.ax.yaxis.offsetText.get_text().replace('e', '×10^'))
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(label=r'$10^{-8} s^{-1}$' if j > 1 else r'$ms^{-1}$', fontsize=16)
        row_axes.append(ax)
    axs.append(row_axes)

# Add row labels
for i in range(4):
    ax_label = fig.add_subplot(gs[i+1, 0])
    ax_label.text(0.5, 0.5, row_labels[i], va='center', ha='center', fontsize=24, transform=ax_label.transAxes)
    ax_label.set_frame_on(False)
    ax_label.set_xticks([])
    ax_label.set_yticks([])

# Add column headings
for j in range(4):
    ax_label = fig.add_subplot(gs[0, j+1])
    ax_label.text(0.5, -0.5, flow_types[j], va='bottom', ha='center', fontsize=28, transform=ax_label.transAxes)
    ax_label.set_frame_on(False)
    ax_label.set_xticks([])
    ax_label.set_yticks([])
# plt.savefig('2d_eigenfunctions.pdf', bbox_inches='tight')
plt.show()

# %%
