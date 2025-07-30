# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from datetime import datetime
from matplotlib.dates import DateFormatter
sys.path.insert(0, '/data/seismo/zhichao/codes/pypkg')
sys.path.append('/data/seismo/joshin/pipeline-test/python_modules/')
import pandas as pd
from zclpy3.remap import from_cyl_to_tan, get_tan_from_lnglat
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'sans-serif'
# %%
# Function to get clip mask for LCT
def return_clip_mask(arr, radius_arr, radius_ratio, rsun_obs, pad = False):
    clipradius = radius_ratio * rsun_obs
    mask = np.zeros_like(arr)
    mask[radius_arr < clipradius] = 1
    mask[radius_arr >= clipradius] = 0
    if pad:
        mask = np.pad(mask, [(0,0),(0,0),(36, 35)], mode = 'constant', constant_values = 0)
    # arr[~(radius_arr < clipradius[:, None, None])] = np.nan
    # if pad:
    #     arr = np.pad(arr, [(0,0),(0,0),(36, 35)], mode = 'constant', constant_values = np.nan)
    return mask
# %%
# Load the LCT and RDA data
uphi_gran = np.load('/data/seismo/joshin/pipeline-test/paper_lct/processed_data/uphi_hmi_ic_45s_processed.npy')
uthe_gran = np.load('/data/seismo/joshin/pipeline-test/paper_lct/processed_data/utheta_hmi_ic_45s_processed.npy')
uthe_mf = np.load('/data/seismo/joshin/pipeline-test/paper_lct/processed_data/utheta_hmi_m_720s_dt_1h_processed.npy')
uphi_mf = np.load('/data/seismo/joshin/pipeline-test/paper_lct/processed_data/uphi_hmi_m_720s_dt_1h_processed.npy')
mask = np.load('/data/seismo/joshin/pipeline-test/paper_lct/mask_rda_for_figures.npz')['mask']

# %%
# Get the LCT mask
t_array = np.load('/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.m_1h/t_all.npy')
crlt_obs = np.load('/scratch/seismo/joshin/pipeline-test/key_arrays_extracted/crlt_obs_10_22.npy')
rsun_obs = np.load('/scratch/seismo/joshin/pipeline-test/key_arrays_extracted/rsun_obs_10_22.npy')
# %%
t = 1608 # This is where B angle is nearly zero
lon_og = np.linspace(-90, 90, 73)
lat_og = np.linspace(-90, 90, 73)
nlat = len(lat_og)
nlng = len(lon_og)
mask_new = mask[:, 36:-35]
lats = np.linspace(-90, 90, 73)
lons = np.linspace(-90, 90, 73)
dP = 0
lng_, lat_ = np.meshgrid(lon_og, lat_og)
xdisk, ydisk = get_tan_from_lnglat(lng_.flatten(), lat_.flatten(), rsun_obs[t], np.nan_to_num(crlt_obs[t]), dP)
r = np.hypot(xdisk.reshape((nlat, nlng)), ydisk.reshape((nlat, nlng)))
clip_mask = return_clip_mask(uphi_gran[t], r, 0.99, rsun_obs[t], pad = False)


# %%
# Plot the figure

fig_path = '/data/seismo/joshin/pipeline-test/local_correlation_tracking/pdfs'

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax[0, 0].tick_params(which='both', top=True, bottom=True, right=True, left=True)
ax[0, 1].tick_params(which='both', top=True, bottom=True, right=True, left=True)
ax[1, 0].tick_params(which='both', top=True, bottom=True, right=True, left=True)
ax[1, 1].tick_params(which='both', top=True, bottom=True, right=True, left=True)
im0 = ax[0, 0].imshow(uphi_gran[t], origin='lower', cmap='bwr', vmin=-2e2, vmax=2e2, extent=[-90, 90, -90, 90])
ax[0, 0].contour(lons, lats, mask_new, levels=[0.5], colors='yellow', linewidths=2)
ax[0, 0].contour(lons, lats, clip_mask, levels=[0.5], colors='lime', linewidths=2)
ax[0, 0].set_title(r'$u_\phi$ (LCTGran)', pad = 20)
ax[0, 0].set_xlabel(r'Stonyhurst Longitude [$\degree$]')
ax[0, 0].set_ylabel(r'Latitude [$\degree$]')
ax[0, 0].set_xticks([-90, -45, 0, 45, 90])
ax[0, 0].set_yticks([-90, -45, 0, 45, 90])
ax[0, 0].set_xticks(np.arange(-90, 90, 5), minor=True)
ax[0, 0].set_yticks(np.arange(-90, 90, 5), minor=True)
ax[0, 0].tick_params(which='minor', length=4, color='gray')
ax[0, 0].tick_params(which='major', length=7, color='black')

ax[0, 1].imshow(uthe_gran[t], origin='lower', cmap='bwr', vmin=-2e2, vmax=2e2, extent=[-90, 90, -90, 90])
ax[0, 1].contour(lons, lats, mask_new, levels=[0.5], colors='yellow', linewidths=2)
ax[0, 1].contour(lons, lats, clip_mask, levels=[0.5], colors='lime', linewidths=2)
ax[0, 1].set_title(r'$u_\theta$ (LCTGran)', pad = 20)
ax[0, 1].set_xlabel(r'Stonyhurst Longitude [$\degree$]')
ax[0, 1].set_ylabel(r'Latitude [$\degree$]')
ax[0, 1].set_xticks([-90, -45, 0, 45, 90])
ax[0, 1].set_yticks([-90, -45, 0, 45, 90])
ax[0, 1].set_xticks(np.arange(-90, 90, 5), minor=True)
ax[0, 1].set_yticks(np.arange(-90, 90, 5), minor=True)
ax[0, 1].tick_params(which='minor', length=4, color='gray')
ax[0, 1].tick_params(which='major', length=7, color='black')

ax[1, 0].imshow(uphi_mf[t], origin='lower', cmap='bwr', vmin=-2e2, vmax=2e2, extent=[-90, 90, -90, 90])
ax[1, 0].contour(lons, lats, mask_new, levels=[0.5], colors='yellow', linewidths=2)
ax[1, 0].contour(lons, lats, clip_mask, levels=[0.5], colors='lime', linewidths=2)
ax[1, 0].set_title(r'$u_\phi$ (LCTMag)', pad = 20)
ax[1, 0].set_xlabel(r'Stonyhurst Longitude [$\degree$]')
ax[1, 0].set_ylabel(r'Latitude [$\degree$]')
ax[1, 0].set_xticks([-90, -45, 0, 45, 90])
ax[1, 0].set_yticks([-90, -45, 0, 45, 90])
ax[1, 0].set_xticks(np.arange(-90, 90, 5), minor=True)
ax[1, 0].set_yticks(np.arange(-90, 90, 5), minor=True)
ax[1, 0].tick_params(which='minor', length=4, color='gray')
ax[1, 0].tick_params(which='major', length=7, color='black')

ax[1, 1].imshow(uthe_mf[t], origin='lower', cmap='bwr', vmin=-2e2, vmax=2e2, extent=[-90, 90, -90, 90])
ax[1, 1].contour(lons, lats, mask_new, levels=[0.5], colors='yellow', linewidths=2)
ax[1, 1].contour(lons, lats, clip_mask, levels=[0.5], colors='lime', linewidths=2)
ax[1, 1].set_title(r'$u_\theta$ (LCTMag)', pad = 20)
ax[1, 1].set_xlabel(r'Stonyhurst Longitude [$\degree$]')
ax[1, 1].set_ylabel(r'Latitude [$\degree$]')
ax[1, 1].set_xticks([-90, -45, 0, 45, 90])
ax[1, 1].set_yticks([-90, -45, 0, 45, 90])
ax[1, 1].set_xticks(np.arange(-90, 90, 5), minor=True)
ax[1, 1].set_yticks(np.arange(-90, 90, 5), minor=True)
ax[1, 1].tick_params(which='minor', length=4, color='gray')
ax[1, 1].tick_params(which='major', length=7, color='black')

cbar_ax = fig.add_axes([1, 0.15, 0.03, 0.7])  # Position: [left, bottom, width, height]
fig.colorbar(im0, cax=cbar_ax, label=r'ms$^{-1}$')
fig.tight_layout()
fig.savefig(f'{fig_path}/u_gran_mf.pdf', bbox_inches='tight')

# %%
