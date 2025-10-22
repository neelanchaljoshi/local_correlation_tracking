# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'sans-serif'
from astropy.io import fits
import h5py

# %%
# Load the data
f = h5py.File('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/diff_rot/data/2018_dspan_360_dstep_30_dt_45_ccf_width_test_5deg_gran_4k.hdf5')
# f = h5py.File('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/diff_rot/data/2018_dspan_360_dstep_30_dt_45_ccf_widths_5deg_mag_4k.hdf5')
ccf_width_x_gran = f['ccf_width_x'][:]
ccf_width_y_gran = f['ccf_width_y'][:]
lons = f['longitude'][:]
lats = f['latitude'][:]

f = h5py.File('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/diff_rot/data/2018_dspan_360_dstep_120_dt_45_ccf_widths_5deg_mag_4k.hdf5')
ccf_width_x_mag = f['ccf_width_x'][:]
ccf_width_y_mag = f['ccf_width_y'][:]

# %%
# Change the width at each latitude from degrees to Mm on the sun
R_sun = 696340.0  # in km
lat_rad = np.radians(lats)
deg_to_km = (np.pi / 180) * R_sun * np.cos(lat_rad)  # km per degree
ccf_width_x_gran = ccf_width_x_gran * deg_to_km[:, np.newaxis] / 1000.0  # Convert to Mm
ccf_width_y_gran = ccf_width_y_gran * deg_to_km[:, np.newaxis] / 1000.0  # Convert to Mm
ccf_width_x_mag = ccf_width_x_mag * deg_to_km[:, np.newaxis] / 1000.0  # Convert to Mm
ccf_width_y_mag = ccf_width_y_mag * deg_to_km[:, np.newaxis] / 1000.0  # Convert to Mm
# %%
# Plot the data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import seaborn as sns

# Replace NaN or Inf values in the data arrays
ccf_width_x_gran = np.nan_to_num(ccf_width_x_gran, nan=0.0, posinf=0.0, neginf=0.0)
ccf_width_y_gran = np.nan_to_num(ccf_width_y_gran, nan=0.0, posinf=0.0, neginf=0.0)
ccf_width_x_mag = np.nan_to_num(ccf_width_x_mag, nan=0.0, posinf=0.0, neginf=0.0)
ccf_width_y_mag = np.nan_to_num(ccf_width_y_mag, nan=0.0, posinf=0.0, neginf=0.0)

# Define the custom colormap for granulation-based data
gran_extra_colors = ['yellow', 'orange', 'red']  # Example extra colors
gran_colors = [
    (0, 0, 0, 1),  # Black at -1
    (1, 1, 1, 1),  # White at 0
    (0, 1, 0, 1)   # Green at 2
] + [plt.cm.viridis(i) for i in np.linspace(0, 1, len(gran_extra_colors))]
gran_cmap = LinearSegmentedColormap.from_list('gran_cmap', gran_colors)

# Define the custom colormap for magnetogram-based data
mag_extra_colors = ['purple', 'pink', 'brown']  # Example extra colors
mag_colors = [
    (0, 0, 0, 1),  # Black at -1
    (1, 1, 1, 1),  # White at 0
    (0, 1, 1, 1),  # Cyan at 2.5 (midpoint of 0-5)
    (0, 0, 1, 1)   # Blue at 5
] + [plt.cm.plasma(i) for i in np.linspace(0, 1, len(mag_extra_colors))]
mag_cmap = LinearSegmentedColormap.from_list('mag_cmap', mag_colors)

# Define the boundaries for the colormaps (ensure no NaN or Inf in bounds)
gran_bounds = [-1, 0.001, 2] + list(np.linspace(2, 10, len(gran_extra_colors) + 1)[1:])
mag_bounds = [-1, 0.001, 5] + list(np.linspace(5, 10, len(mag_extra_colors) + 1)[1:])

gran_norm = BoundaryNorm(gran_bounds, gran_cmap.N)
mag_norm = BoundaryNorm(mag_bounds, mag_cmap.N)

# Plot the data
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Granulation-based plots
c1 = axs[0, 0].pcolormesh(lons, lats, ccf_width_x_mag[0], shading='auto', cmap=mag_cmap, norm=mag_norm)
fig.colorbar(c1, ax=axs[0, 0], orientation='vertical', label = 'Width (Mm)')
axs[0, 0].set_title('CCF Width X (Magnetogram-based LCT)')

c2 = axs[0, 1].pcolormesh(lons, lats, ccf_width_y_mag[0], shading='auto', cmap=mag_cmap, norm=mag_norm)
fig.colorbar(c2, ax=axs[0, 1], orientation='vertical', label = 'Width (Mm)')
axs[0, 1].set_title('CCF Width Y (Magnetogram-based LCT)')

# Magnetogram-based plots
c3 = axs[1, 0].pcolormesh(lons, lats, ccf_width_x_gran[0], shading='auto', cmap=gran_cmap, norm=gran_norm)
fig.colorbar(c3, ax=axs[1, 0], orientation='vertical', label = 'Width (Mm)')
axs[1, 0].set_title('CCF Width X (Granulation-based LCT)')

c4 = axs[1, 1].pcolormesh(lons, lats, ccf_width_y_gran[0], shading='auto', cmap=gran_cmap, norm=gran_norm)
fig.colorbar(c4, ax=axs[1, 1], orientation='vertical', label = 'Width (Mm)')
axs[1, 1].set_title('CCF Width Y (Granulation-based LCT)')

plt.tight_layout()
plt.show()
# %%
# Set the seaborn style
sns.set(style="ticks")

# Plot the data
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Set the ticks for latitude and longitude
lat_ticks = np.arange(-90, 91, 30)
lon_ticks = np.arange(-90, 91, 30)

# Granulation-based plots (top panel)
c3 = axs[0, 0].pcolormesh(lons, lats, ccf_width_x_gran[0], shading='auto', cmap="gist_ncar", vmin=0, vmax=25)
fig.colorbar(c3, ax=axs[0, 0], orientation='vertical', label='Width (Mm)')
axs[0, 0].set_title('CCF Width X (Granulation-based LCT)', fontsize=16)  # Increased font size
axs[0, 0].set_xticks(lon_ticks)
axs[0, 0].set_yticks(lat_ticks)
axs[0, 0].set_xlabel(r'Longitude [$\degree$]', fontsize=16)  # Increased font size
axs[0, 0].set_ylabel(r'Latitude [$\degree$]', fontsize=16)  # Increased font size
axs[0, 0].tick_params(axis='both', labelsize=14)  # Increased tick label size

c4 = axs[0, 1].pcolormesh(lons, lats, ccf_width_y_gran[0], shading='auto', cmap="gist_ncar", vmin=0, vmax=25)
fig.colorbar(c4, ax=axs[0, 1], orientation='vertical', label='Width (Mm)')
axs[0, 1].set_title('CCF Width Y (Granulation-based LCT)', fontsize=16)  # Increased font size
axs[0, 1].set_xticks(lon_ticks)
axs[0, 1].set_yticks(lat_ticks)
axs[0, 1].set_xlabel(r'Longitude [$\degree$]', fontsize=16)  # Increased font size
axs[0, 1].set_ylabel(r'Latitude [$\degree$]', fontsize=16)  # Increased font size
axs[0, 1].tick_params(axis='both', labelsize=14)  # Increased tick label size

# Magnetogram-based plots (bottom panel)
c1 = axs[1, 0].pcolormesh(lons, lats, ccf_width_x_mag[0], shading='auto', cmap="gist_ncar", vmin=0, vmax=25)
fig.colorbar(c1, ax=axs[1, 0], orientation='vertical', label='Width (Mm)')
axs[1, 0].set_title('CCF Width X (Magnetogram-based LCT)', fontsize=16)  # Increased font size
axs[1, 0].set_xticks(lon_ticks)
axs[1, 0].set_yticks(lat_ticks)
axs[1, 0].set_xlabel(r'Longitude [$\degree$]', fontsize=16)  # Increased font size
axs[1, 0].set_ylabel(r'Latitude [$\degree$]', fontsize=16)  # Increased font size
axs[1, 0].tick_params(axis='both', labelsize=14)  # Increased tick label size

c2 = axs[1, 1].pcolormesh(lons, lats, ccf_width_y_mag[0], shading='auto', cmap="gist_ncar", vmin=0, vmax=25)
fig.colorbar(c2, ax=axs[1, 1], orientation='vertical', label='Width (Mm)')
axs[1, 1].set_title('CCF Width Y (Magnetogram-based LCT)', fontsize=16)  # Increased font size
axs[1, 1].set_xticks(lon_ticks)
axs[1, 1].set_yticks(lat_ticks)
axs[1, 1].set_xlabel(r'Longitude [$\degree$]', fontsize=16)  # Increased font size
axs[1, 1].set_ylabel(r'Latitude [$\degree$]', fontsize=16)  # Increased font size
axs[1, 1].tick_params(axis='both', labelsize=14)  # Increased tick label size

plt.tight_layout()
plt.savefig('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pdfs/ccf_widths_comparison.pdf')
plt.show()
# %%
