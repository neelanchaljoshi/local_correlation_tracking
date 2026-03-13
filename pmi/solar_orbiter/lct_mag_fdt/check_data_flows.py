# %% imports
import numpy as np
import matplotlib.pyplot as plt
import h5py

# %% Load data
file = h5py.File('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/solar_orbiter/lct_mag_fdt/data/data_2k_2015/2015.01.01_00:00:00_TAI_nt_4_dspan_360_dstep_60_dt_45_diff_rot_15deg_mag_2k.hdf5')
uphi = file['uphi'][:]
utheta = file['utheta'][:]
tstart = file['tstart'][:]
longitude = file['longitude'][:]
latitude = file['latitude'][:]
# %% Select flow index
flow_index = 0
# %% Plot the flows in subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
im1 = ax[0].pcolormesh(longitude, latitude, uphi[flow_index, :, :], cmap='bwr', vmin=-10000, vmax=10000)
fig.colorbar(im1, ax=ax[0], label='Uphi (m/s)')
ax[0].set_title(f'Uphi at time {tstart[flow_index]}')
ax[0].set_xlabel('Longitude (degrees)')
ax[0].set_ylabel('Latitude (degrees)')
im2 = ax[1].pcolormesh(longitude, latitude, utheta[flow_index, :, :], cmap='bwr', vmin=-10000, vmax=10000)
fig.colorbar(im2, ax=ax[1], label='Utheta (m/s)')
ax[1].set_title(f'Utheta at time {tstart[flow_index]}')
ax[1].set_xlabel('Longitude (degrees)')
ax[1].set_ylabel('Latitude (degrees)')
plt.tight_layout()
plt.show()
# %% Get mean of flows and plot
mean_uphi = np.nanmean(uphi, axis=0)
mean_utheta = np.nanmean(utheta, axis=0)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
im1 = ax[0].pcolormesh(longitude, latitude, mean_uphi, cmap='bwr', vmin=-10000, vmax=10000)
fig.colorbar(im1, ax=ax[0], label='Mean Uphi (m/s)')
ax[0].set_title('Mean Uphi over time')
ax[0].set_xlabel('Longitude (degrees)')
ax[0].set_ylabel('Latitude (degrees)')
im2 = ax[1].pcolormesh(longitude, latitude, mean_utheta, cmap='bwr', vmin=-10000, vmax=10000)
fig.colorbar(im2, ax=ax[1], label='Mean Utheta (m/s)')
ax[1].set_title('Mean Utheta over time')
ax[1].set_xlabel('Longitude (degrees)')
ax[1].set_ylabel('Latitude (degrees)')
plt.tight_layout()
plt.show()
# %% Close the file
