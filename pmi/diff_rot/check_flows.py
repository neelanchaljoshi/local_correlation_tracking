# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
# %%
def load_file(path):
    f = h5py.File(path)
    return f

def get_data_arrays_from_file(path):
    f = load_file(path)
    data = {}
    for key in f.keys():
        data[key] = np.array(f[key][:])
    return data

# %%
# Load the data
files = sorted(glob.glob('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/diff_rot/data/data_2017/*.hdf5'))
for file in files:
    data = get_data_arrays_from_file(file)
    print(f"Loaded {file} with uphi shape {data['uphi'].shape} and utheta shape {data['utheta'].shape}")
    # concatenate the uphi and utheta arrays along the time axis
    if 'uphi_all' not in locals():
        uphi_all = data['uphi']
        utheta_all = data['utheta']
    else:
        uphi_all = np.concatenate((uphi_all, data['uphi']), axis=0)
        utheta_all = np.concatenate((utheta_all, data['utheta']), axis=0)

uphi_mean = np.nanmean(uphi_all, axis=0)
utheta_mean = np.nanmean(utheta_all, axis=0)
longitude = data['longitude']
latitude = data['latitude']
# %%
# Plot the mean flows
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(uphi_mean, cmap = 'bwr', vmax = 1000, vmin = -1000, origin='lower')
ax[0].set_title(r'$u_\phi$')
ax[1].imshow(utheta_mean, cmap = 'bwr', vmax = 1000, vmin = -1000, origin='lower')
ax[1].set_title(r'$u_\theta$')
plt.show()
# %%
# make cut through the centre of the mean flows
uphi_cut = uphi_mean[:, uphi_mean.shape[1]//2]
utheta_cut = utheta_mean[utheta_mean.shape[0]//2, :]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(longitude, uphi_cut, label=r'$u_\phi$')
# ax.plot(longitude, utheta_cut, label=r'$u_\theta$')
ax.set_xlabel('Longitude (degrees)')
ax.set_ylabel('Velocity (m/s)')
ax.set_title('Cut through the centre of the mean flows')
ax.legend()
# %%
