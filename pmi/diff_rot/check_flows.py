# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
# %%
def load_file(path):
    try:
        f = h5py.File(path, "r")
        return f
    except OSError as e:
        print(f"Error loading file {path}: {e}")
        return None


def get_data_arrays_from_file(path):
    f = load_file(path)
    if f is None:
        return None

    data = {}
    try:
        for key in f.keys():
            data[key] = np.array(f[key][:])
    finally:
        f.close()

    return data


# %%
# Load the data
# mag data
# files = sorted(glob.glob('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/diff_rot/data/data_mag/data_2k_2017/*.hdf5'))
files = sorted(glob.glob('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/diff_rot/data/data_mag/data_4k_2017/*.hdf5'))

# gran data
# files = sorted(glob.glob('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/diff_rot/data/data_2k_2017/*.hdf5'))
# files = sorted(glob.glob('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/diff_rot/data/data_4k_2017/*.hdf5'))


for file in files:
    data = get_data_arrays_from_file(file)
    if data is None:
        continue

    print(
        f"Loaded {file} with "
        f"uphi shape {data['uphi'].shape} and "
        f"utheta shape {data['utheta'].shape}"
    )

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
ax[0].pcolormesh(longitude, latitude, uphi_mean, cmap = 'bwr', vmax = 1000, vmin = -1000)
ax[0].set_title(r'$u_\phi$')
ax[1].pcolormesh(longitude, latitude, utheta_mean, cmap = 'bwr', vmax = 1000, vmin = -1000)
ax[1].set_title(r'$u_\theta$')
plt.show()
# %% print shapes of everything
print(uphi_all.shape)
print(utheta_all.shape)
print(longitude.shape)
print(latitude.shape)

# %%
