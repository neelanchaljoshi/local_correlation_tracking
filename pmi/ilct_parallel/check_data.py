# %%
import numpy as np
import h5py
import matplotlib.pyplot as plt
# %%
f = h5py.File('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/ilct_parallel/data/2018.01.08_00:00:00_TAI_nt_1_dspan_360_dstep_30_dt_60_diff_rot_5deg_gran_4k.hdf5')
# %%
uphi = f['uphi'][:]
utheta = f['utheta'][:]
lons = f['longitude'][:]
lats = f['latitude'][:]
tstart = f['tstart'][:]
# %%
# Plot the flow data
fig, axs = plt.subplots(2, 1, figsize=(12, 10))
im1 = axs[0].imshow(uphi[0, :, :], extent=(lons.min(), lons.max(), lats.min(), lats.max()), origin='lower', cmap='bwr', vmax = 2000, vmin = -2000)
axs[0].set_title('Zonal Flow (uphi)')
fig.colorbar(im1, ax=axs[0], orientation='vertical', label='Flow Speed')
im2 = axs[1].imshow(utheta[0, :, :], extent=(lons.min(), lons.max(), lats.min(), lats.max()), origin='lower', cmap='bwr', vmax = 2000, vmin = -2000)
axs[1].set_title('Meridional Flow (utheta)')
fig.colorbar(im2, ax=axs[1], orientation='vertical', label='Flow Speed')
plt.tight_layout()
plt.show()
# %%
# Load the npz data
debug_img = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/ilct_parallel/data_test/debug_img.npz')
debug_imgp = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/ilct_parallel/data_test/debug_imgp.npz')
# %%
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
im1 = axs[0, 0].imshow(debug_img['img1'], origin='lower', cmap='gray')
axs[0, 0].set_title('Debug Image 1')
fig.colorbar(im1, ax=axs[0, 0], orientation='vertical')
im2 = axs[0, 1].imshow(debug_img['img2'], origin='lower', cmap='gray')
axs[0, 1].set_title('Debug Image 2')
fig.colorbar(im2, ax=axs[0, 1], orientation='vertical')
im3 = axs[1, 0].imshow(debug_img['img3'], origin='lower', cmap='gray')
axs[1, 0].set_title('Debug Image 3')
fig.colorbar(im3, ax=axs[1, 0], orientation='vertical')
im4 = axs[1, 1].imshow(debug_img['img4'], origin='lower', cmap='gray')
axs[1, 1].set_title('Debug Image 4')
fig.colorbar(im4, ax=axs[1, 1], orientation='vertical')
plt.tight_layout()
# %%
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
im1 = axs[0, 0].imshow(debug_imgp['img1p'], origin='lower', cmap='gray')
axs[0, 0].set_title('Debug Imagep 1')
fig.colorbar(im1, ax=axs[0, 0], orientation='vertical')
im2 = axs[0, 1].imshow(debug_imgp['img2p'], origin='lower', cmap='gray')
axs[0, 1].set_title('Debug Imagep 2')
fig.colorbar(im2, ax=axs[0, 1], orientation='vertical')
im3 = axs[1, 0].imshow(debug_imgp['img3p'], origin='lower', cmap='gray')
axs[1, 0].set_title('Debug Imagep 3')
fig.colorbar(im3, ax=axs[1, 0], orientation='vertical')
im4 = axs[1, 1].imshow(debug_imgp['img4p'], origin='lower', cmap='gray')
axs[1, 1].set_title('Debug Imagep 4')
fig.colorbar(im4, ax=axs[1, 1], orientation='vertical')
plt.tight_layout()
# %%
# check if there is any nan in imgp
print(np.isnan(debug_imgp['img1p']).any())
print(np.isnan(debug_imgp['img2p']).any())
print(np.isnan(debug_imgp['img3p']).any())
print(np.isnan(debug_imgp['img4p']).any())
# %%
# check where the nan is in img4p
nan_indices = np.argwhere(np.isnan(debug_imgp['img4p']))
print(nan_indices)
# %%
