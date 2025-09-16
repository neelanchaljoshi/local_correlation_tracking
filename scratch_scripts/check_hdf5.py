# %%
import numpy as np
import h5py
import matplotlib.pyplot as plt
# %%
file = '/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.ic_45s/2021_ntry_3_grid_len_5_dspan_6_dstep_30_extent_73.hdf5'
f = h5py.File(file)
t = f['tstart'][()]
flow = f['uphi'][()]
lat = f['latitude'][()]
lon = f['longitude'][()]
# %%
print(t.shape, flow.shape, lat.shape, lon.shape)
# %%
print(flow.shape)
plt.figure()
plt.imshow(flow[1000], vmin=-200, vmax=200)
# %%
