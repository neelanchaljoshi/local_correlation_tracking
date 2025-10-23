# %% import
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from vorticity_func import calculate_vorticity_and_divergence

# %% Load real flow data
uphi_all_real_mag = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uphi_r_2010_2024_hmi_m_720s_dt_1h.npy')
uthe_all_real_mag = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_r_2010_2024_hmi_m_720s_dt_1h.npy')

uphi_all_real_gran = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uphi_r_2010_2024_hmi_ic_45s_granule.npy')
uthe_all_real_gran = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_r_2010_2024_hmi_ic_45s_granule.npy')

# %%
# Plot a sample of the real flow data
time_index = 100  # Example time index
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.imshow(uphi_all_real_mag[time_index], cmap='jet', vmax = 1000, vmin = -1000)
plt.title('Real Magnetic Uphi')
plt.colorbar()
plt.subplot(2, 2, 2)
plt.imshow(uthe_all_real_mag[time_index], cmap='jet', vmax = 1000, vmin = -1000)
plt.title('Real Magnetic Uthe')
plt.colorbar()
plt.subplot(2, 2, 3)
plt.imshow(uphi_all_real_gran[time_index], cmap='jet', vmax = 1000, vmin = -1000)
plt.title('Real Granular Uphi')
plt.colorbar()
plt.subplot(2, 2, 4)
plt.imshow(uthe_all_real_gran[time_index], cmap='jet', vmax = 1000, vmin = -1000)
plt.title('Real Granular Uthe')
plt.colorbar()
plt.tight_layout()
plt.show()
# %%
# Calculate vorticity and divergence for real magnetic data
rvort_all = np.zeros_like(uphi_all_real_mag, dtype = np.complex128)
hdiv_all = np.zeros_like(uphi_all_real_mag, dtype = np.complex128)
nlat = 73
nlng = 144
lmax = 35
mmax = 35
Rsun = 696e6
for k in tqdm(range(rvort_all.shape[0])):
    _, _, rvort, hdiv = calculate_vorticity_and_divergence(uphi_all_real_mag[k], uthe_all_real_mag[k], nlat, nlng, lmax, mmax, Rsun)
    rvort_all[k, :, :] = rvort
    hdiv_all[k, :, :] = hdiv
# Save the results
np.save('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/rvort_r_2010_2024_hmi_m_720s_dt_1h.npy', rvort_all)
np.save('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/hdiv_r_2010_2024_hmi_m_720s_dt_1h.npy', hdiv_all)
# %%
rvort_all = np.zeros_like(uphi_all_real_gran, dtype = np.complex128)
hdiv_all = np.zeros_like(uphi_all_real_gran, dtype = np.complex128)
nlat = 73
nlng = 144
lmax = 35
mmax = 35
Rsun = 696e6
for k in tqdm(range(rvort_all.shape[0])):
    _, _, rvort, hdiv = calculate_vorticity_and_divergence(uphi_all_real_gran[k], uthe_all_real_gran[k], nlat, nlng, lmax, mmax, Rsun)
    rvort_all[k, :, :] = rvort
    hdiv_all[k, :, :] = hdiv
# Save the results
np.save('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/rvort_r_2010_2024_hmi_ic_45s_granule.npy', rvort_all)
np.save('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/hdiv_r_2010_2024_hmi_ic_45s_granule.npy', hdiv_all)


# %%
