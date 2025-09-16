# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'sans-serif'

# %%
# Load flow data
uphi_mag = np.load('/data/seismo/joshin/pipeline-test/paper_lct/processed_data/uphi_hmi_m_720s_dt_1h_processed.npy')
uthe_mag = np.load('/data/seismo/joshin/pipeline-test/paper_lct/processed_data/utheta_hmi_m_720s_dt_1h_processed.npy')
uphi_gran = np.load('/data/seismo/joshin/pipeline-test/paper_lct/processed_data/uphi_hmi_ic_45s_processed.npy')
uthe_gran = np.load('/data/seismo/joshin/pipeline-test/paper_lct/processed_data/utheta_hmi_ic_45s_processed.npy')
# %%


def calculate_variance(flow_array):
    """
    Calculate the variance of the flow data across the time dimension.

    Parameters:
        flow_array (numpy.ndarray): The flow data array with shape (time, lat, lon).

    Returns:
        numpy.ndarray: The variance of the flow data with shape (lat, lon).
    """
    return np.nanvar(flow_array, axis=0)
# %%
# Calculate variance for both components
uphi_mag_variance = calculate_variance(uphi_mag)
uthe_mag_variance = calculate_variance(uthe_mag)
uphi_gran_variance = calculate_variance(uphi_gran)
uthe_gran_variance = calculate_variance(uthe_gran)

# %%
lon = np.linspace(-90, 90, uphi_mag.shape[2])
lat = np.linspace(-90, 90, uphi_mag.shape[1])
# Plot the variance maps
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
c1 = axs[0, 0].pcolormesh(lon, lat, uphi_mag_variance, shading='auto', cmap='jet', vmax=100000)
fig.colorbar(c1, ax=axs[0, 0], orientation='vertical')
axs[0, 0].set_title('Variance of $u_{\\phi}$ (Magnetogram-based LCT)')
c2 = axs[0, 1].pcolormesh(lon, lat, uthe_mag_variance, shading='auto', cmap='jet', vmax=100000)
fig.colorbar(c2, ax=axs[0, 1], orientation='vertical')
axs[0, 1].set_title('Variance of $u_{\\theta}$ (Magnetogram-based LCT)')
c3 = axs[1, 0].pcolormesh(lon, lat, uphi_gran_variance, shading='auto', cmap='jet', vmax=100000)
fig.colorbar(c3, ax=axs[1, 0], orientation='vertical')
axs[1, 0].set_title('Variance of $u_{\\phi}$ (Granulation-based LCT)')
c4 = axs[1, 1].pcolormesh(lon, lat, uthe_gran_variance, shading='auto', cmap='jet', vmax=100000)
fig.colorbar(c4, ax=axs[1, 1], orientation='vertical')
axs[1, 1].set_title('Variance of $u_{\\theta}$ (Granulation-based LCT)')
plt.tight_layout()

# %%
