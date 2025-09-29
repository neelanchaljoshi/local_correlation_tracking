# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'sans-serif'
from astropy.io import fits

# %%
# Load flow data
uphi_mag = np.load('/data/seismo/joshin/pipeline-test/paper_lct/processed_data/uphi_hmi_m_720s_dt_1h_processed.npy')
uthe_mag = np.load('/data/seismo/joshin/pipeline-test/paper_lct/processed_data/utheta_hmi_m_720s_dt_1h_processed.npy')
uphi_gran = np.load('/data/seismo/joshin/pipeline-test/paper_lct/processed_data/uphi_hmi_ic_45s_processed.npy')
uthe_gran = np.load('/data/seismo/joshin/pipeline-test/paper_lct/processed_data/utheta_hmi_ic_45s_processed.npy')
inp= '/scratch/seismo/mandowara/HMIRDA05'
rdv= np.load(inp+'/metadata.npz')
uphi_rda = fits.getdata(inp+'/BBux_stony_intp.fits')
uthe_rda = fits.getdata(inp+'/BBuy_stony_intp.fits')
uphi_rda = uphi_rda[:, 0, :, :]
uthe_rda = uthe_rda[:, 0, :, :]
t  = rdv['t_mid_dec']

# %%

t_rec = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/t_rec.npy')
# index of 2019-01-01 00:00:00
start_index = np.where(t_rec == np.array(['2018.11.20_00:00:00_TAI'], dtype = 'S32'))[0][0]
print("Start index for 2018-12-01 is ", start_index)
end_index = np.where(t_rec == np.array(['2019.01.01_00:00:00_TAI'], dtype = 'S32'))[0][0]
print("End index for 2019-01-01 is ", end_index)

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
uphi_mag_variance = calculate_variance(uphi_mag[start_index:end_index])
uthe_mag_variance = calculate_variance(uthe_mag[start_index:end_index])
uphi_gran_variance = calculate_variance(uphi_gran[start_index:end_index])
uthe_gran_variance = calculate_variance(uthe_gran[start_index:end_index])
uphi_rda

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
# cut through the central meridian
central_lon_index = lon.size // 2
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(lat, np.sqrt(uphi_mag_variance[:, central_lon_index]), label='$u_{\\phi}$ (Magnetogram-based LCT)', color='blue')
ax.semilogy(lat, np.sqrt(uthe_mag_variance[:, central_lon_index]), label='$u_{\\theta}$ (Magnetogram-based LCT)', color='orange')
ax.semilogy(lat, np.sqrt(uphi_gran_variance[:, central_lon_index]), label='$u_{\\phi}$ (Granulation-based LCT)', color='green')
ax.semilogy(lat, np.sqrt(uthe_gran_variance[:, central_lon_index]), label='$u_{\\theta}$ (Granulation-based LCT)', color='red')
ax.set_xlabel('Latitude (degrees)')
ax.set_ylabel('Std Dev (m/s)')
ax.set_title('Variance Cut Through Central Meridian (Longitude = 0°)')
ax.set_ylim(-10, 1000)
ax.legend()

# %%
