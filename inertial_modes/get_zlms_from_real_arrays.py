# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import shtns
from tqdm import tqdm

# %%
def get_zlms_for_array(array, lmax, mmax, nlat, nlng, isvort = False):
    """
    Project the input array onto the sectoral zonal and meridional modes (zlm) up to a given lmax and mmax.

    Parameters:
        array (ndarray): Input array to be projected.
        lmax (int): Maximum degree of the sectoral modes.
        mmax (int): Maximum order of the sectoral modes.
        nlat (int): Number of latitudes in the grid.
        nlng (int): Number of longitudes in the grid.

    Returns:
        ndarray: Array projected onto the sectoral zonal and meridional modes.
    """
    array = np.flip(array, axis=1)  # flip the array in the latitude direction
    l_array = np.arange(0, lmax+1)
    sh = shtns.sht(lmax, mmax)
    # set the grid
    nlat, nphi = sh.set_grid(nlat, nlng, 5, 1e-20) # flag is set to 2 to indicate that the grid is equally spaced in theta

    # calculate the spectral coefficients
    # analys gives the spectral coefficients of the input field
    zlm_array = []
    for k in tqdm(range(array.shape[0])):
        if isvort:
            zlm_array.append(sh.analys(np.clip(np.nan_to_num(array[k]), a_max = 50, a_min = -50)))
        else:
            zlm_array.append(sh.analys(np.clip(np.nan_to_num(array[k]), a_max = 5000, a_min = -5000)))
        # print(zlm)
    zlm_array = np.asarray(zlm_array)
    # print(zlm_array[:10, 0])
    return zlm_array

# %%
# Load the real flow data
uphi_mag = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uphi_r_2010_2024_hmi_m_720s_dt_1h.npy')
uthe_mag = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_r_2010_2024_hmi_m_720s_dt_1h.npy')
uphi_gran = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uphi_r_2010_2024_hmi_ic_45s_granule.npy')
uthe_gran = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_r_2010_2024_hmi_ic_45s_granule.npy')

# %%
# Define parameters
nlat = 73
nlng = 144
lmax = 35
mmax = 35
# %%
# Get zlms for magnetic data
zlm_uphi_mag = get_zlms_for_array(uphi_mag, lmax, mmax, nlat, nlng, isvort = False)
zlm_uthe_mag = get_zlms_for_array(uthe_mag, lmax, mmax, nlat, nlng, isvort = False)
print(zlm_uphi_mag.shape)
print(zlm_uthe_mag.shape)
# %%Save the results
np.save('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/zlm_uphi_r_2010_2024_hmi_m_720s_dt_1h.npy', zlm_uphi_mag)
np.save('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/zlm_uthe_r_2010_2024_hmi_m_720s_dt_1h.npy', zlm_uthe_mag)
# %%
# Get zlms for granular data
zlm_uphi_gran = get_zlms_for_array(uphi_gran, lmax, mmax, nlat, nlng, isvort = False)
zlm_uthe_gran = get_zlms_for_array(uthe_gran, lmax, mmax, nlat, nlng, isvort = False)
print(zlm_uphi_gran.shape)
print(zlm_uthe_gran.shape)
# %% Save the results
np.save('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/zlm_uphi_r_2010_2024_hmi_ic_45s_granule.npy', zlm_uphi_gran)
np.save('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/zlm_uthe_r_2010_2024_hmi_ic_45s_granule.npy', zlm_uthe_gran)
# %%
# Load the rvort and hdiv data
rvort_mag = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/rvort_r_2010_2024_hmi_m_720s_dt_1h.npy')
hdiv_mag = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/hdiv_r_2010_2024_hmi_m_720s_dt_1h.npy')
rvort_gran = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/rvort_r_2010_2024_hmi_ic_45s_granule.npy')
hdiv_gran = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/hdiv_r_2010_2024_hmi_ic_45s_granule.npy')
# %%
# Get zlms for magnetic rvort and hdiv data
zlm_rvort_mag = get_zlms_for_array(rvort_mag.real, lmax, mmax, nlat, nlng, isvort = True)
zlm_hdiv_mag = get_zlms_for_array(hdiv_mag.real, lmax, mmax, nlat, nlng, isvort = False)
print(zlm_rvort_mag.shape)
print(zlm_hdiv_mag.shape)
# %% Save the results
np.save('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/zlm_rvort_r_2010_2024_hmi_m_720s_dt_1h.npy', zlm_rvort_mag)
np.save('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/zlm_hdiv_r_2010_2024_hmi_m_720s_dt_1h.npy', zlm_hdiv_mag)
# %%
# Get zlms for granular rvort and hdiv data
zlm_rvort_gran = get_zlms_for_array(rvort_gran.real, lmax, mmax, nlat, nlng, isvort = True)
zlm_hdiv_gran = get_zlms_for_array(hdiv_gran.real, lmax, mmax, nlat, nlng, isvort = False)
print(zlm_rvort_gran.shape)
print(zlm_hdiv_gran.shape)
# %% Save the results
np.save('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/zlm_rvort_r_2010_2024_hmi_ic_45s_granule.npy', zlm_rvort_gran)
np.save('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/zlm_hdiv_r_2010_2024_hmi_ic_45s_granule.npy', zlm_hdiv_gran)

# %%
