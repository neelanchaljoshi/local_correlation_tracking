# %% import
import numpy as np
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import shtns
from tqdm import tqdm
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'sans-serif'
from scipy.ndimage import gaussian_filter1d

# %% Functions
def rebin_2d_vertical(arr, bin_factor):
    assert arr.shape[0] % bin_factor == 0, "Number of rows must be divisible by the bin factor"
    return np.median(arr.reshape(arr.shape[0] // bin_factor, bin_factor, arr.shape[1]), axis=1)
def rebin_1d(arr, bin_factor):
    assert len(arr) % bin_factor == 0, "Array length must be divisible by the bin factor"
    return np.median(arr.reshape(len(arr) // bin_factor, bin_factor), axis=1)
def smooth_array_gaussian(arr, sigma=2):
    """
    Smooths a 2D array along the first axis using a Gaussian filter.

    Parameters:
        arr (ndarray): Input array of shape (18000, 34).
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        ndarray: Smoothed array.
    """
    return gaussian_filter1d(arr, sigma=sigma, axis=0)


def smooth_array(arr, window_size=5, mode='nearest'):
    """
    Smooths a 2D array along the first axis using a moving average filter.

    Parameters:
        arr (ndarray): Input array of shape (18000, 34).
        window_size (int): Size of the moving average window (should be odd for symmetry).
        mode (str): Boundary mode for convolution (e.g., 'nearest', 'mirror', 'constant').

    Returns:
        ndarray: Smoothed array of the same shape as input.
    """
    kernel = np.ones(window_size) / window_size
    smoothed_arr = convolve1d(arr, kernel, axis=0, mode=mode)
    return smoothed_arr


# %% Load zlm data
zlm_uphi = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/zlm_uphi_r_2010_2024_hmi_m_720s_dt_1h.npy')
zlm_uthe = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/zlm_uthe_r_2010_2024_hmi_m_720s_dt_1h.npy')
zlm_rvort = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/zlm_rvort_r_2010_2024_hmi_m_720s_dt_1h.npy')
zlm_hdiv = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/zlm_hdiv_r_2010_2024_hmi_m_720s_dt_1h.npy')

# %%  Make np arrays
zlm_uphi_array = np.asarray(zlm_uphi, dtype=np.complex128)
zlm_uthe_array = np.asarray(zlm_uthe, dtype=np.complex128)
zlm_hdiv_array = np.asarray(zlm_hdiv, dtype=np.complex128)
zlm_rvort_array = np.asarray(zlm_rvort, dtype=np.complex128)

# %% Set params
lmax = 35
mmax = 35
sh = shtns.sht(lmax, mmax)

# %% FFT
freqs = np.fft.fftfreq(zlm_uphi_array.shape[0], d=6*3600)
freqs_shifted = -np.fft.fftshift(freqs)*1e9
zlm_uphi_ft = np.fft.fft(zlm_uphi_array, axis=0)
zlm_uthe_ft = np.fft.fft(zlm_uthe_array, axis=0)
zlm_hdiv_ft = np.fft.fft(zlm_hdiv_array, axis=0)
zlm_rvort_ft = np.fft.fft(zlm_rvort_array, axis=0)

# %% Compute power spectra for each m
flist_rvort = []
flist_uthe = []
flist_hdiv = []
for m in range(100):
    try:
        flist_rvort.append(zlm_rvort_ft[:, sh.idx(m+1, m)])
        flist_uthe.append(zlm_uthe_ft[:, sh.idx(m+1, m)])
        flist_hdiv.append(zlm_hdiv_ft[:, sh.idx(m, m)])
    except:
        break

flist_rvort = np.asarray(flist_rvort).T
flist_uthe = np.asarray(flist_uthe).T
flist_hdiv = np.asarray(flist_hdiv).T
flist_rvort_shifted = np.fft.fftshift(flist_rvort, axes=0)
flist_uthe_shifted = np.fft.fftshift(flist_uthe, axes=0)
flist_hdiv_shifted = np.fft.fftshift(flist_hdiv, axes=0)
power_rvort = np.abs(flist_rvort_shifted)**2
power_uthe = np.abs(flist_uthe_shifted)**2
power_hdiv = np.abs(flist_hdiv_shifted)**2
frqs_binned = rebin_1d(freqs_shifted, 1)
power_rvort_binned = rebin_2d_vertical(power_rvort, 1)
power_uthe_binned = rebin_2d_vertical(power_uthe, 1)
power_hdiv_binned = rebin_2d_vertical(power_hdiv, 1)
M_arr = np.arange(0,35)
print(power_uthe_binned.shape)
freqs_for_median = (frqs_binned>-400) & (frqs_binned<-50)
power_median_uthe = np.nanmedian(power_uthe_binned[freqs_for_median, :], axis=0)
power_median_rvort = np.nanmedian(power_rvort_binned[freqs_for_median, :], axis=0)
power_median_hdiv = np.nanmedian(power_hdiv_binned[freqs_for_median, :], axis=0)
print(power_median_uthe.shape)
for m in range(35):
    power_uthe_binned[:, m] = power_uthe_binned[:, m]/power_median_uthe[m]
    power_rvort_binned[:, m] = power_rvort_binned[:, m]/power_median_rvort[m]
    power_hdiv_binned[:, m] = power_hdiv_binned[:, m]/power_median_hdiv[m]
power_uthe_binned = smooth_array(power_uthe_binned, window_size=5)
power_rvort_binned = smooth_array(power_rvort_binned, window_size=5)
power_hdiv_binned = smooth_array(power_hdiv_binned, window_size=5)


# %% Plot
fig, ax = plt.subplots(1, 3, figsize = (10, 5))
im = ax[0].pcolormesh(np.arange(flist_rvort_shifted.shape[1]), frqs_binned, power_rvort_binned, cmap = 'binary', vmax = 3, rasterized = True)
# im = ax[0].pcolormesh(np.arange(flist_rvort_shifted.shape[1]), frqs_binned, power_rvort_binned, cmap = 'binary', rasterized = True, vmax = 5e-9)
ax[0].plot(M_arr, -2*456/(M_arr+1), 'darkorange',label = r'$\omega = -2\Omega/(m+1)$')
ax[0].plot(M_arr, -6.5*456/(M_arr+1), 'cyan',label = r'$\omega = -6.5\Omega/(m+1)$')
ax[0].set_ylim(-400, 100)
ax[0].set_xlim(0, 35)
ax[0].set_xticks(np.arange(0, 34, 1), minor = True)
ax[0].set_yticks(np.arange(-400, 101, 100), minor = False)
ax[0].set_yticks(np.arange(-400, 101, 50), minor = True)
ax[0].tick_params(which='minor', length=4, color='gray')
ax[0].tick_params(which='major', length=8, color='black')
ax[0].set_title(r'Power Spectrum of $\zeta_\mathrm{r}$')
ax[0].set_xlabel(r'$m$')
ax[0].set_ylabel('Frequency [nHz]')
im = ax[2].pcolormesh(np.arange(flist_uthe_shifted.shape[1]), frqs_binned, power_uthe_binned, cmap = 'binary', vmax = 3, rasterized = True)
# im = ax[2].pcolormesh(np.arange(flist_uthe_shifted.shape[1]), frqs_binned, power_uthe_binned, cmap = 'binary', rasterized = True, vmax = 1e7)
ax[2].plot(M_arr, -2*456/(M_arr+1), 'darkorange',label = r'$\omega = -2\Omega/(m+1)$')
ax[2].plot(M_arr, -6.5*456/(M_arr+1), 'cyan',label = r'$\omega = -6.5\Omega/(m+1)$')
ax[2].set_ylim(-400, 100)
ax[2].set_xlim(0, 35)
ax[2].set_xticks(np.arange(0, 34, 1), minor = True)
ax[2].set_yticks(np.arange(-400, 101, 100), minor = False)
ax[2].set_yticks(np.arange(-400, 101, 50), minor = True)
ax[2].tick_params(which='minor', length=4, color='gray')
ax[2].tick_params(which='major', length=8, color='black')
ax[2].set_title(r'Power Spectrum of $u_{\theta}$')
ax[2].set_xlabel(r'$m$')
im = ax[1].pcolormesh(np.arange(flist_uthe_shifted.shape[1]), frqs_binned, power_hdiv_binned, cmap = 'binary', vmax = 3, rasterized = True)
# im = ax[1].pcolormesh(np.arange(flist_uthe_shifted.shape[1]), frqs_binned, power_hdiv_binned, cmap = 'binary', rasterized = True, vmax = 5e-9)
ax[1].plot(M_arr, -2*456/(M_arr+1), 'darkorange',label = r'$\omega = -2\Omega/(m+1)$')
ax[1].plot(M_arr, -6.5*456/(M_arr+1), 'cyan',label = r'$\omega = -6.5\Omega/(m+1)$')
ax[1].set_ylim(-400, 100)
ax[1].set_xlim(0, 35)
ax[1].set_xticks(np.arange(0, 34, 1), minor = True)
ax[1].set_yticks(np.arange(-400, 101, 100), minor = False)
ax[1].set_yticks(np.arange(-400, 101, 50), minor = True)
ax[1].tick_params(which='minor', length=4, color='gray')
ax[1].tick_params(which='major', length=8, color='black')
ax[1].set_title(r'Power Spectrum of $(\nabla \cdot \mathbf{u})_{\mathrm{h}}$')
ax[1].set_xlabel(r'$m$')
fig.tight_layout()
# fig.savefig('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pdfs/hfr_ps_rvort_uthe_hdiv.pdf', bbox_inches = 'tight')

# %%
