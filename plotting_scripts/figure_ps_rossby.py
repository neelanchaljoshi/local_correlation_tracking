# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family'] = 'sans-serif'
# %%
# Load the data for mag and gran utheta symmetric
ft_uthe_sym_mag = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_ft_2010_2024_sym_hmi_m_720s_dt_1h.npy')
ft_uthe_sym_gran = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_ft_2010_2024_sym_hmi_ic_45s_granule.npy')
# %%
# Set the figure path
fig_path = '/data/seismo/joshin/pipeline-test/local_correlation_tracking/pdfs'
# %%
def ft_to_2d_power_spectrum(ft, mode, dt):
    """
    Convert Fourier coefficients to 2D power spectrum.
    """
    M_arr = np.arange(ft.shape[2])
    freqs = np.fft.fftfreq(len(ft), d=dt)
    freqs = -np.fft.fftshift(freqs)*1e9
    lat_og = np.linspace(-90, 90, ft.shape[1])
    if mode == 'rossby':
        lat_eq = (abs(lat_og) <= 30)
    elif mode == 'highlat':
        lat_eq = (abs(lat_og) >= 45) & (abs(lat_og) <= 75)
    elif mode == 'critlat':
        lat_eq = (abs(lat_og) >= 15) & (abs(lat_og) <= 45)
    nt = len(ft)
    conv_factor = 2/nt*1e-9*dt/144/144
    power = np.nanmean(abs(ft[:, lat_eq, :])**2, axis = 1)*conv_factor
    return freqs, M_arr, power

# %%
# Calculate the 2D power spectra for mag and gran utheta
freqs, M_arr, ps_uthe_sym_mag = ft_to_2d_power_spectrum(ft_uthe_sym_mag, 'rossby', dt = 6 * 3600)
_, _, ps_uthe_sym_gran = ft_to_2d_power_spectrum(ft_uthe_sym_gran, 'rossby', dt = 6 * 3600)

# %%
# Plotting the 2D power spectra
fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
ax[0].tick_params(which = 'both', top=True, bottom=True, right=True, left=True)
im = ax[0].pcolormesh(M_arr, freqs, ps_uthe_sym_mag, cmap = 'binary', shading = 'auto',rasterized = True, vmin=0.0, vmax=0.06)
ax[0].plot(M_arr, -2*456/(M_arr+1), color = 'darkorange',label = r'$\omega = -2\Omega/(m+1)$', linewidth = 2)
ax[0].set_ylim([-500, 100])
ax[0].set_title(r'Power spectrum of $u_\theta^+$ (LCTMag)', fontsize = 14, pad = 20)
ax[0].set_ylabel('Frequency [nHz]')
ax[0].set_xlabel(r'$m$')
ax[0].set_xticks(np.arange(0, 21, 2), minor = False)
ax[0].set_xticks(np.arange(0, 20, 1), minor = True)
ax[0].set_yticks(np.arange(-500, 101, 100), minor = False)
ax[0].set_yticks(np.arange(-500, 101, 50), minor = True)
ax[0].tick_params(which='minor', length=4, color='gray')
ax[0].tick_params(which='major', length=8, color='black')
ax[0].set_xlim([0,20])
ax[0].legend()

im = ax[1].pcolormesh(M_arr, freqs, ps_uthe_sym_gran, cmap = 'binary', shading = 'auto',rasterized = True, vmin=0.0, vmax=0.06)
ax[1].tick_params(which = 'both', top=True, bottom=True, right=True, left=True)
ax[1].plot(M_arr, -2*456/(M_arr+1), color = 'darkorange',label = r'$\omega = -2\Omega/(m+1)$', linewidth = 2)
ax[1].set_ylim([-500, 100])
ax[1].set_xlabel(r'$m$')
ax[1].legend()
# ax[1].set_ylabel('Frequency [nHz]')
ax[1].set_title(r'Power spectrum of $u_\theta^+$ (LCTGran)', fontsize = 14, pad = 20)
ax[1].set_xticks(np.arange(0, 21, 2), minor = False)
ax[1].set_xticks(np.arange(0, 21, 1), minor = True)
ax[1].set_yticks(np.arange(-500, 101, 100), minor = False)
ax[1].set_yticks(np.arange(-500, 101, 50), minor = True)
ax[1].tick_params(which='minor', length=4, color='gray')
ax[1].tick_params(which='major', length=8, color='black')
ax[1].set_xlim([0,20])
cab = fig.colorbar(im, ticks = [0.0, 0.03, 0.06], shrink=0.2, aspect = 10, pad = 0.05, label = r'm$^2$s$^{-2}$nHz$^{-1}$')
fig.savefig(f'{fig_path}/ps_mag_gran_rossby.pdf', bbox_inches='tight')
# plt.tight_layout()
# %%
