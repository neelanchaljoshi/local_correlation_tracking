# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family'] = 'sans-serif'
# %%
# Load the data for mag and gran uphi, sym and anti
ft_uphi_sym_mag = np.load('/data/seismo/joshin/pipeline-test/lct_data_processed/uphi_fourier_hmi_m_720s_dt_1h_sym_2010_2023.npy')
ft_uphi_anti_mag = np.load('/data/seismo/joshin/pipeline-test/lct_data_processed/uphi_fourier_hmi_m_720s_dt_1h_anti_2010_2023.npy')

ft_uphi_sym_gran = np.load('/data/seismo/joshin/pipeline-test/lct_data_processed/uphi_fourier_hmi_ic_45s_sym_2010_2023.npy')
ft_uphi_anti_gran = np.load('/data/seismo/joshin/pipeline-test/lct_data_processed/uphi_fourier_hmi_ic_45s_anti_2010_2023.npy')
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
freqs, M_arr, ps_uphi_sym_mag = ft_to_2d_power_spectrum(ft_uphi_sym_mag, 'highlat', dt = 6 * 3600)
_, _, ps_uphi_anti_mag = ft_to_2d_power_spectrum(ft_uphi_anti_mag, 'highlat', dt = 6 * 3600)
freqs, M_arr, ps_uphi_sym_gran = ft_to_2d_power_spectrum(ft_uphi_sym_gran, 'highlat', dt = 6 * 3600)
_, _, ps_uphi_anti_gran = ft_to_2d_power_spectrum(ft_uphi_anti_gran, 'highlat', dt = 6 * 3600)
# %%
# Plotting the 2D power spectra for the uphi symmetric components
x1, y1 = 0, 0
x2, y2 = 1, -88
# Compute direction vector
dx = x2 - x1
dy = y2 - y1

# Extension factor (change this as needed)
scale = 10  # Extend beyond the original points

# Compute new extended points
x1_new = x1 - scale * dx
y1_new = y1 - scale * dy
x2_new = x2 + scale * dx
y2_new = y2 + scale * dy

fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
ax[0].tick_params(which = 'both', top=True, bottom=True, right=True, left=True) 
ax[0].pcolormesh(M_arr, freqs, ps_uphi_sym_mag, cmap = 'binary', shading = 'auto',rasterized = True, norm = mpl.colors.LogNorm(vmin=0.05, vmax=0.4))
ax[0].plot([x1_new, x2_new], [y1_new, y2_new], color = 'green', linewidth = 2)
ax[0].plot([1, 2, 3, 5], [-86, -171, -224, -282], 'ro', markersize = 5, color = 'red')

ax[0].set_ylim([-500, 100])
ax[0].set_title(r'Power spectrum of $u_\phi^+$ (LCTMag)', fontsize = 14, pad = 20)
ax[0].set_ylabel('Frequency [nHz]')
ax[0].set_xlabel(r'm')
ax[0].set_xticks(np.arange(0, 20, 1), minor = True)
ax[0].set_yticks(np.arange(-500, 101, 100), minor = False)
ax[0].set_yticks(np.arange(-500, 101, 50), minor = True)
ax[0].tick_params(which='minor', length=4, color='gray')
ax[0].tick_params(which='major', length=8, color='black')
ax[0].set_xlim([0,10])

im = ax[1].pcolormesh(M_arr, freqs, ps_uphi_sym_gran, cmap = 'binary', shading = 'auto',rasterized = True, norm = mpl.colors.LogNorm(vmin=0.05, vmax=0.4))
ax[1].tick_params(which = 'both', top=True, bottom=True, right=True, left=True) 
ax[1].plot([x1_new, x2_new], [y1_new, y2_new], color = 'green', linewidth = 2)
ax[1].plot([1, 2, 3, 5], [-86, -171, -224, -282], 'ro', markersize = 5, color = 'red')
ax[1].set_ylim([-500, 100])
ax[1].set_xlabel(r'm')
# ax[1].set_ylabel('Frequency [nHz]')
ax[1].set_title(r'Power spectrum of $u_\phi^+$ (LCTGran)', fontsize = 14, pad = 20)
ax[1].set_xticks(np.arange(0, 20, 1), minor = True)
ax[1].set_yticks(np.arange(-500, 101, 100), minor = False)
ax[1].set_yticks(np.arange(-500, 101, 50), minor = True)
ax[1].tick_params(which='minor', length=4, color='gray')
ax[1].tick_params(which='major', length=8, color='black')
ax[1].set_xlim([0,10])
cab = fig.colorbar(im, ticks = [0.1, 0.4], shrink=0.3, aspect = 10, pad = 0.05, label = r'm$^2$s$^{-2}$nHz$^{-1}$', norm = mpl.colors.LogNorm(vmin=0.05, vmax=0.4))
fig.savefig(f'{fig_path}/ps_mag_gran_highlat_sym.pdf', bbox_inches='tight')
# %%
# Plotting the 2D power spectra for the uphi anti-symmetric components
fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True) 
ax[0].pcolormesh(M_arr, freqs, ps_uphi_anti_mag, cmap = 'binary', shading = 'auto',rasterized = True, norm = mpl.colors.LogNorm(vmin=0.05, vmax=0.4))
ax[0].tick_params(which = 'both', top=True, bottom=True, right=True, left=True) 
ax[0].plot([x1_new, x2_new], [y1_new, y2_new], color = 'green', linewidth = 2)
ax[0].plot([1, 2, 4, 4, 5], [-86, -151, -294, -245, -282], 'ro', markersize = 5, color = 'red')
ax[0].plot([3], [-190], markersize = 10, color = 'green', marker = '+', markeredgewidth = 2)
ax[0].plot([3], [-265], markersize = 10, color = 'blue', marker = 'x', markeredgewidth = 2)
ax[0].set_ylim([-500, 100])
ax[0].set_title(r'Power spectrum of $u_\phi^-$ (LCTMag)', fontsize = 14, pad = 20)
ax[0].set_ylabel('Frequency [nHz]')
ax[0].set_xlabel(r'm')
ax[0].set_xticks(np.arange(0, 20, 1), minor = True)
ax[0].set_yticks(np.arange(-500, 101, 100), minor = False)
ax[0].set_yticks(np.arange(-500, 101, 50), minor = True)
ax[0].tick_params(which='minor', length=4, color='gray')
ax[0].tick_params(which='major', length=8, color='black')
ax[0].set_xlim([0,10])

im = ax[1].pcolormesh(M_arr, freqs, ps_uphi_anti_gran, cmap = 'binary', shading = 'auto',rasterized = True, norm = mpl.colors.LogNorm(vmin=0.05, vmax=0.4))
ax[1].tick_params(which = 'both', top=True, bottom=True, right=True, left=True) 
ax[1].plot([x1_new, x2_new], [y1_new, y2_new], color = 'green', linewidth = 2)
ax[1].plot([1, 2, 4, 4, 5], [-86, -151, -294, -245, -282], 'ro', markersize = 5, color = 'red')
ax[1].plot([3], [-190], markersize = 10, color = 'green', marker = '+', markeredgewidth = 2)
ax[1].plot([3], [-265], markersize = 10, color = 'blue', marker = 'x', markeredgewidth = 2)
ax[1].set_ylim([-500, 100])
ax[1].set_xlabel(r'm')
# ax[1].set_ylabel('Frequency [nHz]')
ax[1].set_title(r'Power spectrum of $u_\phi^-$ (LCTGran)', fontsize = 14, pad = 20)
ax[1].set_xticks(np.arange(0, 20, 1), minor = True)
ax[1].set_yticks(np.arange(-500, 101, 100), minor = False)
ax[1].set_yticks(np.arange(-500, 101, 50), minor = True)
ax[1].tick_params(which='minor', length=4, color='gray')
ax[1].tick_params(which='major', length=8, color='black')
ax[1].set_xlim([0,10])
cab = fig.colorbar(im, ticks = [0.1, 0.4], shrink=0.3, aspect = 10, pad = 0.05, label = r'm$^2$s$^{-2}$nHz$^{-1}$', norm = mpl.colors.LogNorm(vmin=0.05, vmax=0.4))
fig.savefig(f'{fig_path}/ps_mag_gran_highlat_anti.pdf', bbox_inches='tight')

# %%
