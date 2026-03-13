# %% imports
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.titlesize'] = 18

# %% Load data mag
uphi_ft_anti_mag = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uphi_ft_2010_2024_anti_hmi_m_720s_dt_1h.npy')
uthe_ft_anti_mag = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_ft_2010_2024_anti_hmi_m_720s_dt_1h.npy')
uphi_ft_sym_mag = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uphi_ft_2010_2024_sym_hmi_m_720s_dt_1h.npy')
uthe_ft_sym_mag = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_ft_2010_2024_sym_hmi_m_720s_dt_1h.npy')

# %% Load data gran
uphi_ft_anti_gran = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uphi_ft_2010_2024_anti_hmi_ic_45s_granule.npy')
uthe_ft_anti_gran = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_ft_2010_2024_anti_hmi_ic_45s_granule.npy')
uphi_ft_sym_gran = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uphi_ft_2010_2024_sym_hmi_ic_45s_granule.npy')
uthe_ft_sym_gran = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_ft_2010_2024_sym_hmi_ic_45s_granule.npy')

# %%
lat_og = np.linspace(-90, 90, 73)
M_arr = np.arange(uphi_ft_anti_mag.shape[2])
freqs = np.fft.fftfreq(len(uphi_ft_anti_mag), d=6*3600)
freqs = -np.fft.fftshift(freqs)*1e9
# mode = 'highlat'
# if mode == 'rossby':
lat_eq_rossby = (abs(lat_og) <= 30)
# elif mode == 'highlat':
lat_eq_hl = (abs(lat_og) >= 45) & (abs(lat_og) <= 75)
# elif mode == 'critlat':
lat_eq_cl = (abs(lat_og) >= 15) & (abs(lat_og) <= 45)
nt = len(uphi_ft_anti_mag)
dt = 6.*3600
print('Number of time steps: {}'.format(nt))
conv_factor = 2/nt*1e-9*dt/144/144

# %% Get 1d mag powers
power_uphi_m1_mag = np.nanmean(abs(uphi_ft_anti_mag[:, lat_eq_hl, :])**2, axis = 1)*conv_factor
power_uthe_m1_mag = np.nanmean(abs(uthe_ft_sym_mag[:, lat_eq_hl, :])**2, axis = 1)*conv_factor
power_uphi_m2_mag = np.nanmean(abs(uphi_ft_anti_mag[:, lat_eq_cl, :])**2, axis = 1)*conv_factor
power_uthe_m2_mag = np.nanmean(abs(uthe_ft_sym_mag[:, lat_eq_cl, :])**2, axis = 1)*conv_factor
power_uphi_m3_mag = np.nanmean(abs(uphi_ft_anti_mag[:, lat_eq_rossby, :])**2, axis = 1)*conv_factor
power_uthe_m3_mag = np.nanmean(abs(uthe_ft_sym_mag[:, lat_eq_rossby, :])**2, axis = 1)*conv_factor
power_uphi_m8_mag = np.nanmean(abs(uphi_ft_anti_mag[:, lat_eq_rossby, :])**2, axis = 1)*conv_factor
power_uthe_m8_mag = np.nanmean(abs(uthe_ft_sym_mag[:, lat_eq_rossby, :])**2, axis = 1)*conv_factor
power_uphi_m13_mag = np.nanmean(abs(uphi_ft_sym_mag[:, lat_eq_rossby, :])**2, axis = 1)*conv_factor
power_uthe_m13_mag = np.nanmean(abs(uthe_ft_anti_mag[:, lat_eq_rossby, :])**2, axis = 1)*conv_factor

# %% Get 1d gran powers
power_uphi_m1_gran = np.nanmean(abs(uphi_ft_anti_gran[:, lat_eq_hl, :])**2, axis = 1)*conv_factor
power_uthe_m1_gran = np.nanmean(abs(uthe_ft_sym_gran[:, lat_eq_hl, :])**2, axis = 1)*conv_factor
power_uphi_m2_gran = np.nanmean(abs(uphi_ft_anti_gran[:, lat_eq_cl, :])**2, axis = 1)*conv_factor
power_uthe_m2_gran = np.nanmean(abs(uthe_ft_sym_gran[:, lat_eq_cl, :])**2, axis = 1)*conv_factor
power_uphi_m3_gran = np.nanmean(abs(uphi_ft_anti_gran[:, lat_eq_rossby, :])**2, axis = 1)*conv_factor
power_uthe_m3_gran = np.nanmean(abs(uthe_ft_sym_gran[:, lat_eq_rossby, :])**2, axis = 1)*conv_factor
power_uphi_m8_gran = np.nanmean(abs(uphi_ft_anti_gran[:, lat_eq_rossby, :])**2, axis = 1)*conv_factor
power_uthe_m8_gran = np.nanmean(abs(uthe_ft_sym_gran[:, lat_eq_rossby, :])**2, axis = 1)*conv_factor
power_uphi_m13_gran = np.nanmean(abs(uphi_ft_sym_gran[:, lat_eq_rossby, :])**2, axis = 1)*conv_factor
power_uthe_m13_gran = np.nanmean(abs(uthe_ft_anti_gran[:, lat_eq_rossby, :])**2, axis = 1)*conv_factor


# %% Combined Plot
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.5, hspace=0.5)

# LCTMag Plots
ax[0, 0] = plt.subplot(gs[0, 0])
ax[0, 1] = plt.subplot(gs[0, 1])
ax[0, 2] = plt.subplot(gs[0, 2])
ax[0, 3] = plt.subplot(gs[0, 3])

ax[0, 0].plot(freqs, power_uphi_m1_mag[:, 1], label=r'$u_{\phi}$', color='darkblue')
ax[0, 0].plot(freqs, power_uthe_m1_mag[:, 1], label=r'$u_{\theta}$', color='darkorange')
ax[0, 0].axvline(x=-88, color='red', linestyle='--')
ax[0, 0].set_title('LCTMag m=1')
ax[0, 0].set_xlim([-200, 0])
ax[0, 0].legend(loc='upper right')
ax[0, 0].set_xlabel('Frequency (nHz)')
ax[0, 0].set_ylabel(r'Power [$m^2/s^2/nHz$]')
ax[0, 0].grid()

ax[0, 1].plot(freqs, power_uphi_m2_mag[:, 2], label=r'$u_{\phi}$', color='darkblue')
ax[0, 1].plot(freqs, power_uthe_m2_mag[:, 2], label=r'$u_{\theta}$', color='darkorange')
ax[0, 1].axvline(x=-73, color='red', linestyle='--')
ax[0, 1].set_title('LCTMag m=2')
ax[0, 1].set_xlim([-200, 50])
ax[0, 1].set_ylim([0, 0.02])
ax[0, 1].legend()
ax[0, 1].set_xlabel('Frequency (nHz)')
ax[0, 1].set_ylabel(r'Power [$m^2/s^2/nHz$]')
ax[0, 1].grid()

ax[0, 2].plot(freqs, power_uphi_m8_mag[:, 8], label=r'$u_{\phi}$', color='darkblue')
ax[0, 2].plot(freqs, power_uthe_m8_mag[:, 8], label=r'$u_{\theta}$', color='darkorange')
ax[0, 2].axvline(x=-110, color='red', linestyle='--')
ax[0, 2].set_title('LCTMag m=8')
ax[0, 2].set_xlim([-200, 0])
ax[0, 2].set_ylim([0, 0.13])
ax[0, 2].legend()
ax[0, 2].set_xlabel('Frequency (nHz)')
ax[0, 2].set_ylabel(r'Power [$m^2/s^2/nHz$]')
ax[0, 2].grid()

ax[0, 3].plot(freqs, power_uphi_m13_mag[:, 13], label=r'$u_{\phi}$', color='darkblue')
ax[0, 3].plot(freqs, power_uthe_m13_mag[:, 13], label=r'$u_{\theta}$', color='darkorange')
ax[0, 3].axvline(x=-213, color='red', linestyle='--')
ax[0, 3].set_title('LCTMag m=13')
ax[0, 3].set_xlim([-300, -100])
ax[0, 3].set_ylim([0, 0.03])
ax[0, 3].legend()
ax[0, 3].set_xlabel('Frequency (nHz)')
ax[0, 3].set_ylabel(r'Power [$m^2/s^2/nHz$]')
ax[0, 3].grid()

# LCTGran Plots
ax[1, 0] = plt.subplot(gs[1, 0])
ax[1, 1] = plt.subplot(gs[1, 1])
ax[1, 2] = plt.subplot(gs[1, 2])
ax[1, 3] = plt.subplot(gs[1, 3])

ax[1, 0].plot(freqs, power_uphi_m1_gran[:, 1], label=r'$u_{\phi}$', color='green')
ax[1, 0].plot(freqs, power_uthe_m1_gran[:, 1], label=r'$u_{\theta}$', color='purple')
ax[1, 0].axvline(x=-88, color='red', linestyle='--')
ax[1, 0].set_title('LCTGran m=1')
ax[1, 0].set_xlim([-200, 0])
ax[1, 0].set_ylim([0, 4])
ax[1, 0].legend(loc='upper right')
ax[1, 0].set_xlabel('Frequency (nHz)')
ax[1, 0].set_ylabel(r'Power [$m^2/s^2/nHz$]')
ax[1, 0].grid()

ax[1, 1].plot(freqs, power_uphi_m2_gran[:, 2], label=r'$u_{\phi}$', color='green')
ax[1, 1].plot(freqs, power_uthe_m2_gran[:, 2], label=r'$u_{\theta}$', color='purple')
ax[1, 1].axvline(x=-73, color='red', linestyle='--')
ax[1, 1].set_title('LCTGran m=2')
ax[1, 1].set_xlim([-200, 50])
ax[1, 1].set_ylim([0, 0.01])
ax[1, 1].legend()
ax[1, 1].set_xlabel('Frequency (nHz)')
ax[1, 1].set_ylabel(r'Power [$m^2/s^2/nHz$]')
ax[1, 1].grid()

ax[1, 2].plot(freqs, power_uphi_m8_gran[:, 8], label=r'$u_{\phi}$', color='green')
ax[1, 2].plot(freqs, power_uthe_m8_gran[:, 8], label=r'$u_{\theta}$', color='purple')
ax[1, 2].axvline(x=-110, color='red', linestyle='--')
ax[1, 2].set_title('LCTGran m=8')
ax[1, 2].set_xlim([-200, 0])
ax[1, 2].set_ylim([0, 0.15])
ax[1, 2].legend()
ax[1, 2].set_xlabel('Frequency (nHz)')
ax[1, 2].set_ylabel(r'Power [$m^2/s^2/nHz$]')
ax[1, 2].grid()

ax[1, 3].plot(freqs, power_uphi_m13_gran[:, 13], label=r'$u_{\phi}$', color='green')
ax[1, 3].plot(freqs, power_uthe_m13_gran[:, 13], label=r'$u_{\theta}$', color='purple')
ax[1, 3].axvline(x=-213, color='red', linestyle='--')
ax[1, 3].set_title('LCTGran m=13')
ax[1, 3].set_xlim([-300, -100])
ax[1, 3].set_ylim([0, 0.06])
ax[1, 3].legend()
ax[1, 3].set_xlabel('Frequency (nHz)')
ax[1, 3].set_ylabel(r'Power [$m^2/s^2/nHz$]')
ax[1, 3].grid()

# fig.savefig('combined_ps1D_mag_gran.pdf', bbox_inches='tight')

# %%
# Get the power for m=3 rossby and high latitude from mag
power_uphi_m3_rossby = np.nanmean(abs(uphi_ft_anti_mag[:, lat_eq_rossby, 3])**2, axis = 1)*conv_factor
power_uthe_m3_rossby = np.nanmean(abs(uthe_ft_sym_mag[:, lat_eq_rossby, 3])**2, axis = 1)*conv_factor
power_uphi_m3_hl = np.nanmean(abs(uphi_ft_anti_mag[:, lat_eq_hl, 3])**2, axis = 1)*conv_factor
power_uthe_m3_hl = np.nanmean(abs(uthe_ft_sym_mag[:, lat_eq_hl, 3])**2, axis = 1)*conv_factor
# %%
# Save these into npz files
np.savez('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/ps_1d_m3_rossby_hl_lctmag.npz', freqs=freqs, power_uphi_m3_rossby=power_uphi_m3_rossby, power_uthe_m3_rossby=power_uthe_m3_rossby, power_uphi_m3_hl=power_uphi_m3_hl, power_uthe_m3_hl=power_uthe_m3_hl)

# %%
