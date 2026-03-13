# %% imports
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import curve_fit

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.titlesize'] = 18

# %%
uphi_ft_anti_lct_mask = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/for_rda_mask_comparison/uphi_ft_0.96_2010_2024_anti_hmi_m_720s_dt_1h.npy')
uphi_ft_anti_mag = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uphi_ft_2010_2024_anti_hmi_m_720s_dt_1h.npy')
uthe_ft_sym_lct_mask = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/for_rda_mask_comparison/uthe_ft_0.96_2010_2024_sym_hmi_m_720s_dt_1h.npy')
uthe_ft_sym_mag = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_ft_2010_2024_sym_hmi_m_720s_dt_1h.npy')
rda_list = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/for_rda_mask_comparison/rda_fft.npz')
uphi_ft_anti_rda = rda_list['fup']
uthe_ft_sym_rda = rda_list['fut']
freqs_rda = rda_list['freq']
# %%
lat_og = np.linspace(-90, 90, 73)
M_arr = np.arange(uphi_ft_anti_mag.shape[2])
freqs = np.fft.fftfreq(len(uphi_ft_anti_mag), d=6*3600)
freqs = -np.fft.fftshift(freqs)*1e9
freqs_rda = np.fft.fftshift(freqs_rda)
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
nt_rda = len(uphi_ft_anti_rda[:, -1, 0, 0])
print('Number of time steps RDA: {}'.format(nt_rda))
dt_rda = 27.3/3*3600
conv_factor_rda = 2/nt_rda*1e-9*dt_rda/144/144

# %%
power_uphi_m1_mag = np.nanmean(abs(uphi_ft_anti_mag[:, lat_eq_hl, :])**2, axis = 1)*conv_factor
power_uphi_m1_lct_mask = np.nanmean(abs(uphi_ft_anti_lct_mask[:, lat_eq_hl, :])**2, axis = 1)*conv_factor


power_uthe_m8_mag = np.nanmean(abs(uthe_ft_sym_mag[:, lat_eq_rossby, :])**2, axis = 1)*conv_factor
power_uthe_m8_lct_mask = np.nanmean(abs(uthe_ft_sym_lct_mask[:, lat_eq_rossby, :])**2, axis = 1)*conv_factor

power_uphi_m1_rda = np.nanmean(abs(uphi_ft_anti_rda[:, -1, lat_eq_hl, :])**2, axis = 1)*conv_factor_rda
power_uthe_m8_rda = np.nanmean(abs(uthe_ft_sym_rda[:, -1, lat_eq_rossby, :])**2, axis = 1)*conv_factor_rda


power_uphi_m1_rda = np.fft.fftshift(power_uphi_m1_rda, axes=0)
power_uthe_m8_rda = np.fft.fftshift(power_uthe_m8_rda, axes=0)

# %% get power in the freq window required
freq_window_uphi_m1 = (freqs >= -200) & (freqs <= 0)
freq_window_rda_m1 = (freqs_rda >= -200) & (freqs_rda <= 0)
freq_window_uthe_m8 = (freqs >= -200) & (freqs <= 0)
freq_window_rda_m8 = (freqs_rda >= -200) & (freqs_rda <= 0)
power_m1_plot_lct_mag = power_uphi_m1_mag[freq_window_uphi_m1, 1]
power_m1_plot_lct_mask = power_uphi_m1_lct_mask[freq_window_uphi_m1, 1]
power_m1_plot_rda = power_uphi_m1_rda[freq_window_rda_m1, 1]
power_m8_plot_lct_mag = power_uthe_m8_mag[freq_window_uthe_m8, 8]
power_m8_plot_lct_mask = power_uthe_m8_lct_mask[freq_window_uthe_m8, 8]
power_m8_plot_rda = power_uthe_m8_rda[freq_window_rda_m8, 8]

# %%
# Fit lorentzians to the peaks

def lorentzian(x, A, x0, gamma, C):
    return A * gamma**2 / ((x - x0)**2 + gamma**2) + C

def fit_lorentzian(x, y):
    # Initial guess: [A, x0, gamma, C]
    p0 = [
        y.max() - y.min(),     # amplitude
        x[np.argmax(y)],       # center
        (x.max() - x.min())/20,# width
        y.min()                # background
    ]
    # Initial guess: [A, x0, gamma, C]

    popt, pcov = curve_fit(lorentzian, x, y, p0=p0)

    A, x0, gamma, C = popt
    perr = np.sqrt(np.diag(pcov))  # 1-sigma uncertainties

    print("Best-fit parameters:")
    print(f"A     = {A:.4f} ± {perr[0]:.4f}")
    print(f"x0    = {x0:.4f} ± {perr[1]:.4f}")
    print(f"gamma = {gamma:.4f} ± {perr[2]:.4f}")
    print(f"C     = {C:.4f} ± {perr[3]:.4f}")
    print(f"SNR (A/C)   = {A/C:.2f}")
    print(f"SNR from max value - C / C: {(y.max()-C)/C:.2f}")
    return popt, pcov

print("Fitting LCT mag m=1 uphi:")
popt_lct_mag_m1, pcov_lct_mag_m1 = fit_lorentzian(freqs[freq_window_uphi_m1], power_m1_plot_lct_mag)
print("\nFitting LCT mask m=1 uphi:")
popt_lct_mask_m1, pcov_lct_mask_m1 = fit_lorentzian(freqs[freq_window_uphi_m1], power_m1_plot_lct_mask)
print("\nFitting RDA m=1 uphi:")
popt_rda_m1, pcov_rda_m1 = fit_lorentzian(freqs_rda[freq_window_rda_m1], power_m1_plot_rda)
print("\nFitting LCT mag m=8 uthe:")
popt_lct_mag_m8, pcov_lct_mag_m8 = fit_lorentzian(freqs[freq_window_uthe_m8], power_m8_plot_lct_mag)
print("\nFitting LCT mask m=8 uthe:")
popt_lct_mask_m8, pcov_lct_mask_m8 = fit_lorentzian(freqs[freq_window_uthe_m8], power_m8_plot_lct_mask)
print("\nFitting RDA m=8 uthe:")
popt_rda_m8, pcov_rda_m8 = fit_lorentzian(freqs_rda[freq_window_rda_m8], power_m8_plot_rda)



# %% Plotting
freqs_lorentzian_m1 = np.linspace(-200, 0, 100)
freqs_lorentzian_m8 = np.linspace(-200, 0, 200)
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 1, figure=fig)
ax = fig.add_subplot(gs[0, 0])
ax.plot(freqs, power_uphi_m1_mag[:, 1], label=r'LCTMag $u_\phi^-$', color='blue', lw=2)
ax.plot(freqs, power_uphi_m1_lct_mask[:, 1], label=r'LCTMag $u_\phi^-$ RDA mask', color='orange', lw=2)
ax.plot(freqs_rda, power_uphi_m1_rda[:, 1], label=r'RDA $u_\phi^-$', color='green', lw=2, ls='--')
ax.plot(freqs_lorentzian_m1, lorentzian(freqs_lorentzian_m1, *popt_lct_mag_m1), label='LCTMag Fit', color='blue', lw=1.5, ls=':')
ax.plot(freqs_lorentzian_m1, lorentzian(freqs_lorentzian_m1, *popt_lct_mask_m1), label='LCTMag with RDA Mask Fit', color='orange', lw=1.5, ls=':')
ax.plot(freqs_lorentzian_m1, lorentzian(freqs_lorentzian_m1, *popt_rda_m1), label='RDA Fit', color='green', lw=1.5, ls=':')
ax.set_xlim(-200, 0)
# ax.set_ylim(-0.001, 5.5)
ax.set_title('m=1 High Latitude')
# ax.set_xlabel('Frequency (nHz)')
ax.set_ylabel(r'Power [$m^2/s^2/nHz$]')
ax.grid()
ax.legend(loc = 'upper right', fontsize=10)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(freqs, power_uthe_m8_mag[:, 8], label=r'LCTMag $u_\theta^+$', color='blue', lw=2)
ax2.plot(freqs, power_uthe_m8_lct_mask[:, 8], label=r'LCTMag $u_\theta^+$ with RDA mask', color='orange', lw=2)
ax2.plot(freqs_rda, power_uthe_m8_rda[:, 8], label=r'RDA $u_\theta^+$', color='green', lw=2, ls='--')
ax2.plot(freqs_lorentzian_m8, lorentzian(freqs_lorentzian_m8, *popt_lct_mag_m8), label='LCTMag Fit', color='blue', lw=1.5, ls=':')
ax2.plot(freqs_lorentzian_m8, lorentzian(freqs_lorentzian_m8, *popt_lct_mask_m8), label='LCTMag with RDA Mask Fit', color='orange', lw=1.5, ls=':')
ax2.plot(freqs_lorentzian_m8, lorentzian(freqs_lorentzian_m8, *popt_rda_m8), label='RDA Fit', color='green', lw=1.5, ls=':')
ax2.set_xlim(-200, 0)
# ax2.set_ylim(0.000, 0.13)
ax2.set_title('m=8 Equatorial Rossby')
ax2.set_xlabel('Frequency (nHz)')
ax2.set_ylabel(r'Power [$m^2/s^2/nHz$]')
ax2.grid()
ax2.legend(loc = 'upper right', fontsize=10)
plt.tight_layout()
fig.savefig('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pdfs/figure_lct_rda_compare_ps.pdf', dpi=300)

# %%
