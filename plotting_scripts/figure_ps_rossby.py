# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
plt.rcParams.update({'font.size': 15})
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
m_plot = M_arr[M_arr >= 3]
fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
ax[0].tick_params(which = 'both', top=True, bottom=True, right=True, left=True)
im = ax[0].pcolormesh(M_arr, freqs, ps_uthe_sym_mag, cmap = 'binary', shading = 'auto',rasterized = True, vmin=0.0, vmax=0.06)
ax[0].plot(m_plot, -2*456/(m_plot+1), color = 'darkorange',label = r'$\omega = -2\Omega/(m+1)$', linewidth = 2.5)
ax[0].set_ylim([-500, 100])
ax[0].set_title(r'Power spectrum of $u_\theta^+$ (LCTMag)', fontsize = 16, pad = 20)
ax[0].set_ylabel('Frequency [nHz]')
ax[0].set_xlabel(r'm')
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
ax[1].plot(m_plot, -2*456/(m_plot+1), color = 'darkorange',label = r'$\omega = -2\Omega/(m+1)$', linewidth = 2.5)
ax[1].set_ylim([-500, 100])
ax[1].set_xlabel(r'm')
ax[1].legend()
# ax[1].set_ylabel('Frequency [nHz]')
ax[1].set_title(r'Power spectrum of $u_\theta^+$ (LCTGran)', fontsize = 16, pad = 20)
ax[1].set_xticks(np.arange(0, 21, 2), minor = False)
ax[1].set_xticks(np.arange(0, 21, 1), minor = True)
ax[1].set_yticks(np.arange(-500, 101, 100), minor = False)
ax[1].set_yticks(np.arange(-500, 101, 50), minor = True)
ax[1].tick_params(which='minor', length=4, color='gray')
ax[1].tick_params(which='major', length=8, color='black')
ax[1].set_xlim([0,20])
cab = fig.colorbar(im, ticks = [0.0, 0.03, 0.06], shrink=0.2, aspect = 10, pad = 0.05, label = r'm$^2$s$^{-2}$nHz$^{-1}$')
# fig.savefig(f'{fig_path}/ps_mag_gran_rossby.pdf', bbox_inches='tight')
# plt.tight_layout()
# %%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family'] = 'sans-serif'

# %%
# Load the data for mag and gran utheta symmetric
ft_uthe_sym_mag  = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_ft_2010_2024_sym_hmi_m_720s_dt_1h.npy')
ft_uthe_sym_gran = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_ft_2010_2024_sym_hmi_ic_45s_granule.npy')

# %%
fig_path = '/data/seismo/joshin/pipeline-test/local_correlation_tracking/pdfs'

# %%
def ft_to_2d_power_spectrum(ft, mode, dt):
    """Convert Fourier coefficients to 2D power spectrum."""
    M_arr  = np.arange(ft.shape[2])
    freqs  = np.fft.fftfreq(len(ft), d=dt)
    freqs  = -np.fft.fftshift(freqs) * 1e9
    lat_og = np.linspace(-90, 90, ft.shape[1])
    if mode == 'rossby':
        lat_mask = (abs(lat_og) <= 30)
    elif mode == 'highlat':
        lat_mask = (abs(lat_og) >= 45) & (abs(lat_og) <= 75)
    elif mode == 'critlat':
        lat_mask = (abs(lat_og) >= 15) & (abs(lat_og) <= 45)
    nt           = len(ft)
    conv_factor  = 2 / nt * 1e-9 * dt / 144 / 144
    power        = np.nanmean(abs(ft[:, lat_mask, :]) ** 2, axis=1) * conv_factor
    return freqs, M_arr, power

# %%
freqs, M_arr, ps_uthe_sym_mag  = ft_to_2d_power_spectrum(ft_uthe_sym_mag,  'rossby', dt=6 * 3600)
_,     _,     ps_uthe_sym_gran = ft_to_2d_power_spectrum(ft_uthe_sym_gran, 'rossby', dt=6 * 3600)
m_plot = M_arr[M_arr >= 3]
# %%
# --- Layout: 3 columns, right column split 2 rows ---
fig = plt.figure(figsize=(18, 7), constrained_layout=True)
gs  = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.85])

ax0  = fig.add_subplot(gs[:, 0])   # full left column  — LCTMag 2D PSD
ax1  = fig.add_subplot(gs[:, 1])   # full mid column   — LCTGran 2D PSD
ax2t = fig.add_subplot(gs[0, 2])   # top-right         — cut m = 3
ax2b = fig.add_subplot(gs[1, 2])   # bottom-right      — cut m = 8

# ---- shared pcolormesh settings ----
pcm_kw = dict(cmap='binary', shading='auto', rasterized=True, vmin=0.0, vmax=0.06)

def style_2d(ax, title):
    ax.tick_params(which='both', top=True, bottom=True, right=True, left=True)
    ax.set_ylim([-500, 100])
    ax.set_xlim([0, 20])
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel(r'$m$', fontsize=14)
    ax.set_xticks(np.arange(0, 21, 2),  minor=False)
    ax.set_xticks(np.arange(0, 20, 1),  minor=True)
    ax.set_yticks(np.arange(-500, 101, 100), minor=False)
    ax.set_yticks(np.arange(-500, 101,  50), minor=True)
    ax.tick_params(which='minor', length=4, color='gray')
    ax.tick_params(which='major', length=8, color='black')
    ax.plot(m_plot, -2 * 456 / (m_plot + 1),
            color='darkorange', label=r'$\omega = -2\Omega/(m+1)$', linewidth=3)
    ax.legend(fontsize=12)

# ---- Col 0: LCTMag ----
ax0.pcolormesh(M_arr, freqs, ps_uthe_sym_mag, **pcm_kw)
style_2d(ax0, r'Power spectrum of $u_\theta^+$ (LCTMag)')
ax0.set_ylabel('Frequency [nHz]')

# ---- Col 1: LCTGran ----
im = ax1.pcolormesh(M_arr, freqs, ps_uthe_sym_gran, **pcm_kw)
style_2d(ax1, r'Power spectrum of $u_\theta^+$ (LCTGran)')
# colorbar attached to col-1 panel
fig.colorbar(im, ax=ax1, ticks=[0.0, 0.03, 0.06],
             shrink=0.5, aspect=15, pad=0.03,
             label=r'm$^2$s$^{-2}$nHz$^{-1}$')

# ---- Helper: find closest m index ----
def m_idx(m_val):
    return np.argmin(abs(M_arr - m_val))

# ---- Col 2 top: cut at m = 3 ----
mi3 = m_idx(3)
ax2t.plot(freqs, ps_uthe_sym_mag[:, mi3],  color='steelblue',   lw=1.8, label='LCTMag')
ax2t.plot(freqs, ps_uthe_sym_gran[:, mi3], color='darkorange',  lw=1.8, label='LCTGran', ls='--')
ax2t.axvline(-2 * 456 / (3 + 1), color='gray', lw=1, ls=':', label=r'Theoretical $\omega_{R}$')
ax2t.set_xlim([-500, 100])
ax2t.set_ylim([0, 0.2])
ax2t.set_title(r'Cut at $m=3$', fontsize=16)
# ax2t.set_ylabel(r'm$^2$s$^{-2}$nHz$^{-1}$', fontsize=11)
ax2t.tick_params(which='both', top=True, right=True)
ax2t.set_xticks(np.arange(-500, 101, 100), minor=False)
ax2t.set_xticks(np.arange(-500, 101,  50), minor=True)
ax2t.tick_params(which='minor', length=3, color='gray')
ax2t.legend(fontsize=10)
# suppress x-tick labels on top panel (shared x-axis feel)
ax2t.set_xticklabels([])

# ---- Col 2 bottom: cut at m = 8 ----
mi8 = m_idx(8)
ax2b.plot(freqs, ps_uthe_sym_mag[:, mi8],  color='steelblue',  lw=1.8, label='LCTMag')
ax2b.plot(freqs, ps_uthe_sym_gran[:, mi8], color='darkorange', lw=1.8, label='LCTGran', ls='--')
ax2b.axvline(-2 * 456 / (8 + 1), color='gray', lw=1, ls=':', label=r'Theoretical $\omega_{R}$')
ax2b.set_xlim([-500, 100])
ax2b.set_ylim([0, 0.2])
ax2b.set_title(r'Cut at $m=8$', fontsize=16)
ax2b.set_xlabel('Frequency [nHz]', fontsize=13)
# ax2b.set_ylabel(r'm$^2$s$^{-2}$nHz$^{-1}$', fontsize=11)
ax2b.tick_params(which='both', top=True, right=True)
ax2b.set_xticks(np.arange(-500, 101, 100), minor=False)
ax2b.set_xticks(np.arange(-500, 101,  50), minor=True)
ax2b.tick_params(which='minor', length=3, color='gray')
ax2b.legend(fontsize=10)

fig.savefig(f'{fig_path}/ps_mag_gran_rossby_3col.pdf', bbox_inches='tight')
plt.show()
# %%
