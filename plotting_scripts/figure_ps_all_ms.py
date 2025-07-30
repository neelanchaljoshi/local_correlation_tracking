# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family'] = 'sans-serif'
# %%
# Load the data for mag for all uphi and uthe components
ft_uphi_sym_mag = np.load('/data/seismo/joshin/pipeline-test/lct_data_processed/uphi_fourier_hmi_m_720s_dt_1h_sym_2010_2023.npy')
ft_uphi_anti_mag = np.load('/data/seismo/joshin/pipeline-test/lct_data_processed/uphi_fourier_hmi_m_720s_dt_1h_anti_2010_2023.npy')
ft_uthe_sym_mag = np.load('/data/seismo/joshin/pipeline-test/lct_data_processed/uthe_fourier_hmi_m_720s_dt_1h_sym_2010_2023.npy')
ft_uthe_anti_mag = np.load('/data/seismo/joshin/pipeline-test/lct_data_processed/uthe_fourier_hmi_m_720s_dt_1h_anti_2010_2023.npy')
# %%
# Set the figure path
fig_path = '/data/seismo/joshin/pipeline-test/local_correlation_tracking/pdfs/ps_all_ms'
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
def ft_to_2d_power_spectrum_lat_freq(ft, m, dt):
    """
    Convert Fourier coefficients to 2D power spectrum.
    """
    M_arr = np.arange(ft.shape[2])
    freqs = np.fft.fftfreq(len(ft), d=dt)
    freqs = -np.fft.fftshift(freqs)*1e9
    lat_og = np.linspace(-90, 90, ft.shape[1])
    nt = len(ft)
    conv_factor = 2/nt*1e-9*dt/144/144
    power = abs(ft[:, :, m])**2 * conv_factor
    return freqs, power
# %%
# Calculate the 2D power spectra for mag uphi and uthe components for different latitude bands

freqs, M_arr, ps_uphi_sym_low = ft_to_2d_power_spectrum(ft_uphi_sym_mag, 'rossby', dt = 6 * 3600)
_, _, ps_uphi_sym_mid = ft_to_2d_power_spectrum(ft_uphi_sym_mag, 'critlat', dt = 6 * 3600)
_, _, ps_uphi_sym_high = ft_to_2d_power_spectrum(ft_uphi_sym_mag, 'highlat', dt = 6 * 3600)

_, _, ps_uphi_anti_low = ft_to_2d_power_spectrum(ft_uphi_anti_mag, 'rossby', dt = 6 * 3600)
_, _, ps_uphi_anti_mid = ft_to_2d_power_spectrum(ft_uphi_anti_mag, 'critlat', dt = 6 * 3600)
_, _, ps_uphi_anti_high = ft_to_2d_power_spectrum(ft_uphi_anti_mag, 'highlat', dt = 6 * 3600)

_, _, ps_uthe_sym_low = ft_to_2d_power_spectrum(ft_uthe_sym_mag, 'rossby', dt = 6 * 3600)
_, _, ps_uthe_sym_mid = ft_to_2d_power_spectrum(ft_uthe_sym_mag, 'critlat', dt = 6 * 3600)
_, _, ps_uthe_sym_high = ft_to_2d_power_spectrum(ft_uthe_sym_mag, 'highlat', dt = 6 * 3600)

_, _, ps_uthe_anti_low = ft_to_2d_power_spectrum(ft_uthe_anti_mag, 'rossby', dt = 6 * 3600)
_, _, ps_uthe_anti_mid = ft_to_2d_power_spectrum(ft_uthe_anti_mag, 'critlat', dt = 6 * 3600)
_, _, ps_uthe_anti_high = ft_to_2d_power_spectrum(ft_uthe_anti_mag, 'highlat', dt = 6 * 3600)
# %%
def plot_ps_for_each_m(m, lat, freq):
    '''
    Plot the 2D power spectrum and 1D ps for each m value. in a grid like fashion.
    '''
    # Get the 2d lat vs freq power spectra for each component
    _, ps_uphi_sym_lat_freq = ft_to_2d_power_spectrum_lat_freq(ft_uphi_sym_mag, m, dt = 6 * 3600)
    _, ps_uphi_anti_lat_freq = ft_to_2d_power_spectrum_lat_freq(ft_uphi_anti_mag, m, dt = 6 * 3600)
    _, ps_uthe_sym_lat_freq = ft_to_2d_power_spectrum_lat_freq(ft_uthe_sym_mag, m, dt = 6 * 3600)
    _, ps_uthe_anti_lat_freq = ft_to_2d_power_spectrum_lat_freq(ft_uthe_anti_mag, m, dt = 6 * 3600)

    # print(ps_uphi_sym_lat_freq.shape, ps_uphi_anti_lat_freq.shape, ps_uthe_sym_lat_freq.shape, ps_uthe_anti_lat_freq.shape)
    # print(ps_uphi_anti_low.shape, ps_uphi_anti_mid.shape, ps_uphi_anti_high.shape)

    # Get 1d line plots for each component
    ps_1d_uphi_sym_low = ps_uphi_sym_low[:, m]
    ps_1d_uphi_sym_mid = ps_uphi_sym_mid[:, m]
    ps_1d_uphi_sym_high = ps_uphi_sym_high[:, m]

    ps_1d_uphi_anti_low = ps_uphi_anti_low[:, m]
    ps_1d_uphi_anti_mid = ps_uphi_anti_mid[:, m]
    ps_1d_uphi_anti_high = ps_uphi_anti_high[:, m]
    
    ps_1d_uthe_sym_low = ps_uthe_sym_low[:, m]
    ps_1d_uthe_sym_mid = ps_uthe_sym_mid[:, m]
    ps_1d_uthe_sym_high = ps_uthe_sym_high[:, m]
    
    ps_1d_uthe_anti_low = ps_uthe_anti_low[:, m]
    ps_1d_uthe_anti_mid = ps_uthe_anti_mid[:, m]
    ps_1d_uthe_anti_high = ps_uthe_anti_high[:, m]


    # Create the power_data array
    power_data = np.array([ps_uphi_sym_lat_freq, ps_uthe_anti_lat_freq,
                            ps_uphi_anti_lat_freq, ps_uthe_sym_lat_freq])


    # Create the line_data array
    line_data = np.array([ps_1d_uphi_sym_low, ps_1d_uthe_anti_low, ps_1d_uphi_anti_low, ps_1d_uthe_sym_low,
                          ps_1d_uphi_sym_mid, ps_1d_uthe_anti_mid, ps_1d_uphi_anti_mid, ps_1d_uthe_sym_mid,
                          ps_1d_uphi_sym_high, ps_1d_uthe_anti_high, ps_1d_uphi_anti_high, ps_1d_uthe_sym_high])

    # Set up square grid: 4x4 panels, all square
    panel_size = 4  # inches per subplot
    fig = plt.figure(figsize=(4 * panel_size, 4 * panel_size))
    gs = fig.add_gridspec(4, 4, wspace=0.3, hspace=0.3)

    # --- Top row: 2D power spectra ---
    axes_2d = []
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        if m >= 20:
            vmax = 0.01
        else:
            vmax = 0.02
        im = ax.pcolormesh(freq, lat, power_data[i].T, shading='nearest', cmap='gray_r', vmax = vmax, vmin = 0.00)
        ax.set_xlim(-400, 400)
        ax.set_ylim(-75, 75)
        ax.set_xlabel('Frequency [nHz]')
        if i == 0:
            ax.set_ylabel(r'Latitude [$\degree$]')
        if i == 0:
            ax.set_title(r'$u_\phi^+$', fontsize=10)
        elif i == 1:
            ax.set_title(r'$u_\theta^-$', fontsize=10)
        elif i == 2:
            ax.set_title(r'$u_\phi^-$', fontsize=10)
        elif i == 3:
            ax.set_title(r'$u_\theta^+$', fontsize=10)
        # ax.set_title(f"Power {i+1}", fontsize=10)
        ax.tick_params(top=True, right=True, direction='out')
        axes_2d.append(ax)

    # Shared colorbar to the right
    cbar_ax = fig.add_axes([0.92, 0.76, 0.015, 0.15])
    fig.colorbar(im, cax=cbar_ax, label='Power')

    # --- Bottom 3 rows: Line plots ---
    for row in range(1, 4):
        for col in range(4):
            idx = (row - 1) * 4 + col
            ax = fig.add_subplot(gs[row, col])
            ax.plot(freq, line_data[idx], color='black')
            ax.set_xlim(-400, 400)
            ax.set_xlabel('Frequency [nHz]' if row == 3 else '')
            if col == 0:
                if row == 1:
                    ax.set_ylabel('Low latitude power')
                elif row == 2:
                    ax.set_ylabel('Mid latitude power')
                elif row == 3:
                    ax.set_ylabel('High latitude power')

            ax.tick_params(top=True, right=True, direction='out')
    fig.suptitle(f'Power spectra for m = {m}', fontsize=16)
    plt.subplots_adjust(left=0.07, right=0.9, top=0.95, bottom=0.07)
    plt.savefig(f'{fig_path}/ps_all_ms_m_{m}.png')
    plt.close()
    # plt.show()
    return None
# %%
# Plot for each m value
lats = np.linspace(-90, 90, ft_uphi_sym_mag.shape[1])
freqs = np.fft.fftfreq(len(ft_uphi_sym_mag), d=6 * 3600)
# Convert to nHz
freqs = -np.fft.fftshift(freqs) * 1e9
# Loop through m values from 0 to 20
for m in range(35, 73):
    plot_ps_for_each_m(m, lats, freqs)
# %%
