# %%
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'sans-serif'
# %%
# Load the data for mag for all uphi and uthe components
ft_uphi_sym_mag = np.load('/data/seismo/joshin/pipeline-test/lct_data_processed/uphi_fourier_hmi_m_720s_dt_1h_sym_2010_2023.npy')
ft_uphi_anti_mag = np.load('/data/seismo/joshin/pipeline-test/lct_data_processed/uphi_fourier_hmi_m_720s_dt_1h_anti_2010_2023.npy')
ft_uthe_sym_mag = np.load('/data/seismo/joshin/pipeline-test/lct_data_processed/uthe_fourier_hmi_m_720s_dt_1h_sym_2010_2023.npy')
ft_uthe_anti_mag = np.load('/data/seismo/joshin/pipeline-test/lct_data_processed/uthe_fourier_hmi_m_720s_dt_1h_anti_2010_2023.npy')
# %%
# Set the figure path
fig_path = '/data/seismo/joshin/pipeline-test/local_correlation_tracking/pdfs/ps_all_ms/test'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
# %%
def ft_to_3d_power_spectrum(ft, dt):
    """
    Convert Fourier coefficients to 2D power spectrum.
    """
    M_arr = np.arange(ft.shape[2])
    freqs = np.fft.fftfreq(len(ft), d=dt)
    freqs = -np.fft.fftshift(freqs)*1e9
    nt = len(ft)
    conv_factor = 2/nt*1e-9*dt/144/144
    power = abs(ft[:, :, :])**2*conv_factor
    return freqs, M_arr, power

# %%
# Calculate the 2D power spectra for mag uphi and uthe components for different latitude bands

freqs, M_arr, ps_uphi_sym = ft_to_3d_power_spectrum(ft_uphi_sym_mag, dt = 6 * 3600)
_, _, ps_uthe_sym = ft_to_3d_power_spectrum(ft_uthe_sym_mag, dt = 6 * 3600)
_, _, ps_uphi_anti = ft_to_3d_power_spectrum(ft_uphi_anti_mag, dt = 6 * 3600)
_, _, ps_uthe_anti = ft_to_3d_power_spectrum(ft_uthe_anti_mag, dt = 6 * 3600)

# Limit the power spectrum values between -/+400 nHz
freqs_mask = np.where((freqs >= -400) & (freqs <= 400))[0]
ps_uphi_sym = ps_uphi_sym[freqs_mask, :, :]
ps_uthe_sym = ps_uthe_sym[freqs_mask, :, :]
ps_uphi_anti = ps_uphi_anti[freqs_mask, :, :]
ps_uthe_anti = ps_uthe_anti[freqs_mask, :, :]
# %%
def get_1d_ps_list_from_lat_freq(ps, mode_type = 'rossby'):
    lat = np.linspace(-90, 90, ps.shape[1])
    if mode_type == 'rossby':
        lat_eq = (abs(lat) <= 30)
        lat_list = [10, 20, 30]
    elif mode_type == 'critlat':
        lat_eq = (abs(lat) >= 15) & (abs(lat) <= 45)
        lat_list = [20, 30, 40]
    elif mode_type == 'highlat':
        lat_eq = (abs(lat) >= 45) & (abs(lat) <= 75)
        lat_list = [50, 60, 70]
    ps_1d_avg = np.nanmean(ps[:, lat_eq], axis=1)  # Average over the latitude band
    ps_1d_lat1 = ps[:, lat == lat_list[0]]
    ps_1d_lat2 = ps[:, lat == lat_list[1]]
    ps_1d_lat3 = ps[:, lat == lat_list[2]]
    ps_1d_list = [ps_1d_avg, ps_1d_lat1, ps_1d_lat2, ps_1d_lat3]
    return ps_1d_list
# %%
def plot_ps_for_each_m(m, lat, freq):
    '''
    Plot the 2D power spectrum and 1D ps for each m value. in a grid like fashion.
    '''
    # First 4 panels are 2D power spectra
    ps_uphi_sym_lat_freq = ps_uphi_sym[:, :, m]
    ps_uthe_sym_lat_freq = ps_uthe_sym[:, :, m]
    ps_uphi_anti_lat_freq = ps_uphi_anti[:, :, m]
    ps_uthe_anti_lat_freq = ps_uthe_anti[:, :, m]

    # Get the 1D power spectra for each component
    ps_1d_uphi_sym_low = get_1d_ps_list_from_lat_freq(ps_uphi_sym_lat_freq, mode_type = 'rossby')
    ps_1d_uthe_sym_low = get_1d_ps_list_from_lat_freq(ps_uthe_sym_lat_freq, mode_type = 'rossby')
    ps_1d_uphi_anti_low = get_1d_ps_list_from_lat_freq(ps_uphi_anti_lat_freq, mode_type = 'rossby')
    ps_1d_uthe_anti_low = get_1d_ps_list_from_lat_freq(ps_uthe_anti_lat_freq, mode_type = 'rossby')

    ps_1d_uphi_sym_mid = get_1d_ps_list_from_lat_freq(ps_uphi_sym_lat_freq, mode_type = 'critlat')
    ps_1d_uthe_sym_mid = get_1d_ps_list_from_lat_freq(ps_uthe_sym_lat_freq, mode_type = 'critlat')
    ps_1d_uphi_anti_mid = get_1d_ps_list_from_lat_freq(ps_uphi_anti_lat_freq, mode_type = 'critlat')
    ps_1d_uthe_anti_mid = get_1d_ps_list_from_lat_freq(ps_uthe_anti_lat_freq, mode_type = 'critlat')

    ps_1d_uphi_sym_high = get_1d_ps_list_from_lat_freq(ps_uphi_sym_lat_freq, mode_type = 'highlat')
    ps_1d_uthe_sym_high = get_1d_ps_list_from_lat_freq(ps_uthe_sym_lat_freq, mode_type = 'highlat')
    ps_1d_uphi_anti_high = get_1d_ps_list_from_lat_freq(ps_uphi_anti_lat_freq, mode_type = 'highlat')
    ps_1d_uthe_anti_high = get_1d_ps_list_from_lat_freq(ps_uthe_anti_lat_freq, mode_type = 'highlat')




    # Create the power_data array
    power_data = np.array([ps_uphi_sym_lat_freq, ps_uthe_anti_lat_freq,
                            ps_uphi_anti_lat_freq, ps_uthe_sym_lat_freq])


    # Create the line_data array
    line_data = [ps_1d_uphi_sym_low, ps_1d_uthe_anti_low, ps_1d_uphi_anti_low, ps_1d_uthe_sym_low,
                          ps_1d_uphi_sym_mid, ps_1d_uthe_anti_mid, ps_1d_uphi_anti_mid, ps_1d_uthe_sym_mid,
                          ps_1d_uphi_sym_high, ps_1d_uthe_anti_high, ps_1d_uphi_anti_high, ps_1d_uthe_sym_high]

    # Set up square grid: 4x4 panels, all square
    panel_size = 4  # inches per subplot
    fig = plt.figure(figsize=(4 * panel_size, 4 * panel_size))
    gs = fig.add_gridspec(4, 4, wspace=0.3, hspace=0.3)

    # --- Top row: 2D power spectra ---
    axes_2d = []
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        if m >= 20:
            vmax = 0.02
        else:
            vmax = 0.04
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
    cbar_ax = fig.add_axes([0.92, 0.82, 0.015, 0.10])  # Adjusted position higher
    fig.colorbar(im, cax=cbar_ax, label='Power')

    # --- Bottom 3 rows: Line plots ---
    first_line_ax = None
    for row in range(1, 4):
        for col in range(4):
            idx = (row - 1) * 4 + col
            ax = fig.add_subplot(gs[row, col])

            ax.plot(freq, line_data[idx][1], color='red', linewidth = 0.5, label = '1')
            ax.plot(freq, line_data[idx][2], color='blue', linewidth = 0.5, label = '2')
            ax.plot(freq, line_data[idx][3], color='green', linewidth = 0.5, label = '3')
            ax.plot(freq, line_data[idx][0], color='black', linewidth = 1.5, label = 'range average')
            # Take 4 year average for line_data[idx][0]
            arr = line_data[idx][0]
            n = len(arr) // 4 * 4  # largest multiple of 4
            result = arr[:n].reshape(-1, 4).mean(axis=1)
            arr = freq
            freq_av = arr[:n].reshape(-1, 4).mean(axis=1)
            ax.plot(freq_av, result, color='cyan', linewidth = 1.0, label = '4 year average')
            if first_line_ax is None:
                first_line_ax = ax

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
    # Get handles and labels from the first axis for the shared legend
    handles, labels = first_line_ax.get_legend_handles_labels()

    # Add the shared legend at the bottom center
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=20, frameon=False)
    plt.subplots_adjust(left=0.07, right=0.9, top=0.95, bottom=0.10)
    plt.savefig(f'{fig_path}/ps_all_ms_m_{m}.png', dpi = 300)
    plt.close()
    # plt.show()
    return None
# %%
# Plot for each m value
lats = np.linspace(-90, 90, ft_uphi_sym_mag.shape[1])
freqs = np.fft.fftfreq(len(ft_uphi_sym_mag), d=6 * 3600)
# Convert to nHz
freqs = -np.fft.fftshift(freqs) * 1e9
freqs_masked = np.where((freqs >= -400) & (freqs <= 400))[0]
freqs = freqs[freqs_masked]
# Loop through m values from 0 to 20
for m in range(0, 73):
    print(f"Plotting for m = {m}")
    plot_ps_for_each_m(m, lats, freqs)
# %%
