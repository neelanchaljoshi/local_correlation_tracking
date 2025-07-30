# %%
import matplotlib.pyplot as plt
import numpy as np

# Dummy data
lat = np.linspace(-90, 90, 100)
freq = np.linspace(0, 5, 100)
power_data = [np.random.rand(100, 100) for _ in range(4)]
line_data = [np.random.rand(100) for _ in range(12)]

# Set up square grid: 4x4 panels, all square
# Set up square grid: 4x4 panels, all square
panel_size = 2.5  # inches per subplot
fig = plt.figure(figsize=(4 * panel_size, 4 * panel_size))
gs = fig.add_gridspec(4, 4, wspace=0.3, hspace=0.3)

# --- Top row: 2D power spectra ---
axes_2d = []
for i in range(4):
    ax = fig.add_subplot(gs[0, i])
    im = ax.pcolormesh(freq, lat, power_data[i], shading='auto', cmap='viridis')
    ax.set_xlabel('Frequency [nHz]')
    if i == 0:
        ax.set_ylabel(r'Latitude [$\degree$]')
    ax.set_title(f"Power {i+1}", fontsize=10)
    ax.tick_params(top=True, right=True, direction='in')
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
        ax.set_xlabel('Frequency [nHz]' if row == 3 else '')
        if col == 0:
            if row == 1:
                ax.set_ylabel('Low latitude power')
            elif row == 2:
                ax.set_ylabel('Mid latitude power')
            elif row == 3:
                ax.set_ylabel('High latitude power')

        ax.tick_params(top=True, right=True, direction='in')

plt.subplots_adjust(left=0.07, right=0.9, top=0.95, bottom=0.07)
plt.show()

# %%
