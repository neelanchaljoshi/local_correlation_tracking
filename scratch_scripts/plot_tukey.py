# %%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def tukey_twoD(width, alpha=0.6):
    """2D tukey lowpass window with a circular support"""
    base = np.zeros((width, width))
    tukey = signal.windows.tukey(width, alpha)
    tukey = tukey[int(len(tukey)/2)-1:]  # Second half of tukey window
    x = np.linspace(-width/2, width/2, width)
    y = np.linspace(-width/2, width/2, width)
    for x_index in range(0, width):
        for y_index in range(0, width):
            if int(np.hypot(x[x_index], y[y_index])) <= width/2:
                base[x_index, y_index] = tukey[int(np.hypot(x[x_index], y[y_index]))]
    return base

width = 168
alpha = 0.6
px_to_deg = 0.03  # 1 px = 0.03 deg on solar surface

# --- 1D Tukey ---
tukey_full = signal.windows.tukey(width, alpha)
tukey_half = tukey_full[int(len(tukey_full)/2)-1:]  # second half used in 2D

# --- 2D Tukey ---
tukey_2d = tukey_twoD(width, alpha)

# --- Flat region boundary ---
flat_half = int(width * (1 - alpha) / 2)  # ~17 px radius
flat_half_deg = flat_half * px_to_deg
width_deg = width * px_to_deg

# --- Plot ---
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#ffffff')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])  # Full 1D window
ax2 = fig.add_subplot(gs[0, 1])  # Half window (used in 2D radial mapping)
ax3 = fig.add_subplot(gs[1, 0])  # 2D window image
ax4 = fig.add_subplot(gs[1, 1])  # 2D window cross-section

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor('#f7f7f7')
    ax.tick_params(colors='#333333')
    ax.xaxis.label.set_color('#333333')
    ax.yaxis.label.set_color('#333333')
    ax.title.set_color('#111111')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')

# Plot 1: Full 1D Tukey window
x_full_deg = np.linspace(0, width_deg, width)
flat_indices = np.where(tukey_full == 1.0)[0]
flat_start_deg = x_full_deg[flat_indices[0]]
flat_end_deg   = x_full_deg[flat_indices[-1]]
ax1.plot(x_full_deg, tukey_full, color='#0077cc', lw=2)
ax1.axvspan(flat_start_deg, flat_end_deg,
            alpha=0.15, color='#0077cc', label=f'Flat region ({flat_start_deg:.2f}–{flat_end_deg:.2f}°)')
ax1.axvline(width_deg/2, color='#cc3300', lw=1, linestyle='--', label='Center')
ax1.set_title('1D Tukey Window (full)', fontsize=11)
ax1.set_xlabel('Position (°)')
ax1.set_ylabel('Amplitude')
ax1.legend(fontsize=8, facecolor='#ffffff', edgecolor='#cccccc', labelcolor='#333333')
ax1.set_xlim(0, width_deg)

# Plot 2: Half window (radial profile used in 2D)
radii_deg = np.arange(len(tukey_half)) * px_to_deg
ax2.plot(radii_deg, tukey_half, color='#7733cc', lw=2)
ax2.axvspan(0, flat_half_deg, alpha=0.15, color='#7733cc', label=f'Flat region (r ≤ {flat_half_deg:.2f}°)')
ax2.axvline(flat_half_deg, color='#cc8800', lw=1.5, linestyle='--', label=f'Taper start (~{flat_half_deg:.2f}°)')
ax2.set_title('Radial Profile (half window → 2D)', fontsize=11)
ax2.set_xlabel('Radius (°)')
ax2.set_ylabel('Amplitude')
ax2.legend(fontsize=8, facecolor='#ffffff', edgecolor='#cccccc', labelcolor='#333333')
ax2.set_xlim(0, (len(tukey_half) - 1) * px_to_deg)

# Plot 3: 2D window image
extent = [-width_deg/2, width_deg/2, -width_deg/2, width_deg/2]
im = ax3.imshow(tukey_2d, cmap='viridis', origin='lower', vmin=0, vmax=1, extent=extent)
circle_flat = plt.Circle((0, 0), flat_half_deg,
                          color='#ffffff', fill=False, lw=1.5, linestyle='--', label=f'Flat boundary (r={flat_half_deg:.2f}°)')
circle_full = plt.Circle((0, 0), width_deg/2,
                          color='#ffffff', fill=False, lw=1, linestyle=':', alpha=0.6, label=f'Full radius ({width_deg/2:.2f}°)')
ax3.add_patch(circle_flat)
ax3.add_patch(circle_full)
ax3.set_title('2D Tukey Window', fontsize=11)
ax3.set_xlabel('x (°)')
ax3.set_ylabel('y (°)')
ax3.legend(fontsize=8, facecolor='#ffffff', edgecolor='#cccccc', labelcolor='#333333', loc='lower right')
cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color='#333333')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#333333')

# Plot 4: Cross-section through center of 2D window
center_row = tukey_2d[width // 2, :]
x_coords_deg = np.linspace(-width_deg/2, width_deg/2, width)
ax4.plot(x_coords_deg, center_row, color='#007755', lw=2)
ax4.axvspan(-flat_half_deg, flat_half_deg, alpha=0.15, color='#007755', label=f'Flat region (±{flat_half_deg:.2f}°)')
ax4.axvline(-flat_half_deg, color='#cc8800', lw=1.5, linestyle='--')
ax4.axvline( flat_half_deg, color='#cc8800', lw=1.5, linestyle='--', label=f'Taper starts (±{flat_half_deg:.2f}°)')
ax4.set_title('2D Window — Horizontal Cross-Section', fontsize=11)
ax4.set_xlabel('x (°)')
ax4.set_ylabel('Amplitude')
ax4.legend(fontsize=8, facecolor='#ffffff', edgecolor='#cccccc', labelcolor='#333333')
ax4.set_xlim(-width_deg/2, width_deg/2)

fig.suptitle(f'Tukey Window  |  width={width_deg:.2f}°, α={alpha}  (1 px = {px_to_deg} °)', fontsize=14, color='#111111', y=1.01)

# plt.savefig('/mnt/user-data/outputs/tukey_plots.png', dpi=150, bbox_inches='tight',
#             facecolor=fig.get_facecolor())
# print("Saved to tukey_plots.png")
plt.show()
# %%
