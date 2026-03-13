# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import odr
import matplotlib.gridspec as gridspec

# %%
# Load flow data
vlos = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/supergran/data/detrended_vlos/vlos_cleaned_langfellner_90s.npy')
f = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/supergran/data/detrended_vlos/detrended_vlos_from_lct_mean_flows_dspan_90s.npz')
lct = f['v_los_detrended']
longitude = f['longitude']
latitude = f['latitude']

print(longitude, latitude)

# Plot the comparison with vlos on x axis and lct on y
plt.figure(figsize=(8, 8))
plt.scatter(vlos, lct, s=1, alpha=0.5)
plt.xlabel('vlos (m/s)')
plt.ylabel('LCT-derived LOS Velocity (m/s)')
plt.title('Comparison of LCT-derived LOS Velocity with vlos')
plt.xlim(-500, 500)
plt.ylim(-500, 500)
plt.plot([-500, 500], [-500, 500], 'r--', label='y=x')
plt.legend()
plt.grid()
plt.show()

# %%
# Fit a line to the scatter plot using least square fit

A = np.vstack([vlos.flatten(), np.ones_like(vlos.flatten())]).T
slope, intercept = np.linalg.lstsq(A, lct.flatten(), rcond=None)[0]

# %% ODR fit
def linear_func(B, x):
    return B[0] * x + B[1]
model = odr.Model(linear_func)
data = odr.RealData(vlos.flatten(), lct.flatten())
odr_instance = odr.ODR(data, model, beta0=[slope, intercept])
odr_output = odr_instance.run()
slope, intercept = odr_output.beta

# %%

fig = plt.figure(figsize=(16, 4))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

# First subplot - pcolormesh
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.pcolormesh(longitude, latitude, vlos, cmap='coolwarm', shading='auto', vmax = 500, vmin = -500)
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title('Doppler LOS Velocity [m/s]')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('[m/s]')

# Second subplot - pcolormesh
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.pcolormesh(longitude, latitude, lct, cmap='coolwarm', shading='auto', vmax = 500, vmin = -500)
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.set_title('LCT-derived LOS Velocity [m/s]')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('[m/s]')


# Third subplot - scatter
ax3 = fig.add_subplot(gs[0, 2])
scatter = ax3.scatter(vlos.flatten(), lct.flatten(), c=lct.flatten(), cmap='coolwarm', s=1, alpha=0.5)
line = ax3.plot(vlos.flatten(), slope * vlos.flatten() + intercept, 'r-', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('Scatter Plot')
ax3.grid(True, alpha=0.3)
cbar3 = plt.colorbar(scatter, ax=ax3)
cbar3.set_label('Y values')
ax3.legend()

plt.tight_layout()
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patches

# Set style for better aesthetics
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

fig = plt.figure(figsize=(18, 5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35, hspace=0.3)

# First subplot - Doppler velocity
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.pcolormesh(longitude, latitude, vlos, cmap='RdBu_r',
                      shading='auto', vmax=500, vmin=-500, rasterized=True)
ax1.set_xlabel('Longitude [degrees]', fontsize=12, fontweight='bold')
ax1.set_ylabel('Latitude [degrees]', fontsize=12, fontweight='bold')
ax1.set_title('Doppler LOS Velocity', fontsize=13, fontweight='bold', pad=10)
ax1.set_aspect('equal', adjustable='box')
cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
cbar1.set_label('Velocity [m/s]', fontsize=11, fontweight='bold')
cbar1.ax.tick_params(labelsize=10)
ax1.tick_params(labelsize=10)

# Second subplot - LCT velocity
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.pcolormesh(longitude, latitude, lct, cmap='RdBu_r',
                      shading='auto', vmax=500, vmin=-500, rasterized=True)
ax2.set_xlabel('Longitude [degrees]', fontsize=12, fontweight='bold')
ax2.set_ylabel('Latitude [degrees]', fontsize=12, fontweight='bold')
ax2.set_title('LCT-Derived LOS Velocity', fontsize=13, fontweight='bold', pad=10)
ax2.set_aspect('equal', adjustable='box')
cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02)
cbar2.set_label('Velocity [m/s]', fontsize=11, fontweight='bold')
cbar2.ax.tick_params(labelsize=10)
ax2.tick_params(labelsize=10)

# Third subplot - correlation scatter
ax3 = fig.add_subplot(gs[0, 2])

# Create 2D histogram for better visualization of dense data
vlos_flat = vlos.flatten()
lct_flat = lct.flatten()

# Remove NaN values
mask = ~(np.isnan(vlos_flat) | np.isnan(lct_flat))
vlos_clean = vlos_flat[mask]
lct_clean = lct_flat[mask]

# 2D histogram
h = ax3.hexbin(vlos_clean, lct_clean, gridsize=50, cmap='viridis',
               mincnt=1, alpha=0.8, edgecolors='none')

# Fit line
line = ax3.plot(vlos_clean, slope * vlos_clean + intercept, 'r-',
                linewidth=2.5, label=f'y = {slope:.3f}x + {intercept:.2f}',
                zorder=10)

# 1:1 reference line
ax3.plot([-500, 500], [-500, 500], 'k--', linewidth=1.5,
         alpha=0.5, label='1:1 Reference', zorder=5)

ax3.set_xlabel('Doppler LOS Velocity [m/s]', fontsize=12, fontweight='bold')
ax3.set_ylabel('LCT-Derived LOS Velocity [m/s]', fontsize=12, fontweight='bold')
ax3.set_title('Method Comparison', fontsize=13, fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax3.set_xlim(-500, 500)
ax3.set_ylim(-500, 500)
ax3.set_aspect('equal', adjustable='box')

# Colorbar for hexbin
cbar3 = plt.colorbar(h, ax=ax3, pad=0.02)
cbar3.set_label('Point Density', fontsize=11, fontweight='bold')
cbar3.ax.tick_params(labelsize=10)

# Legend with better styling
legend = ax3.legend(loc='upper left', fontsize=10, framealpha=0.9,
                    edgecolor='black', fancybox=True, shadow=True)
ax3.tick_params(labelsize=10)

# Add correlation coefficient if you have it
# r_squared = 0.85  # Replace with your actual value
# ax3.text(0.05, 0.95, f'$R^2$ = {r_squared:.3f}',
#          transform=ax3.transAxes, fontsize=11,
#          verticalalignment='top', bbox=dict(boxstyle='round',
#          facecolor='wheat', alpha=0.5))

plt.tight_layout()
# plt.savefig('velocity_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Simple, clean settings
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.2,
})

# Create figure
fig = plt.figure(figsize=(16, 4))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

# First subplot - Doppler velocity
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.pcolormesh(longitude, latitude, vlos, cmap='RdBu_r',
                      shading='auto', vmax=500, vmin=-500)
ax1.set_xlabel('Longitude [deg]', fontsize=14)
ax1.set_ylabel('Latitude [deg]', fontsize=14)
ax1.set_title('Doppler LOS Velocity', fontsize=15, pad=15)
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('m/s', fontsize=13)

# Second subplot - LCT velocity
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.pcolormesh(longitude, latitude, lct, cmap='RdBu_r',
                      shading='auto', vmax=500, vmin=-500)
ax2.set_xlabel('Longitude [deg]', fontsize=14)
ax2.set_ylabel('Latitude [deg]', fontsize=14)
ax2.set_title('LCT-Derived LOS Velocity', fontsize=15, pad=15)
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('m/s', fontsize=13)

# Third subplot - correlation
ax3 = fig.add_subplot(gs[0, 2])

# Remove NaN values
vlos_flat = vlos.flatten()
lct_flat = lct.flatten()
mask = ~(np.isnan(vlos_flat) | np.isnan(lct_flat))
vlos_clean = vlos_flat[mask]
lct_clean = lct_flat[mask]

# Hexbin for density
h = ax3.hexbin(vlos_clean, lct_clean, gridsize=50, cmap='viridis',
               mincnt=1, linewidths=0)

# Fit line
ax3.plot([-500, 500], [slope * -500 + intercept, slope * 500 + intercept],
         'r-', linewidth=2.5, label=f'Fit: y = {slope:.2f}x + {intercept:.1f}')

# 1:1 reference line
ax3.plot([-500, 500], [-500, 500], 'k--', linewidth=2, alpha=0.6, label='1:1')

ax3.set_xlabel('Doppler LOS [m/s]', fontsize=14)
ax3.set_ylabel('LCT LOS [m/s]', fontsize=14)
ax3.set_title('Correlation', fontsize=15, pad=15)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-500, 500)
ax3.set_ylim(-500, 500)

# Colorbar
cbar3 = plt.colorbar(h, ax=ax3)
cbar3.set_label('Density', fontsize=13)

# Legend
ax3.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig('figures/velocity_comparison_15min.pdf', bbox_inches='tight')
plt.show()
# %%
