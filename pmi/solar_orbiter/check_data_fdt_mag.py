# %% imports
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import sys
sys.path.insert(0, '/data/seismo/zhichao/codes/pypkg')
from zclpy3.remap import rot_tan_map, from_tan_to_cyl, from_tan_to_postel
# %% Load the solar orbiter continuum intensity data
files = sorted(glob.glob('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/solar_orbiter/FDT_data/blos/*/*.fits.gz'))
file1 = files[1100]
file2 = files[1100]


# %% Read the FITS file
with fits.open(file1) as hdul:
    print(file1)
    header = hdul[0].header
    data = hdul[0].data

with fits.open(file2) as hdul:
    print(file2)
    header2 = hdul[0].header
    data2 = hdul[0].data
# %% Print header information
print(header.tostring(sep='\n'))
# %% Plot the data
plt.figure(figsize=(8, 8))
plt.imshow(data, cmap='gray', origin='lower', vmax = 100, vmin = -100)
plt.colorbar(label='Magnetic Field (Gauss)', shrink=0.8)
plt.title('Solar MF at time {}'.format(header['date']))
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.show()
print(data.shape)
# Print some key header information
# %%
print(header['crota'])
print(header['cdelt1'])
print(header['cdelt2'])
print(header['crpix1'])
print(header['crpix2'])
print(header['crval1'])
print(header['crval2'])
print(header['date'])
print(header['rsun_arc'])
# %%
for key, value in header.items():
    print(f"{key} = {value}")

# %%
print(281*3.75)

# %%
plt.plot(data[int(header['crpix2']), :])

# %% Rotate and remap the data
from astropy.wcs import WCS
ncyl = 1024
nx_out = ny_out = 1024
wcs_cyl  = WCS(naxis=2)
wcs_cyl.wcs.ctype = 'longitude', 'latitude'
wcs_cyl.wcs.cunit = 'deg', 'deg'
wcs_cyl.wcs.crpix = 0.5*(1+ncyl), 0.5*(1+ncyl)
wcs_cyl.wcs.crval = 0, 0
wcs_cyl.wcs.cdelt = 0.167, 0.167
dL = 0
interp_method = 'bilinear'
verbose = 1
nthr = 1
crpix1, crpix2 = header['crpix1'], header['crpix2']
crval1, crval2 = header['crval1'], header['crval2']
cdelt1, cdelt2 = header['cdelt1'], header['cdelt2']
rsun_obs = header['rsun_arc']
dB = header['crlt_obs']
dP = -header['crota']
print('rsun_obs for cyl', rsun_obs)
patch_size = 30
clng = 0
clat = 0
pixel_size = 0.167
# img_no_blur_helio = from_tan_to_cyl(data[np.newaxis, :, :],
#         crpix1, crpix2, crval1, crval2,
#         cdelt1, cdelt2,
#         rsun_obs, dB, dP, dL,
#         nx_out, ny_out, wcs_cyl,
#         interp_method, verbose, nthr, header=False)
img_no_blur_helio = from_tan_to_postel(data[np.newaxis, :, :], [crpix1], [crpix2],0, 0, cdelt1, cdelt2, [rsun_obs], dB, dP, dL, nx_out = patch_size,
							ny_out = patch_size, lngc_out = clng,latc_out = clat, pixscale_out = pixel_size,
							interp_method = 'bilinear', verbose = 1, nthr = 1, header = False)
# %% Second image
crpix1_2, crpix2_2 = header2['crpix1'], header2['crpix2']
crval1_2, crval2_2 = header2['crval1'], header2['crval2']
cdelt1_2, cdelt2_2 = header2['cdelt1'], header2['cdelt2']
rsun_obs_2 = header2['rsun_arc']
dB_2 = header2['crlt_obs']
dP_2 = -header2['crota']
dL = header['crln_obs'] - header2['crln_obs']
print('rsun_obs for cyl', rsun_obs_2)
# img_no_blur_helio_2 = from_tan_to_cyl(data2[np.newaxis, :, :],
#         crpix1_2, crpix2_2, crval1_2, crval2_2,
#         cdelt1_2, cdelt2_2,
#         rsun_obs_2, dB_2, dP_2, dL,
#         nx_out, ny_out, wcs_cyl,
#         interp_method, verbose, nthr, header=False)
img_no_blur_helio_2 = from_tan_to_postel(data2[np.newaxis, :, :], crpix1_2, crpix2_2,0, 0, cdelt1_2, cdelt2_2,rsun_obs_2, dB_2, dP_2, dL, nx_out = patch_size,
                            ny_out = patch_size, lngc_out = clng,latc_out = clat, pixscale_out = pixel_size,
                            interp_method = 'bilinear', verbose = 1, nthr = 1, header = False)
# %% Plot the remapped data
lng, = wcs_cyl.sub([1]).wcs_pix2world(range(ncyl), 0)
lat, = wcs_cyl.sub([2]).wcs_pix2world(range(ncyl), 0)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(img_no_blur_helio[0, :, :], extent=(lng[0], lng[-1], lat[0], lat[-1]),
               cmap='gray', origin='lower', vmax = 100, vmin = -100)
fig.colorbar(im, ax=ax, label='Magnetic Field (units)')
# ax.set_xlim(-7.5, 7.5)
# ax.set_ylim(-7.5, 7.5)
ax.set_title('Remapped Solar MF Data')
ax.set_xlabel('Longitude (degrees)')
ax.set_ylabel('Latitude (degrees)')
plt.show()
# %% get cross correlation function of 5 deg patch
from scipy import signal
def tukey_twoD(width, alpha = 0.8):
	"""2D tukey lowpass window with a circular support
	"""
	base = np.zeros((width, width))
	tukey = signal.windows.tukey(width, alpha)
	tukey = tukey[int(len(tukey)/2)-1:]  # Second half of tukey window
	x = np.linspace(-width/2, width/2, width)
	y = np.linspace(-width/2, width/2, width)
	for x_index in range(0, width):
		for y_index in range(0, width):
			# Only plot tukey value with in circle of radius width
			if int(np.hypot(x[x_index], y[y_index])) <= width/2:
				base[x_index, y_index] = tukey[int(np.hypot(x[x_index], y[y_index]))]
					# Based on k**2 find tukey window value and place in matrix
	return base
def get_lct_map(patch1, patch2, patch_size, gkern1):

	patch1 = gkern1*(patch1 - np.nanmean(patch1))
	patch2 = gkern1*(patch2 - np.nanmean(patch2))

	#patch1 = patch1*gkern1
	#patch2 = patch2*gkern1

	f = np.fft.rfft2(patch1)
	g = np.fft.rfft2(patch2)
	c = np.conj(f)*g

	ccf = np.fft.irfft2(c, s= (patch_size, patch_size), axes = [0,1])
	ccf = np.fft.fftshift(ccf, axes = [0,1])
	return ccf, patch1, patch2

patch1 = img_no_blur_helio[0]
patch2 = img_no_blur_helio_2[0]
gkern1 = tukey_twoD(patch_size, alpha = 0.8)
ccf, patch1_win, patch2_win = get_lct_map(patch1, patch2, patch_size, gkern1)
# %% plot cross correlation function and cuts through the centre with x and y in Mm


# Parameters
N = patch_size                   # number of points
deg_per_pixel = 0.167      # angular size per pixel
R_sun = 696                # solar radius in Mm

# Convert pixel size from degrees to Mm
pix_size_mm = R_sun * np.deg2rad(deg_per_pixel)

# Create array centered at zero
x_mm = (np.arange(N) - (N - 1) / 2) * pix_size_mm
y_mm = (np.arange(N) - (N - 1) / 2) * pix_size_mm
# Plot ccf with x and y in Mm
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.pcolormesh(x_mm, y_mm, ccf, cmap='viridis', shading='auto')
fig.colorbar(im, ax=ax, label='Cross-Correlation')
ax.set_title('Cross-Correlation Function')
ax.set_xlabel('X (Mm)')
ax.set_ylabel('Y (Mm)')
plt.show()



# %% plot the two patches in Mm
# Get t1 and t2 of observation
t1 = header['date']
t2 = header2['date']
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].pcolormesh(x_mm, y_mm, patch1, cmap='gray', shading='auto', vmax = 50, vmin = -50)
ax[0].set_title(f'{t1}')
ax[0].set_xlabel('X (Mm)')
ax[0].set_ylabel('Y (Mm)')
ax[1].pcolormesh(x_mm, y_mm, patch2, cmap='gray', shading='auto', vmax = 50, vmin = -50)
ax[1].set_title(f'{t2}')
ax[1].set_xlabel('X (Mm)')
ax[1].set_ylabel('Y (Mm)')
plt.show()


# %% Plot cuts through the max of the ccf in x and y
max_index = np.unravel_index(np.argmax(ccf), ccf.shape)
print("Max index of CCF:", max_index)
x_cut = ccf[max_index[0], :]
y_cut = ccf[:, max_index[1]]
fig, ax = plt.subplots(figsize=(8, 5))

# Plot lines with nicer aesthetics
ax.plot(x_mm, x_cut, lw=2, alpha=0.9, label='X Cut through Max')
ax.plot(y_mm, y_cut, lw=2, alpha=0.9, label='Y Cut through Max')

# Labels and title
ax.set_xlabel('Distance (Mm)', fontsize=14)
ax.set_ylabel('Cross-Correlation', fontsize=14)
ax.set_title('Cuts through Maximum of Cross-Correlation Function', fontsize=16, pad=12)

# Grid for readability (light and subtle)
ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.5)

# Legend styling
ax.legend(fontsize=12, frameon=False)

# Tick parameters
ax.tick_params(axis='both', labelsize=12, length=6)
ax.minorticks_on()
ax.tick_params(which='minor', length=3)

# Tight layout for nice spacing
plt.tight_layout()

plt.show()

# %% Print max index and corresponding distance in Mm
print(y_mm)
# %% Find FWHM of the cuts
def find_fwhm(x, y):
    half_max = np.max(y) / 2.0
    indices = np.where(y >= half_max)[0]
    if len(indices) < 2:
        return None  # FWHM not found
    fwhm = x[indices[-1]] - x[indices[0]]
    return fwhm
fwhm_x = find_fwhm(x_mm, x_cut)
fwhm_y = find_fwhm(y_mm, y_cut)
print(f"FWHM in X: {fwhm_x} Mm")
print(f"FWHM in Y: {fwhm_y} Mm")


# %%
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

# Plot lines with nicer aesthetics
ax.plot(x_mm, x_cut, lw=2, alpha=0.9, label='X Cut through Max')
ax.plot(y_mm, y_cut, lw=2, alpha=0.9, label='Y Cut through Max')

# Labels and title
ax.set_xlabel('Distance (Mm)', fontsize=14)
ax.set_ylabel('Cross-Correlation', fontsize=14)
ax.set_title('Cuts through Maximum of Cross-Correlation Function', fontsize=16, pad=12)

# Grid for readability
ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.5)

# Legend styling
ax.legend(fontsize=12, frameon=False)

# Tick styling
ax.tick_params(axis='both', labelsize=12, length=6)
ax.minorticks_on()
ax.tick_params(which='minor', length=3)

# ----- FWHM calculation -----
def find_fwhm(x, y):
    half_max = np.max(y) / 2.0
    indices = np.where(y >= half_max)[0]
    if len(indices) < 2:
        return None
    return x[indices[-1]] - x[indices[0]]

fwhm_x = find_fwhm(x_mm, x_cut)
fwhm_y = find_fwhm(y_mm, y_cut)

# ----- Add FWHM text inside the figure -----
textstr = (
    f"FWHM X = {fwhm_x:.2f} Mm\n"
    f"FWHM Y = {fwhm_y:.2f} Mm"
)

ax.text(
    0.02, 0.95, textstr,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
)

plt.tight_layout()
plt.show()


# %%
