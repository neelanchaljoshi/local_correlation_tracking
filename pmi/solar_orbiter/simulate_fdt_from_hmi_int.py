# %% imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.special import j1 as J1
from scipy.ndimage import zoom
from astropy.io import fits
from astropy.table import Table
import sys
sys.path.insert(0, '/data/seismo/zhichao/codes/pypkg')
from zclpy3.remap import rot_tan_map, from_tan_to_cyl
from astropy.wcs import WCS
from scipy import signal
# %% functions
def airy_disk_psf(shape, airy_radius_pixels):
    """Generate normalized 2D Airy disk PSF centered in array of given shape."""
    y, x = np.indices(shape)
    cy, cx = shape[0] // 2, shape[1] // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r[cy, cx] = 1e-10  # avoid division by zero
    k = np.pi * r / airy_radius_pixels
    psf = (2 * J1(k) / k)**2
    psf /= psf.sum()
    return psf

def compute_airy_radius_pixels(wavelength_m, aperture_m, pixel_scale_arcsec):
    """Compute Airy disk radius in pixels based on aperture and pixel scale."""
    theta_rad = 1.22 * wavelength_m / aperture_m  # angular radius of Airy disk
    pixel_scale_rad = pixel_scale_arcsec / 206265.0
    return theta_rad / pixel_scale_rad

def compute_pixel_scale_fdt(distance_au, base_pixel_scale_arcsec=3.75):
    """
    Compute the effective pixel scale (arcsec/pix) for the simulated FDT image.
    The instrument pixel size in µm is fixed, so the same angular pixel scale applies.
    But since you rescaled the Sun to the correct apparent size, you keep 3.75″/px.
    """
    return base_pixel_scale_arcsec  # fixed per design; no distance scaling

def radial_profile(psf):
    cy, cx = np.array(psf.shape) // 2
    y, x = np.indices(psf.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), psf.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)

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

def prepare_fdt_simulated_hmi(
    hmi,
    keys,
    pixel_scale_hmi=0.5,        # arcsec/px for HMI
    pixel_scale_fdt=3.57,       # arcsec/px for FDT
    distance_au_fdt=0.28,       # distance of FDT to Sun
    target_size=2048,
    pad_frac=0.05
):
    """
    Simulate PHI/FDT-like data from an HMI image.
    - Pads image slightly outside solar limb
    - Scales angular radius for given FDT distance
    - Downsamples with flux conservation
    """

    # --- Load HMI image ---
    data = hmi
    ny, nx = data.shape
    rsun_obs = keys['rsun_obs'][0]
    cdelt1 = keys['cdelt1'][0]
    R_hmi_px = rsun_obs / abs(cdelt1)
    print('rsun_obs', rsun_obs, 'cdelt1', cdelt1)
    print(f"HMI solar radius ≈ {R_hmi_px:.1f} px")

    # --- Compute geometric scaling ---
    scale_factor = (pixel_scale_hmi / pixel_scale_fdt) * (1.0 / distance_au_fdt)
    R_fdt_px = R_hmi_px * scale_factor
    print(f"Expected FDT solar radius ≈ {R_fdt_px:.1f} px (diam ≈ {2*R_fdt_px:.1f})")

    # --- Mask outside the solar disk + small margin ---
    y, x = np.indices((ny, nx))
    cy, cx = ny / 2, nx / 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    margin = R_hmi_px * pad_frac
    data[r > (R_hmi_px + margin)] = np.nan

    # --- Pad image symmetrically ---
    pad = int(np.ceil(margin))
    data_padded = np.pad(data, pad, constant_values=np.nan)
    cy_p, cx_p = np.array(data_padded.shape) / 2

    # --- Determine downsample (zoom-out) factor ---
    current_radius = R_hmi_px + pad
    downsample_factor = current_radius / R_fdt_px  # how much to shrink
    print(f"Downsample (zoom-out) factor ≈ {downsample_factor:.3f}")

    # --- Flux-conserving rescale ---
    # zoom expects inverse factor: new = old / factor
    # To conserve flux per unit area, divide by factor^2
    valid = np.isfinite(data_padded)
    data_filled = np.nan_to_num(data_padded, nan=0.0)
    rebinned = zoom(data_filled, 1.0 / downsample_factor, order=1) / (downsample_factor**2)
    mask_zoomed = zoom(valid.astype(float), 1.0 / downsample_factor, order=0)
    rebinned[mask_zoomed < 0.5] = np.nan

    # --- Center crop/pad to target size ---
    ny2, nx2 = rebinned.shape
    y0 = (ny2 - target_size) // 2
    x0 = (nx2 - target_size) // 2
    if y0 < 0 or x0 < 0:
        # pad if smaller
        pad_y = max(0, -y0)
        pad_x = max(0, -x0)
        rebinned = np.pad(rebinned, ((pad_y, pad_y), (pad_x, pad_x)), constant_values=np.nan)
    else:
        rebinned = rebinned[y0:y0+target_size, x0:x0+target_size]

    return rebinned

# %% Get PSF for FDT
# Inputs
wavelength = 617.3e-9       # meters
aperture_fdt = 0.0175       # meters
distance_au = 0.5          # e.g. 0.28 AU
pixel_scale_fdt = compute_pixel_scale_fdt(distance_au)

# Compute PSF radius on *FDT image grid*
radius_fdt_px = compute_airy_radius_pixels(
    wavelength_m=wavelength,
    aperture_m=aperture_fdt,
    pixel_scale_arcsec=pixel_scale_fdt
)
print(f"Airy radius on FDT image = {radius_fdt_px:.2f} px")

# Make PSF of suitable size for convolution
psf_shape = (64, 64)
psf_fdt = airy_disk_psf(psf_shape, radius_fdt_px)

# %% Plot PSF and radial profile
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im = ax[0].imshow(psf_fdt, cmap='viridis', norm=plt.Normalize(vmin=0, vmax=psf_fdt.max()))
ax[0].set_title('FDT Airy Disk PSF')
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
r_prof = radial_profile(psf_fdt)
ax[1].plot(r_prof, 'b-')
ax[1].set_title('Radial Profile of PSF')
ax[1].set_xlabel('Radius [px]')
ax[1].set_ylabel('Normalized Intensity')
plt.tight_layout()
plt.show()

# %% Load HMI data
keys = Table.read('/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.ic_45s/keys_new_swan/keys-2014.fits')
paths = keys['path']
hmi_image = fits.open(paths[0][:-1] + '/continuum.fits')[1].data

# %% Make the HMI into FDT-like data
fdt_no_blur = prepare_fdt_simulated_hmi(
    np.nan_to_num(hmi_image),
    keys=keys,
    pixel_scale_hmi=0.5,        # arcsec / px
    pixel_scale_fdt=3.57,       # arcsec / px
    distance_au_fdt=0.5,       # in AU
    target_size=2048,
    pad_frac=0.005               # small margin outside limb
)
# Convolve with FDT PSF
fdt_blurred = fftconvolve(np.nan_to_num(fdt_no_blur), psf_fdt, mode='same')

# %% Plot results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
im0 = ax[0].imshow(hmi_image, cmap='gray', vmin=np.nanpercentile(hmi_image, 5), vmax=np.nanpercentile(hmi_image, 95), origin = 'lower')
ax[0].set_title('Original HMI Image')
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(fdt_no_blur, cmap='gray', vmin=np.nanpercentile(fdt_no_blur, 5), vmax=np.nanpercentile(fdt_no_blur, 95), origin = 'lower')
ax[1].set_title('Simulated FDT (No Blur)')
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
im2 = ax[2].imshow(fdt_blurred, cmap='gray', vmin=np.nanpercentile(fdt_blurred, 5), vmax=np.nanpercentile(fdt_blurred, 95), origin = 'lower')
ax[2].set_title('Simulated FDT (With PSF Blur)')
fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# %% Pick a region to zoom in for all 3 images and they have to be the same physical region
# for hmi the region will be twice as big since it is 4k by 4k
# the zoom center would be different for hmi and fdt images
zoom_center_hmi = (2048, 2048)  # center of HMI image
zoom_size_hmi = 1000          # size of zoomed region in HMI
zoom_center_fdt = (1024, 1024)  # center of FDT image
zoom_size_fdt = 256            # size of zoomed region in FDT
hmi_zoomed = hmi_image[
    zoom_center_hmi[0]-zoom_size_hmi//2:zoom_center_hmi[0]+zoom_size_hmi//2,
    zoom_center_hmi[1]-zoom_size_hmi//2:zoom_center_hmi[1]+zoom_size_hmi//2
]
fdt_no_blur_zoomed = fdt_no_blur[
    zoom_center_fdt[0]-zoom_size_fdt//2:zoom_center_fdt[0]+zoom_size_fdt//2,
    zoom_center_fdt[1]-zoom_size_fdt//2:zoom_center_fdt[1]+zoom_size_fdt//2
]
fdt_blurred_zoomed = fdt_blurred[
    zoom_center_fdt[0]-zoom_size_fdt//2:zoom_center_fdt[0]+zoom_size_fdt//2,
    zoom_center_fdt[1]-zoom_size_fdt//2:zoom_center_fdt[1]+zoom_size_fdt//2
]
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
im0 = ax[0].imshow(hmi_zoomed, cmap='gray', vmin=np.nanpercentile(hmi_zoomed, 1), vmax=np.nanpercentile(hmi_zoomed, 99), origin = 'lower')
ax[0].set_title('Zoomed HMI Image')
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(fdt_no_blur_zoomed, cmap='gray', vmin=np.nanpercentile(fdt_no_blur_zoomed, 1), vmax=np.nanpercentile(fdt_no_blur_zoomed, 99), origin = 'lower')
ax[1].set_title('Zoomed Simulated FDT (No Blur)')
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
im2 = ax[2].imshow(fdt_blurred_zoomed, cmap='gray', vmin=np.nanpercentile(fdt_blurred_zoomed, 1), vmax=np.nanpercentile(fdt_blurred_zoomed, 99), origin = 'lower')
ax[2].set_title('Zoomed Simulated FDT (With PSF Blur)')
fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
# %% Make the radius same for FDT at different distances
img = hmi_image
wcs_in = WCS(naxis=2)
wcs_in.wcs.crpix = keys['crpix1'][0], keys['crpix2'][0]
wcs_in.wcs.cdelt = keys['cdelt1'][0], keys['cdelt2'][0]
wcs_in.wcs.crval = keys['crval1'][0], keys['crval2'][0]

wcs_out = WCS(naxis=2)
nx_out, ny_out = 2048, 2048
wcs_out.wcs.crpix = 0.5*(1+nx_out), 0.5*(1+ny_out)
wcs_out.wcs.cdelt = 3.57, 3.57  # FDT pixel scale
wcs_out.wcs.crval = 0.0, 0.0
wcs_out.wcs.cunit = 'arcsec', 'arcsec'

# compute radius of sun in arcsec at given distance using hmi and fdt parameters
dsun_au = 0.5
rsun_hmi_arcsec = keys['rsun_obs'][0]  # in arcsec
rsun_fdt_arcsec = rsun_hmi_arcsec * (1.0 / dsun_au)  # in arcsec
rsun_obs_out = rsun_fdt_arcsec
print('rsun_obs_out', rsun_obs_out)
print('rsun_hmi_arcsec', rsun_hmi_arcsec)

rsun_obs_in = keys['rsun_obs'][0]
dB = 0
dP = 0
print('dB', dB, 'dP', dP)
img_out = rot_tan_map(img, wcs_in, rsun_obs_in, dB, dP,
        nx_out, ny_out, wcs_out, rsun_obs_out,
        interp_method=b'cubconv')

# %% Plot final result
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im0 = ax[0].imshow(hmi_image, cmap='gray', vmin=np.nanpercentile(hmi_image, 5), vmax=np.nanpercentile(hmi_image, 95), origin = 'lower')
ax[0].set_title('Original HMI Image')
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(img_out, cmap='gray', vmin=np.nanpercentile(img_out, 5), vmax=np.nanpercentile(img_out, 95), origin = 'lower')
ax[1].set_title('Simulated FDT Image at 0.5 AU')
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# %% Convolve with PSF
img_blurred = fftconvolve(np.nan_to_num(img_out), psf_fdt, mode='same')
# %% Plot final blurred result
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im0 = ax[0].imshow(np.nan_to_num(img_out), cmap='gray', vmin=np.nanpercentile(img_out, 5), vmax=np.nanpercentile(img_out, 95), origin = 'lower')
ax[0].set_title('Simulated FDT Image at 0.5 AU (No Blur)')
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(img_blurred, cmap='gray', vmin=np.nanpercentile(img_blurred, 5), vmax=np.nanpercentile(img_blurred, 95), origin = 'lower')
ax[1].set_title('Simulated FDT Image at 0.5 AU (With PSF Blur)')
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# %% zoom in on both images
zoom_center = (1024, 1024)  # center of FDT image
zoom_size = 256            # size of zoomed region in FDT
img_out_zoomed = img_out[
    zoom_center[0]-zoom_size//2:zoom_center[0]+zoom_size//2,
    zoom_center[1]-zoom_size//2:zoom_center[1]+zoom_size//2
]
img_blurred_zoomed = img_blurred[
    zoom_center[0]-zoom_size//2:zoom_center[0]+zoom_size//2,
    zoom_center[1]-zoom_size//2:zoom_center[1]+zoom_size//2
]
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im0 = ax[0].imshow(np.nan_to_num(img_out_zoomed), cmap='gray', vmin=np.nanpercentile(img_out_zoomed, 1), vmax=np.nanpercentile(img_out_zoomed, 99), origin = 'lower')
ax[0].set_title('Zoomed Simulated FDT Image at 0.5 AU (No Blur)')
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(np.nan_to_num(img_blurred_zoomed), cmap='gray', vmin=np.nanpercentile(img_blurred_zoomed, 1), vmax=np.nanpercentile(img_blurred_zoomed, 99), origin = 'lower')
ax[1].set_title('Zoomed Simulated FDT Image at 0.5 AU (With PSF Blur)')
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()


# %% Convert blurred and non blurred images to cylindrical maps
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
crpix1, crpix2 = wcs_out.wcs.crpix
crval1, crval2 = wcs_out.wcs.crval
cdelt1, cdelt2 = wcs_out.wcs.cdelt
rsun_obs = rsun_obs_out
print('rsun_obs for cyl', rsun_obs)
img_no_blur_helio = from_tan_to_cyl(img_out[np.newaxis, :, :],
        crpix1, crpix2, crval1, crval2,
        cdelt1, cdelt2,
        rsun_obs, dB, dP, dL,
        nx_out, ny_out, wcs_cyl,
        interp_method, verbose, nthr, header=False)

img_blurred_helio = from_tan_to_cyl(img_blurred[np.newaxis, :, :],
        crpix1, crpix2, crval1, crval2, cdelt1, cdelt2,
        rsun_obs, dB, dP, dL,
        nx_out, ny_out, wcs_cyl,
        interp_method, verbose, nthr, header=False)

lng, = wcs_cyl.sub([1]).wcs_pix2world(range(ncyl), 0)
lat, = wcs_cyl.sub([2]).wcs_pix2world(range(ncyl), 0)

print(lng, lat)
# %% plot maps using lng, lat pcolormesh

fig, ax = plt.subplots(1, 2, figsize=(15, 7))
im0 = ax[0].pcolormesh(lng, lat, img_no_blur_helio[0], cmap='gray', shading='auto',
                       vmin=np.nanpercentile(img_no_blur_helio, 5), vmax=np.nanpercentile(img_no_blur_helio, 95))
ax[0].set_title('Cylindrical Map from Simulated FDT Image (No Blur)')
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].pcolormesh(lng, lat, img_blurred_helio[0], cmap='gray', shading='auto',
                       vmin=np.nanpercentile(img_blurred_helio, 5), vmax=np.nanpercentile(img_blurred_helio, 95))
ax[1].set_title('Cylindrical Map from Simulated FDT Image (With PSF Blur)')
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
# %% Zoom in on both maps
zoom_lat_center = 0
zoom_lng_center = 0
zoom_size_deg = 5  # degrees
lat_mask = (lat >= zoom_lat_center - zoom_size_deg/2) & (lat <= zoom_lat_center + zoom_size_deg/2)
lng_mask = (lng >= zoom_lng_center - zoom_size_deg/2) & (lng <= zoom_lng_center + zoom_size_deg/2)
img_no_blur_zoomed = img_no_blur_helio[0][np.ix_(lat_mask, lng_mask)]
img_blurred_zoomed = img_blurred_helio[0][np.ix_(lat_mask, lng_mask)]
lng_zoomed = lng[lng_mask]
lat_zoomed = lat[lat_mask]
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
im0 = ax[0].pcolormesh(lng_zoomed, lat_zoomed, img_no_blur_zoomed, cmap='gray', shading='auto',
                       vmin=np.nanpercentile(img_no_blur_zoomed, 1), vmax=np.nanpercentile(img_no_blur_zoomed, 99))
ax[0].set_title('Zoomed Cylindrical Map from Simulated FDT Image (No Blur)')
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].pcolormesh(lng_zoomed, lat_zoomed, img_blurred_zoomed, cmap='gray', shading='auto',
                       vmin=np.nanpercentile(img_blurred_zoomed, 1), vmax=np.nanpercentile(img_blurred_zoomed, 99))
ax[1].set_title('Zoomed Cylindrical Map from Simulated FDT Image (With PSF Blur)')
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
# %% Get ccf
patch_size = img_blurred_zoomed.shape[0]
gkern1 = tukey_twoD(patch_size, alpha = 0.8)
ccf_no_blur, patch1, patch2 = get_lct_map(img_no_blur_zoomed, img_no_blur_zoomed, patch_size, gkern1)
ccf_blurred, patch1, patch2 = get_lct_map(img_blurred_zoomed, img_blurred_zoomed, patch_size, gkern1)
# Plot the ccf and cuts through the centre of the ccf in a 3 panel figure
# convert longitude and latitude from deg to Mm only for the 15 deg
R_sun_m = 6.957e8  # meters
deg_to_m = (np.pi/180) * R_sun_m
lng_m = lng_zoomed * deg_to_m / 1e6  # in Mm
lat_m = lat_zoomed * deg_to_m / 1e6  # in Mm

# %% ccf plot for no blur
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
im = ax[0].pcolormesh(lng_m, lat_m, ccf_no_blur, cmap='viridis', shading='auto')
ax[0].set_title('Cross-Correlation Function (CCF)')
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
ax[1].plot(lng_m, ccf_no_blur[patch_size//2, :], 'b-')
ax[1].set_title('CCF Cut Along Longitude')
ax[1].set_xlabel('Longitude [Mm]')
ax[1].set_ylabel('CCF Amplitude')
ax[2].plot(lat_m, ccf_no_blur[:, patch_size//2], 'r-')
ax[2].set_title('CCF Cut Along Latitude')
ax[2].set_xlabel('Latitude [Mm]')
ax[2].set_ylabel('CCF Amplitude')
plt.tight_layout()
plt.show()


# %% plot for blurred
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
im = ax[0].pcolormesh(lng_m, lat_m, ccf_blurred, cmap='viridis', shading='auto')
ax[0].set_title('Cross-Correlation Function (CCF) - Blurred')
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
ax[1].plot(lng_m, ccf_blurred[patch_size//2, :], 'b-')
ax[1].set_title('CCF Cut Along Longitude - Blurred')
ax[1].set_xlabel('Longitude [Mm]')
ax[1].set_ylabel('CCF Amplitude')
ax[2].plot(lat_m, ccf_blurred[:, patch_size//2], 'r-')
ax[2].set_title('CCF Cut Along Latitude - Blurred')
ax[2].set_xlabel('Latitude [Mm]')
ax[2].set_ylabel('CCF Amplitude')
plt.tight_layout()
plt.show()
# %%
