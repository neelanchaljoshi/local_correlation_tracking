#!/usr/bin/env python
# %% import
import os, sys, socket
import argparse
import logging
from datetime import datetime, timedelta
import numpy as np
from astropy.io import fits
from scipy import signal
from astropy.table import Table
from scipy.linalg import lstsq
from scipy.ndimage import shift
from scipy.ndimage.interpolation import zoom
from scipy.interpolate import CubicSpline
import h5py
sys.path.insert(0, '/data/seismo/joshin/pypkg')
from josh.misc import xdays
sys.path.insert(0, '/data/seismo/zhichao/codes/pypkg')
from zclpy3.remap import from_tan_to_postel

# %% functions
def fast_cubic_spline_interpolate_stack(img_stack, target_time=60.0, times=[0, 45, 90, 135]):
	"""
	Fast cubic spline interpolation from a stack of 4 images taken at specified times.

	Parameters:
	img_stack : np.ndarray
		Array of shape (4, n, n) with images taken at 0s, 45s, 90s, and 135s.
	target_time : float
		Time (in seconds) at which to interpolate the image.
	times : list or np.ndarray
		List of time points corresponding to the images in img_stack. Default is [0, 45, 90, 135].

	Returns:
	np.ndarray
		Interpolated image at target_time, shape (n, n).
	"""
	if img_stack.shape[0] != 4:
		raise ValueError("img_stack must have shape (4, n, n)")

	n, m = img_stack.shape[1], img_stack.shape[2]
	images_flat = img_stack.reshape(4, -1)  # shape: (4, n*n)

	# Spline interpolation along the time axis
	cs = CubicSpline(times, images_flat, axis=0)
	interpolated_flat = cs(target_time)  # shape: (n*n,)

	return interpolated_flat.reshape(n, m)


def gather_bigsize(comm, obj, root):
	"""
	This routine is to replace comm.gather() because comm.gather() gets stuck on swan when size > 30.

	Parameters
	----------
	comm: MPI.COMM_WORLD
	obj: any python object
	root: int
		rank of dest

	Returns
	-------
	out: list
		list of obj gathered from all ranks

	Examples
	--------
	msg = 'rank %d on %s' % (rank, host)
	#msg = comm.gather(msg, root=0)
	msg = gather_bigsize(comm, msg, root=0)
	"""
	rank = comm.Get_rank()
	size = comm.Get_size()
	out = [] if rank == root else None
	for src in range(size):
		if src == root:
			buf = obj
		else:
			if rank == src:
				comm.send(obj, dest=root)
			elif rank == root:
				buf = comm.recv(source=src)
		if rank == root:
			out.append(buf)
	return out

def downsample_img(img, factor):
	# Downsample the image by a factor of factor
	down_img = zoom(np.nan_to_num(img), 1/factor, order = 1)
	return down_img


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

def get_lct_map(patch1, patch2):

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



def get_ux_uy_ellipsoid(ccf_av, grid_len, patch_size, pixel_size, R_sun, ntry = 3):
	# Finds the ux and uy from the ccf_av
	dx_tot = 0
	dy_tot = 0

	assert grid_len%2 == 1

	for n in range(ntry):
		ym, xm = np.where(ccf_av==ccf_av[1:-1, 1:-1].max())
		xmax = xm[0]
		ymax = ym[0]

		fxy = ccf_av[ymax-grid_len//2:ymax+(grid_len+1)//2, xmax-grid_len//2:xmax+(grid_len+1)//2]
		val_arr = fxy.flatten()

		x = y = np.linspace(-grid_len//2+1, grid_len//2, grid_len)

		coeff_arr = []
		for j, i in [[_i, _j] for _i in x for _j in y]:
			coeff_arr.append([i**2, j**2, i, j, i*j, 1])

		coeff_array = np.array(coeff_arr)
		p, res, rnk, s = lstsq(coeff_array, val_arr)

		a,b,c,d,e,f = p

		ypar = ((e*c)-(2*a*d))/((4*a*b-(e**2)))
		xpar= (-e*ypar - c)/(2*a)

		del_x = xmax  - patch_size//2 + xpar
		del_y = ymax  - patch_size//2 + ypar

		ccf_av = shift(ccf_av,[-del_y, -del_x], mode = 'reflect')
		dx_tot = dx_tot + del_x
		dy_tot = dy_tot + del_y

	#Calculate ux, uy
	ux = R_sun * dx_tot * np.deg2rad(pixel_size)/(cadence_interp) * 1e6
	uy = R_sun * dy_tot * np.deg2rad(pixel_size)/(cadence_interp) * 1e6

	return dx_tot, dy_tot, ux, uy

from scipy.signal import fftconvolve
from scipy.special import j1 as J1


def airy_disk_psf(shape, airy_radius_pixels):
    """Generate a normalized 2D Airy disk PSF."""
    y, x = np.indices(shape)
    cy, cx = shape[0] // 2, shape[1] // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r[cy, cx] = 1e-10  # avoid division by zero
    k = np.pi * r / airy_radius_pixels
    psf = (2 * J1(k) / k)**2
    psf /= psf.sum()
    return psf


def compute_airy_radius_pixels(wavelength_m, aperture_m, pixel_scale_arcsec):
    """Calculate Airy disk radius in image pixels."""
    theta_rad = 1.22 * wavelength_m / aperture_m
    pixel_scale_rad = pixel_scale_arcsec / 206265.0
    return theta_rad / pixel_scale_rad

def simulate_pmi_from_hmi(hmi_image, psf_rel):
    """
    Simulate a PMI image from an HMI-like image using relative PSF and 2x2 box mean downsampling.

    Parameters:
        hmi_image (ndarray): 2D input image (e.g., 4096x4096 HMI image)
        wavelength_m (float): Observation wavelength in meters
        aperture_hmi (float): Aperture size of HMI instrument in meters
        aperture_pmi (float): Aperture size of PMI instrument in meters
        pixel_scale (float): Pixel scale in arcseconds/pixel (same for HMI and PMI assumed)

    Returns:
        ndarray: 2x2 downsampled and PSF-blurred image (e.g., 2048x2048)
    """
    # Compute Airy disk radii


    # Convolve with relative PSF
    blurred = fftconvolve(hmi_image, psf_rel, mode='same')

    # 2x2 box mean downsampling (non-overlapping)
    h, w = blurred.shape
    downsampled = blurred[:h - h % 2, :w - w % 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

    return downsampled

# %% argparse
Ntry = 1
nthr = int(os.getenv('OMP_NUM_THREADS', default=1))
date_fmt = '%Y.%m.%d_%H:%M:%S_TAI'

# --- arguments ---
parser = argparse.ArgumentParser(description='compute flows from granulation tracking')
parser.add_argument('start', type=str, help=f'start time %%Y.%%m.%%d_%%H:%%M:%%S_TAI')
parser.add_argument('stop', type=str, help=f'stop time (excluded) %%Y.%%m.%%d_%%H:%%M:%%S_TAI')
parser.add_argument('dspan', type=int, help='time span in minutes')
parser.add_argument('dstep', type=int, help='time step in minutes')
parser.add_argument('--downsample', help='downsample or not', action = 'store_true')
parser.add_argument('--interp', help='interpolate or not', action='store_true')
parser.add_argument('-l', '--loglevel', type=str, default='info',
		help='logging level: critical | error | warning | info | debug (default: info)')
args = parser.parse_args()
dstart = datetime.strptime(args.start, date_fmt)
dstop  = datetime.strptime(args.stop, date_fmt)
downsample = args.downsample
interp = args.interp
save_ccf = False
if downsample:
	NY, NX = 2048, 2048
else:
	NY, NX = 4096, 4096
loglevel = getattr(logging, args.loglevel.upper(), None)
if not isinstance(loglevel, int):
	print('Unknown logging level: %s' % (args.loglevel), file=sys.stderr)

# --- logger ---
logging.basicConfig(format='%(levelname)s %(message)s',
		level=loglevel, stream=sys.stdout)
logger = logging.getLogger()
logger.info('host: %s', socket.gethostname())
logger.info('start time: %s', dstart)
logger.info('stop time: %s', dstop)
logger.info('num_threads: %d', nthr)
begtime = datetime.now()

# %% === config: main ===
## input parameters
dspan = timedelta(minutes = args.dspan)
cadence = 45 # [s] int time interval between two images used for cross-correlation
cadence_keys = 45 # from the data series
if interp:
	cadence_interp = 60 # [s] int
else:
	cadence_interp = cadence_keys
njump = cadence//cadence_keys
dstep = timedelta(minutes = args.dstep)

nt = (dstop - dstart).total_seconds()/dspan.total_seconds()
assert nt.is_integer()
nt = int(nt)

if downsample:
	resolution = '2k'
	infile_fmt = '/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.ic_45s/keys_all_bad_excluded/keys_2k_2k/keys-%Y-2k.fits'
else:
	resolution = '4k'
	infile_fmt = '/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.ic_45s/keys_new_swan/keys-%Y.fits'
outfile = f'/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/diff_rot/data/data_{dstart.year}/{args.start}_nt_{nt}_dspan_{args.dspan}_dstep_{args.dstep}_dt_{cadence_interp}_diff_rot_5deg_gran_{resolution}.hdf5'
segname = 'continuum.fits'

## mapping parameters
dI = -0.08
t_ref_b0 = datetime(2010, 6, 7, 14, 17, 20)

#----- change tracking rate to anything other than carrington rotation rate ---------#
change_track = False
if change_track:
	def get_dL_change(lamda, time_diff):
		return (A + B*np.sin(np.deg2rad(lamda))**2 + C*np.sin(np.deg2rad(lamda))**4 - CRrate)*time_diff/(3600*24)
else:
	def get_dL_change(lamda, time_diff):
		return 0

if segname == 'magnetogram.fits':
	#----- Parameters from Hathaway 2011 in deg/day -----------------------#
	A = 14.437
	B = -1.48
	C = -2.99
	CRrate = 14.184
	logger.info('Using parameters from Hathaway 2011 for magnetogram data.')
elif segname == 'continuum.fits':
	#----- Parameters from Snodgrass 1984 B ------------------------------------#
	A = 14.034
	B = -1.702
	C = -2.494
	CRrate = 14.184
	logger.info('Using parameters from Snodgrass 1984 for continuum data.')

#------ Parameters for the tukey window and the CCF grid search ---------------------#
# We use 17 px to match size for 1 degree - 8 to match 0.5 degree for 2k
R_sun = 695.7 #Radius of sun in megametres
pixel_size = 0.03 #Degrees per pixel in the remapped image

if downsample:
	patch_size = 84 #Size of the patch in pixels for 2k
	pixel_size = 0.06 #Degrees per pixel in the remapped image for 2k
else:
	patch_size = 168
	pixel_size = 0.03 #Degrees per pixel in the remapped image for 4k


gkern1 = tukey_twoD(patch_size)

## PSF Parameters for HMI and PMI
wavelength_m = 617.3e-9  # Wavelength in meters
aperture_hmi = 0.14  # HMI aperture size in meters
aperture_pmi = 0.075  # PMI aperture size in meters
pixel_scale = 0.5  # Pixel scale in arcseconds/pixel (same for HMI and PMI assumed)


radius_hmi = compute_airy_radius_pixels(wavelength_m, aperture_hmi, pixel_scale)
radius_pmi = compute_airy_radius_pixels(wavelength_m, aperture_pmi, pixel_scale)

# Generate PSFs
psf_size = 64
psf_hmi = airy_disk_psf((psf_size, psf_size), radius_hmi)
psf_pmi = airy_disk_psf((psf_size, psf_size), radius_pmi)

# Compute relative PSF: PMI blur - HMI PSF
psf_rel = fftconvolve(psf_pmi, psf_hmi[::-1, ::-1], mode='same')

## output parameters
clatarr = np.arange(-60, 61, 2.5)
clngarr = np.arange(-60, 61, 2.5)
nlng = len(clngarr)
nlat = len(clatarr)
# clngarr = np.linspace(-90, +90, nlng)
xylist = [(x,y) for y in clatarr for x in clngarr]
ijlist = [(i,j) for j in range(clatarr.size) for i in range(clngarr.size)]
npatch = len(xylist)

h5file = h5py.File(outfile, 'w')
utheta = h5file.create_dataset('utheta', (nt, nlat, nlng), dtype='f4')
uphi = h5file.create_dataset('uphi', (nt, nlat, nlng), dtype='f4')
tstart = h5file.create_dataset('tstart', (nt), dtype='S19')
h5file.create_dataset('longitude', dtype = 'f8', data = clngarr)
h5file.create_dataset('latitude', dtype = 'f8', data = clatarr)
logger.critical('create %s', outfile)


T = []

# %% --- loop over chunks ---
for it,dstart_chunk in enumerate(xdays(dstart, dstop, dspan)):

	# --- read keys ---
	infile = dstart_chunk.strftime(infile_fmt)
	keys = Table.read(infile)

	logger.critical('read %s', infile)

	# Modified to get ccfs for each different dstep
	ccfs = np.zeros((npatch, patch_size, patch_size), dtype='f8')
	nums = np.zeros((npatch), dtype='i4')

	T0_chunk = datetime.now()

	for dat in xdays(dstart_chunk, dstart_chunk+dspan, dstep):

		tstart[it] = bytes(dstart_chunk.strftime(date_fmt), encoding='utf-8')

		# --- log ---
		T.append(datetime.now())
		logger.info('--- %s ---', dat.strftime(date_fmt))
		logger.info('%s', T[-1])

		dat0 = datetime.strptime(keys['t_rec'][0], date_fmt)
		ii = (dat - dat0).total_seconds()/cadence_keys
		assert ii.is_integer()
		ii = int(ii)

		# --- read data ---
		infile1 = os.path.join(keys['path'][ii][:-1], segname)
		infile2 = os.path.join(keys['path'][ii+njump][:-1], segname)
		infile3 = os.path.join(keys['path'][ii+2*njump][:-1], segname)
		infile4 = os.path.join(keys['path'][ii+3*njump][:-1], segname)
		if keys['isbad'][ii] or keys['isbad'][ii+njump]:
			logger.error( 'skipping %s and %s (either one is bad)',
					keys['t_rec'][ii], keys['t_rec'][ii+njump])
			isbad = True
		elif not os.path.exists(infile1) or not os.path.exists(infile2) or not os.path.exists(infile3) or not os.path.exists(infile4):
			logger.error( 'skipping %s, %s, %s, and %s (either one does not exist)',
					keys['t_rec'][ii], keys['t_rec'][ii+njump], keys['t_rec'][ii+2*njump], keys['t_rec'][ii+3*njump])
			isbad = True
		else:
			img1 = fits.getdata(infile1)
			img2 = fits.getdata(infile2)
			img3 = fits.getdata(infile3)
			img4 = fits.getdata(infile4)
			if downsample:
				img1 = simulate_pmi_from_hmi(np.nan_to_num(img1), psf_rel)
				img2 = simulate_pmi_from_hmi(np.nan_to_num(img2), psf_rel)
				img3 = simulate_pmi_from_hmi(np.nan_to_num(img3), psf_rel)
				img4 = simulate_pmi_from_hmi(np.nan_to_num(img4), psf_rel)
			isbad = False

		if isbad:
			continue
		else:
			assert img1.shape == img2.shape == (NY, NX)
		# raise SystemExit(0)
		T.append(datetime.now())
		logger.info('%s (ET read)', T[-1]-T[-2])

		assert keys['crval1'][ii] == keys['crval1'][ii+njump] == keys['crval1'][ii+2*njump] == keys['crval1'][ii+3*njump] == 0
		assert keys['crval2'][ii] == keys['crval2'][ii+njump] == keys['crval2'][ii+2*njump] == keys['crval2'][ii+3*njump] == 0
		crpix1 = keys['crpix1'][ii], keys['crpix1'][ii+njump], keys['crpix1'][ii+2*njump], keys['crpix1'][ii+3*njump]
		crpix2 = keys['crpix2'][ii], keys['crpix2'][ii+njump], keys['crpix2'][ii+2*njump], keys['crpix2'][ii+3*njump]
		cdelt1 = keys['cdelt1'][ii], keys['cdelt1'][ii+njump], keys['cdelt1'][ii+2*njump], keys['cdelt1'][ii+3*njump]
		cdelt2 = keys['cdelt2'][ii], keys['cdelt2'][ii+njump], keys['cdelt2'][ii+2*njump], keys['cdelt2'][ii+3*njump]
		rsun_obs = keys['rsun_obs'][ii], keys['rsun_obs'][ii+njump], keys['rsun_obs'][ii+2*njump], keys['rsun_obs'][ii+3*njump]

		t_rec_1 = datetime.strptime(keys['t_rec'][ii], date_fmt)
		t_rec_2 = datetime.strptime(keys['t_rec'][ii+njump], date_fmt)
		t_rec_3 = datetime.strptime(keys['t_rec'][ii+2*njump], date_fmt)
		t_rec_4 = datetime.strptime(keys['t_rec'][ii+3*njump], date_fmt)
		dt_b0_1 = (t_rec_1-t_ref_b0).total_seconds()/86400./365.25
		dt_b0_2 = (t_rec_2-t_ref_b0).total_seconds()/86400./365.25
		dt_b0_3 = (t_rec_3-t_ref_b0).total_seconds()/86400./365.25
		dt_b0_4 = (t_rec_4-t_ref_b0).total_seconds()/86400./365.25

		# --- correction to dB, dP ---
		dB = keys['crlt_obs'][ii] + dI*np.sin(2*np.pi*dt_b0_1), keys['crlt_obs'][ii+njump] + dI*np.sin(2*np.pi*dt_b0_2), keys['crlt_obs'][ii+2*njump] + dI*np.sin(2*np.pi*dt_b0_3), keys['crlt_obs'][ii+3*njump] + dI*np.sin(2*np.pi*dt_b0_4)
		dP = -keys['crota2'][ii] - dI*np.cos(2*np.pi*dt_b0_1), -keys['crota2'][ii+njump] - dI*np.cos(2*np.pi*dt_b0_2), -keys['crota2'][ii+2*njump] - dI*np.cos(2*np.pi*dt_b0_3), -keys['crota2'][ii+3*njump] - dI*np.cos(2*np.pi*dt_b0_4)

		T.append(datetime.now())
		T0 = datetime.now()
		for ipatch,(clng,clat) in enumerate(xylist):

			dL_change = 0
			dL = 0, keys['crln_obs'][ii] - keys['crln_obs'][ii+njump] + get_dL_change(clat, 45), keys['crln_obs'][ii] - keys['crln_obs'][ii+2*njump] + get_dL_change(clat, 90), keys['crln_obs'][ii] - keys['crln_obs'][ii+3*njump] + get_dL_change(clat, 135)

			img1p, img2p, img3p, img4p = from_tan_to_postel([img1, img2, img3, img4], np.array(crpix1), np.array(crpix2),0, 0, cdelt1, cdelt2, np.array(rsun_obs), dB, dP, dL, nx_out = patch_size,
							ny_out = patch_size, lngc_out = clng,latc_out = clat, pixscale_out = pixel_size,
							interp_method = 'bilinear', verbose = 1, nthr = 1, header = False)
			if interp:
				T_bf_interp = datetime.now()
				img_interp = fast_cubic_spline_interpolate_stack(np.nan_to_num(np.array([img1p, img2p, img3p, img4p])), target_time = cadence_interp, times = [0, 45, 90, 135])
				logger.info('%s (interp done)', datetime.now()-T_bf_interp)
				T_bf_ccf = datetime.now()
				ccf, _, _ = get_lct_map(img1p, img_interp)
				logger.info('%s (ccf calculation done)', datetime.now()-T_bf_ccf)
			else:
				T_bf_ccf = datetime.now()
				ccf, _, _ = get_lct_map(img1p, img2p)
				logger.info('%s (ccf calculation done)', datetime.now()-T_bf_ccf)


			if np.isnan(ccf).any():
				logger.warning('nan in ccf at %f, %f', clng, clat)
				continue
			ccfs[ipatch] += ccf
			nums[ipatch] += 1
			logger.info('got ccf at %d/%d: (%f, %f)', ipatch+1, npatch, clng, clat)
			# raise SystemExit(0)

		T1 = datetime.now()
		logger.info('%s (loop over patches)', T1-T0)

	ccfs /= nums[:,None,None]
	logger.info('(loop over dstep)')

	for ixy, (ccf, (clng,clat), (i,j)) in enumerate(zip(ccfs, xylist, ijlist)):
		if np.isnan(ccf).any():
			ux = uy = np.nan
		else:
			_,_,ux, uy = get_ux_uy_ellipsoid(ccf, grid_len = 5, ntry= 3, patch_size=patch_size, pixel_size = pixel_size, R_sun = R_sun)
		utheta[it,j,i] = -uy
		uphi[it,j,i] = ux
		logger.info('  %d, %d: utheta = %f, uphi = %f', i, j, utheta[it,j,i], uphi[it,j,i])
	T1_chunk = datetime.now()
	logger.critical('%s (loop over one chunk)', T1_chunk-T0_chunk)

h5file.close()
logger.critical('close %s', outfile)
