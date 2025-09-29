#!/usr/bin/env python
import os, sys, socket
import time
import argparse
import logging
from datetime import datetime, timedelta
import numpy as np
np.seterr(all='ignore')
from astropy.io import fits
from scipy import signal
from astropy.table import Table
from scipy.linalg import lstsq
from scipy.ndimage import shift
from scipy.ndimage.interpolation import zoom

sys.path.insert(0, '/data/seismo/joshin/pypkg')

from mpi4py import MPI

import h5py

import string
def strip_nonprintable(s):
	printable = set(string.printable)
	return ''.join([x for x in s if x in printable])

from josh.misc import xdays, strip_nonprintable

sys.path.insert(0, '/data/seismo/zhichao/codes/pypkg')
from zclpy3.remap import from_tan_to_postel
from zclpy3.mpi_util import get_delimiters

from scipy.interpolate import CubicSpline

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


MPI.Get_version()
mpi_lib_ver = strip_nonprintable(MPI.Get_library_version())

host = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

msg = 'rank %d on %s' % (rank, host)
# msg = comm.gather(msg, root=0)
msg = gather_bigsize(comm, msg, root=0)
if rank == 0:
	print(mpi_lib_ver)
	assert len(msg) == size
	print('\n'.join(msg))

Ntry = 1
nthr = int(os.getenv('OMP_NUM_THREADS', default=1))

# --- arguments ---
parser = argparse.ArgumentParser(description='compute flows from granulation tracking')
parser.add_argument('yr_start', type=int, help='start year')
parser.add_argument('month_start', type=int, help='start month')
parser.add_argument('yr_stop', type=int, help='end year')
parser.add_argument('month_stop', type=int, help='end month')
parser.add_argument('dspan', type=int, help='time span in minutes')
parser.add_argument('dstep', type=int, help='time step in minutes')
parser.add_argument('downsample', type=int, help='downsample or not (0 or 1)')
parser.add_argument('interpolate', type=int, help='interpolate or not (0 or 1)')
parser.add_argument('-l', '--loglevel', type=str, default='info',
		help='logging level: critical | error | warning | info | debug (default: info)')
args = parser.parse_args()
ystart = args.yr_start
ystop  = args.yr_stop
downsample = args.downsample
interpolate = args.interpolate
save_ccf = False
if args.downsample == 1:
	NY, NX = 2048, 2048
else:
	NY, NX = 4096, 4096
if args.interpolate == 0:
	interp = False
else:
	interp = True
loglevel = getattr(logging, args.loglevel.upper(), None)
if not isinstance(loglevel, int):
	print('Unknown logging level: %s' % (args.loglevel), file=sys.stderr)

# --- logger ---
logging.basicConfig(format='%(levelname)s %(message)s',
		level=loglevel, stream=sys.stdout)
logger = logging.getLogger()
if rank == 0:
	logger.info('host: %s', socket.gethostname())
	logger.info('start year: %d', ystart)
	logger.info('end year: %d', ystop)
	logger.info('num_threads: %d', nthr)
	begtime = datetime.now()

# === config: main ===
date_fmt = '%Y.%m.%d_%H:%M:%S_TAI'
dspan = timedelta(minutes = args.dspan)
cadence = 45 # [s] int
cadence_keys = 45
if interp:
	cadence_interp = 60 # [s] int
else:
	cadence_interp = cadence_keys
njump = cadence//cadence_keys
dstep = timedelta(minutes = args.dstep)

if downsample == 1:
	infile_fmt = '/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.ic_45s/keys_all_bad_excluded/keys_2k_2k/keys-%Y-2k.fits'
	outfile_fmt = '/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/diff_rot/data/%d_dspan_{}_dstep_{}_dt_{}_ccf_width_test_5deg_gran_2k.hdf5'.format(args.dspan, args.dstep, cadence_interp)
	segname = 'continuum.fits'
else:
	infile_fmt = '/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.ic_45s/keys_new_swan/keys-%Y.fits'
	outfile_fmt = '/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/diff_rot/data/%d_dspan_{}_dstep_{}_dt_{}_ccf_width_test_5deg_gran_4k.hdf5'.format(args.dspan, args.dstep, cadence_interp)
	segname = 'continuum.fits'

date_shift = timedelta(minutes=0)

dI = -0.08
t_ref_b0 = datetime(2010, 6, 7, 14, 17, 20)

#----- change tracking rate to anything other than carrington rotation rate ---------#
change_track = False

if segname == 'magnetogram.fits':
	#----- Parameters from Hathaway 2011 in deg/day -----------------------#
	A = 14.437
	B = -1.48
	C = -2.99
	CRrate = 14.184
	if rank == 0:
		print('Using parameters from Hathaway 2011 for magnetogram data.')
elif segname == 'continuum.fits':
	#----- Parameters from Snodgrass 1984 B ------------------------------------#
	A = 14.034
	B = -1.702
	C = -2.494
	CRrate = 14.184
	if rank == 0:
		print('Using parameters from Snodgrass 1984 for continuum data.')

#------ Parameters for the tukey window and the CCF grid search ---------------------#
# We use 17 px to match size for 1 degree - 8 to match 0.5 degree for 2k
R_sun = 695.7 #Radius of sun in megametres
pixel_size = 0.03 #Degrees per pixel in the remapped image

if downsample == 1:
	patch_size = 84 #Size of the patch in pixels for 2k
	pixel_size = 0.06 #Degrees per pixel in the remapped image for 2k
else:
	patch_size = 168
	pixel_size = 0.03 #Degrees per pixel in the remapped image for 4k

degTorad = np.pi/180
radtodeg = 180/np.pi

alpha = 0.8

def downsample_img(img, factor):
	# Downsample the image by a factor of factor
	down_img = zoom(np.nan_to_num(img), 1/factor, order = 1)
	return down_img


def tukey_twoD(width = patch_size, alpha = 0.8):
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


gkern1 = tukey_twoD()



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

def fft_shift(img,shift):
	try:
		import pyfftw.interfaces.numpy_fft as fft
	except:
		import numpy.fft as fft
	sz = img.shape
	#ky = fft.ifftshift(np.linspace(-np.fix(sz[0]/2),np.ceil(sz[0]/2)-1,sz[0]))
	#kx = fft.ifftshift(np.linspace(-np.fix(sz[1]/2),np.ceil(sz[1]/2)-1,sz[1]))
	yf = fft.fftfreq(img.shape[1], d = 1/img.shape[0])
	xf = fft.fftfreq(img.shape[0], d = 1/img.shape[1])
	#print(kx - xf)
	#print(xf)
	img_fft = fft.fft2(img)
	shf = np.exp(-2j*np.pi*(yf[:,np.newaxis]*shift[1]/sz[1]+xf[np.newaxis]*shift[0]/sz[0]))

	img_fft *= shf
	img_shf = fft.ifft2(img_fft).real

	return img_shf



def get_ux_uy_ellipsoid(ccf_av, grid_len = 7, patch_size = patch_size, pixel_size = pixel_size, R_sun = 696, ntry = 3):
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
		# ccf_av = fft_shift(ccf_av,[-del_x, -del_y])
		dx_tot = dx_tot + del_x
		dy_tot = dy_tot + del_y

	#Calculate ux, uy
	ux = R_sun * dx_tot * np.deg2rad(pixel_size)/(cadence_interp) * 1e6
	uy = R_sun * dy_tot * np.deg2rad(pixel_size)/(cadence_interp) * 1e6

	return dx_tot, dy_tot, ux, uy


# function to get ccf widths

def fwhm_from_xy(x, y, baseline=None, x_sorted=False):
    """
    Compute FWHM (full width at half maximum) of a single-peaked, Lorentzian-like curve.

    Parameters
    ----------
    x : array-like
        Independent variable values (x-axis). Can be unsorted.
    y : array-like
        Dependent variable values (y-axis).
    baseline : float or None, optional
        Baseline (offset) to subtract from y before computing half-maximum.
        If None, baseline = min(y).
    x_sorted : bool
        If True, assumes x is already strictly increasing. If False, function will sort x,y.

    Returns
    -------
    fwhm : float or np.nan
        Full width at half maximum in units of x (x_right - x_left). np.nan if not computable.
    x_left : float or None
        x position of left half-maximum crossing (interpolated).
    x_right : float or None
        x position of right half-maximum crossing (interpolated).
    peak_x : float
        x position of the peak.
    peak_y : float
        y value of the peak (after baseline subtraction).

    Notes
    -----
    - Works for a single peak (one maximum). If the data contain multiple peaks, supply sliced data around the peak.
    - Uses linear interpolation between samples to estimate crossing points.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # sort by x unless told sorted
    if not x_sorted:
        order = np.argsort(x)
        x = x[order]
        y = y[order]

    # baseline
    if baseline is None:
        baseline = np.min(y)

    y0 = y - baseline
    # avoid negative or zero plateau: if entire y0 <= 0, cannot compute
    if np.all(y0 <= 0):
        return np.nan, None, None, None, None

    # find peak
    peak_idx = np.argmax(y0)
    peak_y = y0[peak_idx]
    peak_x = x[peak_idx]

    half = peak_y / 2.0

    # left side: search indices i where y0[i] <= half < y0[i+1] (i runs 0..peak_idx-1)
    x_left = None
    if peak_idx == 0:
        x_left = None
    else:
        left_slice_y = y0[:peak_idx+1]
        left_slice_x = x[:peak_idx+1]
        # find last index i before peak where y0[i] <= half
        # we look for i such that left_slice_y[i] <= half <= left_slice_y[i+1]
        idxs = np.where(left_slice_y <= half)[0]
        if idxs.size == 0:
            # maybe the values on left are all above half (rare) -> cannot find crossing
            x_left = None
        else:
            i = idxs[-1]
            # handle boundary: if i == peak_idx then no crossing found
            if i == peak_idx:
                x_left = None
            else:
                x1, y1 = left_slice_x[i], left_slice_y[i]
                x2, y2 = left_slice_x[i+1], left_slice_y[i+1]
                if y2 == y1:
                    x_left = x1
                else:
                    t = (half - y1) / (y2 - y1)
                    x_left = x1 + t * (x2 - x1)

    # right side: search indices i where y0[i] >= half > y0[i+1] (i runs peak_idx..end-2)
    x_right = None
    if peak_idx == len(x) - 1:
        x_right = None
    else:
        right_slice_y = y0[peak_idx:]
        right_slice_x = x[peak_idx:]
        # find first index j (relative to peak_idx) where right_slice_y[j] <= half
        idxs = np.where(right_slice_y <= half)[0]
        if idxs.size == 0:
            x_right = None
        else:
            j = idxs[0]
            # if j == 0, the first point at peak is already <= half -> can't interpolate to right
            if j == 0:
                x_right = None
            else:
                i = peak_idx + (j - 1)
                x1, y1 = x[i], y0[i]
                x2, y2 = x[i+1], y0[i+1]
                if y2 == y1:
                    x_right = x2
                else:
                    t = (half - y1) / (y2 - y1)
                    x_right = x1 + t * (x2 - x1)

    if (x_left is None) or (x_right is None):
        return np.nan, x_left, x_right, peak_x, peak_y

    fwhm = x_right - x_left
    return fwhm, x_left, x_right, peak_x, peak_y



import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import convolve
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

# Parameters for HMI and PMI
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


# === config: main ===
clatarr = np.arange(-90, 91, 2.5)
clngarr = np.arange(-90, 91, 2.5)
# clatarr = np.arange(-10, 11, 1)
# clngarr = np.arange(-10, 11, 1)
# clatarr = np.arange(30, 50, 0.1)
# clngarr = np.arange(-10, 10, 0.1)
nlng = len(clngarr)
nlat = len(clatarr)
# clngarr = np.linspace(-90, +90, nlng)
xylist = [(x,y) for y in clatarr for x in clngarr]
ijlist = [(i,j) for j in range(clatarr.size) for i in range(clngarr.size)]
delim, chunks = get_delimiters(len(xylist), size, return_chunks=True)
lo, hi = delim[rank]

k = 1

T = []

if ystart == ystop:
	ystop += 1
	flag = 1
else:
	flag = 0

# --- loop over days ---
for yr in range(ystart,ystop):

	# --- read keys ---
	infile = datetime(yr,1,1).strftime(infile_fmt)
	keys = Table.read(infile)

	if rank == 0:
		logger.critical('read %s', infile)

	assert not ystart == 2010

	dstart_yr = datetime(yr, args.month_start, 15, 0, 0, 0)
	dstop_yr = datetime(yr, args.month_start, 15, 6, 0, 0)  # changed here warning
	# if flag == 1:
	# 	dstop_yr = datetime(ystop+1, args.month_stop, 1, 0, 0, 0) + date_shift ##changed here warning
	# else:
	# 	dstop_yr = datetime(ystop, args.month_stop, 1, 0, 0, 0) + date_shift


	# dstart_yr = datetime(yr, 1 if args.which_half == 1 else 7, 1, 0, 0, 0) + date_shift
	# dstop_yr = datetime(yr if args.which_half==1 else yr+1, 7 if args.which_half==1 else 1, 1, 0, 0, 0) + date_shift
	# dstart_yr = datetime(yr, 5, 1, 0, 0, 0) if yr == 2010 else datetime(yr, 1, 1, 0, 0, 0)
	# dstop_yr = datetime(yr+1, 1, 1, 0, 0, 0)


	nt = (dstop_yr - dstart_yr).total_seconds()/dspan.total_seconds()
	assert nt.is_integer()
	nt = int(nt)

	if rank == 0:
		outfile = outfile_fmt % (yr)
		h5file = h5py.File(outfile, 'w')
		utheta = h5file.create_dataset('utheta', (nt, nlat, nlng), dtype='f4')
		uphi = h5file.create_dataset('uphi', (nt, nlat, nlng), dtype='f4')
		tstart = h5file.create_dataset('tstart', (nt), dtype='S19')
		ccf_width_x = h5file.create_dataset('ccf_width_x', (nt, nlat, nlng), dtype='f4')
		ccf_width_y = h5file.create_dataset('ccf_width_y', (nt, nlat, nlng), dtype='f4')
		h5file.create_dataset('longitude', dtype = 'f8', data = clngarr)
		h5file.create_dataset('latitude', dtype = 'f8', data = clatarr)
		logger.critical('create %s', outfile)

	for it,dstart_chunk in enumerate(xdays(dstart_yr, dstop_yr, dspan)):
		# Modified to get ccfs for each different dstep
		# ccfs_30m = np.zeros((hi-lo, patch_size, patch_size), dtype='f8')
		# nums_30m = np.zeros((hi-lo), dtype='i4')
		# ccfs_1h = np.zeros((hi-lo, patch_size, patch_size), dtype='f8')
		# nums_1h = np.zeros((hi-lo), dtype='i4')
		ccfs = np.zeros((hi-lo, patch_size, patch_size), dtype='f8')
		nums = np.zeros((hi-lo), dtype='i4')
		# ccfs_6h = np.zeros((hi-lo, patch_size, patch_size), dtype='f8')
		# nums_6h = np.zeros((hi-lo), dtype='i4')
		if rank == 0:
			tstart[it] = bytes(dstart_chunk.strftime('%Y.%m.%d_%H:%M:%S'), encoding='utf-8')

		T0_chunk = datetime.now()
		# k = 0
		for dat in xdays(dstart_chunk, dstart_chunk+dspan, dstep):
			try:

				if rank == 0:
					# --- log ---
					T.append(datetime.now())
					logger.info('--- %s ---', dat.strftime(date_fmt))
					logger.info('%s', T[-1])

				dat0 = datetime.strptime(keys['t_rec'][0], '%Y.%m.%d_%H:%M:%S_TAI')
				ii = (dat - dat0).total_seconds()/cadence_keys
				assert ii.is_integer()
				ii = int(ii)

				if rank == 0:
					# --- read data ---
					if keys['isbad'][ii] or keys['isbad'][ii+njump]:
						logger.error( 'skipping %s and %s (either one is bad)',
								keys['t_rec'][ii], keys['t_rec'][ii+njump])
						isbad = True
						img1, img2 = None, None
					else:
						for itry in range(Ntry):
							try:
								img1 = fits.getdata(os.path.join(keys['path'][ii][:-1], segname))
								img2 = fits.getdata(os.path.join(keys['path'][ii+njump][:-1], segname))
								if downsample == 1:
									img1 = simulate_pmi_from_hmi(np.nan_to_num(img1), psf_rel)
									img2 = simulate_pmi_from_hmi(np.nan_to_num(img2), psf_rel)
								if interp:
									img3 = fits.getdata(os.path.join(keys['path'][ii+2*njump][:-1], segname))
									img4 = fits.getdata(os.path.join(keys['path'][ii+3*njump][:-1], segname))
									if downsample == 1:
										img3 = simulate_pmi_from_hmi(np.nan_to_num(img3), psf_rel)
										img4 = simulate_pmi_from_hmi(np.nan_to_num(img4), psf_rel)
								# if k < 5:
								# 	np.save('/data/seismo/joshin/pipeline-test/pmi_test/supergranular_flow/4k/data/{}_{}_{}_4k_for_ccfs.npy'.format(dat.strftime('%Y%m%d_%H%M%S'), 'img1', k), img1)
								# 	np.save('/data/seismo/joshin/pipeline-test/pmi_test/supergranular_flow/4k/data/{}_{}_{}_4k_for_ccfs.npy'.format(dat.strftime('%Y%m%d_%H%M%S'), 'img2', k), img2)
								# 	k = k+1
								# img1 = downsample_img(img1, 2)
								# img2 = downsample_img(img2, 2)
								isbad = False
								break
							except IOError as e: # [Errno 5] Input/output error
								logger.warning( '%s (%s or %s)', e, keys['t_rec'][ii], keys['t_rec'][ii+njump])
								if itry != Ntry-1:
									time.sleep(5)
						else:
							logger.error( 'skipping %s (%s) and %s (%s)',
								keys['path'][ii], keys['t_rec'][ii], keys['path'][ii+njump], keys['t_rec'][ii+njump])
							isbad = True
							if interp:
								img1, img2, img3, img4 = None, None, None, None
							img1, img2 = None, None
				else:
					isbad = False
					if interp:
						img1, img2, img3, img4 = None, None, None, None
					else:
						img1, img2 = None, None

				if interp:
					isbad, img1, img2, img3, img4 = comm.bcast([isbad, img1, img2, img3, img4], root=0)
				else:
					isbad, img1, img2 = comm.bcast([isbad, img1, img2], root=0)
				if isbad:
					continue
				else:
					assert img1.shape == img2.shape == (NY, NX)
				if rank == 0:
					T.append(datetime.now())
					logger.info('%s (ET read)', T[-1]-T[-2])


				if interp:
					assert keys['crval1'][ii] == keys['crval1'][ii+njump] == keys['crval1'][ii+2*njump] == keys['crval1'][ii+3*njump] == 0
					assert keys['crval2'][ii] == keys['crval2'][ii+njump] == keys['crval2'][ii+2*njump] == keys['crval2'][ii+3*njump] == 0
					crpix1 = keys['crpix1'][ii], keys['crpix1'][ii+njump], keys['crpix1'][ii+2*njump], keys['crpix1'][ii+3*njump]
					crpix2 = keys['crpix2'][ii], keys['crpix2'][ii+njump], keys['crpix2'][ii+2*njump], keys['crpix2'][ii+3*njump]
					cdelt1 = keys['cdelt1'][ii], keys['cdelt1'][ii+njump], keys['cdelt1'][ii+2*njump], keys['cdelt1'][ii+3*njump]
					cdelt2 = keys['cdelt2'][ii], keys['cdelt2'][ii+njump], keys['cdelt2'][ii+2*njump], keys['cdelt2'][ii+3*njump]
					rsun_obs = keys['rsun_obs'][ii], keys['rsun_obs'][ii+njump], keys['rsun_obs'][ii+2*njump], keys['rsun_obs'][ii+3*njump]

					t_rec_1 = datetime.strptime(keys['t_rec'][ii], '%Y.%m.%d_%H:%M:%S_TAI')
					t_rec_2 = datetime.strptime(keys['t_rec'][ii+njump], '%Y.%m.%d_%H:%M:%S_TAI')
					t_rec_3 = datetime.strptime(keys['t_rec'][ii+2*njump], '%Y.%m.%d_%H:%M:%S_TAI')
					t_rec_4 = datetime.strptime(keys['t_rec'][ii+3*njump], '%Y.%m.%d_%H:%M:%S_TAI')
					dt_b0_1 = (t_rec_1-t_ref_b0).total_seconds()/86400./365.25
					dt_b0_2 = (t_rec_2-t_ref_b0).total_seconds()/86400./365.25
					dt_b0_3 = (t_rec_3-t_ref_b0).total_seconds()/86400./365.25
					dt_b0_4 = (t_rec_4-t_ref_b0).total_seconds()/86400./365.25

					dB = keys['crlt_obs'][ii] + dI*np.sin(2*np.pi*dt_b0_1), keys['crlt_obs'][ii+njump] + dI*np.sin(2*np.pi*dt_b0_2), keys['crlt_obs'][ii+2*njump] + dI*np.sin(2*np.pi*dt_b0_3), keys['crlt_obs'][ii+3*njump] + dI*np.sin(2*np.pi*dt_b0_4)
					dP = -keys['crota2'][ii] - dI*np.cos(2*np.pi*dt_b0_1), -keys['crota2'][ii+njump] - dI*np.cos(2*np.pi*dt_b0_2), -keys['crota2'][ii+2*njump] - dI*np.cos(2*np.pi*dt_b0_3), -keys['crota2'][ii+3*njump] - dI*np.cos(2*np.pi*dt_b0_4)

				else:
					assert keys['crval1'][ii] == keys['crval1'][ii+njump] == 0
					assert keys['crval2'][ii] == keys['crval2'][ii+njump] == 0
					crpix1 = keys['crpix1'][ii], keys['crpix1'][ii+njump]
					crpix2 = keys['crpix2'][ii], keys['crpix2'][ii+njump]
					cdelt1 = keys['cdelt1'][ii], keys['cdelt1'][ii+njump]
					cdelt2 = keys['cdelt2'][ii], keys['cdelt2'][ii+njump]
					rsun_obs = keys['rsun_obs'][ii], keys['rsun_obs'][ii+njump]
					t_rec_1 = datetime.strptime(keys['t_rec'][ii], '%Y.%m.%d_%H:%M:%S_TAI')
					t_rec_2 = datetime.strptime(keys['t_rec'][ii+njump], '%Y.%m.%d_%H:%M:%S_TAI')
					dt_b0_1 = (t_rec_1-t_ref_b0).total_seconds()/86400./365.25
					dt_b0_2 = (t_rec_2-t_ref_b0).total_seconds()/86400./365.25
					dB = keys['crlt_obs'][ii] + dI*np.sin(2*np.pi*dt_b0_1), keys['crlt_obs'][ii+njump] + dI*np.sin(2*np.pi*dt_b0_2)
					dP = -keys['crota2'][ii] - dI*np.cos(2*np.pi*dt_b0_1), -keys['crota2'][ii+njump] - dI*np.cos(2*np.pi*dt_b0_2)


				# --- correction to dB, dP ---



				T.append(datetime.now())
				T0 = datetime.now()
				# if rank == 0:
				# 	print('k = {}'.format(k))
				for ipatch,(clng,clat) in enumerate(xylist[lo:hi]):

					if interp:
						dL_change = 0
						if change_track:
							lamda = clat
							def get_dL_change(lamda, time_diff):
								return (A + B*np.sin(np.deg2rad(lamda))**2 + C*np.sin(np.deg2rad(lamda))**4 - CRrate)*time_diff/(3600*24)
						else:
							lamda = clat
							def get_dL_change(lamda, time_diff):
								return 0

						dL = 0, keys['crln_obs'][ii] - keys['crln_obs'][ii+njump] + get_dL_change(lamda, 45), keys['crln_obs'][ii] - keys['crln_obs'][ii+2*njump] + get_dL_change(lamda, 90), keys['crln_obs'][ii] - keys['crln_obs'][ii+3*njump] + get_dL_change(lamda, 135)
						# img1 = np.nan_to_num(img1)
						# img2 = np.nan_to_num(img2)
						# img3 = np.nan_to_num(img3)
						# img4 = np.nan_to_num(img4)

						img1p, img2p, img3p, img4p = from_tan_to_postel([img1, img2, img3, img4], np.array(crpix1), np.array(crpix2),0, 0, cdelt1, cdelt2, np.array(rsun_obs), dB, dP, dL, nx_out = patch_size,
										ny_out = patch_size, lngc_out = clng,latc_out = clat, pixscale_out = pixel_size,
										interp_method = 'cubconv', verbose = 1, nthr = 1, header = False)
						if np.isnan(img1p).any() or np.isnan(img2p).any() or np.isnan(img3p).any() or np.isnan(img4p).any():
							raise RuntimeError('nan in remapped image')
							exit(1)
						# if rank == 0:
						# 	print('img1 = ', img1)
						# 	print('img2 = ', img2)
						# 	print('img3 = ', img3)
						# 	print('img4 = ', img4)

						img_interp = fast_cubic_spline_interpolate_stack(np.array([img1p, img2p, img3p, img4p]), target_time = cadence_interp, times = [0, 45, 90, 135])
						ccf, _, _ = get_lct_map(img1p, img_interp)

					else:
						dL_change = 0
						if change_track:
							lamda = clat
							dL_change = (A + B*np.sin(np.deg2rad(lamda))**2 + C*np.sin(np.deg2rad(lamda))**4 - CRrate)*cadence/(3600*24)

						dL = 0, keys['crln_obs'][ii] - keys['crln_obs'][ii+njump] + dL_change
						img1p, img2p = from_tan_to_postel([img1, img2], crpix1, crpix2,0, 0, cdelt1, cdelt2, rsun_obs, dB, dP, dL,
										nx_out = patch_size, ny_out = patch_size, lngc_out = clng, latc_out = clat,
										pixscale_out = pixel_size, interp_method = 'cubconv', verbose = 1,
										nthr = 1, header = False)
						ccf, _, _ = get_lct_map(img1p, img2p)


					if np.isnan(ccf).any():
						continue
					# print('clat = ', clat)
					# print('clng = ', clng)
					if (abs(clat) < 0.1) and (abs(clng) < 0.1) and downsample == 0 and save_ccf == True:
						np.save('/data/seismo/joshin/pipeline-test/pmi_test/supergranular_flow/final_sg_compare/ccfs/gran/{}_{}_{}_{}_{}_{}_4k_gran_interp_no_av.npy'.format(dat.strftime('%Y%m%d_%H%M%S'), 'ccf', dspan, dstep, round(clat, 1), round(clng, 1)), ccf)
					if (abs(clat) < 0.1) and (abs(clng) < 0.1) and downsample == 1 and save_ccf == True:
						np.save('/data/seismo/joshin/pipeline-test/pmi_test/supergranular_flow/final_sg_compare/ccfs/gran/{}_{}_{}_{}_{}_{}_2k_gran_interp_no_av.npy'.format(dat.strftime('%Y%m%d_%H%M%S'), 'ccf', dspan, dstep, round(clat, 1), round(clng, 1)), ccf)
					ccfs[ipatch] += ccf
					nums[ipatch] += 1
				# k = k + 1
				comm.Barrier()
				T1 = datetime.now()
				if rank == 0:
					logger.info('%s (loop over patches)', T1-T0)

			except RuntimeError as e:
				if rank == 0:
					logger.error('%s: %s', dat.date(), e)
				continue

		ccfs /= nums[:,None,None]

		ux1 = np.zeros((hi-lo), dtype='f8')
		uy1 = np.zeros((hi-lo), dtype='f8')
		ccf_width_x1 = np.zeros((hi-lo), dtype='f8')
		ccf_width_y1 = np.zeros((hi-lo), dtype='f8')


		for ixy, (ccf, (clng,clat)) in enumerate(zip(ccfs, xylist[lo:hi])):
			if (abs(clat) < 0.1) and (abs(clng) < 0.1) and downsample == 0 and save_ccf == True:
				np.save('/data/seismo/joshin/pipeline-test/pmi_test/supergranular_flow/final_sg_compare/ccfs/gran/{}_{}_{}_{}_{}_{}_4k_gran_interp_av.npy'.format(dat.strftime('%Y%m%d_%H%M%S'), 'ccf', dspan, dstep, round(clat, 1), round(clng, 1)), ccf)
			if (abs(clat) < 0.1) and (abs(clng) < 0.1) and downsample == 1 and save_ccf == True:
				np.save('/data/seismo/joshin/pipeline-test/pmi_test/supergranular_flow/final_sg_compare/ccfs/gran/{}_{}_{}_{}_{}_{}_2k_gran_interp_av.npy'.format(dat.strftime('%Y%m%d_%H%M%S'), 'ccf', dspan, dstep, round(clat, 1), round(clng, 1)), ccf)
			if np.isnan(ccf).any():
				ux1[ixy] = uy1[ixy] = np.nan
				continue
			_,_,ux1[ixy], uy1[ixy] = get_ux_uy_ellipsoid(ccf, grid_len = 5, ntry= 4)
			ccf_width_x1[ixy], _, _, _, _ = fwhm_from_xy(np.arange(patch_size)*pixel_size, ccf[patch_size//2, :], baseline=None, x_sorted=True)
			ccf_width_y1[ixy], _, _, _, _ = fwhm_from_xy(np.arange(patch_size)*pixel_size, ccf[:, patch_size//2], baseline=None, x_sorted=True)
		ux_all = np.empty((len(xylist)), dtype='f8') if rank == 0 else None
		uy_all = np.empty((len(xylist)), dtype='f8') if rank == 0 else None
		ccf_width_x_all = np.empty((len(xylist)), dtype='f8') if rank == 0 else None
		ccf_width_y_all = np.empty((len(xylist)), dtype='f8') if rank == 0 else None
		comm.Gatherv(ux1, [ux_all, chunks], root=0)
		comm.Gatherv(uy1, [uy_all, chunks], root=0)
		comm.Gatherv(ccf_width_x1, [ccf_width_x_all, chunks], root=0)
		comm.Gatherv(ccf_width_y1, [ccf_width_y_all, chunks], root=0)

		if rank == 0:
			for (i,j), ux, uy in zip(ijlist, ux_all, uy_all):
				utheta[it,j,i] = -uy
				uphi[it,j,i] = ux
			for (i,j), widx, widy in zip(ijlist, ccf_width_x_all, ccf_width_y_all):
				ccf_width_x[it,j,i] = widx
				ccf_width_y[it,j,i] = widy
			T1_chunk = datetime.now()
			logger.critical('%s (loop over one chunk)', T1_chunk-T0_chunk)

	if rank == 0:
		h5file.close()
		logger.critical('close %s', outfile)
