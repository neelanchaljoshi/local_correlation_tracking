#!/usr/bin/env python
import os, sys, shutil, socket
import time
from traceback import print_exc
import argparse
import logging
from datetime import datetime, timedelta
import numpy as np
np.seterr(all='ignore')
from astropy.wcs import WCS
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

from josh.remap import from_tan_to_cyl_mit_trk
from josh.misc import xdays, append_suffix_number_push_back, strip_nonprintable

sys.path.insert(0, '/data/seismo/zhichao/codes/pypkg')
from zclpy3.remap import from_tan_to_postel
from zclpy3.mpi_util import get_delimiters

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
# NX, NY = 2048, 2048
NX, NY = 4096, 4096
nthr = int(os.getenv('OMP_NUM_THREADS', default=1))

# --- arguments ---
parser = argparse.ArgumentParser(description='compute flows from granulation tracking')
parser.add_argument('yr_start', type=int, help='start year')
parser.add_argument('month_start', type=int, help='start month')
parser.add_argument('yr_stop', type=int, help='end year')
parser.add_argument('month_stop', type=int, help='end month')
parser.add_argument('dspan', type=int, help='time span in minutes')
parser.add_argument('dstep', type=int, help='time step in minutes')
parser.add_argument('-l', '--loglevel', type=str, default='info',
		help='logging level: critical | error | warning | info | debug (default: info)')
args = parser.parse_args()
ystart = args.yr_start
ystop  = args.yr_stop
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
njump = cadence//cadence_keys
dstep = timedelta(minutes = args.dstep)
# infile_fmt = '/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.ic_45s/keys_all_bad_excluded/keys_2k_2k/keys-%Y-2k.fits'
infile_fmt = '/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.ic_45s/keys_new_swan/keys-%Y.fits'
outfile_fmt = '/data/seismo/joshin/pipeline-test/pmi_test/granules/data/%d_{}_dspan_{}_dstep_{}_test_ccfs_4k.hdf5'.format(args.month_start, args.dspan, args.dstep)
segname = 'continuum.fits'
# date_shift = timedelta(seconds = 2*args.shift*cadence_keys)
date_shift = timedelta(minutes=0)

dI = -0.08
t_ref_b0 = datetime(2010, 6, 7, 14, 17, 20)

#----- change tracking rate to anything other than carrington rotation rate ---------#
change_track = False
#----- Parameters from Hathaway 2011 in deg/day -----------------------#
A = 14.437
B = -1.48
C = -2.99
CRrate = 14.184

#------ Parameters for the tukey window and the CCF grid search ---------------------#
R_sun = 695.7 #Radius of sun in megametres
pixel_size = 0.03 #Degrees per pixel in the remapped image
degTorad = np.pi/180
radtodeg = 180/np.pi 
patch_size = 168 #Size of the patch in pixels
alpha = 0.6 #Taper on the edges in the tukey window


def downsample_img(img, factor):
    # Downsample the image by a factor of factor
    down_img = zoom(np.nan_to_num(img), 1/factor, order = 1)
    return down_img


def tukey_twoD(width = patch_size, alpha = 0.6):
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

def fft_shift(ccf, shift):
	# Shifts the ccf by the amount shift
	sz = ccf.shape
	yf = np.fft.fftfreq(ccf.shape[1], d = 1/ccf.shape[0])
	xf = np.fft.fftfreq(ccf.shape[0], d = 1/ccf.shape[1])
	img_fft = np.fft.fft2(ccf)
	shf = np.exp(-2j*np.pi*(yf[:,np.newaxis]*shift[1]/sz[1]+xf[np.newaxis]*shift[0]/sz[0]))
	
	ccf_fft *= shf
	ccf_shf = np.fft.ifft2(ccf_fft).real
	
	return ccf_shf



def get_ux_uy_ellipsoid(ccf_av, grid_len = 7, patch_size = patch_size, pixel_size = pixel_size, R_sun = 696, ntry = 3):
	# Finds the ux and uy from the ccf_av
	dx_tot = 0
	dy_tot = 0

	assert grid_len%2 == 1

	for n in range(ntry):
		ym, xm = np.where(ccf_av==ccf_av[grid_len:-grid_len, grid_len:-grid_len].max())
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
	ux = R_sun * dx_tot * np.deg2rad(pixel_size)/(cadence) * 1e6
	uy = R_sun * dy_tot * np.deg2rad(pixel_size)/(cadence) * 1e6

	return dx_tot, dy_tot, ux, uy


# === config: main ===
nlng = 3
# nlat = 13
nlat = 3
clatarr = np.array([-5, 0, 5])
clngarr = np.array([-5, 0, 5])
# clatarr = np.linspace(-90, +90, nlat)
# clatarr = np.array([-75, -65, -60, -50, -10, -5, 0, 5, 10, 50, 60, 65, 75])
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

	dstart_yr = datetime(yr, args.month_start, 23, 0, 0, 0) + date_shift
	dstop_yr = datetime(yr, args.month_start, 23, 12, 0, 0) + date_shift
	# if flag == 1:
	# 	dstop_yr = datetime(ystop+1, args.month_stop, 1, 0, 0, 0) + date_shift ##changed here warning
	# else:
	# 	dstop_yr = datetime(ystop, args.month_stop, 1, 0, 0, 0) + date_shift


	# dstart_yr = datetime(yr, 1 if args.which_half == 1 else 7, 1, 0, 0, 0) + date_shift
	# dstop_yr = datetime(yr if args.which_half==1 else yr+1, 7 if args.which_half==1 else 1, 1, 0, 0, 0) + date_shift



	nt = (dstop_yr - dstart_yr).total_seconds()/dspan.total_seconds()
	assert nt.is_integer()
	nt = int(nt)

	if rank == 0:
		outfile = outfile_fmt % (yr)
		h5file = h5py.File(outfile, 'w')
		utheta = h5file.create_dataset('utheta', (nt, nlat, nlng), dtype='f4')
		uphi = h5file.create_dataset('uphi', (nt, nlat, nlng), dtype='f4')
		tstart = h5file.create_dataset('tstart', (nt), dtype='S19')
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
								# if k < 5:
									# np.save('/data/seismo/joshin/pipeline-test/pmi_test/granules/data/{}_{}_{}_4k_for_ccfs.npy'.format(dat.strftime('%Y%m%d_%H%M%S'), 'img1', k), img1)
									# np.save('/data/seismo/joshin/pipeline-test/pmi_test/granules/data/{}_{}_{}_4k_for_ccfs.npy'.format(dat.strftime('%Y%m%d_%H%M%S'), 'img2', k), img2)
									# k = k+1
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
							img1, img2 = None, None
				else:
					isbad = False
					img1, img2 = None, None
					
				isbad, img1, img2 = comm.bcast([isbad, img1, img2], root=0)
				if isbad:
					continue
				else:
					assert img1.shape == img2.shape == (NY, NX)
				if rank == 0:
					T.append(datetime.now())
					logger.info('%s (ET read)', T[-1]-T[-2])

				assert keys['crval1'][ii] == keys['crval1'][ii+njump] == 0
				assert keys['crval2'][ii] == keys['crval2'][ii+njump] == 0
				crpix1 = keys['crpix1'][ii], keys['crpix1'][ii+njump]
				crpix2 = keys['crpix2'][ii], keys['crpix2'][ii+njump]
				cdelt1 = keys['cdelt1'][ii], keys['cdelt1'][ii+njump]
				cdelt2 = keys['cdelt2'][ii], keys['cdelt2'][ii+njump]
				rsun_obs = keys['rsun_obs'][ii], keys['rsun_obs'][ii+njump]
				# print('crpix1 = ', crpix1)
				# print('crpix2 = ', crpix2)
				
				# print('cdelt1 = ', cdelt1)
				# print('cdelt2 = ', cdelt2)
				# print('rsun_obs = ', rsun_obs)

				# --- correction to dB, dP ---


				t_rec_1 = datetime.strptime(keys['t_rec'][ii], '%Y.%m.%d_%H:%M:%S_TAI')
				t_rec_2 = datetime.strptime(keys['t_rec'][ii+njump], '%Y.%m.%d_%H:%M:%S_TAI')
				dt_b0_1 = (t_rec_1-t_ref_b0).total_seconds()/86400./365.25
				dt_b0_2 = (t_rec_2-t_ref_b0).total_seconds()/86400./365.25

				dB = keys['crlt_obs'][ii] + dI*np.sin(2*np.pi*dt_b0_1), keys['crlt_obs'][ii+njump] + dI*np.sin(2*np.pi*dt_b0_2)
				dP = -keys['crota2'][ii] - dI*np.cos(2*np.pi*dt_b0_1), -keys['crota2'][ii+njump] - dI*np.cos(2*np.pi*dt_b0_2)

				T.append(datetime.now())
				T0 = datetime.now()
				# if rank == 0:
				# 	print('k = {}'.format(k))
				for ipatch,(clng,clat) in enumerate(xylist[lo:hi]):
					dL_change = 0
					if change_track:
						lamda = clat
						dL_change = (A + B*np.sin(np.deg2rad(lamda))**2 + C*np.sin(np.deg2rad(lamda))**4 - CRrate)*cadence/(3600*24)
					
					dL = 0, keys['crln_obs'][ii] - keys['crln_obs'][ii+njump] + dL_change
					
					img1p, img2p = from_tan_to_postel([img1, img2], crpix1, crpix2,0, 0, cdelt1, cdelt2, rsun_obs, dB, dP, dL, nx_out = patch_size, 
									   ny_out = patch_size, lngc_out = clng,latc_out = clat, pixscale_out = pixel_size, 
									   interp_method = 'cubconv', verbose = 1, nthr = 1, header = False) 
					
					ccf, _, _ = get_lct_map(img1p, img2p)
					

					if rank == 0:
						T.append(datetime.now())
						logger.info('%s (Postel remap and CCF for lat: %s and lng : %s)', T[-1]-T[-2], clat, clng)  

					if np.isnan(ccf).any():
						continue
					print('clat = ', clat, 'clng = ', clng)
					if (clat == 0.) and (clng == 0.):
						np.save('/data/seismo/joshin/pipeline-test/pmi_test/granules/data/data_ccfs/{}_{}_{}_{}_{}_4k_no_av.npy'.format(dat.strftime('%Y%m%d_%H%M%S'), 'ccf', k, clat, clng), ccf)
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

		
		for ixy, (ccf, (clng,clat)) in enumerate(zip(ccfs, xylist[lo:hi])):
			if clat == 0 and clng == 0:
				np.save('/data/seismo/joshin/pipeline-test/pmi_test/granules/data/data_ccfs/{}_{}_{}_{}_{}_4k_av.npy'.format(dat.strftime('%Y%m%d_%H%M%S'), 'ccf', k, clat, clng), ccf)
			if np.isnan(ccf).any():
				ux1[ixy] = uy1[ixy] = np.nan
				continue
			_,_,ux1[ixy], uy1[ixy] = get_ux_uy_ellipsoid(ccf, grid_len = 5, ntry= 3)

		ux_all = np.empty((len(xylist)), dtype='f8') if rank == 0 else None
		uy_all = np.empty((len(xylist)), dtype='f8') if rank == 0 else None
		comm.Gatherv(ux1, [ux_all, chunks], root=0)
		comm.Gatherv(uy1, [uy_all, chunks], root=0)

		if rank == 0:
			for (i,j), ux, uy in zip(ijlist, ux_all, uy_all):	
				utheta[it,j,i] = -uy
				uphi[it,j,i] = ux
			T1_chunk = datetime.now()
			logger.critical('%s (loop over one chunk)', T1_chunk-T0_chunk)

	if rank == 0:
		h5file.close()
		logger.critical('close %s', outfile)
