#!/usr/bin/env python
import os, sys, shutil, socket
from traceback import print_exc
import argparse
import logging
import configparser

from datetime import datetime, timedelta
import numpy as np
np.seterr(all='ignore')
from astropy.wcs import WCS
from astropy.io import fits
from scipy import signal
from astropy.table import Table

# sys.path.insert(0, '/data/seismo/joshin/opt/gcc-9.3.0/mpi4py-3.1.3_openmpi-4.0.5/lib/python3.6/site-packages')
sys.path.insert(0, '/data/seismo/joshin/pypkg')
# import mpi4py
# mpi4py.rc.recv_mprobe = False
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


MPI.Get_version()
mpi_lib_ver = strip_nonprintable(MPI.Get_library_version())

host = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

msg = 'rank %d on %s' % (rank, host)
msg = comm.gather(msg, root=0)
if rank == 0:
	print(mpi_lib_ver)
	assert len(msg) == size
	print('\n'.join(msg))

Ntry = 1
NX, NY = 4096, 4096
nthr = int(os.getenv('OMP_NUM_THREADS', default=1))

# --- arguments ---
parser = argparse.ArgumentParser(description='compute flow maps using correlation tracking')
parser.add_argument('cfgfile', type = str, metavar='<cfgfile>', help='configuration file')
#parser.add_argument('yr_start', type=int, help='start year')
#parser.add_argument('yr_stop', type=int, help='end year')
parser.add_argument('-l', '--loglevel', type=str, default='info',
		help='logging level: critical | error | warning | info | debug (default: info)')
args = parser.parse_args()

#ystart = args.yr_start
#ystop  = args.yr_stop

loglevel = getattr(logging, args.loglevel.upper(), None)
if not isinstance(loglevel, int):
	print('Unknown logging level: %s' % (args.loglevel), file=sys.stderr)
    
# --- import configuration file ---
cfgfile = args.cfgfile
cfg = configparser.ConfigParser()
cfg.read(cfgfile)

# === config: main ===
Ntry = cfg.getint('main', 'Ntry')
NX = cfg.getint('main', 'NX')
NY = cfg.getint('main', 'NY')

date_fmt = cfg.get('main', 'date_fmt')
ystart = cfg.getint('main', 'yr_start')
ystop = cfg.getint('main', 'yr_stop')
dspan_cfg = cfg.getfloat('main', 'dspan')
dspan = timedelta(hours = dspan_cfg)
cadence = cfg.getint('main', 'cadence')
dataset_cadence = cfg.getint('main', 'dataset_cadence')
njump = float(cadence)//float(dataset_cadence)
dstep = cfg.get('main', 'dstep')

dataset = cfg.get('main', 'dataset')
segname = cfg.get('main', 'segname')

rootdir_in = cfg.get('main', 'rootdir_in')
rootdir_out = cfg.get('main', 'rootdir_out')
infile_fmt = rootdir_in + dataset + '/keys-%Y.fits'
outfile_fmt = rootdir_in + dataset + '/%d_flowmap_debug.hdf5'



dI = cfg.getfloat('main', 'dI')
t_ref_b0 = datetime.strptime(cfg.get('main', 't_ref_b0'), date_fmt)

R_sun = cfg.getfloat('main', 'R_sun')
pixel_size = cfg.getfloat('main', 'pixel_size')
patch_size = cfg.getint('main', 'patch_size')
alpha = cfg.getfloat('main', 'alpha')


#======logger=========
logging.basicConfig(format='%(levelname)s %(message)s',
		level=loglevel, stream=sys.stdout)
logger = logging.getLogger()
if rank == 0:
	logger.info('host: %s', socket.gethostname())
	logger.info('start year: %d', ystart)
	logger.info('end year: %d', ystop)
	logger.info('num_threads: %d', nthr)
	begtime = datetime.now()

	# Based on k**2 find tukey window value and place in matrix
#	return base
from tukey_window import tukey_twoD
from get_lct_map import get_lct_map
from get_ux_uy import get_ux_uy_ellipsoid

gkern = tukey_twoD(patch_size, alpha) #See tukey_window.py for details



nlng, nlat = 35, 35
clatarr = np.linspace(-85, +85, nlat)
clngarr = np.linspace(-85, +85, nlng)
xylist = [(x,y) for y in clatarr for x in clngarr]
ijlist = [(i,j) for j in range(clatarr.size) for i in range(clngarr.size)]
delim, chunks = get_delimiters(len(xylist), size, return_chunks=True)
lo, hi = delim[rank]

T = []

#print('HERE out of the loop')

# --- loop over days ---
for yr in range(ystart,ystop): # HERE
	#print('HERE in the loop')
	# --- read keys ---
	infile = datetime(yr,1,1).strftime(infile_fmt)
	keys = Table.read(infile)

	if rank == 0:
		logger.critical('read %s', infile)

	dstart_yr = datetime(yr, 1 if yr > 2010 else 5, 1, 0, 0, 0)
	#dstop_yr = datetime(yr+1, 1, 1, 0, 0, 0)
	dstop_yr = datetime(yr+1, 1, 1, 0, 0, 0) # HERE
	#print(dstart_yr, dstop_yr)

	#dats_total = [dat for dat in xdays(dstart, dstop, dspan)]
	# dats = []
	# for i in range(len(dats_total)-1):
	# 	dats_temp = [dat for dat in xdays(dats_total[i], dats_total[i+1], dstep)]
	# 	dats.append(dats_temp)
	nt = (dstop_yr - dstart_yr).total_seconds()/dspan.total_seconds()
	assert nt.is_integer()
	nt = int(nt)

	if rank == 0:
		# f1 = h5py.File('/scratch/seismo/joshin/pipeline-test/hmi.ic_45s/%d_u_phi.hdf5' % (yr), 'w', driver='mpio', comm=MPI.COMM_WORLD)
		# f2 = h5py.File('/scratch/seismo/joshin/pipeline-test/hmi.ic_45s/%d_u_theta.hdf5' % (yr), 'w', driver='mpio', comm=MPI.COMM_WORLD)
		outfile = outfile_fmt % (yr)
		# h5file = h5py.File(outfile, 'w', driver='mpio', comm=MPI.COMM_WORLD)
		h5file = h5py.File(outfile, 'w')
		utheta = h5file.create_dataset('utheta', (nt, nlat, nlng), dtype='f4')
		uphi = h5file.create_dataset('uphi', (nt, nlat, nlng), dtype='f4')
		tstart = h5file.create_dataset('tstart', (nt,), dtype='S19')
		h5file.create_dataset('longitude', dtype = 'f8', data = clngarr)
		h5file.create_dataset('latitude', dtype = 'f8', data = clatarr)
		logger.critical('create %s', outfile)

	#print('HERE 3')
	# for i in range(len(dats_total)):
	for it,dstart_chunk in enumerate(xdays(dstart_yr, dstop_yr, dspan)):

		ccfs = np.zeros((hi-lo, patch_size, patch_size), dtype='f8')
		nums = np.zeros((hi-lo), dtype='i4')
		if rank == 0:
			tstart[it] = bytes(dstart_chunk.strftime('%Y.%m.%d_%H:%M:%S'), encoding='utf-8')

		T0_chunk = datetime.now()
		# for j in range(dspan/dstep):
		for dat in xdays(dstart_chunk, dstart_chunk+dspan, dstep):
			# dat = dats[i][j]
			# outdir = dat.strftime(outdir_fmt)
			try:

				if rank == 0:
					# --- log ---
					T.append(datetime.now())
					logger.info('--- %s ---', dat.strftime(date_fmt))
					logger.info('%s', T[-1])
				# --- outdir ---
					#if not os.path.exists(outdir):
					#    os.makedirs(outdir)
				
				

				dat0 = datetime.strptime(keys['t_rec'][0], '%Y.%m.%d_%H:%M:%S_TAI')
				ii = (dat - dat0).total_seconds()/cadence
				assert ii.is_integer()
				ii = int(ii)
				# il = int(il)
				# ih = il + ntotal
				# keys = tabs[yr][il:ih]


				# for it in range(nt):
					# ii = it*step
				if rank == 0:
					# --- read data ---
					if keys['isbad'][ii] or keys['isbad'][ii+1]:
						logger.error( 'skipping %s and %s (either one is bad)',
								keys['t_rec'][ii], keys['t_rec'][ii+1])
						isbad = True
						img1, img2 = None, None
					else:
						for itry in range(Ntry):
							try:
								img1 = fits.getdata(os.path.join(keys['path'][ii], segname))
								img2 = fits.getdata(os.path.join(keys['path'][ii+1], segname))
								isbad = False
								break
							except IOError as e: # [Errno 5] Input/output error
								logger.warning( '%s (%s or %s)', e, keys['t_rec'][ii], keys['t_rec'][ii+1])
								if itry != Ntry-1:
									time.sleep(5)
						else:
							logger.error( 'skipping %s (%s) and %s (%s)',
								keys['path'][ii], keys['t_rec'][ii], keys['path'][ii+1], keys['t_rec'][ii+1])
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

				assert keys['crval1'][ii] == keys['crval1'][ii+1] == 0
				assert keys['crval2'][ii] == keys['crval2'][ii+1] == 0
				crpix1 = keys['crpix1'][ii], keys['crpix1'][ii+1]
				crpix2 = keys['crpix2'][ii], keys['crpix2'][ii+1]
				cdelt1 = keys['cdelt1'][ii], keys['cdelt1'][ii+1]
				cdelt2 = keys['cdelt2'][ii], keys['cdelt2'][ii+1]
				rsun_obs = keys['rsun_obs'][ii], keys['rsun_obs'][ii+1]

				# --- correction to dB, dP ---


				t_rec_1 = datetime.strptime(keys['t_rec'][ii], '%Y.%m.%d_%H:%M:%S_TAI')
				t_rec_2 = datetime.strptime(keys['t_rec'][ii+1], '%Y.%m.%d_%H:%M:%S_TAI')
				dt_b0_1 = (t_rec_1-t_ref_b0).total_seconds()/86400./365.25
				dt_b0_2 = (t_rec_2-t_ref_b0).total_seconds()/86400./365.25

				dB = keys['crlt_obs'][ii] + dI*np.sin(2*np.pi*dt_b0_1), keys['crlt_obs'][ii+1] + dI*np.sin(2*np.pi*dt_b0_2)
				dP = -keys['crota2'][ii] - dI*np.cos(2*np.pi*dt_b0_1), -keys['crota2'][ii+1] - dI*np.cos(2*np.pi*dt_b0_2)
				dL = 0, keys['crln_obs'][ii] - keys['crln_obs'][ii+1]


				T.append(datetime.now())
				T0 = datetime.now()
				for ipatch,(clng,clat) in enumerate(xylist[lo:hi]):
					img1p, img2p = from_tan_to_postel([img1, img2], crpix1, crpix2,0, 0, cdelt1, cdelt2, rsun_obs, dB, dP, dL, nx_out = patch_size, ny_out = patch_size, lngc_out = clng,latc_out = clat, pixscale_out = pixel_size, interp_method = 'cubconv', verbose = 1, nthr = 1, header = False) 
					ccf, _, _ = get_lct_map(img1p, img2p, patch_size, gkern)
					if rank == 0 and ipatch%100==0:
						T.append(datetime.now())
						logger.info('%s (Postel remap and CCF for lat: %s and lng : %s)', T[-1]-T[-2], clat, clng)  
					# outfile = '/scratch/seismo/joshin/pipeline-test/hmi.ic_45s/ii%d_lng%g_lat%g.fits' % (ii, clng, clat)
					# fits.writeto(outfile, ccf, overwrite=True)
					# print('output', outfile)
					if np.isnan(ccf[0,0]):
						continue
					ccfs[ipatch] += ccf
					nums[ipatch] += 1
					# outfile = '/scratch/seismo/joshin/pipeline-test/hmi.ic_45s/ccfs_ii%d_lng%g_lat%g.fits' % (ii, clng, clat)
					# fits.writeto(outfile, ccfs, overwrite=True)
					# print('output', outfile)
				
				comm.Barrier()
				T1 = datetime.now()
				if rank == 0:
					logger.info('%s (loop over patches)', T1-T0)


				# if output_cube_remap:
				#     cube = np.empty((nt,nlat_map,nlng_map), dtype='f4') if rank == 0 else None
				#     gatherv_chunky(comm, cube1, cube, delim, root=0)
				#     rank0_has_cube = True
				#     if rank == 0:
				#         T.append(datetime.now())
				#         logger.info('%s (gatherv_chunky cube1[%g GB] -> cube[%g GB])',
				#                 T[-1]-T[-2], cube1.nbytes/1024.**3, cube.nbytes/1024.**3)
				#         outfile = os.path.join(outdir, 'cube_remap.fits')
				#         fits.writeto(outfile, cube, hdr, overwrite=True)
				#         T.append(datetime.now())
				#         logger.info('%s (output %s)', T[-1]-T[-2], outfile)

			except RuntimeError as e:
				if rank == 0:
					logger.error('%s: %s', dat.date(), e)
				continue

		ccfs /= nums[:,None,None]
		ux1 = np.empty((hi-lo), dtype='f8')
		uy1 = np.empty((hi-lo), dtype='f8')
		for ixy, (ccf, (clng,clat)) in enumerate(zip(ccfs, xylist[lo:hi])):
			_,_,ux1[ixy], uy1[ixy] = get_ux_uy_ellipsoid(clat, clng, np.nan_to_num(ccf), patch_size, R_sun, pixel_size, cadence)
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

		# ccfs = ccfs.astype('f4')
		# counts = [npatch*patch_size*patch_size for npatch in chunks]
		# recv = np.empty((sum(chunks), patch_size, patch_size), dtype='f4') if rank == 0 else None
		# comm.Gatherv(ccfs, [recv, counts], root=0)
		# if rank == 0:
		#     T.append(datetime.now())
		#     logger.info('%s (Gatherv)', T[-1]-T[-2])
		# 	ux_list = []
		# 	uy_list = []
		# 	for ccf,(clng,clat) in zip(recv, xylist):
		# 		_,_,ux, uy = get_ux_uy_ellipsoid(clat, clng, ccf)
		# 		ux_list.append(ux)
		# 		uy_list.append(uy)
		# 	uphi_map = (np.reshape(np.array(ux_list), (35, 35)))
		# 	utheta_map = (np.reshape(np.array(-uy_list), (35, 35)))
		# 	f1.create_dataset(str(dat), dtype = 'float', data = uphi_map)
		# 	f2.create_dataset(str(dat), dtype = 'float', data = utheta_map)





# if rank == 0:
#     hdulist = fits.HDUList()
#     for ccf,(clng,clat) in zip(recv, xylist):
#         h = wcs_out.to_header()
#         h['latc'] = format(clat, '.3f')
#         h['lngc'] = format(clng, '.3f')
#         hdulist.append(fits.ImageHDU(ccf, h, name=str(clng)+'_'+str(clat)))
#     outfile_p = '/scratch/seismo/joshin/pipeline-test/hmi.ic_45s/ccf_postel_all.fits'
#     hdulist.writeto(outfile_p, overwrite=True)
#     hdulist.close()
#     logger.info('%s (output %s)', T[-1]-T[-2], outfile_p)
#     logger.info('%s (total elapsed time)', T[-1]-T[0])


	#except:
		#comm.Abort()
		#if rank == 0:
		#    etype, e, tb = sys.exc_info()
		#    print >>sys.stderr, '%s: %s' % (etype.__name__, e)
		#    print_tb(tb)
		#    print_exception(etype, e, tb)
		#    print_exc(file=sys.stderr)
		#    print >>sys.stderr, '%s: %s' % (sys.exc_type.__name__, sys.exc_value)
		#    raise
		# if rank == 0:
		#     print_exc(file=sys.stderr)
		# raise SystemExit(1)

	# finally:
	#     if os.path.exists(outdir) and len(os.listdir(outdir)) == 0:
	#         os.removedirs(outdir)
	#     # --- log ---
	#     T.append(datetime.now())
	#     logger.info('%s ET', T[-1]-T[0])

# } end of date loop

# --- log ---
# endtime = datetime.now()
# logger.critical('=== The End ===')
# logger.info('Beg Time: %s', begtime)
# logger.info('End Time: %s', endtime)
# logger.info('Total Elapsed Time: %s', endtime-begtime)
# raise SystemExit

# vim: set foldlevel=0:

