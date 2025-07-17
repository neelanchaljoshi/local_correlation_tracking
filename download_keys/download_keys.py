#!/usr/bin/env python
import os
from datetime import datetime, timedelta
import subprocess as subp
import numpy as np
from astropy.table import Table, Column

# QbitsPass = 0b01111111111111111111001010011111
QbitsPass = 0b00000000000000000000000000000000

# All bits set to zero indicates no quality issues.

seriesname = 'hmi.ic_45s' # HMI Continuum Intensity, 45s cadence
outdir = '/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.ic_45s' # Output directory for the keys files
cadence = 45 # Cadence in seconds


KeyList = [ 
        ('t_rec', bytes),
        ('t_obs', bytes),
        ('quality', str),
        ('crpix1', float),
        ('crpix2', float),
        ('crval1', float),
        ('crval2', float),
        ('cdelt1', float),
        ('cdelt2', float),
        ('crota2', float),
        ('crln_obs', float),
        ('crlt_obs', float),
        ('rsun_obs', float),
        # ('dsun_obs', float),
        # ('obs_vr', float),
        # ('obs_vw', float),
        # ('obs_vn', float),
        # ('datamean', float),
        # ('datarms', float),
        ]
def get_info(ds, keylist=KeyList):
    """
    Get the keys and path information for a given dataset.
    Parameters:
        ds (str): The dataset identifier.
        keylist (list of tuples): List of keys and their types to retrieve.
    Returns:
        keys (dict): Dictionary of keys with their values.
        path (list): List of file paths associated with the dataset.
    """
    knames = ','.join([nam for nam, typ in keylist])
    p = subp.Popen('show_info ds=%s key=%s -q' % (ds, knames), shell=True, stdout=subp.PIPE, encoding='utf-8')  
    lines = [line.rstrip() for line in p.stdout.readlines()]
    keys_str = np.array([line.split() for line in lines])
    keys = {}
    for i, (nam,typ) in enumerate(keylist):
        keys[nam] = keys_str[:,i].astype(typ)
    path = subp.Popen('show_info ds=%s -Pq' % (ds), shell=True, stdout=subp.PIPE, encoding='utf-8').stdout.readlines()
    return keys, path


T = [datetime.now()]
for yr in range(2022, 2023):
	if yr == 2010:
		dstart = datetime(2010,5,1)
	else:
		dstart = datetime(yr,1,1)
	dstop = datetime(yr+1,1,1)

	dspan = dstop - dstart
	nt = dspan.total_seconds()/cadence
	assert nt.is_integer()
	nt = int(nt)
	ds = '%s[%s/%ds@%ds]' % (seriesname,
				dstart.strftime('%Y.%m.%d_%H:%M:%S_TAI'),
				dspan.total_seconds(), cadence)
	keys, path = get_info(ds)
	T.append(datetime.now())
	print(T[-1] - T[-2], 'get_info', yr)
	if len(path) != nt:
		raise RuntimeError('acquired data length %d is not equal to %d' % (len(path), nt))
	if len(keys['quality']) < nt:
		raise RuntimeError('acquired data length ({}) is less than expected ({})'.format(len(keys['quality']), nt))
	quality = np.array([int(s, 16) for s in keys['quality']])

	isbad = (quality | QbitsPass != QbitsPass)

	tab = Table()
	for nam, typ in KeyList:
		tab[nam] = Column(keys[nam], dtype=typ)
	tab['isbad'] = Column(isbad, dtype=bool)
	tab['path'] = Column(path, dtype=bytes)

	# outfile = os.path.join(outdir, 'keys.hdf5')
	# tab.write(outfile, format='hdf5', path='table', overwrite=True)
	# T.append(datetime.now())
	# print(T[-1] - T[-2], 'output', outfile)

	outfile = os.path.join(outdir, 'keys-%d.fits') % (yr)
	tab.write(outfile, format='fits', overwrite=True)
	T.append(datetime.now())
	print(T[-1] - T[-2], 'output', outfile)