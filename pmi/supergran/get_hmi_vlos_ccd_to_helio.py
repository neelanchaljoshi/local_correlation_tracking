# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/data/seismo/zhichao/codes/pypkg/swan/remap')
from wrapper_tan2cyl import from_tan_to_cyl
from datetime import datetime, timedelta
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
from scipy.ndimage.filters import convolve1d
# %%

'''
Usage for the function from_tan_to_cyl:
def from_tan_to_cyl(cube,
        crpix1, crpix2, crval1, crval2, cdelt1, cdelt2,
        rsun_obs, dB, dP, dL,
        nx_out, ny_out, wcs_out,
        interp_method, verbose, nthr, header=False):
For getting the comparitive vlos, we need to remove the large scale trends
and track at differential rotation - see how i did it in LCT
dspan = 2h starting at 2019-05-15 00:00:00 to 02:00:00
dstep = 30 mins
dt = 1 min
'''
def nansmooth_gaussian(data, fwhm, return_se=False, return_sem=False, **kwargs):
    """
    --- Parameters ---
    data: numpy.ndarray
    sigma: float
    --- Returns ---
    out: numpy.ndarray
        gaussian-weighted-smoothing results
    se: numpy.ndarray
        gaussian-weighted-smoothing (unbiased) standard deviation (standard error)
    sem: numpy.ndarray
        standard error of the weighted mean
    --- Usage ---
    arr_o, sem_o = nansmooth_gaussian(arr_i, fwhm, return_sem=True, axis=?, mode='constant', cval=0)
    --- Reference ---
    gsl_stats_wvariance
    http://en.wikipedia.org/wiki/Mean_square_weighted_deviation
    """
    sigma = fwhm/np.sqrt(8*np.log(2))
    lw = int(4.0 * sigma + 0.5)
    w = np.exp(-0.5*((np.arange(2*lw+1, dtype='f8') - lw)/sigma)**2)
    #w /= w.sum()

    valid = ~np.isnan(data)
    b = valid.astype('f8')
    a = data.copy()
    a[~valid] = 0

    sw = convolve1d(b, w, **kwargs)
    swa = convolve1d(a, w, **kwargs)

    if return_se or return_sem:
        ret = [swa/sw]
        sww = convolve1d(b, w**2, **kwargs)
        swaa = convolve1d(a**2, w, **kwargs)
        se = np.sqrt((swaa*sw-swa*swa)/(sw*sw-sww))
        sem = np.sqrt(sww)/sw * se
        if return_se: ret.append(se)
        if return_sem: ret.append(sem)
        return ret
    else:
        return swa/sw

def nansmooth_boxcar(data, size, return_se=False, return_sem=False, **kwargs):
    """
    --- Parameters ---
    data: numpy.ndarray
    size: int
    --- Returns ---
    out: numpy.ndarray
        boxcar-smoothing results
    std: numpy.ndarray
        boxcar-smoothing (unbiased) standard deviation
    --- Usage ---
    arr_o, sem_o = nansmooth_boxcar(arr_i, size, return_sem=True, axis=?, mode='constant', cval=0)
    arr_o, sem_o = nansmooth_boxcar(arr_i, size, return_sem=True, axis=?, mode='constant', cval=np.nan)
    --- Reference ---
    gsl_stats_wvariance
    http://en.wikipedia.org/wiki/Mean_square_weighted_deviation
    """
    #if size % 2 == 0:
    #    raise RuntimeError('size must be an odd number')
    #w = np.ones(size, dtype='f8')
    if size % 2 == 0: # even number
        w = np.ones(size+1, dtype='f8')
        w[[0,-1]] *= 0.5
    else: # odd number
        w = np.ones(size, dtype='f8')

    valid = ~np.isnan(data)
    b = valid.astype('f8')
    a = data.copy()
    a[~valid] = 0

    sw = convolve1d(b, w, **kwargs)
    swa = convolve1d(a, w, **kwargs)

    if return_se or return_sem:
        ret = [swa/sw]
        sww = convolve1d(b, w**2, **kwargs)
        swaa = convolve1d(a**2, w, **kwargs)
        se = np.sqrt((swaa*sw-swa*swa)/(sw*sw-sww))
        #sem = np.sqrt(np.sum(w**2))/np.sum(w) * se
        sem = np.sqrt(sww)/sw * se
        if return_se: ret.append(se)
        if return_sem: ret.append(sem)
        return ret
    else:
        return swa/sw
# %%
# constants for differential rotation rate
A = 14.034
B = -1.702
C = -2.494
CRrate = 14.184
dI = -0.08
t_ref_b0 = datetime(2010, 6, 7, 14, 17, 20)

def get_dL_change(clat, time_diff):
    return (A + B*np.sin(np.deg2rad(clat))**2 + C*np.sin(np.deg2rad(clat))**4 - CRrate)*time_diff/(3600*24)

# %%
keys_2019 = Table.read('/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.v_45s/keys-2019.fits')
# %%
# First create the average vlos image from a span of 5 days that will be removed from the averagde vlos image of two hours
total_time_in_secs = 5*24*3600
dt = 45
time_jump_in_secs = 30*60
njump = time_jump_in_secs//dt
num_images = total_time_in_secs//dt
start_time = datetime(2019, 5, 15, 0, 0, 0)
end_time = start_time + timedelta(seconds=total_time_in_secs)
time_list = [start_time + timedelta(seconds=i*dt) for i in range(num_images)]
time_list_str = [i.strftime('%Y.%m.%d_%H:%M:%S_TAI') for i in time_list]

# Find the index of the time 2010-05-15 00:00:00 in keys_2019
idx_start = np.where(keys_2019['t_rec'] == time_list_str[0])[0][0]
# This index will be the starting point to get the next num_images with a step of njump

# %%
# Create a WCS out to which we will remap
ncyl = 6000
wcs_cyl  = WCS(naxis=2)
wcs_cyl.wcs.ctype = 'longitude', 'latitude'
wcs_cyl.wcs.cunit = 'deg', 'deg'
wcs_cyl.wcs.crpix = 0.5*(1+ncyl), 0.5*(1+ncyl)
wcs_cyl.wcs.crval = 35, 35
wcs_cyl.wcs.cdelt = 0.03, 0.03

total_img = np.zeros((ncyl, ncyl))
# %%
for i in tqdm(range(idx_start, idx_start + num_images, njump)):
    img = fits.getdata(keys_2019['path'][i][:-1] + '/Dopplergram.fits')
    obs_vr = keys_2019['obs_vr'][i]
    img_without_vr = img - obs_vr
    crpix1 = keys_2019['crpix1'][i]
    crpix2 = keys_2019['crpix2'][i]
    crval1 = keys_2019['crval1'][i]
    crval2 = keys_2019['crval2'][i]
    cdelt1 = keys_2019['cdelt1'][i]
    cdelt2 = keys_2019['cdelt2'][i]

    # Now there is a small correction to dB and dP required
    t_rec = datetime.strptime(keys_2019['t_rec'][i], '%Y.%m.%d_%H:%M:%S_TAI')
    dt_b0 = (t_rec-t_ref_b0).total_seconds()/86400./365.25

    dB = keys_2019['crlt_obs'][i] + dI*np.sin(2*np.pi*dt_b0)
    dP = -keys_2019['crota2'][i] - dI*np.cos(2*np.pi*dt_b0)

    rsun_obs = keys_2019['rsun_obs'][i]

    nx_out = ny_out = ncyl
    wcs_out = wcs_cyl

    interp_method = 'bilinear'

    clat = 35.0 # latitude at which we want to track

    # dL = keys_2019['crln_obs'][idx_start] - keys_2019['crln_obs'][i] + get_dL_change(clat, (i - idx_start)*dt)
    dL = get_dL_change(clat, (i - idx_start)*dt)
    img_smoothed_x = nansmooth_boxcar(img_without_vr, size=3, axis=1, mode='constant', cval=np.nan)
    img_smoothed_xy = nansmooth_boxcar(img_smoothed_x, size=3, axis=0, mode='constant', cval=np.nan)
    img_helio = from_tan_to_cyl(img_smoothed_xy[np.newaxis, :, :],
                                crpix1, crpix2, crval1, crval2, cdelt1, cdelt2,
                                rsun_obs, dB, dP, dL,
                                nx_out, ny_out, wcs_out,
                                interp_method, verbose = 1, nthr = 1, header=False)
    total_img += np.nan_to_num(img_helio[0, :, :])
# %%
avg_img_5days = total_img/(num_images//njump)
# %%
# Plot this image in pcolormesh with latitude and longitude axes
plt.figure(figsize=(10, 8))
plt.subplot()
plt.pcolormesh(np.linspace(30, 40, ncyl), np.linspace(30, 40, ncyl), avg_img_5days, cmap='bwr', vmin=-1000, vmax=1000)
plt.colorbar(label='Vlos (m/s)')
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('Average Vlos Image from 2019-05-15 to 2019-05-20')
# %%
# We now create an average image over a span of 2 hours with a step of 30 mins
total_time_in_secs = 2*3600
dt = 45
time_jump_in_secs = 30*60
njump = time_jump_in_secs//dt
num_images = total_time_in_secs//dt
start_time = datetime(2019, 5, 15, 0, 0, 0)
end_time = start_time + timedelta(seconds=total_time_in_secs)

# %%
total_img = np.zeros((ncyl, ncyl))
for i in tqdm(range(idx_start, idx_start + num_images, njump)):
    img = fits.getdata(keys_2019['path'][i][:-1] + '/Dopplergram.fits')
    obs_vr = keys_2019['obs_vr'][i]
    img_without_vr = img - obs_vr
    crpix1 = keys_2019['crpix1'][i]
    crpix2 = keys_2019['crpix2'][i]
    crval1 = keys_2019['crval1'][i]
    crval2 = keys_2019['crval2'][i]
    cdelt1 = keys_2019['cdelt1'][i]
    cdelt2 = keys_2019['cdelt2'][i]

    # Now there is a small correction to dB and dP required
    t_rec = datetime.strptime(keys_2019['t_rec'][i], '%Y.%m.%d_%H:%M:%S_TAI')
    dt_b0 = (t_rec-t_ref_b0).total_seconds()/86400./365.25

    dB = keys_2019['crlt_obs'][i] + dI*np.sin(2*np.pi*dt_b0)
    dP = -keys_2019['crota2'][i] - dI*np.cos(2*np.pi*dt_b0)

    rsun_obs = keys_2019['rsun_obs'][i]

    nx_out = ny_out = ncyl
    wcs_out = wcs_cyl

    interp_method = 'bilinear'

    clat = 35.0 # latitude at which we want to track

    # dL = keys_2019['crln_obs'][idx_start] - keys_2019['crln_obs'][i] + get_dL_change(clat, (i - idx_start)*dt)
    # dL = get_dL_change(clat, (i - idx_start)*dt)
    dL = 0
    img_smoothed_gaussian = nansmooth_gaussian(img_without_vr, fwhm=20, axis=1, mode='constant', cval=np.nan)
    img_smoothed_gaussian = nansmooth_gaussian(img_smoothed_gaussian, fwhm=20, axis=0, mode='constant', cval=np.nan)
    # img_smoothed_x = nansmooth_boxcar(img_without_vr, size=3, axis=1, mode='constant', cval=np.nan)
    # img_smoothed_xy = nansmooth_boxcar(img_smoothed_x, size=3, axis=0, mode='constant', cval=np.nan)
    img_helio = from_tan_to_cyl(img_smoothed_gaussian[np.newaxis, :, :],
                                crpix1, crpix2, crval1, crval2, cdelt1, cdelt2,
                                rsun_obs, dB, dP, dL,
                                nx_out, ny_out, wcs_out,
                                interp_method, verbose = 1, nthr = 1, header=False)
    total_img += img_helio[0, :, :]
# %%
avg_img_2hrs = total_img/(num_images//njump)
# %%
# Plot these two images side by side
lat = np.linspace(30, 40, ncyl)
lon = np.linspace(30, 40, ncyl)
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.pcolormesh(lon, lat, avg_img_5days, cmap='bwr', vmin=-1000, vmax=1000)
plt.colorbar(label='Vlos (m/s)')
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('Average Vlos Image from 2019-05-15 to 2019-05-20')
plt.subplot(1, 2, 2)
plt.pcolormesh(lon, lat, avg_img_2hrs, cmap='bwr', vmin=-1000, vmax=1000)
plt.colorbar(label='Vlos (m/s)')
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('Average Vlos Image from 2019-05-15 00:00:00 to 02:00:00')
# %%
# Subtract the two images to get the final vlos image for LCT
# final_vlos = avg_img_2hrs - avg_img_5days
final_vlos = avg_img_2hrs - np.nanmean(avg_img_2hrs)
# also remove a linear function in x direction from each row
for i in range(final_vlos.shape[0]):
    p = np.polyfit(lon, final_vlos[i, :], 1)
    final_vlos[i, :] -= np.polyval(p, lon)
# %%
# Plot this image
plt.figure(figsize=(10, 8))
plt.pcolormesh(lon, lat, final_vlos, cmap='bwr', vmin=-500, vmax=500)
plt.colorbar(label='Vlos (m/s)')
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('Final Vlos Image for LCT (2hrs - 5days)')
# %%
