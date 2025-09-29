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
from scipy.ndimage import zoom
from scipy.ndimage.filters import convolve1d
from scipy.ndimage import gaussian_filter
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
dspan = 2h starting at 2018-05-15 00:00:00 to 03:00:00
dstep = 30 mins
dt = 1 min
'''

# Fit and remove the polynomial in the x-direction (first) then average in y direction and the subtract it
def remove_x_fit(data, x, order=1, average_over_y=True):
    """
    Remove a polynomial fit from the data along the x dimension.
    data: 2D array
    x: 1D array of the same length as data.shape[1]
    order: order of the polynomial fit
    average_over_y: if True, remove the fit averaged over the y dimension
    """
    Y = np.arange(data.shape[0])
    X = np.meshgrid(x, Y)[0]

    if average_over_y:
        # Remove the fit for each y independently
        if order == 1:
            A = np.c_[X.ravel(), np.ones(X.size)]
        elif order == 2:
            A = np.c_[X.ravel()**2, X.ravel(), np.ones(X.size)]
        else:
            raise ValueError("Order must be 1 or 2")
        C, _, _, _ = np.linalg.lstsq(A, data.ravel(), rcond=None)
        fit = A.dot(C).reshape(data.shape[0], -1)
        data_avg = np.nanmean(fit, axis=0)  # Average the fit
        data_fit_removed = data - data_avg  # Subtract the averaged fit from the data

        # Average the data over the y dimension after removing the fit
        data_avg = np.nanmean(data_fit_removed, axis=0)
        return data_fit_removed - data_avg
    else:
        # Average the data over the y dimension
        data_avg = np.nanmean(data, axis=0)
        if order == 1:
            A = np.c_[x, np.ones(len(x))]
        elif order == 2:
            A = np.c_[x**2, x, np.ones(len(x))]
        else:
            raise ValueError("Order must be 1 or 2")
        C, _, _, _ = np.linalg.lstsq(A, data_avg, rcond=None)
        fit = A.dot(C)
        return data - fit

def remove_y_fit(data, y, order=1):
    """
    Remove a polynomial fit from the data along the y dimension.
    data: 2D array
    y: 1D array of the same length as data.shape[0]
    order: order of the polynomial fit
    """
    X = np.arange(data.shape[1])
    Y = np.meshgrid(X, y)[1]
    if order == 1:
        A = np.c_[Y.ravel(), np.ones(Y.size)]
    elif order == 2:
        A = np.c_[Y.ravel()**2, Y.ravel(), np.ones(Y.size)]
    else:
        raise ValueError("Order must be 1 or 2")
    C, _, _, _ = np.linalg.lstsq(A, data.ravel(), rcond=None)
    fit = A.dot(C).reshape(-1, data.shape[1])
    return data - fit

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
keys_2018 = Table.read('/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.v_45s/keys-2018.fits')
# %%
# We now create an average image over a span of 2 hours with a step of 30 mins
total_time_in_secs = 3*3600
dt = 45
time_jump_in_secs = 30*60
njump = time_jump_in_secs//dt
num_images = total_time_in_secs//dt
start_time = datetime(2018, 5, 15, 0, 0, 0)
end_time = start_time + timedelta(seconds=total_time_in_secs)
time_list = [start_time + timedelta(seconds=i*dt) for i in range(num_images)]
time_list_str = [i.strftime('%Y.%m.%d_%H:%M:%S_TAI') for i in time_list]

# Find the index of the time 2010-05-15 00:00:00 in keys_2018
idx_start = np.where(keys_2018['t_rec'] == time_list_str[0])[0][0]
# This index will be the starting point to get the next num_images with a step of njump

# %%
# Create a WCS out to which we will remap
ncyl = 6000
wcs_cyl  = WCS(naxis=2)
wcs_cyl.wcs.ctype = 'longitude', 'latitude'
wcs_cyl.wcs.cunit = 'deg', 'deg'
wcs_cyl.wcs.crpix = 0.5*(1+ncyl), 0.5*(1+ncyl)
wcs_cyl.wcs.crval = 0, 0
wcs_cyl.wcs.cdelt = 0.03, 0.03
total_img = np.zeros((ncyl, ncyl))
image_cube = np.zeros((num_images, ncyl, ncyl))
for i in tqdm(range(idx_start, idx_start + num_images, njump)):
    img = fits.getdata(keys_2018['path'][i][:-1] + '/Dopplergram.fits')
    obs_vr = keys_2018['obs_vr'][i]
    img_without_vr = img - obs_vr
    crpix1 = keys_2018['crpix1'][i]
    crpix2 = keys_2018['crpix2'][i]
    crval1 = keys_2018['crval1'][i]
    crval2 = keys_2018['crval2'][i]
    cdelt1 = keys_2018['cdelt1'][i]
    cdelt2 = keys_2018['cdelt2'][i]

    # Now there is a small correction to dB and dP required
    t_rec = datetime.strptime(keys_2018['t_rec'][i], '%Y.%m.%d_%H:%M:%S_TAI')
    dt_b0 = (t_rec-t_ref_b0).total_seconds()/86400./365.25

    dB = keys_2018['crlt_obs'][i] + dI*np.sin(2*np.pi*dt_b0)
    dP = -keys_2018['crota2'][i] - dI*np.cos(2*np.pi*dt_b0)
    print('dB = ', dB, 'dP = ', dP)
    rsun_obs = keys_2018['rsun_obs'][i]

    nx_out = ny_out = ncyl
    wcs_out = wcs_cyl

    interp_method = 'nearest'

    clat = 35.0 # latitude at which we want to track
    dL = keys_2018['crln_obs'][i] - keys_2018['crln_obs'][i+1]
    # dL = keys_2018['crln_obs'][idx_start] - keys_2018['crln_obs'][i] #+ get_dL_change(clat, (i - idx_start)*dt)
    # dL = get_dL_change(clat, (i - idx_start)*dt)
    # dL = 0
    # img_smoothed_gaussian = nansmooth_gaussian(img_without_vr, fwhm=20, axis=1, mode='constant', cval=np.nan)
    # img_smoothed_gaussian = nansmooth_gaussian(img_smoothed_gaussian, fwhm=20, axis=0, mode='constant', cval=np.nan)
    # img_smoothed_x = nansmooth_boxcar(img_without_vr, size=3, axis=1, mode='constant', cval=np.nan)
    # img_smoothed_xy = nansmooth_boxcar(img_smoothed_x, size=3, axis=0, mode='constant', cval=np.nan)
    img_helio = from_tan_to_cyl(img[np.newaxis, :, :],
                                crpix1, crpix2, crval1, crval2, cdelt1, cdelt2,
                                rsun_obs, dB, dP, dL,
                                nx_out, ny_out, wcs_out,
                                interp_method, verbose = 1, nthr = 1, header=False)
    image_cube[(i - idx_start)//njump, :, :] = img_helio[0, :, :]

    total_img += np.nan_to_num(img_helio[0, :, :])
# %%
# Save the image cube
# np.save('dopplergram_3h_helio_cube.npy', image_cube)
# fits.writeto('dopplergram_3h_helio.fits', image_cube, overwrite=True)
avg_img = total_img/(num_images//njump)
# print(avg_img)
# Crop out the square between lat 30 to 50 and lon -10 to 10
# Plot the average image with pcolormesh


lats = np.linspace(-90, 90, ncyl)
lons = np.linspace(-90, 90, ncyl)

plt.figure(figsize=(8, 6))
plt.pcolormesh(lons, lats, avg_img, cmap='jet', vmin=-2000, vmax=2000)
plt.colorbar(label='Doppler Velocity (m/s)')
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
plt.title('Average Dopplergram (3 hours)')
plt.show()


lat_mask = (lats >= 30) & (lats <= 50)
lon_mask = (lons >= -10) & (lons <= 10)
avg_img = avg_img[lat_mask, :][:, lon_mask]
print(avg_img.shape)

def reshape_to_200_zoom(arr):
    """
    Resize a 666x666 array to 200x200 using scipy.ndimage.zoom.
    """
    assert arr.shape == (666, 666), "Input must be 666x666"

    zoom_factor = 200 / 666
    return zoom(arr, zoom_factor, order=1)

# Resize the image to 200x200
def smooth_2d_gaussian(Z, sigma=1):
    return gaussian_filter(Z, sigma=sigma, mode='constant', cval=0)
# smooth the image with a gaussian filter with sigma = 1.4 degrees

avg_img = reshape_to_200_zoom(avg_img)

avg_img = avg_img - np.nanmean(avg_img)

# Remove a polyfit of order 1 in x and y
# Remove a 2D polynomial fit of order 1 in x and y
x = np.arange(avg_img.shape[1])  # x-coordinates (columns)
y = np.arange(avg_img.shape[0])  # y-coordinates (rows)
avg_img = remove_x_fit(avg_img, x, order=1, average_over_y=True)
avg_img = remove_y_fit(avg_img, y, order=1)
avg_img = smooth_2d_gaussian(avg_img, sigma=2)
# Remove mean over image

# print(avg_img)



# %%
# Plot the figure with pcolormesh
lons = np.linspace(-10, 10, avg_img.shape[1])
lats = np.linspace(30, 50, avg_img.shape[0])
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
c = ax.pcolormesh(lons, lats, avg_img, cmap='jet', vmin=-400, vmax=400)
ax.set_xlabel('Longitude (degrees)')
ax.set_ylabel('Latitude (degrees)')
# ax.set_xlim(-50, 50)
# ax.set_ylim(-50, 50)
ax.set_title('Average Dopplergram cleaned and zoomed (3 hours)')
fig.colorbar(c, ax=ax, label='Doppler Velocity (m/s)')
plt.show()

# %%
# save the avg image as npy
# np.save('vlos_cleaned_langfellner.npy', avg_img)

# %%
