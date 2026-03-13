# %% imports
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter
import glob
from astropy.table import Table
from datetime import datetime
# %% Functions
# Load the data
def load_file(path):
    f = h5py.File(path)
    return f

def get_data_arrays_from_file(path):
    f = load_file(path)
    data = {}
    for key in f.keys():
        data[key] = np.array(f[key][:])
    return data

def smooth_2d_gaussian(Z, sigma=1):
    return gaussian_filter(Z, sigma=sigma, mode='constant', cval=0)

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
    else:
        # Remove the fit for each y independently
        if order == 1:
            A = np.c_[X.ravel(), np.ones(X.size)]
        elif order == 2:
            A = np.c_[X.ravel()**2, X.ravel(), np.ones(X.size)]
        else:
            raise ValueError("Order must be 1 or 2")
        C, _, _, _ = np.linalg.lstsq(A, data.ravel(), rcond=None)
        fit = A.dot(C).reshape(data.shape[0], -1)
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

def vlos_from_u(b, l, B0, u_theta, u_phi, u_r=None, away_positive=False):
    """
    b: heliographic latitude [rad], +north
    l: Stonyhurst/CMD longitude from disk center [rad], +west
    B0: solar B-angle [rad]
    u_theta: colatitude (southward +) surface speed [same units as output]
    u_phi: longitude (westward +) surface speed
    u_r: optional radial (outward +) speed
    away_positive: if False, return toward-observer positive (Doppler blue +)
    """
    sinb, cosb = np.sin(b), np.cos(b)
    sinl, cosl = np.sin(l), np.cos(l)
    sinB0, cosB0 = np.sin(B0), np.cos(B0)

    # Horizontal contributions
    vlos = u_phi * (sinl * cosb) - u_theta * (sinb * cosB0 - cosb * sinB0 * cosl)

    # Optional radial contribution
    if u_r is not None:
        mu = sinb * sinB0 + cosb * cosB0 * cosl
        vlos += u_r * mu

    # By default we return away-from-observer positive; flip if you prefer toward +
    return vlos if away_positive else -vlos


def compute_los_velocity_lonlat(utheta, uphi, longitude, latitude, B0):
    """
    Compute line-of-sight velocity from 2D velocity arrays using longitude/latitude coordinates.

    This version is designed for 2D arrays of solar surface velocities with
    longitude/latitude coordinate grids, typical for solar physics data analysis.

    Parameters:
    -----------
    utheta : 2D array
        Meridional velocity component (positive towards south) [km/s]
        Shape: (nlat, nlon)
    uphi : 2D array
        Zonal velocity component (positive towards east) [km/s]
        Shape: (nlat, nlon)
    longitude : 2D array or 1D array
        Longitude coordinates in degrees
        If 1D: will be broadcast to match utheta/uphi shape
        If 2D: must match utheta/uphi shape
    latitude : 2D array or 1D array
        Latitude coordinates in degrees
        If 1D: will be broadcast to match utheta/uphi shape
        If 2D: must match utheta/uphi shape
    B0 : float
        Solar B-angle [rad]

    Returns:
    --------
    v_los : 2D array
        Line-of-sight velocity [km/s]
        Same shape as input velocity arrays
        Positive: motion towards observer (blueshift)
        Negative: motion away from observer (redshift)
    """

    # Convert to numpy arrays
    utheta = np.asarray(utheta)
    uphi = np.asarray(uphi)
    longitude = np.asarray(longitude)
    latitude = np.asarray(latitude)

    # Handle 1D coordinate arrays - create meshgrid if needed
    if longitude.ndim == 1 and latitude.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    elif longitude.ndim == 1:
        # Broadcast longitude to match latitude
        lon_grid = np.broadcast_to(longitude[np.newaxis, :], utheta.shape)
        lat_grid = latitude
    elif latitude.ndim == 1:
        # Broadcast latitude to match longitude
        lat_grid = np.broadcast_to(latitude[:, np.newaxis], utheta.shape)
        lon_grid = longitude
    else:
        # Both are 2D arrays
        lon_grid = longitude
        lat_grid = latitude

    # Convert degrees to radians
    lon_rad = np.deg2rad(lon_grid)
    lat_rad = np.deg2rad(lat_grid)

    # Convert latitude to colatitude (θ = π/2 - lat)
    theta = np.pi/2 - lat_rad
    phi = lon_rad

    # Line-of-sight unit vector components in spherical coordinates
    # For observer along +z axis (Earth's perspective)
    los_theta_comp = -(np.cos(B0) * np.cos(theta) * np.cos(phi) - np.sin(B0) * np.sin(theta))  # θ component
    los_phi_comp = np.cos(B0) * np.sin(phi)    # φ component


    # Project horizontal velocities onto line-of-sight
    # Note: utheta is positive southward, uphi is positive eastward
    # Need to flip sign of utheta since θ increases southward but los_theta_comp uses standard spherical convention
    v_los = utheta * los_theta_comp + uphi * los_phi_comp

    return v_los

# %% Load files

# path = "/data/seismo/joshin/pipeline-test/pmi_test/supergranular_flow/final_sg_compare/data/2018_5_dspan_60_dstep_30_dt_60_sg_flows_010_res_2deg_save_ccf_gran_40lat_0lon_4k.hdf5"
# path = "/data/seismo/joshin/pipeline-test/pmi_test/supergranular_flow/final_sg_compare/data/2018_5_dspan_60_dstep_30_dt_60_sg_flows_010_res_2deg_save_ccf_gran_40lat_0lon_carr_tracked_4k.hdf5"
path = '/data/seismo/joshin/pipeline-test/pmi_test/supergranular_flow/final_sg_compare/data/2018_5_dspan_60_dstep_30_dt_60_sg_flows_010_res_1,5deg_save_ccf_gran_40lat_0lon_carr_tracked_4k.hdf5'
# files = sorted(glob.glob('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/supergran/data/data_to_be_stitched_test_dspan_6/*.hdf5'))
# files = sorted(glob.glob('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/supergran/data/data_to_be_stitched_test_dspan_90s/*.hdf5'))
files = sorted(glob.glob('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/supergran/data/data_to_be_stitched_test/*.hdf5'))
keys_2010 = Table.read('/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.v_45s/keys-2010.fits')
# for file in files:
#     data = get_data_arrays_from_file(file)
#     print(f"Loaded {file} with uphi shape {data['uphi'].shape} and utheta shape {data['utheta'].shape}")
#     # concatenate the uphi and utheta arrays along the time axis
#     if 'uphi_all' not in locals():
#         uphi_all = data['uphi']
#         utheta_all = data['utheta']
#         t_all = data['tstart']
#     else:
#         uphi_all = np.concatenate((uphi_all, data['uphi']), axis=0)
#         utheta_all = np.concatenate((utheta_all, data['utheta']), axis=0)
#         t_all = np.concatenate((t_all, data['tstart']), axis=0)
f = h5py.File(path)
uphi_all = f['uphi'][:]
utheta_all = f['utheta'][:]
t_all = f['tstart'][:]
longitude1 = f['longitude'][:]
latitude1 = f['latitude'][:]


uphi_mean = np.nanmean(uphi_all, axis=0)
utheta_mean = np.nanmean(utheta_all, axis=0)
# longitude = data['longitude']
# latitude = data['latitude']
t_array_datetime = [datetime.strptime(str(t, 'utf-8'), '%Y.%m.%d_%H:%M:%S') for t in t_all]
print(f"Total number of time steps: {uphi_all.shape[0]}")
print(f"Shape of uphi_mean: {uphi_mean.shape}, Shape of utheta_mean: {utheta_mean.shape}")
print(f"Longitude shape: {longitude.shape}, Latitude shape: {latitude.shape}")
print(f"Time array length: {len(t_array_datetime)}")

# %% Plot mean flows
# Plot the mean flows
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(uphi_mean, cmap = 'bwr', vmax = 1000, vmin = -1000, origin='lower')
ax[0].set_title(r'$u_\phi$')
# ax[0].set_xlim([-4, 4])
# ax[0].set_ylim([36, 44])
ax[1].imshow(utheta_mean, cmap = 'bwr', vmax = 1000, vmin = -1000, origin='lower')
ax[1].set_title(r'$u_\theta$')
# ax[1].set_xlim([-4, 4])
# ax[1].set_ylim([36, 44])
plt.show()
# %%
# Compute the LOS velocity from the mean flows for each uphi and utheta
for i in range(uphi_all.shape[0]):
    dI = -0.08
    t_ref_b0 = datetime(2010, 6, 7, 14, 17, 20)
    # t_idx = np.where(keys_2010['t_rec'] == t_array_datetime[i].strftime('%Y.%m.%d_%H:%M:%S_TAI'))[0][0]
    print(t_array_datetime[i])
    # print(t_idx)
    t_rec = datetime.strptime(keys_2010['t_rec'][i], '%Y.%m.%d_%H:%M:%S_TAI')
    dt_b0 = (t_rec-t_ref_b0).total_seconds()/86400./365.25

    # dB = keys_2010['crlt_obs'][t_idx] + dI*np.sin(2*np.pi*dt_b0)
    dB = 0
    v_los = compute_los_velocity_lonlat(utheta_all[i], uphi_all[i], longitude, latitude, B0=np.deg2rad(dB))
    # smooth the vlos
    # v_los = smooth_2d_gaussian(v_los, sigma=1.63*3)
    print(v_los.shape)
    if 'v_los_all' not in locals():
        v_los_all = v_los[np.newaxis, :, :]
    else:
        v_los_all = np.concatenate((v_los_all, v_los[np.newaxis, :, :]), axis=0)
v_los_mean = np.nanmean(v_los_all, axis=0)
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(v_los_mean, cmap = 'jet', vmax = 1000, vmin = -1000, origin='lower')
ax.set_title(r'$v_{los}$ from mean flows')
# ax.set_xlim([-4, 4])
# ax.set_ylim([36, 44])
fig.colorbar(im, ax=ax, label='v_los (m/s)')
plt.show()
# %%
# Remove avg from the img
# Smooth the mean v_los
#
v_los_mean = v_los_mean - np.nanmean(v_los_mean)
# Remove x and y fits
x = np.arange(v_los_mean.shape[1])  # x-coordinates (columns)
y = np.arange(v_los_mean.shape[0])
v_los_detrended = remove_x_fit(v_los_mean, x, order=1, average_over_y=True)
v_los_detrended = remove_y_fit(v_los_detrended, y, order=1)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.pcolormesh(longitude, latitude, v_los_detrended, cmap = 'jet', vmax = 500, vmin = -500)
ax.set_title(r'Detrended $v_{los}$ from mean flows')
# ax.set_xlim([-4, 4])
# ax.set_ylim([36, 44])
fig.colorbar(im, ax=ax, label='v_los (m/s)')
plt.show()
# %% Save the detrended v_los
np.savez('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/supergran/data/detrended_vlos/detrended_vlos_from_lct_mean_flows_dspan_15min.npz', v_los_detrended=v_los_detrended, longitude=longitude, latitude=latitude)

# %%
