# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter
# %%
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
# %%


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

# %%

# path = "/data/seismo/joshin/pipeline-test/pmi_test/supergranular_flow/final_sg_compare/data/2018_5_dspan_60_dstep_30_dt_60_sg_flows_010_res_2deg_save_ccf_gran_40lat_0lon_4k.hdf5"
path = "/data/seismo/joshin/pipeline-test/pmi_test/supergranular_flow/final_sg_compare/data/2018_5_dspan_60_dstep_30_dt_60_sg_flows_010_res_2deg_save_ccf_gran_40lat_0lon_carr_tracked_4k.hdf5"
# path = '/data/seismo/joshin/pipeline-test/pmi_test/supergranular_flow/final_sg_compare/data/2018_5_dspan_60_dstep_30_dt_60_sg_flows_010_res_1,5deg_save_ccf_gran_40lat_0lon_carr_tracked_4k.hdf5'
data = get_data_arrays_from_file(path)

uphi_mean = np.nanmean(data['uphi'][0:3], axis=0)
utheta_mean = np.nanmean(data['utheta'][0:3], axis=0)

# Check the keys in the file
print("Keys in the file:", data.keys())

# Check the shape of the data arrays
for key, value in data.items():
    print(f"Shape of {key}: {value.shape}")
print("The time start for this array is ", data['tstart'][0])

# Remove large-scale trends from the flow data
uphi_mean = remove_x_fit(uphi_mean, data['longitude'][:], order=1)
uphi_mean = remove_y_fit(uphi_mean, data['latitude'][:], order=1)
utheta_mean = remove_x_fit(utheta_mean, data['longitude'][:], order=1)
utheta_mean = remove_y_fit(utheta_mean, data['latitude'][:], order=1)


# Plot the data
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].pcolormesh(data['longitude'], data['latitude'], uphi_mean, cmap = 'jet', vmax = 500, vmin = -500)
ax[0].set_title(r'$u_\phi$')
ax[1].pcolormesh(data['longitude'], data['latitude'], utheta_mean, cmap = 'jet', vmax = 500, vmin = -500)
ax[1].set_title(r'$u_\theta$')




# %%
lat_rad = np.radians(data['latitude'][:])
lon_rad = np.radians(data['longitude'][:])
B0 = np.deg2rad(-2.65)  # Assuming B0 is 0 radians for simplicity
# %%
# testing the compute_los_velocity_lonlat function
vlos_test2 = compute_los_velocity_lonlat(utheta_mean, uphi_mean, data['longitude'], data['latitude'], B0)
# tesing the vlos_from_u function
# vlos_test2 = vlos_from_u(lat_rad[:, np.newaxis], lon_rad[np.newaxis, :], B0, utheta_mean, uphi_mean, u_r=None, away_positive=False)
# %%
# Remove mean from vlos
# vlos_test2 -= np.nanmean(vlos_test2)
# plot vlos_test2
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.pcolormesh(data['longitude'][:], data['latitude'][:], vlos_test2, cmap='jet', vmin=-500, vmax=500)
ax.set_title('LOS Velocity from compute_los_velocity_lonlat')
ax.set_xlabel('Longitude (degrees)')
ax.set_ylabel('Latitude (degrees)')
ax.set_xlim(-10, 10)
ax.set_ylim(30, 50)
cbar = plt.colorbar(im, ax=ax, label='LOS Velocity (m/s)')
plt.show()

# %%
vlos_test2 = vlos_test2 - np.nanmean(vlos_test2)
x = np.arange(vlos_test2.shape[1])  # x-coordinates (columns)
y = np.arange(vlos_test2.shape[0])  # y-coordinates (rows)
vlos_test2 = remove_x_fit(vlos_test2, x, order=1, average_over_y=True)
vlos_test2 = remove_y_fit(vlos_test2, y, order=1)
vlos_test2 = smooth_2d_gaussian(vlos_test2, sigma=1.4)
# vlos_test2 = smooth_2d_gaussian(vlos_test2, sigma=1.4)
# %%
# plot vlos_test2 again
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.pcolormesh(data['longitude'][:], data['latitude'][:], vlos_test2, cmap='jet', vmin=-400, vmax=400)
ax.set_title('LOS Velocity from compute_los_velocity_lonlat (trend removed)')
ax.set_xlabel('Longitude (degrees)')
ax.set_ylabel('Latitude (degrees)')
ax.set_xlim(-10, 10)
ax.set_ylim(30, 50)
cbar = plt.colorbar(im, ax=ax, label='LOS Velocity (m/s)')
plt.show()
# %%
# save the vlos_test2 array to a npy file
# np.save('vlos_test2.npy', vlos_test2)
# %%
