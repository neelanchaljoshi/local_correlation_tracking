"""
geometry.py
-----------
Solar disk geometry: radius array construction and flow data masking.
No I/O, no Fourier transforms — pure spatial operations.
"""

import sys
import numpy as np
from tqdm import tqdm

from .config import LON_OG, LAT_OG, CLIP_RADIUS, APOD_R_MIN, APOD_R_MAX

# MPS-internal package
sys.path.insert(0, '/data/seismo/zhichao/codes/pypkg')
from zclpy3.remap import get_tan_from_lnglat  # noqa: E402


def make_lon_lat_grids(
    lon_range: tuple = LON_OG,
    lat_range: tuple = LAT_OG,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return 1-D longitude and latitude grids.

    Parameters
    ----------
    lon_range, lat_range : (start, stop, n_points)

    Returns
    -------
    lon_og, lat_og : 1-D ndarrays
    """
    lon_og = np.linspace(*lon_range)
    lat_og = np.linspace(*lat_range)
    return lon_og, lat_og


def build_radius_array(
    crlt_obs: np.ndarray,
    rsun_obs: np.ndarray,
    lon_og: np.ndarray,
    lat_og: np.ndarray,
) -> np.ndarray:
    """
    Compute the projected disk radius at every (time, lat, lon) pixel.

    Parameters
    ----------
    crlt_obs : (nt,) Carrington latitude of disk centre [degrees]
    rsun_obs : (nt,) solar radius [arcsec]
    lon_og   : (nlng,) longitude grid [degrees]
    lat_og   : (nlat,) latitude grid  [degrees]

    Returns
    -------
    r : (nt, nlat, nlng) projected radius in arcsec
    """
    nt   = len(crlt_obs)
    nlat = len(lat_og)
    nlng = len(lon_og)
    r    = np.zeros((nt, nlat, nlng))

    for i, b_angle in tqdm(enumerate(np.nan_to_num(crlt_obs)),
                           total=nt, desc='Building radius array'):
        lng_, lat_ = np.meshgrid(lon_og, lat_og)
        xdisk, ydisk = get_tan_from_lnglat(
            lng_.flatten(), lat_.flatten(),
            rsun_obs[i], b_angle, dP=0)
        r[i] = np.hypot(
            xdisk.reshape(nlat, nlng),
            ydisk.reshape(nlat, nlng))
    return r


def clip_flow_data(
    arr: np.ndarray,
    radius_arr: np.ndarray,
    rsun_obs: np.ndarray,
    radius_ratio: float = CLIP_RADIUS,
    pad: bool = True,
) -> np.ndarray:
    """
    Set pixels beyond radius_ratio * R_sun to NaN.

    Parameters
    ----------
    arr          : (nt, nlat, nlng) flow array
    radius_arr   : (nt, nlat, nlng) projected radius [arcsec]
    rsun_obs     : (nt,) solar radius [arcsec]
    radius_ratio : fraction of R_sun used as the clip boundary
    pad          : if True, pad longitude axis by (36, 35) with NaN

    Returns
    -------
    Clipped (and optionally padded) copy of arr
    """
    out = arr.copy()
    clipradius = radius_ratio * rsun_obs
    out[~(radius_arr < clipradius[:, None, None])] = np.nan
    if pad:
        out = np.pad(out, [(0, 0), (0, 0), (36, 35)],
                     mode='constant', constant_values=np.nan)
    return out


def apodize_flow_data(
    arr: np.ndarray,
    radius_arr: np.ndarray,
    rsun_obs: np.ndarray,
    r_min: float = APOD_R_MIN,
    r_max: float = APOD_R_MAX,
) -> np.ndarray:
    """
    Apply a cosine apodization window between r_min and r_max (in R_sun units).

    Pixels below r_min are kept at full weight; above r_max are zeroed;
    the transition zone gets a half-cosine taper.

    Parameters
    ----------
    arr        : (nt, nlat, nlng) flow array
    radius_arr : (nt, nlat, nlng) projected radius [arcsec]
    rsun_obs   : (nt,) solar radius [arcsec]
    r_min, r_max : inner and outer apodization radii in units of R_sun

    Returns
    -------
    Apodized and padded copy of arr
    """
    out    = arr.copy()
    r_frac = np.clip(radius_arr / rsun_obs[:, None, None], 0, 1.0)
    apod   = np.zeros_like(r_frac)
    apod[r_frac < r_min] = 1.0
    span   = (r_frac >= r_min) & (r_frac < r_max)
    apod[span] = 0.5 * (1 + np.cos(
        np.pi * (r_frac[span] - r_min) / (r_max - r_min)))
    out   *= apod
    out    = np.pad(out, [(0, 0), (0, 0), (36, 35)],
                    mode='constant', constant_values=0)
    return out


def apply_symmetry(
    uphi_all: np.ndarray,
    uthe_all: np.ndarray,
    symmetry: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Symmetrise or anti-symmetrise the flow arrays about the equator.

    Parameters
    ----------
    uphi_all, uthe_all : (nt, nlat, nlng)
    symmetry : 'sym' | 'anti' | 'all'
        'sym'  → u_phi symmetric,  u_theta anti-symmetric
        'anti' → u_phi anti-symmetric, u_theta symmetric
        'all'  → no symmetrisation

    Returns
    -------
    uphi, uthe : symmetrised arrays
    """
    if symmetry == 'sym':
        uphi = (uphi_all + uphi_all[:, ::-1, :]) / 2
        uthe = (uthe_all - uthe_all[:, ::-1, :]) / 2
    elif symmetry == 'anti':
        uphi = (uphi_all - uphi_all[:, ::-1, :]) / 2
        uthe = (uthe_all + uthe_all[:, ::-1, :]) / 2
    elif symmetry == 'all':
        uphi = uphi_all
        uthe = uthe_all
    else:
        raise ValueError(f"symmetry must be 'sym', 'anti', or 'all', got {symmetry!r}")
    return uphi, uthe


def fill_carrington_gaps(crln_obs: np.ndarray) -> np.ndarray:
    """
    Fill NaN gaps in the Carrington longitude series by linear extrapolation
    from the mean step size.

    Parameters
    ----------
    crln_obs : (nt,) raw Carrington longitude array, may contain NaNs

    Returns
    -------
    crln : gap-filled copy
    """
    crln  = crln_obs.copy()
    dcrln = crln[1:] - crln[:-1]
    dphi  = np.nanmean(dcrln[dcrln < 0.])
    # Fallback if no valid negative steps found (e.g. all-NaN differences)
    if np.isnan(dphi):
        dphi = -0.5   # default step matching typical HMI cadence

    nan_pos = np.where(np.isnan(crln))[0].tolist()
    max_iter = len(crln) * 10
    iteration = 0
    while nan_pos and iteration < max_iter:
        for j in nan_pos:
            if j > 0 and not np.isnan(crln[j - 1]):
                crln[j] = crln[j - 1] + dphi
                if crln[j] < 0.:
                    crln[j] += 360.
        nan_pos = np.where(np.isnan(crln))[0].tolist()
        iteration += 1
    return crln


def get_correction_factor(
    arr: np.ndarray,
    nlng_carr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute time and longitude amplitude correction factors for missing data.

    Parameters
    ----------
    arr       : (nt, nlat, nlng) masked flow array (NaN = missing)
    nlng_carr : number of Carrington longitude bins

    Returns
    -------
    cft : time correction factor     (nt, nlat, 1)
    cfl : longitude correction factor (nt, nlat, 1)
    """
    win    = np.isfinite(arr).astype(int)
    nlon_p = np.sum(np.nan_to_num(win), axis=2)[:, :, None]
    nt_p   = np.sum(nlon_p > 0, axis=0)[None, :]
    cft    = win.shape[0] / nt_p
    cfl    = np.nan_to_num(nlng_carr / nlon_p)
    cft[cft > 1e200] = np.inf
    cfl[cfl > 1e200] = np.inf
    return cft, cfl
