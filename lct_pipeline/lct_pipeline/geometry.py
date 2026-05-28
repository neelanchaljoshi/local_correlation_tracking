"""
geometry.py
-----------
Optical PSF generation, PMI image simulation from HMI, and
a thin wrapper around the MPS postel remapping routine.

PSFs are computed once from Config at startup and reused throughout
the pipeline — not recomputed inside the main loop.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import j1 as J1

from .config import Config


# ── PSF utilities ─────────────────────────────────────────────────────────

def compute_airy_radius_pixels(
    wavelength_m: float,
    aperture_m: float,
    pixel_scale_arcsec: float,
) -> float:
    """
    Compute the Airy disk radius in image pixels.

    Parameters
    ----------
    wavelength_m        : observation wavelength [m]
    aperture_m          : instrument aperture diameter [m]
    pixel_scale_arcsec  : pixel scale [arcsec/pixel]

    Returns
    -------
    radius in pixels (float)
    """
    theta_rad      = 1.22 * wavelength_m / aperture_m
    pixel_scale_rad = pixel_scale_arcsec / 206265.0
    return theta_rad / pixel_scale_rad


def airy_disk_psf(shape: tuple[int, int], airy_radius_pixels: float) -> np.ndarray:
    """
    Generate a normalised 2D Airy disk PSF centred in the array.

    Parameters
    ----------
    shape               : (H, W) output array shape
    airy_radius_pixels  : Airy disk radius [pixels]

    Returns
    -------
    (H, W) float array normalised to unit sum
    """
    y, x   = np.indices(shape)
    cy, cx = shape[0] // 2, shape[1] // 2
    r      = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r[cy, cx] = 1e-10   # avoid division by zero at centre
    k   = np.pi * r / airy_radius_pixels
    psf = (2 * J1(k) / k) ** 2
    psf /= psf.sum()
    return psf


def build_psfs(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute HMI PSF, PMI PSF, and the relative PSF (PMI convolved with
    flipped HMI) from config parameters.

    Parameters
    ----------
    cfg : Config object

    Returns
    -------
    psf_hmi, psf_pmi, psf_rel : each (psf_size, psf_size) float arrays
    """
    shape = (cfg.psf_size, cfg.psf_size)
    r_hmi = compute_airy_radius_pixels(
        cfg.wavelength_m, cfg.aperture_hmi_m, cfg.pixel_scale_arcsec)
    r_pmi = compute_airy_radius_pixels(
        cfg.wavelength_m, cfg.aperture_pmi_m, cfg.pixel_scale_arcsec)
    psf_hmi = airy_disk_psf(shape, r_hmi)
    psf_pmi = airy_disk_psf(shape, r_pmi)
    psf_rel = fftconvolve(psf_pmi, psf_hmi[::-1, ::-1], mode='same')
    return psf_hmi, psf_pmi, psf_rel


# ── PMI simulation ────────────────────────────────────────────────────────

def simulate_pmi_from_hmi(
    hmi_image: np.ndarray,
    psf_rel: np.ndarray,
) -> np.ndarray:
    """
    Simulate a PMI image from an HMI image by:
      1. Convolving with the relative PSF (PMI blur relative to HMI)
      2. 2×2 box-mean downsampling to PMI pixel scale

    Parameters
    ----------
    hmi_image : (H, W) input HMI image (e.g. 4096×4096)
    psf_rel   : (K, K) relative PSF from build_psfs()

    Returns
    -------
    (H/2, W/2) simulated PMI image
    """
    blurred = fftconvolve(hmi_image, psf_rel, mode='same')
    h, w    = blurred.shape
    h2, w2  = h - h % 2, w - w % 2
    return blurred[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))


# ── Differential rotation rate ────────────────────────────────────────────

def differential_rotation_rate(
    lat_deg: float,
    A: float,
    B: float,
    C: float,
) -> float:
    """
    Compute sidereal differential rotation rate at a given latitude.

    Rate [deg/day] = A + B*sin²(lat) + C*sin⁴(lat)

    Parameters
    ----------
    lat_deg : heliographic latitude [degrees]
    A, B, C : rotation profile coefficients

    Returns
    -------
    rotation rate [deg/day]
    """
    s = np.sin(np.deg2rad(lat_deg))
    return A + B * s**2 + C * s**4


def carrington_longitude_shift(
    lat_deg: float,
    dt_seconds: float,
    cfg: Config,
) -> float:
    """
    Compute the longitude shift due to differential rotation relative to
    the Carrington frame over a time interval dt_seconds.

    Parameters
    ----------
    lat_deg    : heliographic latitude [degrees]
    dt_seconds : time interval [seconds]
    cfg        : Config (provides A, B, C, CRrate)

    Returns
    -------
    longitude shift [degrees]
    """
    rate     = differential_rotation_rate(lat_deg, cfg.A, cfg.B, cfg.C)
    dt_days  = dt_seconds / 86400.0
    return (rate - cfg.CRrate) * dt_days


# ── B0 correction ─────────────────────────────────────────────────────────

def compute_b0_correction(
    t_rec: object,   # datetime
    cfg: Config,
) -> tuple[float, float]:
    """
    Compute the B0 angle and P angle corrections for a given observation time.

    Parameters
    ----------
    t_rec : datetime of the observation
    cfg   : Config (provides dI, t_ref_b0)

    Returns
    -------
    dB : B0 correction [degrees]
    dP : P angle correction [degrees]
    """
    dt_years = (t_rec - cfg.t_ref_b0).total_seconds() / 86400.0 / 365.25
    phase    = 2 * np.pi * dt_years
    dB       = cfg.dI * np.sin(phase)
    dP       = -cfg.dI * np.cos(phase)
    return dB, dP


# ── Postel remap wrapper ──────────────────────────────────────────────────

def remap_patches(
    images: list[np.ndarray],
    crpix1: tuple, crpix2: tuple,
    cdelt1: tuple, cdelt2: tuple,
    rsun_obs: tuple,
    dB: tuple, dP: tuple, dL: tuple,
    cfg: Config,
    clng: float,
    clat: float,
):
    """
    Thin wrapper around zclpy3.remap.from_tan_to_postel.

    Remaps a list of images to a Postel-projected patch centred at
    (clng, clat) using the given astrometric parameters.

    Returns list of remapped patches, same length as images.
    """
    import sys
    sys.path.insert(0, '/data/seismo/zhichao/codes/pypkg')
    from zclpy3.remap import from_tan_to_postel

    return from_tan_to_postel(
        images,
        np.array(crpix1), np.array(crpix2),
        0, 0,
        cdelt1, cdelt2,
        np.array(rsun_obs),
        dB, dP, dL,
        nx_out=cfg.patch_size,
        ny_out=cfg.patch_size,
        lngc_out=clng,
        latc_out=clat,
        pixscale_out=cfg.pixel_size,
        interp_method='bilinear',
        verbose=1,
        nthr=1,
        header=False,
    )
