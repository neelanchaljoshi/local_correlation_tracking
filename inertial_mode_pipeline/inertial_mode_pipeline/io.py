"""
io.py
-----
All data loading and saving for the inertial modes pipeline.
Nothing here does any computation — pure I/O only.
"""

import pathlib
from datetime import datetime

import numpy as np
import pandas as pd

from .config import DATA_ROOT, PROC_DATA, EF_OUT, EF_FILENAME


# ── Time utilities ────────────────────────────────────────────────────────

def year_fraction(date: datetime) -> float:
    """Convert a datetime to a decimal year (e.g. 2015.5)."""
    start       = datetime(date.year, 1, 1).toordinal()
    year_length = datetime(date.year + 1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length


def parse_time_array(t_raw: np.ndarray) -> np.ndarray:
    """
    Convert a binary TAI time string array to decimal years.

    Parameters
    ----------
    t_raw : ndarray of bytes
        Strings in the format b'%Y.%m.%d_%H:%M:%S_TAI'

    Returns
    -------
    ndarray of float
    """
    dts = [datetime.strptime(str(t, 'utf-8'), '%Y.%m.%d_%H:%M:%S_TAI')
           for t in t_raw]
    return np.array([year_fraction(d) for d in dts])


# ── Flow data loading ─────────────────────────────────────────────────────

def load_flow_data(data_name: str) -> dict:
    """
    Load pre-processed LCT flow maps and observational metadata.

    Parameters
    ----------
    data_name : str
        Dot-replaced data product label, e.g. 'hmi_m_720s_dt_1h'

    Returns
    -------
    dict with keys:
        uphi_all, uthe_all  – raw flow arrays (nt, nlat, nlon)
        t_array             – decimal year array (nt,)
        crln_obs            – Carrington longitude (nt,)
        crlt_obs            – Carrington latitude  (nt,)
        rsun_obs            – solar radius in arcsec (nt,)
    """
    uphi_all = np.load(PROC_DATA / f'uphi_{data_name}_processed.npy')
    uthe_all = np.load(PROC_DATA / f'utheta_{data_name}_processed.npy')
    t_raw    = np.load(DATA_ROOT / 't_rec.npy')
    crln_obs = np.load(DATA_ROOT / 'crln_obs.npy')
    crlt_obs = np.load(DATA_ROOT / 'crlt_obs.npy')
    rsun_obs = np.load(DATA_ROOT / 'rsun_obs.npy')

    t_array  = parse_time_array(t_raw)

    # Interpolate missing metadata linearly
    df = pd.DataFrame({'t': t_array, 'crln': crln_obs,
                       'crlt': crlt_obs, 'rsun': rsun_obs})
    df.interpolate(method='linear', inplace=True)

    return {
        'uphi_all': uphi_all,
        'uthe_all': uthe_all,
        't_array':  t_array,
        'crln_obs': crln_obs,
        'crlt_obs': crlt_obs,
        'rsun_obs': df['rsun'].values,
    }


# ── Eigenfunction saving ──────────────────────────────────────────────────

def save_eigenfunction(result: dict, m: int, cent_freq: float,
                       mode: str, symmetry: str, data_name: str) -> pathlib.Path:
    """
    Save the cleaned eigenfunction and errors to an .npz file.

    Parameters
    ----------
    result : dict
        Must contain keys: ef_uphi, ef_uthe, ef_uphi_sm, ef_uthe_sm,
        uphi_err_real, uphi_err_imag, uthe_err_real, uthe_err_imag,
        lats, final_td
    m, cent_freq, mode, symmetry, data_name : metadata for filename

    Returns
    -------
    pathlib.Path — path to the saved file
    """
    EF_OUT.mkdir(parents=True, exist_ok=True)
    filename = EF_FILENAME.format(
        m=m, freq=cent_freq, mode=mode,
        symmetry=symmetry, data=data_name)
    out_path = EF_OUT / filename
    np.savez(out_path, **result)
    return out_path


def load_eigenfunction(m: int, cent_freq: float, mode: str,
                       symmetry: str, data_name: str) -> dict:
    """
    Load a previously saved eigenfunction .npz file as a dict.
    """
    filename = EF_FILENAME.format(
        m=m, freq=cent_freq, mode=mode,
        symmetry=symmetry, data=data_name)
    return dict(np.load(EF_OUT / filename))
