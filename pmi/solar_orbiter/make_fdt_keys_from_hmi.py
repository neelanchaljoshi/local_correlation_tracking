# %%
import numpy as np
from astropy.table import Table
from astropy.wcs import WCS
from tqdm import tqdm
# %%
from astropy.table import Table
from astropy.io import fits
import numpy as np
from pathlib import Path

def make_fdt_headers_from_hmi(keys_table, fdt_distance_au=0.5):
    """
    Generate synthetic FDT headers for all timestamps in an HMI keys.fits table.
    Handles masked/missing values gracefully.
    """

    def safe_value(val, default=0.0):
        """Return a clean scalar instead of a masked value."""
        if np.ma.is_masked(val):
            return default
        if isinstance(val, (np.ma.MaskedArray,)):
            return val.item() if val.size == 1 else default
        return val

    headers_fdt = []
    image_paths = []
    scale_factor = 1.0 / fdt_distance_au

    for i, row in tqdm(enumerate(keys_table)):
        hmi_keys = {k.upper(): safe_value(row[k]) for k in keys_table.colnames}

        hdr = fits.Header()

        # ---- WCS geometry ----
        hdr['CTYPE1'] = hmi_keys.get('CTYPE1', 'HPLN-TAN')
        hdr['CTYPE2'] = hmi_keys.get('CTYPE2', 'HPLT-TAN')
        hdr['CUNIT1'] = 'arcsec'
        hdr['CUNIT2'] = 'arcsec'

        # pixel scale (keep sign)
        hdr['CDELT1'] = safe_value(hmi_keys.get('CDELT1', -0.5)) / scale_factor
        hdr['CDELT2'] = safe_value(hmi_keys.get('CDELT2', 0.5)) / scale_factor

        hdr['CROTA2'] = safe_value(hmi_keys.get('CROTA2', 0.0))
        hdr['CRVAL1'] = safe_value(hmi_keys.get('CRVAL1', 0.0))
        hdr['CRVAL2'] = safe_value(hmi_keys.get('CRVAL2', 0.0))

        hdr['NAXIS1'] = 2048
        hdr['NAXIS2'] = 2048
        hdr['CRPIX1'] = hdr['NAXIS1'] / 2 + 0.5
        hdr['CRPIX2'] = hdr['NAXIS2'] / 2 + 0.5

        # ---- Observer geometry ----
        hdr['RSUN_OBS'] = safe_value(hmi_keys.get('RSUN_OBS', 975.0)) * scale_factor
        hdr['RSUN_REF'] = safe_value(hmi_keys.get('RSUN_REF', 6.957e8))
        hdr['DSUN_OBS'] = safe_value(hmi_keys.get('DSUN_OBS', 1.496e11)) * fdt_distance_au

        hdr['CRLT_OBS'] = safe_value(hmi_keys.get('CRLT_OBS', 0.0))
        hdr['CRLN_OBS'] = safe_value(hmi_keys.get('CRLN_OBS', 0.0))
        hdr['HGLN_OBS'] = safe_value(hmi_keys.get('HGLN_OBS', hdr['CRLN_OBS']))
        hdr['HGLT_OBS'] = safe_value(hmi_keys.get('HGLT_OBS', hdr['CRLT_OBS']))

        # ---- Timing ----
        hdr['DATE-OBS'] = str(hmi_keys.get('DATE-OBS', '2025-01-01T00:00:00'))
        hdr['T_OBS'] = hdr['DATE-OBS']

        # ---- Instrument info ----
        hdr['INSTRUME'] = 'SO/PHI-FDT (synthetic)'
        hdr['ORIGIN'] = 'Simulated from HMI'
        hdr['COMMENT'] = f"Derived from HMI keys for synthetic FDT at {fdt_distance_au:.2f} AU"

        # ---- Optional observer velocities ----
        for key in ['OBS_VR', 'OBS_VW', 'OBS_VN']:
            if key in hmi_keys:
                hdr[key] = safe_value(hmi_keys[key])

        # ---- Image path ----
        img_path = hmi_keys.get('PATH', None)
        if img_path and not np.ma.is_masked(img_path):
            image_paths.append(str(Path(img_path).resolve()))
        else:
            image_paths.append(None)

        headers_fdt.append(hdr)

    return headers_fdt, image_paths



# %%
# Get keys from HMI file
f = Table.read('/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.m_45s/keys_new_swan/keys-2014.fits')
headers_fdt, image_paths = make_fdt_headers_from_hmi(f, fdt_distance_au=0.5)

print(f"Generated {len(headers_fdt)} synthetic headers.")

# Example: create a WCS for the first timestamp
wcs_fdt = WCS(headers_fdt[0])
print(wcs_fdt)
print("HMI image path:", image_paths[0])
# %%
# Save keys to file
t_out = Table()
t_out['HEADER'] = [hdr.tostring() for hdr in headers_fdt]
t_out['PATH'] = image_paths
t_out.write('data/2014_fdt_keys_from_hmi_0.5au.fits', overwrite=True)
# %%
# check saved file if it has all keys for all times
t_check = Table.read('data/2014_fdt_keys_from_hmi_0.5au.fits')
print(t_check)
print(len(t_check))

# %%
