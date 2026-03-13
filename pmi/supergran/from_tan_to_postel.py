
import numpy as np
import sys
sys.path.insert(0, '/data/seismo/zhichao/codes/pypkg')
from zclpy3.remap import rot_lonlat
# from .wrapper_tan2cyl import _remap_tan_to_cyl

def get_lng_lat_in_postel(nx_out, ny_out, lngc_out, latc_out, pixscale_out):
    # --- output coordinates: postel ---
    iy, ix = np.indices((ny_out, nx_out))
    crpix1_out = 0.5*(1+nx_out)
    crpix2_out = 0.5*(1+ny_out)
    dx = ix+1 - crpix1_out
    dy = iy+1 - crpix2_out
    # clat == hypot(dx, dy)*imscale
    lats = np.pi/2 - np.hypot(dy, dx)*np.deg2rad(pixscale_out)
    lngs = np.arctan2(dy, dx) + np.deg2rad(90) # angle added here corresponds to rot_angle
    lngs, lats = rot_lonlat(lngs, lats,
            0., np.deg2rad(latc_out-90), 0)
    lngs += np.deg2rad(lngc_out) # adding angle here is the same as adding angle to dL
    return np.rad2deg(lngs), np.rad2deg(lats)

# === config: main ===
nx_out = 466
ny_out = 466
lngc_out = 0.0
latc_out = 40.0
pixscale_out = 0.03
clngarr, clatarr = get_lng_lat_in_postel(nx_out, ny_out, lngc_out, latc_out, pixscale_out)
print(clngarr, clatarr)
