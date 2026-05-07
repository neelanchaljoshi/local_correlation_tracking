import numpy as np

def get_tan_from_lnglat(lng, lat, rsun, b_angle, dP):
    """Mock: return synthetic disk coordinates."""
    lng = np.asarray(lng)
    lat = np.asarray(lat)
    x = rsun * np.deg2rad(lng) * 0.5
    y = rsun * np.deg2rad(lat) * 0.5
    return x, y
