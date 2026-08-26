import numpy as np
import shtns
from tqdm import tqdm   

def flip_data_array_in_latitude(data_array):
    """
    Flip data array in latitude direction.
    """
    data_array_flipped = np.flip(data_array, axis=0)
    return data_array_flipped

def generate_stream_function(l_array, m_array, nlat, nlng, lmax, mmax, Rsun):
    # theta = np.linspace(0, np.pi, nlat)
    # phi = np.linspace(-np.pi, np.pi, nlng, endpoint=False)
    sh = shtns.sht(lmax, mmax)
    # set the grid
    nlat, nphi = sh.set_grid(nlat, nlng, 5, 1.e-20) # flag is set to 2 to indicate that the grid is equally spaced in theta

    #generate the spectral coefficients according to the l and m values
    zlm = sh.spec_array()
    print(l_array, m_array)
    for l in l_array:
        for m in m_array:
            if m <= l:
                zlm[sh.idx(int(l),int(m))] = 1.0
    
    # zlm += np.random.normal(0, 0.01, zlm.shape) + 1j*np.random.normal(0, 0.01, zlm.shape)
    
    # reconstruct the field
    # synth gives the field given the spectral coefficients
    stream_function = sh.synth(zlm)
    print(zlm)
    return stream_function


def calculate_flow_from_stream_function(stream_function, nlat, nlng, lmax, mmax, Rsun):
    theta = np.linspace(0, np.pi, nlat)
    phi = np.linspace(-np.pi, np.pi, nlng, endpoint=False)

    sh = shtns.sht(lmax, mmax)
    # set the grid
    # nlat, nphi = sh.set_grid(nlat, nlng, shtns.sht_reg_dct | shtns.SHT_PHI_CONTIGUOUS, 1.e-10) # flag is set to 2 to indicate that the grid is equally spaced in theta
    nlat, nphi = sh.set_grid(nlat, nlng, 5, 1e-20)
    print(np.rad2deg(np.arccos(sh.cos_theta)))

    # calculate the spectral coefficients
    # analys gives the spectral coefficients of the input field
    zlm = sh.analys(stream_function)

    # calculate the gradient of the field
    # synth gives the gradient of the field given the spectral coefficients
    # for zlm_x_sin, the dy_dp_x output already has the 1/sin(theta) factor included after differentiation
    dy_dt, dy_dp = sh.synth_grad(zlm)
    uphi = dy_dt/Rsun
    utheta = dy_dp/Rsun
    return uphi, utheta

def calculate_vorticity_and_divergence(uphi_carr, uthe_carr, nlat, nlng, lmax, mmax, Rsun):
    """
    Calculate vorticity and divergence of a vector field on a spherical grid.

    Args:
        uphi_carr (ndarray): Array representing the phi component of the vector field.
        uthe_carr (ndarray): Array representing the theta component of the vector field.
        nlat (int): Number of latitude grid points.
        nlng (int): Number of longitude grid points.
        lmax (int): Maximum spherical harmonic degree.
        mmax (int): Maximum spherical harmonic order.
        Rsun (float): Solar radius.

    Returns:
        tuple: A tuple containing the following arrays:
            - uphi_carr_recon (ndarray): Reconstructed phi component of the vector field.
            - uthe_carr_recon (ndarray): Reconstructed theta component of the vector field.
            - r_vort (ndarray): Vorticity of the vector field.
            - h_divg (ndarray): Divergence of the vector field.
    """
    
    # lat_grid = np.linspace(180,0, nlat)
    # First flip the uphi and uthe arrays in latitude direction
    uphi_carr = flip_data_array_in_latitude(uphi_carr)
    uthe_carr = flip_data_array_in_latitude(uthe_carr)

    lat_grid = np.linspace(0, 180, nlat)
    sin_lat_grid = np.sin(np.deg2rad(lat_grid))
    sin2d = np.outer(sin_lat_grid, np.ones(nlng))

    sh = shtns.sht(lmax, mmax)
    # set the grid
    nlat, nphi = sh.set_grid(nlat, nlng, 5, 1e-20) # flag is set to 2 to indicate that the grid is equally spaced in theta
    # print(np.rad2dessg(np.arccos(sh.cos_theta)))
    # calculate the spectral coefficients
    zlm_phi_sin = sh.analys(uphi_carr*sin2d)
    zlm_the_sin = sh.analys(uthe_carr*sin2d)
    zlm_the = sh.analys(uthe_carr)
    zlm_phi = sh.analys(uphi_carr)

    # Reconstruct the field
    uphi_carr_recon = sh.synth(zlm_phi)
    uthe_carr_recon = sh.synth(zlm_the)

    # calculate the gradient of the field
    dy_dt_the, dy_dp_the = sh.synth_grad(zlm_the)
    dy_dt_phi, dy_dp_phi = sh.synth_grad(zlm_phi)
    dy_dt_phi_sin, dy_dp_phi_sin = sh.synth_grad(zlm_phi_sin)
    dy_dt_the_sin, dy_dp_the_sin = sh.synth_grad(zlm_the_sin)

    r_vort = (dy_dt_phi_sin/sin2d - dy_dp_the)/Rsun
    h_divg = (dy_dt_the_sin/sin2d + dy_dp_phi)/Rsun

    # changed to match the sign convention in the Bastian code
    # r_vort = (-dy_dt_phi_sin/sin2d - dy_dp_the)/Rsun
    # h_divg = (-dy_dt_the_sin/sin2d + dy_dp_phi)/Rsun

    # Now flip the reconstructed arrays back in latitude direction
    uphi_carr_recon = flip_data_array_in_latitude(uphi_carr_recon)
    uthe_carr_recon = flip_data_array_in_latitude(uthe_carr_recon)
    r_vort = flip_data_array_in_latitude(r_vort)
    h_divg = flip_data_array_in_latitude(h_divg)

    return uphi_carr_recon, uthe_carr_recon, r_vort, h_divg

# def project_onto_zlm_sectoral(array, lmax, mmax, nlat, nlng):
#     """
#     Project the input array onto the sectoral zonal and meridional modes (zlm) up to a given lmax and mmax.
    
#     Parameters:
#         array (ndarray): Input array to be projected.
#         lmax (int): Maximum degree of the sectoral modes.
#         mmax (int): Maximum order of the sectoral modes.
#         nlat (int): Number of latitudes in the grid.
#         nlng (int): Number of longitudes in the grid.
    
#     Returns:
#         ndarray: Array projected onto the sectoral zonal and meridional modes.
#     """
#     l_array = np.arange(0, lmax+1)
#     sh = shtns.sht(lmax, mmax)
#     # set the grid
#     nlat, nphi = sh.set_grid(nlat, nlng, 5, 1e-20) # flag is set to 2 to indicate that the grid is equally spaced in theta

#     # calculate the spectral coefficients
#     # analys gives the spectral coefficients of the input field
#     zlm = sh.analys(np.clip(np.nan_to_num(array), a_max = 5, a_min = -5))
#     # print(zlm)
#     for l in l_array:
#         m_array = np.arange(0, l+1)
#         for m in m_array:
#             if m != l:
#                 zlm[sh.idx(int(l),int(m))] = 0.0
#     # print(zlm)
#     array_proj = sh.synth(zlm)
#     return array_proj

# def project_onto_zlm_hanson(array, lmax, mmax, nlat, nlng):
#     """
#     Project the input array onto the sectoral zonal and meridional modes (zlm) up to a given lmax and mmax.
    
#     Parameters:
#         array (ndarray): Input array to be projected.
#         lmax (int): Maximum degree of the sectoral modes.
#         mmax (int): Maximum order of the sectoral modes.
#         nlat (int): Number of latitudes in the grid.
#         nlng (int): Number of longitudes in the grid.
    
#     Returns:
#         ndarray: Array projected onto the sectoral zonal and meridional modes.
#     """
#     l_array = np.arange(0, lmax+1)
#     sh = shtns.sht(lmax, mmax)
#     # set the grid
#     nlat, nphi = sh.set_grid(nlat, nlng, 5, 1e-20) # flag is set to 2 to indicate that the grid is equally spaced in theta

#     # calculate the spectral coefficients
#     # analys gives the spectral coefficients of the input field
#     zlm = sh.analys(np.clip(np.nan_to_num(array), a_max = 5, a_min = -5))
#     # print(zlm)
#     for l in l_array:
#         m_array = np.arange(0, l+1)
#         for m in m_array:
#             if l != m+1:
#                 zlm[sh.idx(int(l),int(m))] = 0.0
#     # print(zlm)
#     array_proj = sh.synth(zlm)
#     return array_proj


# def project_onto_zlm_high_m(array, lmax, mmax, nlat, nlng):
#     """
#     Project the input array onto the sectoral zonal and meridional modes (zlm) up to a given lmax and mmax.
    
#     Parameters:
#         array (ndarray): Input array to be projected.
#         lmax (int): Maximum degree of the sectoral modes.
#         mmax (int): Maximum order of the sectoral modes.
#         nlat (int): Number of latitudes in the grid.
#         nlng (int): Number of longitudes in the grid.
    
#     Returns:
#         ndarray: Array projected onto the sectoral zonal and meridional modes.
#     """
#     l_array = np.arange(0, lmax+1)
#     sh = shtns.sht(lmax, mmax)
#     # set the grid
#     nlat, nphi = sh.set_grid(nlat, nlng, 5, 1e-20) # flag is set to 2 to indicate that the grid is equally spaced in theta

#     # calculate the spectral coefficients
#     # analys gives the spectral coefficients of the input field
#     zlm = sh.analys(np.clip(np.nan_to_num(array), a_max = 5, a_min = -5))
#     # print(zlm)
#     for l in l_array:
#         m_array = np.arange(0, l+1)
#         for m in m_array:
#             if l != m+1:
#                 zlm[sh.idx(int(l),int(m))] = 0.0
#     # print(zlm)
#     array_proj = sh.synth(zlm)
#     return array_proj



# def project_onto_zlm_tesseral(array, lmax, mmax, nlat, nlng):
#     """
#     Project the input array onto the sectoral zonal and meridional modes (zlm) up to a given lmax and mmax.
    
#     Parameters:
#         array (ndarray): Input array to be projected.
#         lmax (int): Maximum degree of the sectoral modes.
#         mmax (int): Maximum order of the sectoral modes.
#         nlat (int): Number of latitudes in the grid.
#         nlng (int): Number of longitudes in the grid.
    
#     Returns:
#         ndarray: Array projected onto the sectoral zonal and meridional modes.
#     """
#     l_array = np.arange(0, lmax+1)
#     sh = shtns.sht(lmax, mmax)
#     # set the grid
#     nlat, nphi = sh.set_grid(nlat, nlng, 5, 1e-20) # flag is set to 2 to indicate that the grid is equally spaced in theta

#     # calculate the spectral coefficients
#     # analys gives the spectral coefficients of the input field
#     zlm = sh.analys(np.clip(np.nan_to_num(array), a_max = 5, a_min = -5))
#     # print(zlm)
#     for l in l_array:
#         m_array = np.arange(0, l+1)
#         for m in m_array:
#             if m == l:
#                 zlm[sh.idx(int(l),int(m))] = 0.0
#     # print(zlm)
#     array_proj = sh.synth(zlm)
#     return array_proj


# def symmetrize_data_using_zlm(array, lmax, mmax, nlat, nlng):
#     """
#     Project the input array onto the sectoral zonal and meridional modes (zlm) up to a given lmax and mmax.
    
#     Parameters:
#         array (ndarray): Input array to be projected.
#         lmax (int): Maximum degree of the sectoral modes.
#         mmax (int): Maximum order of the sectoral modes.
#         nlat (int): Number of latitudes in the grid.
#         nlng (int): Number of longitudes in the grid.
    
#     Returns:
#         ndarray: Array projected onto the sectoral zonal and meridional modes.
#     """
#     l_array = np.arange(0, lmax+1)
#     sh = shtns.sht(lmax, mmax)
#     # set the grid
#     nlat, nphi = sh.set_grid(nlat, nlng, 5, 1e-25) # flag is set to 2 to indicate that the grid is equally spaced in theta

#     # calculate the spectral coefficients
#     # analys gives the spectral coefficients of the input field
#     zlm = sh.analys(np.clip(np.nan_to_num(array), a_max = 5, a_min = -5))
#     # print(zlm)
#     zlm_l_sym = np.zeros_like(zlm)
#     zlm_l_anti = np.zeros_like(zlm)

#     for l in l_array:
#         m_array = np.arange(0, l+1)
#         for m in m_array:
#             if l%2==0:
#                 zlm_l_sym = zlm[sh.idx(int(l),int(m))]
#             else:
#                 zlm_l_anti = zlm[sh.idx(int(l),int(m))]
#     # # print(zlm)
#     array_proj_sym = sh.synth(zlm_l_sym)
#     array_proj_anti = sh.synth(zlm_l_anti)
#     return array_proj_sym, array_proj_anti

def get_zlms_for_array(array, lmax, mmax, nlat, nlng, isvort = False):
    """
    Project the input array onto the sectoral zonal and meridional modes (zlm) up to a given lmax and mmax.
    
    Parameters:
        array (ndarray): Input array to be projected.
        lmax (int): Maximum degree of the sectoral modes.
        mmax (int): Maximum order of the sectoral modes.
        nlat (int): Number of latitudes in the grid.
        nlng (int): Number of longitudes in the grid.
    
    Returns:
        ndarray: Array projected onto the sectoral zonal and meridional modes.
    """
    # First flip the array in latitude direction
    # array = flip_data_array_in_latitude(array)

    l_array = np.arange(0, lmax+1)
    sh = shtns.sht(lmax, mmax)
    # set the grid
    nlat, nphi = sh.set_grid(nlat, nlng, 5, 1e-20) # flag is set to 2 to indicate that the grid is equally spaced in theta

    # calculate the spectral coefficients
    # analys gives the spectral coefficients of the input field
    zlm_array = []
    for k in tqdm(range(array.shape[0])):
        if isvort:
            zlm_array.append(sh.analys(np.clip(np.nan_to_num(array[k]), a_max = 50, a_min = -50)))
        else:
            zlm_array.append(sh.analys(np.clip(np.nan_to_num(array[k]), a_max = 5000, a_min = -5000)))
        # print(zlm)
    zlm_array = np.asarray(zlm_array)
    # print(zlm_array[:10, 0])
    return zlm_array

# def project_onto_zlm_lm2(array, lmax, mmax, nlat, nlng):
#     """
#     Project the input array onto the sectoral zonal and meridional modes (zlm) up to a given lmax and mmax.
    
#     Parameters:
#         array (ndarray): Input array to be projected.
#         lmax (int): Maximum degree of the sectoral modes.
#         mmax (int): Maximum order of the sectoral modes.
#         nlat (int): Number of latitudes in the grid.
#         nlng (int): Number of longitudes in the grid.
    
#     Returns:
#         ndarray: Array projected onto the sectoral zonal and meridional modes.
#     """
#     l_array = np.arange(0, lmax+1)
#     sh = shtns.sht(lmax, mmax)
#     # set the grid
#     nlat, nphi = sh.set_grid(nlat, nlng, 5, 1e-20) # flag is set to 2 to indicate that the grid is equally spaced in theta

#     # calculate the spectral coefficients
#     # analys gives the spectral coefficients of the input field
#     zlm = sh.analys(np.clip(np.nan_to_num(array), a_max = 5, a_min = -5))
#     # print(zlm)
#     for l in l_array:
#         m_array = np.arange(0, l+1)
#         for m in m_array:
#             if l != m-2:
#                 zlm[sh.idx(int(l),int(m))] = 0.0
#     # print(zlm)
#     array_proj = sh.synth(zlm)
#     return array_proj