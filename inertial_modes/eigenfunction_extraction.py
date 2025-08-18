import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime
from scipy.fftpack import rfft, fftfreq, fftshift
from scipy.linalg import lstsq
from astropy.wcs import WCS
import sys
from tqdm import tqdm
sys.path.insert(0, '/data/seismo/zhichao/codes/pypkg')
sys.path.append('/data/seismo/joshin/pipeline-test/python_modules/')
import pandas as pd

from zclpy3.remap import from_cyl_to_tan, get_tan_from_lnglat
from matplotlib.patches import Circle
import matplotlib.colors as colors


plt.style.use('default')
from numpy import linalg


# Arguments
m = int(sys.argv[1])
cent_freq = float(sys.argv[2])
mode = sys.argv[3]
data = sys.argv[4]
symmetry = sys.argv[5]

if symmetry == 'sym':
    sym_uphi = 'sym'
    sym_uthe = 'anti'
elif symmetry == 'anti':
    sym_uphi = 'anti'
    sym_uthe = 'sym'
elif symmetry == 'all':
    sym_uphi = 'all'
    sym_uthe = 'all'

#Configurations
# fig_path = '/data/seismo/joshin/pipeline-test/paper_lct/figures_new/{}/m{}/{}/{}/'.format(data, m, mode, symmetry)
fig_path = '/data/seismo/joshin/pipeline-test/paper_lct/data_plots_paper/from_script/'
pathlib.Path(fig_path).mkdir(parents=True, exist_ok=True)

# m = 2
# mode = 'highlat'
# data = 'hmi.m_1h'
reject_type = 'clip' # 'clip' or 'noclip'
span_lower = 2013
span_upper = 2018
# symmetry = 'all' # symmetry in uphi


# Replace . by _
data_name = data.replace('.', '_')

# Load Data
# uphi_all = np.load('/scratch/seismo/joshin/pipeline-test/IterativeLCT/{}/uphi_99_5_perc_inclusion.npy'.format(data))
# uthe_all = np.load('/scratch/seismo/joshin/pipeline-test/IterativeLCT/{}/uthe_99_5_perc_inclusion_test_TAC.npy'.format(data))
uphi_all = np.load('/data/seismo/joshin/pipeline-test/paper_lct/processed_data/uphi_{}_processed.npy'.format(data_name))
uthe_all = np.load('/data/seismo/joshin/pipeline-test/paper_lct/processed_data/utheta_{}_processed.npy'.format(data_name))
t_array = np.load('/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.m_1h/t_all.npy')
crln_obs = np.load('/scratch/seismo/joshin/pipeline-test/key_arrays_extracted/crln_obs_10_22.npy')
crlt_obs = np.load('/scratch/seismo/joshin/pipeline-test/key_arrays_extracted/crlt_obs_10_22.npy')
rsun_obs = np.load('/scratch/seismo/joshin/pipeline-test/key_arrays_extracted/rsun_obs_10_22.npy')

#Symmetrize
uphi_sym = (uphi_all + uphi_all[:, ::-1, :])/2
uphi_anti = (uphi_all - uphi_all[:, ::-1, :])/2
uthe_sym = (uthe_all + uthe_all[:, ::-1, :])/2
uthe_anti = (uthe_all - uthe_all[:, ::-1, :])/2

# Interpolate for missing values
df = pd.DataFrame({'t': t_array, 'crln': crln_obs, 'crlt': crlt_obs, 'rsun': rsun_obs})
df.interpolate(method='linear', inplace=True)
rsun_obs = df['rsun'].values

# Choose array based on symmetry
if symmetry == 'sym':
    uphi = uphi_sym
    uthe = uthe_anti
elif symmetry == 'anti':
    uphi = uphi_anti
    uthe = uthe_sym
elif symmetry == 'all':
    uphi = uphi_all
    uthe = uthe_all

# filling in data gaps in crln data series
crln = crln_obs
dcrln = crln[1:] - crln[:-1]
dphi_obs = np.nanmean(dcrln[dcrln<0.])
nan_pos  = np.where(np.isnan(crln))[0]
while len(nan_pos)>0:
    for j in nan_pos:
        crln[j] = crln[j-1] + dphi_obs
        crln[j] = crln[j]+360 if crln[j]<0. else crln[j]
    nan_pos = np.where(np.isnan(crln))[0].tolist()

# Define variables
nt = uthe_all.shape[0]
nlat = uthe_all.shape[1]
nlng_stony = uthe_all.shape[2]
nlng_carr = 2*(nlng_stony-1)
longs_all = np.arange(-180, 180, 2.5)
lon_og = np.linspace(-90, 90, 73)
lat_og = np.linspace(-90, 90, 73)
dt = 6*3600 # in seconds

# Get radius array
nt = uthe_all.shape[0]
nlat = uthe_all.shape[1]
nlng = uthe_all.shape[2]
r = np.zeros((nt, nlat, nlng))
# rsun_obs = 960.
for i, b_angle in tqdm(enumerate(np.nan_to_num(crlt_obs))):
    dP = 0
    lng_, lat_ = np.meshgrid(lon_og, lat_og)
    xdisk, ydisk = get_tan_from_lnglat(lng_.flatten(), lat_.flatten(), rsun_obs[i], b_angle, dP)
    r[i] = np.hypot(xdisk.reshape((nlat, nlng)), ydisk.reshape((nlat, nlng)))


def get_correction_factor(arr, nlng_carr):
    # print('Calculating correction factors...')
    # Defining the valid points to calculate the amplitude correction factor
    win = arr.copy()
    win = (np.isfinite(win)).astype(int)
    nlon_p = np.sum(np.nan_to_num(win), axis = 2)[:, :, None]
    nt_p = np.sum(nlon_p>0, axis = 0)[None, :]
    cft = win.shape[0]/nt_p
    cfl = np.nan_to_num(nlng_carr/nlon_p)
    cft[cft>1e200]=np.inf
    cfl[cfl>1e200]=np.inf
    return cft, cfl

def clip_flow_data(arr, radius_arr, radius_ratio, rsun_obs, pad = True):
    """
    Clips the array based on the radius array and the clipradius.

    Parameters:
    arr (ndarray): The input array to be clipped.
    radius_arr (ndarray): The array containing radius values.
    clipradius (float): The radius value used for clipping.

    Returns:
    ndarray: The clipped array.
    """
    print('Clipping...')
    clipradius = radius_ratio * rsun_obs
    arr[~(radius_arr < clipradius[:, None, None])] = np.nan
    if pad:
        arr = np.pad(arr, [(0,0),(0,0),(36, 35)], mode = 'constant', constant_values = np.nan)
    return arr

def apodize_flow_data(arr, radius_arr, r_min, r_max, r_sun):
    """
    Apodizes the array based on the radius array and the apodization parameters.

    Parameters:
    arr (ndarray): The input array to be apodized.
    radius_arr (ndarray): The array containing radius values.
    r_min (float): The inner radius value used for apodization.
    r_max (float): The outer radius value used for apodization.

    Returns:
    ndarray: The apodized array.
    """
    # print('Apodizing...')
    r_frac = radius_arr / r_sun[:, None, None]
    r_frac = np.clip(r_frac, 0, 1.0)
    apod = np.zeros_like(r_frac)
    apod[np.isnan(r_frac)] = 0
    apod[r_frac<r_min] = 1
    apod[r_frac>r_max] = 0
    span = (r_frac >= r_min) & (r_frac < r_max)
    apod[span] = 0.5 *(1+ np.cos(np.pi*(r_frac[span] - r_min)/(r_max - r_min)))
    arr *= apod
    arr = np.pad(arr, [(0,0),(0,0),(36, 35)], mode = 'constant', constant_values = 0)
    return arr

def tukeywin(window_length, alpha=0.0):
    # Special cases
    if alpha <= 0:
        return np.ones(window_length) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)

    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)

    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))

    # second condition already taken care of

    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2)))
    # print('HERE')
    return w


def transform_fourier(arr, crln, cft, cfl, span, dt = 6.*3600, to_Nhz = True, save_carr = True):
    print('Transforming to Fourier space...')
    uphi_fft_m = np.fft.rfft(np.nan_to_num(arr[span]), axis=2)*np.nan_to_num(cfl[span])
    M_arr = np.arange(uphi_fft_m.shape[2])
    carr_conv  = np.exp(-1j*np.deg2rad(crln[span])[:,None] * M_arr[None,:])[:,None,:]
    uphi_m_carr = uphi_fft_m * carr_conv
    uphi_f = np.fft.fft(uphi_m_carr, axis=0)*np.sqrt(np.nan_to_num(cft))
    uphi_r = np.fft.irfft(np.fft.ifft(uphi_f, axis = 0), axis = 2)
    print(uphi_f.shape)
    if to_Nhz:
        freq = -np.fft.fftfreq(uphi_f.shape[0], dt)*1e9
    freq_ffts = np.fft.fftshift(freq)
    uphi_ft  = np.fft.fftshift(uphi_f, axes=0)
    if save_carr:
        return uphi_ft, freq_ffts, uphi_r
    return uphi_ft, freq_ffts

def filter_in_freq(uphi_ft_m, uthe_ft_m, freq_ffts, cent_freq, df =  20):
    lf = cent_freq - df
    rf = cent_freq + df
    # print('HERE 2')
    s = (freq_ffts>lf) & (freq_ffts < rf)
    wl = len(freq_ffts[s])
    window = np.zeros_like(freq_ffts)
    # print('HERE 3')
    w = tukeywin(wl, 0.1)
    # print(len(w))
    # plt.plot(w)
    # print('HERE 4')
    window[s] = w
    uphi_filt = np.zeros_like(uphi_ft_m)
    uthe_filt = np.zeros_like(uthe_ft_m)
    uphi_filt = uphi_ft_m * window[:, np.newaxis]
    uthe_filt = uthe_ft_m * window[:, np.newaxis]
    # print('HERE 5')
    return uphi_filt, uthe_filt


def extract_eigenfunction_lats(uphi_ft, uthe_ft, m, cent_freq, freq_ffts, lat_for_scaling = 0, nlng = 144, df = 30):
    uphi_f_m = uphi_ft[:, :, m]
    uthe_f_m = uthe_ft[:, :, m]
    print(uphi_f_m.shape)
    # print('HERE 1')
    uphi_f_m_filt, uthe_f_m_filt = filter_in_freq(uphi_f_m, uthe_f_m, freq_ffts, cent_freq, df = df)
    nt = uphi_ft.shape[0]
    nlat = uphi_ft.shape[1]
    uphi_filt = np.zeros((nt, nlat, nlng), dtype = np.complex128)
    uthe_filt = np.zeros((nt, nlat, nlng), dtype = np.complex128)
    uphi_filt[:, :, m] = uphi_f_m_filt
    uthe_filt[:, :, m] = uthe_f_m_filt

    lats = np.linspace(-90, 90, nlat)
    # print(uphi_filt.shape)
    # print(uphi_f_m_filt.shape)
    # print('HERE 6')

    uphi_ifftshift = np.fft.ifftshift(np.nan_to_num(uphi_filt), axes = 0)
    uphi_t_m = np.fft.ifft(uphi_ifftshift, axis = 0)

    uthe_ifftshift = np.fft.ifftshift(np.nan_to_num(uthe_filt), axes = 0)
    uthe_t_m = np.fft.ifft(uthe_ifftshift, axis = 0)
    lat_mask_for_svd = np.where(abs(lats)<=75)[0]
    arr_svd = np.concatenate((uphi_t_m[:, lat_mask_for_svd, m], uthe_t_m[:, lat_mask_for_svd, m]), axis = 1)

    U, s, Vh = linalg.svd(arr_svd, full_matrices=False)


    time_dependence = U[:, 0]
    m_factor = 2/nlng
    m_factor/= np.sqrt(np.mean(abs(time_dependence)**2))
    eigenfunction_uphi = uphi_t_m[:, :, m] * np.conj(time_dependence[:,None]) * m_factor
    eigenfunction_uthe = uthe_t_m[:, :, m] * np.conj(time_dependence[:,None]) * m_factor
    # np.save('eigenfunction_time_uphi_m1_all.npy', eigenfunction_uphi)
    # np.save('eigenfunction_time_uthe_m1_all.npy', eigenfunction_uthe)
    ef_uphi = np.mean(eigenfunction_uphi, axis = 0)
    ef_uthe = np.mean(eigenfunction_uthe, axis = 0)
    # ef_uphi, ef_uthe = rotate_eigenfunction_1d(ef_uphi, ef_uthe, 0)

    index = np.where(lats == lat_for_scaling)[0][0]
    final_td = abs(s[0]*Vh[0, index]*abs(time_dependence)*2/nlng)
    return ef_uphi, ef_uthe, final_td



def rotate_eigenfunction_1d(ef_uphi, ef_uthe, lat_for_rotation):
    lats = np.linspace(-90, 90, 73)
    index = np.where(lats == lat_for_rotation)[0][0]
    angle = np.angle(ef_uthe[index])
    ef_uphi = ef_uphi*np.exp(-1j*angle)
    ef_uthe = ef_uthe*np.exp(-1j*angle)
    return ef_uphi, ef_uthe




def run():
    span = (t_array >= span_lower) & (t_array < span_upper)
    rcut = 0.99
    if reject_type == 'clip':
        uphi_clip = clip_flow_data(uphi, r, rcut, rsun_obs, pad=True)
        uthe_clip = clip_flow_data(uthe, r, rcut, rsun_obs, pad=True)
    else:
        uphi_clip = apodize_flow_data(uphi, r, 0.96, 0.99, rsun_obs)
        uthe_clip = apodize_flow_data(uthe, r, 0.96, 0.99, rsun_obs)


    cft1, cfl1 = get_correction_factor(uphi_clip, nlng_carr)
    cft2, cfl2 = get_correction_factor(uthe_clip, nlng_carr)


    dt = 6.*3600


    uphi_ft, freq_ffts, uphi_r = transform_fourier(uphi_clip, crln, cft1, cfl1, span, dt = dt, to_Nhz = True)
    uthe_ft, freq_ffts, uthe_r = transform_fourier(uthe_clip, crln, cft2, cfl2, span, dt = dt, to_Nhz = True)
    np.save(fig_path + 'uphi_ft_m{}_{}_{}_{}_{}_{}_{}.npy'.format(m, mode, cent_freq, span_lower, span_upper-1, sym_uphi, data_name), uphi_ft)
    np.save(fig_path + 'uthe_ft_m{}_{}_{}_{}_{}_{}_{}.npy'.format(m, mode, cent_freq, span_lower, span_upper-1, sym_uthe, data_name), uthe_ft)



    ef_uphi, ef_uthe, final_td = extract_eigenfunction_lats(uphi_ft, uthe_ft, m, cent_freq, freq_ffts, lat_for_scaling = 0, nlng = nlng_carr, df = 10)
    np.save(fig_path + 'eigenfunction_uphi_m{}_{}_{}_{}_{}_{}_{}.npy'.format(m, mode, cent_freq, span_lower, span_upper-1, sym_uphi, data_name), ef_uphi)
    np.save(fig_path + 'eigenfunction_uthe_m{}_{}_{}_{}_{}_{}_{}.npy'.format(m, mode, cent_freq, span_lower, span_upper-1, sym_uthe, data_name), ef_uthe)
    np.save(fig_path + 'final_td_m{}_{}_{}_{}_{}.npy'.format(m, mode, span_lower, span_upper-1, data_name), final_td)
    return None







if __name__ == '__main__':
    run()
    plt.close('all')
