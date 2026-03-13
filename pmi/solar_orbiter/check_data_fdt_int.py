# %% imports
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
# %% Load the solar orbiter continuum intensity data
files = sorted(glob.glob('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/solar_orbiter/FDT_data/icnt/*/*.fits.gz'))
file1 = files[1202]


# %% Read the FITS file
with fits.open(file1) as hdul:
    print(file1)
    header = hdul[0].header
    data = hdul[0].data
# %% Print header information
print(header.tostring(sep='\n'))
# %% Plot the centre 50x50 box
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(data, cmap='gray', origin='lower')
ax.set_title('Solar Intensity at time {}'.format(header['date']))
ax.set_xlabel('X Pixel')
ax.set_ylabel('Y Pixel')
plt.colorbar(im, label='Intensity', shrink=0.8)
plt.show()
print(data.shape)

# Print some key header information
# %%
print(header['crota'])
print(header['cdelt1'])
print(header['cdelt2'])
print(header['crpix1'])
print(header['crpix2'])
print(header['crval1'])
print(header['crval2'])
print(header['date'])
print(header['rsun_arc'])
# %%
for key, value in header.items():
    print(f"{key} = {value}")

# %%
print(281*3.75)

# %%
plt.plot(data[int(header['crpix2']), :])
# %% Get x and y in arcsec
nx, ny = data.shape
x = (np.arange(nx) - header['crpix1'] + 1) * header['cdelt1']  # arcsec
y = (np.arange(ny) - header['crpix2'] + 1) * header['cdelt2']  # arcsec
X, Y = np.meshgrid(x, y, indexing='ij')
# %% plot with arcsec
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(data, cmap='gray', origin='lower',
               extent=[x.min(), x.max(), y.min(), y.max()])
ax.set_title('Solar Intensity at time {}'.format(header['date']))
ax.set_xlabel('X (arcsec)')
ax.set_ylabel('Y (arcsec)')
plt.colorbar(im, label='Intensity', shrink=0.8)
plt.show()
# %% Convert arcsec to radians
