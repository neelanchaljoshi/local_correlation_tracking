# %% imports
import numpy as np
import matplotlib.pyplot as plt

# %% load data
m1_hl_gran = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/eigenfunctions/eigenfunction_m8_-115.0_rossby_anti_hmi_ic_45s_granule.npz')
m1_hl_mag =  np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/eigenfunctions/eigenfunction_m8_-115.0_rossby_anti_hmi_m_720s_dt_1h.npz')

ef_uphi_m1_hl_mag = m1_hl_mag['ef_uphi']
ef_uthe_m1_hl_mag = m1_hl_mag['ef_uthe']

ef_uphi_m1_hl_gran = m1_hl_gran['ef_uphi']
ef_uthe_m1_hl_gran = m1_hl_gran['ef_uthe']

# %% plot
plt.figure(figsize=(12, 8))

# Subplot for granule
plt.subplot(2, 2, 1)
plt.plot(ef_uphi_m1_hl_gran)
plt.title('ef_uphi - Granule')
plt.xlabel('Index')
plt.ylabel('ef_uphi')

plt.subplot(2, 2, 2)
plt.plot(ef_uthe_m1_hl_gran)
plt.title('ef_uthe - Granule')
plt.xlabel('Index')
plt.ylabel('ef_uthe')

# Subplot for magnitude
plt.subplot(2, 2, 3)
plt.plot(ef_uphi_m1_hl_mag)
plt.title('ef_uphi - Magnitude')
plt.xlabel('Index')
plt.ylabel('ef_uphi')

plt.subplot(2, 2, 4)
plt.plot(ef_uthe_m1_hl_mag)
plt.title('ef_uthe - Magnitude')
plt.xlabel('Index')
plt.ylabel('ef_uthe')

plt.tight_layout()
plt.show()

# %%
