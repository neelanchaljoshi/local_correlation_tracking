# %%
import numpy as np
from astropy.table import Table
import pandas as pd

# %%
f = Table.read('/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.ic_45s/keys_new_swan/keys-2010.fits')
print(f.colnames)
# %%
print(f['t_rec'].shape)
# %%
s = pd.Series(f['t_rec'].data)
s[s.duplicated()]
# %%
