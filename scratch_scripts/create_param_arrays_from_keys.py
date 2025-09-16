# %%
import numpy as np
from astropy.table import Table

# %%
def create_arrays_from_keys_for_all_years(key_name, data_series, cadence, dtype=np.float64):
    key_name_array = np.array(np.empty((0,)), dtype=dtype)
    for year in range(2010, 2025):
        f = Table.read(f'/scratch/seismo/joshin/pipeline-test/IterativeLCT/{data_series}/keys_new_swan/keys-{year}.fits')
        njump = int(6*3600/cadence)  # Number of jumps in the cadence
        print(njump)
        key_array = np.array(f[key_name].data[::njump], dtype = dtype)  # Select every nth element based on cadence
        key_name_array = np.append(key_name_array, key_array)
        key_name_array = key_name_array.astype(dtype)
    return key_name_array

# %%
if __name__ == "__main__":
    # Example usage
    key_name = 't_rec'  # Replace with actual key name
    data_series = 'hmi.ic_45s'  # Replace with actual data series
    cadence = 45  # Replace with actual cadence in seconds
    keys_array = create_arrays_from_keys_for_all_years(key_name, data_series, cadence = cadence, dtype='S32')
    print(keys_array)  # Output the array for verification
# %%
print(" The shape of the keys array is: ", keys_array.shape)
print(" The keys array is: ", keys_array)
print(" The key array data type is: ", keys_array.dtype)
# %%
# Save the keys array to a npy file
np.save(f'/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/{key_name}.npy', keys_array)
# %%
