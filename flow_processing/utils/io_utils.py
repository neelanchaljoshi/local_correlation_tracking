import os
import numpy as np

# Output filename suffix per known legacy dataset -- keyed by the same
# which_data string as flow_data.LEGACY_GLOB_PATTERNS, so the input
# selection and output naming can't silently drift out of sync the way
# the old commented-out-line toggle allowed.
LEGACY_OUTPUT_SUFFIX = {
    'hmi.ic_45s': '_granule',
    'hmi.m_720s': '_dt_1h',
}

PROCESSED_DATA_DIR = '/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data'


def save_flow_array(flow_array, which_flow, which_data, suffix=None):
    """
    Save the processed flow array to a file.
    Parameters:
        flow_array (numpy.ndarray): The processed flow data array to save.
        which_flow (str): The type of flow data (e.g., 'uphi', 'utheta').
        which_data (str): The dataset identifier (e.g., 'hmi.m_720s').
        suffix (str, optional): Output filename suffix distinguishing
            this LCT run (e.g. '_granule', '_dt_1h'). Defaults to
            LEGACY_OUTPUT_SUFFIX[which_data] for known legacy datasets;
            raises ValueError otherwise, since there's no safe default
            to guess for a new lct_pipeline run.
    Returns:
        None
    """
    if suffix is None:
        try:
            suffix = LEGACY_OUTPUT_SUFFIX[which_data]
        except KeyError:
            raise ValueError(
                f'No default output suffix known for which_data={which_data!r}. '
                f'Pass suffix= explicitly — known legacy datasets are '
                f'{sorted(LEGACY_OUTPUT_SUFFIX)}.')

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    data_name = which_data.replace('.', '_') + suffix
    file_path = os.path.join(PROCESSED_DATA_DIR, f'{which_flow}_{data_name}_processed.npy')
    np.save(file_path, flow_array)
