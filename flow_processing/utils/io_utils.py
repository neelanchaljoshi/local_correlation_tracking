import os
import numpy as np

def save_flow_array(flow_array, which_flow, which_data):
    """
    Save the processed flow array to a file.
    Parameters:
        flow_array (numpy.ndarray): The processed flow data array to save.
        which_flow (str): The type of flow data (e.g., 'uphi', 'vphi').
        which_data (str): The dataset identifier (e.g., 'hmi.m_720s').
    Returns:
        None
    """
    os.makedirs('processed_data', exist_ok=True)
    data_name = which_data.replace('.', '_') + '_dt_1h'
    file_path = os.path.join('processed_data', f'{which_flow}_{data_name}_processed.npy')
    np.save(file_path, flow_array)
