import os
import matplotlib.pyplot as plt

def make_plot(flow_array, t_array, mad, n, which_plot):
    """
    Create and save plots for flow data analysis.
    Parameters:
        flow_array (numpy.ndarray): The flow data array.
        t_array (numpy.ndarray): The time array corresponding to the flow data.
        mad (numpy.ndarray): The median absolute deviation array, if available.
        n (int): An index for naming the saved plot files.
        which_plot (str): The type of plot to create. Options are:
            - 'flow_histogram': Histogram of flow data.
            - 'time_series': Time series plot of flow data.
            - 'flow_data_plot': Image of flow data for a specific frame.
            - 'mad': Image of the median absolute deviation (MAD) data.
    Returns:
        None
    """
    os.makedirs('figures_processing', exist_ok=True)
    fpath = lambda name: f'figures_processing/{name}_{n}.png'

    if which_plot == 'flow_histogram':
        data = flow_array[~np.isnan(flow_array)]
        plt.hist(data, bins=500, histtype='step', range=(-2000, 2000))
        plt.xlabel('Flow (m/s)')
        plt.ylabel('Frequency')
        plt.title('Flow Histogram')
        plt.savefig(fpath('flow_histogram'))
    elif which_plot == 'time_series':
        plt.plot(t_array, flow_array[:, 35, 35])
        plt.xlabel('Time (decimal years)')
        plt.ylabel('Flow (m/s)')
        plt.title('Time Series')
        plt.savefig(fpath('time_series'))
    elif which_plot == 'flow_data_plot':
        plt.imshow(flow_array[1576], origin='lower')
        plt.title('Flow Map (Sample Frame)')
        plt.colorbar()
        plt.savefig(fpath('flow_data_plot'))
    elif which_plot == 'mad' and mad is not None:
        plt.imshow(mad, origin='lower')
        plt.title('MAD Map')
        plt.colorbar()
        plt.savefig(fpath('mad'))
    plt.close()
