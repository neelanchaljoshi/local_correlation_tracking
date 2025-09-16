import numpy as np
import h5py
import os
from datetime import datetime
from astropy.time import Time
from tqdm import tqdm
from scipy import stats
from scipy.optimize import curve_fit
from utils.fitting import sin_fit
from utils.plotting import make_plot
from utils.io_utils import save_flow_array


class FlowData:
    """
    Class to handle flow data processing for local correlation tracking.
    Attributes:
        crln_obs (numpy.ndarray): Observed central meridian longitude.
        crlt_obs (numpy.ndarray): Observed central meridian latitude.
        rsun_obs (numpy.ndarray): Observed solar radius.
    Methods:
        __init__(which_flow, which_data): Initializes the FlowData object with specified flow and data type.
        getdata(): Loads flow data from HDF5 files and prepares the time and spatial arrays.
        remove_median(): Removes the median from the flow data.
        calculate_mad(): Computes the median absolute deviation of the flow data.
        outlier_rejection(threshold): Rejects outliers based on a specified threshold.
        remove_yearly_variation(): Removes yearly variations from the flow data using sinusoidal fitting.
        save(): Saves the processed flow array to a file.
        plot(n, which_plot): Generates and saves plots for the flow data.
    """
    crln_obs = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/crln_obs.npy')
    crlt_obs = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/crlt_obs.npy')
    rsun_obs = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/rsun_obs.npy')

    def __init__(self, which_flow, which_data):
        """
        Initializes the FlowData object with specified flow and data type.
        Parameters:
            which_flow (str): Specifies the type of flow data to process (e.g., 'uphi', 'utheta').
            which_data (str): Specifies the dataset identifier (e.g., 'hmi.m_720s').
        """
        self.which_flow = which_flow
        self.which_data = which_data

    def getdata(self):
        """
        Loads flow data from HDF5 files and prepares the time and spatial arrays.
        This method reads multiple HDF5 files, extracts flow data, timestamps, and spatial coordinates,
        and combines them into a single flow array.
        Returns:
            self (FlowData): The instance of FlowData with the flow array, time array, and spatial coordinates.
        """

        for i, n in tqdm(enumerate(np.arange(10, 25))):
            # file_path = f'/scratch/seismo/joshin/pipeline-test/IterativeLCT/{self.which_data}/20{n}_dt_1h_dspan_6h_dstep_120m.hdf5' #for m_720s
            file_path = f'/scratch/seismo/joshin/pipeline-test/IterativeLCT/{self.which_data}/20{n}_ntry_3_grid_len_5_dspan_6_dstep_30_extent_73.hdf5' # for ic_45s
            with h5py.File(file_path, 'r') as f1:
                t = f1['tstart'][()]
                flow = f1[self.which_flow][()]
                lat = f1['latitude'][()]
                lon = f1['longitude'][()]

                if i == 0:
                    self.flow_array = flow
                    self.t = t
                    self.lat_og = lat
                    self.lon_og = lon
                else:
                    self.flow_array = np.append(self.flow_array, flow, axis=0)
                    self.t = np.append(self.t, t, axis=0)
                print(n, len(t))
        dats = [datetime.strptime(str(s, encoding='utf-8'), '%Y.%m.%d_%H:%M:%S') for s in self.t]
        self.t_array = Time(dats, format='datetime', scale='tai').decimalyear
        self.nt, self.nlat, self.nlng = len(self.t_array), len(self.lat_og), len(self.lon_og)
        return self

    def remove_median(self):
        """
        Removes the median from the flow data.
        This method calculates the median of the flow array along the 0th axis (time) and subtracts it from the flow array.
        Returns:
            self (FlowData): The instance of FlowData with the median removed from the flow array.
        """
        self.median = np.nanmedian(self.flow_array, axis=0)
        self.flow_array -= self.median
        return self

    def calculate_mad(self):
        """
        Computes the median absolute deviation (MAD) of the flow data.
        The MAD is calculated along the 0th axis (time) of the flow array, ignoring NaN values.
        Returns:
            self (FlowData): The instance of FlowData with the MAD calculated and stored.
        """
        self.mad = stats.median_abs_deviation(self.flow_array, axis=0, nan_policy='omit')
        return self

    def outlier_rejection(self, threshold):
        """
        Rejects outliers in the flow data based on a specified threshold.
        This method calculates the median absolute deviation (MAD) of the flow array and identifies outliers
        as values that exceed a specified number of MADs from the median.
        Parameters:
            threshold (float): The number of MADs to use for outlier rejection.
        Returns:
            self (FlowData): The instance of FlowData with outliers replaced by NaN in the flow array.
        """
        self.calculate_mad()
        k = 1.4826
        mask = np.abs(self.flow_array) > threshold * self.mad * k
        self.flow_array[mask] = np.nan
        return self

    def remove_yearly_variation(self):
        print(self.t_array.shape, self.crlt_obs.shape)
        print(self.t_array)
        print(self.nt)
        pop, _ = curve_fit(sin_fit, self.t_array, np.nan_to_num(self.crlt_obs))
        for i in tqdm(range(self.nlat)):
            for j in range(self.nlng):
                ts = self.flow_array[:, i, j]
                valid = ~np.isnan(ts)
                if np.sum(valid) < 10:
                    self.flow_array[:, i, j] = np.nan
                    continue
                popt, _ = curve_fit(sin_fit, self.t_array[valid], ts[valid], p0=pop)
                fitted = sin_fit(self.t_array, *popt)
                self.flow_array[:, i, j] -= fitted
        return self

    def save(self):
        """
        Saves the processed flow array to a file.
        This method calls the utility function `save_flow_array` to save the flow array,
        which flow type, and which dataset.
        Returns:
            self (FlowData): The instance of FlowData after saving the flow array.
        """
        save_flow_array(self.flow_array, self.which_flow, self.which_data)
        return self

    def plot(self, n, which_plot):
        """
        Generates and saves plots for the flow data.
        This method uses the utility function `make_plot` to create various plots based on the flow data,
        time array, and median absolute deviation (if available).
        Parameters:
            n (int): An index for naming the saved plot files.
            which_plot (str): The type of plot to create. Options include 'flow_histogram', 'time_series',
                              'flow_data_plot', and 'mad'.
        Returns:
            None
        """
        make_plot(self.flow_array, self.t_array, self.mad if hasattr(self, 'mad') else None, n, which_plot)
