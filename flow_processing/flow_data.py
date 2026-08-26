import glob
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


# Legacy per-year LCT output (the pre-refactor lct_pipeline). A single
# directory can hold more than one run's files side by side with
# different naming/cadence (e.g. IterativeLCT/hmi.m_720s/ has both a
# "_dt_1h_dspan_6h_dstep_120m" run and an unrelated
# "_ntry_3_..._extent_73_new" run for the same years) -- which_data
# alone isn't enough to disambiguate, so each known dataset gets an
# explicit glob pattern rather than one hardcoded template.
LEGACY_ROOT = '/scratch/seismo/joshin/pipeline-test/IterativeLCT'
LEGACY_GLOB_PATTERNS = {
    'hmi.ic_45s': '20*_ntry_3_grid_len_5_dspan_6_dstep_30_extent_73.hdf5',
    'hmi.m_720s': '20*_dt_1h_dspan_6h_dstep_120m.hdf5',
}


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

    def getdata(self, data_root=None, pattern=None):
        """
        Loads and concatenates every HDF5 file matching `pattern` under
        `data_root`, sorted by each file's own recorded timestamps (not
        filename order), and prepares the time and spatial arrays.

        This works the same way regardless of whether the source is one
        file per year (the legacy pre-refactor LCT pipeline), one file
        per month (the current lct_pipeline MPI/month mode,
        `pipeline.py`), or one file per chunk (the current lct_pipeline
        non-MPI/chunk mode, `pipeline_chunk.py`) — all three write the
        same tstart/uphi/utheta/latitude/longitude schema
        (`lct_pipeline.io.create_output_hdf5`).

        Parameters:
            data_root (str, optional): Directory to search. Defaults to
                the legacy `IterativeLCT/{which_data}/` layout for
                backward compatibility with existing callers.
            pattern (str, optional): Glob pattern (relative to
                data_root) selecting which files to include. Needed
                because a single directory can hold more than one run's
                files side by side with different naming/cadence.
                Defaults to `LEGACY_GLOB_PATTERNS[which_data]` when
                data_root is not given. Required (raises ValueError if
                omitted) when which_data has no known legacy pattern —
                e.g. any current lct_pipeline output — since there is no
                safe default to guess. For current lct_pipeline output,
                pass the same `rootdir_out` from the `.ini` config as
                data_root, and a pattern like
                `*_gran_dspan*_4k.hdf5` (month mode) or
                `*_gran_dspan*_4k_chunk.hdf5` (chunk mode) — `gran`/`mag`
                and `4k`/`2k` per the config's segname/downsample.
        Returns:
            self (FlowData): The instance of FlowData with the flow array, time array, and spatial coordinates.
        """
        if data_root is None:
            data_root = os.path.join(LEGACY_ROOT, self.which_data)
        if pattern is None:
            try:
                pattern = LEGACY_GLOB_PATTERNS[self.which_data]
            except KeyError:
                raise ValueError(
                    f'No default glob pattern known for which_data={self.which_data!r}. '
                    f'Pass pattern= explicitly — known legacy datasets are '
                    f'{sorted(LEGACY_GLOB_PATTERNS)}; for current lct_pipeline '
                    f"output use something like '*_gran_dspan*_4k.hdf5' "
                    f"(month mode) or '*_gran_dspan*_4k_chunk.hdf5' (chunk mode).")

        search = os.path.join(data_root, pattern)
        files = sorted(glob.glob(search))
        if not files:
            raise FileNotFoundError(f'No files matched {search!r}')

        t_chunks = []
        flow_chunks = []
        ref_lat = ref_lon = None
        for file_path in tqdm(files, desc='Reading HDF5 files'):
            with h5py.File(file_path, 'r') as f1:
                t = f1['tstart'][()]
                flow = f1[self.which_flow][()]
                lat = f1['latitude'][()]
                lon = f1['longitude'][()]

            if ref_lat is None:
                ref_lat, ref_lon = lat, lon
            elif not (np.array_equal(lat, ref_lat) and np.array_equal(lon, ref_lon)):
                raise ValueError(
                    f'{file_path}: latitude/longitude grid does not match the '
                    f'other files being concatenated — these look like output '
                    f'from different LCT runs. Narrow `pattern` to select only '
                    f'one run.')

            t_chunks.append(t)
            flow_chunks.append(flow)
            print(file_path, len(t))

        t_all = np.concatenate(t_chunks, axis=0)
        flow_all = np.concatenate(flow_chunks, axis=0)

        dats = [datetime.strptime(str(s, encoding='utf-8'), '%Y.%m.%d_%H:%M:%S') for s in t_all]
        t_array = Time(dats, format='datetime', scale='tai').decimalyear

        order = np.argsort(t_array)
        self.t_array = t_array[order]
        self.t = t_all[order]
        self.flow_array = flow_all[order]
        self.lat_og = ref_lat
        self.lon_og = ref_lon
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

    def save(self, suffix=None):
        """
        Saves the processed flow array to a file.
        This method calls the utility function `save_flow_array` to save the flow array,
        which flow type, and which dataset.
        Parameters:
            suffix (str, optional): Output filename suffix distinguishing
                this LCT run (e.g. '_granule', '_dt_1h'). Defaults to
                `LEGACY_OUTPUT_SUFFIX[which_data]` for known legacy
                datasets; required otherwise. Pass the same value you'd
                use for `run_pipeline.py`'s `data` argument's suffix, so
                inertial_mode_pipeline can find this output.
        Returns:
            self (FlowData): The instance of FlowData after saving the flow array.
        """
        save_flow_array(self.flow_array, self.which_flow, self.which_data, suffix=suffix)
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
