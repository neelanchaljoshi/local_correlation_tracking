"""
main.py
-------
Entrypoint: python main.py <which_flow> <which_data> [options]

Plain legacy usage is unchanged:
    python main.py uphi hmi.ic_45s
    python main.py utheta hmi.m_720s

To process current lct_pipeline output instead (point at the same
rootdir_out from the .ini config used to produce it):
    python main.py uphi hmi.m_720s \
        --data-root /data/seismo/joshin/pipeline-test/local_correlation_tracking/data/magnetic \
        --pattern '*_mag_dspan*_4k.hdf5' \
        --out-suffix _dspan6h_dstep2h
"""
import argparse
from flow_data import FlowData


def main(which_flow, which_data, data_root=None, pattern=None, out_suffix=None):
    """
    Main function to process flow data.
    Parameters:
        which_flow (str): The type of flow data to process (e.g., 'uphi', 'utheta').
        which_data (str): The dataset identifier (e.g., 'hmi.m_720s').
        data_root (str, optional): Directory to search for input HDF5 files.
            Defaults to the legacy IterativeLCT/{which_data}/ layout.
        pattern (str, optional): Glob pattern selecting which files to read.
            Defaults to the known legacy pattern for which_data.
        out_suffix (str, optional): Output filename suffix. Defaults to the
            known legacy suffix for which_data.
    """
    flow = FlowData(which_flow, which_data)
    flow.getdata(data_root=data_root, pattern=pattern)
    flow.plot(n=1, which_plot='flow_histogram')
    flow.remove_median()
    flow.plot(n=2, which_plot='flow_histogram')
    flow.plot(n=2, which_plot='time_series')
    flow.outlier_rejection(3)
    flow.plot(n=3, which_plot='flow_histogram')
    flow.remove_yearly_variation()
    flow.plot(n=4, which_plot='time_series')
    flow.plot(n=5, which_plot='flow_data_plot')
    flow.save(suffix=out_suffix)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('which_flow', choices=['uphi', 'utheta'],
                    help='Flow component to process')
    p.add_argument('which_data', type=str,
                    help="Dataset identifier, e.g. 'hmi.m_720s'")
    p.add_argument('--data-root', type=str, default=None,
                    help='Directory to search for input HDF5 files. '
                         'Defaults to the legacy IterativeLCT/{which_data}/ '
                         'layout. For current lct_pipeline output, pass the '
                         "same rootdir_out from the .ini config, e.g. "
                         "'.../local_correlation_tracking/data/magnetic'.")
    p.add_argument('--pattern', type=str, default=None,
                    help='Glob pattern (relative to --data-root) selecting '
                         'which files to read. Defaults to the known legacy '
                         "pattern for which_data. For current lct_pipeline "
                         "output: '*_{gran,mag}_dspan*_{4k,2k}.hdf5' (month "
                         "mode) or the same with '_chunk' before '.hdf5' "
                         '(chunk mode).')
    p.add_argument('--out-suffix', type=str, default=None,
                    help='Output filename suffix distinguishing this LCT '
                         "run (e.g. '_granule', '_dt_1h'). Defaults to the "
                         'known legacy suffix for which_data; required '
                         'otherwise.')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.which_flow, args.which_data,
         data_root=args.data_root, pattern=args.pattern,
         out_suffix=args.out_suffix)
