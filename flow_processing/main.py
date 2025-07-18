import sys
from flow_data import FlowData

def main(which_flow, which_data):
    """
    Main function to process flow data.
    Parameters:
        which_flow (str): The type of flow data to process (e.g., 'uphi', 'vphi').
        which_data (str): The dataset identifier (e.g., 'hmi.m_720s').
    """
    flow = FlowData(which_flow, which_data)
    flow.getdata()
    flow.plot(n=1, which_plot='flow_histogram')
    flow.remove_median()
    flow.plot(n=2, which_plot='flow_histogram')
    flow.plot(n=2, which_plot='time_series')
    flow.outlier_rejection(3)
    flow.plot(n=3, which_plot='flow_histogram')
    flow.remove_yearly_variation()
    flow.plot(n=4, which_plot='time_series')
    flow.plot(n=5, which_plot='flow_data_plot')
    flow.save()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <which_flow> <which_data>")
    else:
        main(sys.argv[1], sys.argv[2])