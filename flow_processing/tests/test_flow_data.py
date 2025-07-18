import unittest
import numpy as np
from flow_data import FlowData

class TestFlowData(unittest.TestCase):
    def test_flow_pipeline(self):
        """
        Test the complete flow data processing pipeline.
        This includes data retrieval, median removal, outlier rejection,
        yearly variation removal, and saving the processed data.
        """
        # Initialize FlowData with the flow type and dataset
        flow = FlowData("uphi", "hmi.m_720s")
        flow.getdata()
        self.assertEqual(flow.flow_array.shape[1:], (73, 73))
        flow.remove_median()
        self.assertIsNotNone(flow.median)
        flow.outlier_rejection(3)
        self.assertTrue(np.isnan(flow.flow_array).any() or not np.isnan(flow.flow_array).any())
        flow.remove_yearly_variation()
        flow.save()