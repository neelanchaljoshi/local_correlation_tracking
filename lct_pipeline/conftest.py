"""
conftest.py
-----------
Inject mock packages before any test imports so CI runs without
MPI, zclpy3, or josh installed.
"""

import pathlib
import sys

MOCK_DIR = pathlib.Path(__file__).parent / 'tests' / 'mocks'
sys.path.insert(0, str(MOCK_DIR))
