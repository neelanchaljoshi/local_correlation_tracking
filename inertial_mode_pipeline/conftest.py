"""
conftest.py
-----------
pytest configuration: inject the mock zclpy3 package so tests run
without the MPS-internal dependency.
"""

import sys
import pathlib

# Insert mock package path before any test collection
MOCK_DIR = pathlib.Path(__file__).parent / 'tests' / 'mocks'
sys.path.insert(0, str(MOCK_DIR))

# Also patch the MPS internal path so geometry.py doesn't crash on import
sys.path.insert(0, str(MOCK_DIR))
