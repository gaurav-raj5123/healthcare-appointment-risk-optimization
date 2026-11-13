"""
Pytest configuration and shared fixtures for CarePredict test suite.
"""

import sys
from pathlib import Path
import pytest

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

