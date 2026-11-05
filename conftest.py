"""pytest configuration for bayesDREAM test suite."""

import sys
import os

# Ensure project root is on sys.path so `from bayesDREAM import ...` works
# regardless of where pytest is invoked from.
sys.path.insert(0, os.path.dirname(__file__))
