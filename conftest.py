"""Ensure the repo root is importable so `import maestro` works under pytest
without installing the package."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
