import os
import sys

import matplotlib

matplotlib.use("Agg")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_PATH = os.path.join(ROOT, "dermatology", "dermatology.data")
