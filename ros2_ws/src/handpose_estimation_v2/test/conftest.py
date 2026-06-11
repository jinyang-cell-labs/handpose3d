"""Make both ament_python package roots importable without installation.

Lets the math tests run from any cwd:
    python -m pytest ros2_ws/src/handpose_estimation_v2/test/ -q
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]  # ros2_ws/src
for _pkg in ("handpose_estimation", "handpose_estimation_v2"):
    _root = str(_SRC / _pkg)
    if _root not in sys.path:
        sys.path.insert(0, _root)
