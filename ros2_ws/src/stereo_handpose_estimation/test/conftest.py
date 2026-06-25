"""Make the stereo_handpose_estimation package root importable without install.

Lets the ROS-free math tests run from any cwd:
    python -m pytest ros2_ws/src/stereo_handpose_estimation/test/ -q
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]  # ros2_ws/src
_root = str(_SRC / "stereo_handpose_estimation")
if _root not in sys.path:
    sys.path.insert(0, _root)
