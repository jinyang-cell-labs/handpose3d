"""Small SE(3) helpers built on OpenCV's Rodrigues, shared by the solver.

Conventions
-----------
A transform ``T`` is a 4x4 matrix. ``T_b_a`` maps a point expressed in frame
``a`` into frame ``b``::  X_b = T_b_a @ X_a  (homogeneous).

A pose is parameterized for optimization as a 6-vector ``[rvec(3), tvec(3)]``
where ``rvec`` is an axis-angle rotation (OpenCV Rodrigues).
"""
from __future__ import annotations

import cv2
import numpy as np


def rt_to_T(rvec, tvec):
    """(rvec, tvec) -> 4x4 transform."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def T_to_rt(T):
    """4x4 transform -> (rvec(3,), tvec(3,))."""
    rvec, _ = cv2.Rodrigues(np.asarray(T, dtype=np.float64)[:3, :3])
    return rvec.reshape(3), np.asarray(T, dtype=np.float64)[:3, 3].copy()


def invert_T(T):
    """Inverse of a 4x4 rigid transform."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def compose(*transforms):
    """Matrix-multiply a chain of 4x4 transforms left-to-right."""
    out = np.eye(4, dtype=np.float64)
    for T in transforms:
        out = out @ np.asarray(T, dtype=np.float64)
    return out


def euler_deg_to_R(rx, ry, rz):
    """Euler angles in **degrees** (intrinsic XYZ order) -> 3x3 rotation.

    Intrinsic XYZ: rotate about the body X axis by ``rx``, then about the new
    Y by ``ry``, then about the new Z by ``rz``. Equivalent matrix product is
    ``R = Rx @ Ry @ Rz`` and matches
    ``scipy.spatial.transform.Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)``.
    """
    ax, ay, az = np.deg2rad([float(rx), float(ry), float(rz)])
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return Rx @ Ry @ Rz


def average_transforms(T_list):
    """Robust average of several 4x4 transforms via component-wise median of
    the axis-angle + translation parameterization. Good enough for an initial
    guess that bundle adjustment will refine."""
    rs = []
    ts = []
    for T in T_list:
        r, t = T_to_rt(T)
        rs.append(r)
        ts.append(t)
    r_med = np.median(np.asarray(rs), axis=0)
    t_med = np.median(np.asarray(ts), axis=0)
    return rt_to_T(r_med, t_med)
