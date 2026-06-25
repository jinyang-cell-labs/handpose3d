"""Constant-velocity Kalman filter for smoothing a 3D point track.

Pure numpy (no ROS) so it can be unit-tested with the repo venv, mirroring
``triangulation.py``.

The motion model (constant velocity) and the measurement model (we observe
position directly, ``z = H x`` with ``H = [I | 0]``) are both **linear**, so
the exact linear Kalman filter *is* the EKF for this problem — there is no
nonlinearity to linearise and the Jacobians ``F``, ``H`` are constant. We
implement that exact form; no approximation is made.

State (6-vector): ``x = [px, py, pz, vx, vy, vz]`` — position (m), velocity
(m/s). Each measurement supplies its own ``R`` (the triangulation covariance),
so the Kalman gain down-weights noisy axes automatically: stereo depth has a
large ``R`` and is smoothed hard, while the well-constrained lateral axes are
tracked tightly.
"""

import numpy as np

_I3 = np.eye(3)


class ConstantVelocityKF:
    """6-state (position + velocity) constant-velocity Kalman filter."""

    def __init__(self, q_accel, init_velocity_sigma):
        """
        Args:
            q_accel: process-noise acceleration density (white-noise-accel
                model). Larger -> more responsive / less smoothing; smaller ->
                smoother / laggier. Units (m/s^2)^2.
            init_velocity_sigma: initial velocity std (m/s) for P0.
        """
        self.q_accel = float(q_accel)
        self.init_velocity_sigma = float(init_velocity_sigma)
        self.x = None  # (6,) state
        self.P = None  # (6, 6) covariance

    @property
    def initialized(self):
        return self.x is not None

    def initialize(self, z, R):
        """Seed the state from a first position measurement ``z`` (cov ``R``)."""
        z = np.asarray(z, dtype=float).reshape(3)
        R = np.asarray(R, dtype=float).reshape(3, 3)
        self.x = np.zeros(6)
        self.x[:3] = z
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = R
        self.P[3:, 3:] = (self.init_velocity_sigma ** 2) * _I3

    def predict(self, dt):
        """Advance the state by ``dt`` seconds (constant-velocity model)."""
        dt = float(dt)
        F = np.eye(6)
        F[:3, 3:] = dt * _I3

        # Discrete white-noise-acceleration process noise.
        q = self.q_accel
        d2, d3, d4 = dt * dt, dt ** 3, dt ** 4
        Q = np.zeros((6, 6))
        Q[:3, :3] = (d4 / 4.0) * q * _I3
        Q[:3, 3:] = (d3 / 2.0) * q * _I3
        Q[3:, :3] = (d3 / 2.0) * q * _I3
        Q[3:, 3:] = d2 * q * _I3

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation_mahalanobis2(self, z, R):
        """Squared Mahalanobis distance of ``z`` from the predicted position.

        Used for outlier gating *before* applying the update. Does not mutate
        the filter.
        """
        z = np.asarray(z, dtype=float).reshape(3)
        R = np.asarray(R, dtype=float).reshape(3, 3)
        y = z - self.x[:3]
        S = self.P[:3, :3] + R
        return float(y @ np.linalg.solve(S, y))

    def update(self, z, R):
        """Correct the state with position measurement ``z`` (cov ``R``).

        Returns the squared Mahalanobis distance of the innovation.
        """
        z = np.asarray(z, dtype=float).reshape(3)
        R = np.asarray(R, dtype=float).reshape(3, 3)
        H = np.zeros((3, 6))
        H[:, :3] = _I3

        y = z - self.x[:3]              # innovation
        S = self.P[:3, :3] + R          # innovation covariance
        Sinv = np.linalg.inv(S)
        maha2 = float(y @ Sinv @ y)
        K = self.P @ H.T @ Sinv         # (6, 3) Kalman gain

        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        # Joseph form: stays symmetric positive-definite under round-off.
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return maha2

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:].copy()

    @property
    def position_covariance(self):
        return self.P[:3, :3].copy()
