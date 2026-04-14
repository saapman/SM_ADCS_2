"""
Multiplicative Extended Kalman Filter  (Phase 5 — stub).

State: [δθ(3), δbias(3)]  in the tangent space of SO(3).

Reference: Markley & Crassidis, *Fundamentals of Spacecraft
Attitude Determination and Control*, Chapter 6.
"""

import numpy as np
from . import quaternion as quat


class MEKF:
    def __init__(self, q0, dt, bias0=None):
        self.q    = q0.copy()
        self.bias = bias0 if bias0 is not None else np.zeros(3)
        self.dt   = dt

        self.P = np.diag([1e-2]*3 + [1e-6]*3)
        self.Q = np.diag([1e-8]*3 + [1e-12]*3)
        self.R_sun = np.eye(3) * np.radians(0.5)**2
        self.R_mag = np.eye(3) * (1e-7)**2

    # ------------------------------------------------------------------
    def predict(self, omega_meas):
        """Propagate attitude using bias-corrected gyro."""
        omega = omega_meas - self.bias
        q_dot = 0.5 * quat.omega_matrix(omega) @ self.q
        self.q = quat.normalize(self.q + self.dt * q_dot)

        # TODO: build F matrix, propagate P = F P F^T + Q
        self.P += self.Q * self.dt

    # ------------------------------------------------------------------
    def update_sun(self, sun_meas_body, sun_ref_eci):
        """Sun-sensor measurement update.  TODO: implement."""
        pass

    def update_mag(self, mag_meas_body, mag_ref_eci):
        """Magnetometer measurement update.  TODO: implement."""
        pass

    # ------------------------------------------------------------------
    def get_estimate(self):
        return self.q.copy(), self.bias.copy()