"""
Multiplicative Extended Kalman Filter (MEKF) for attitude + gyro-bias
estimation from a gyroscope propagated between vector-sensor updates.

State: [delta_theta(3), delta_bias(3)] -- an additive error state in the
tangent space of SO(3), defined by

    q_true = delta_q (x) q_hat,     delta_q ~ [1, delta_theta/2]

so delta_theta is a small-angle rotation vector taking the current
estimate q_hat onto the true attitude, and delta_bias is the gyro-bias
error. The quaternion itself is propagated exactly (nonlinearly); only
the *error* is linear, which is the point of the multiplicative
formulation -- it avoids the redundant/singular parameterisation of a
naive additive EKF on the quaternion.

Reference: Markley & Crassidis, *Fundamentals of Spacecraft Attitude
Determination and Control*, Chapter 6 (the "MEKF").
"""

import numpy as np
from . import quaternion as quat


def _skew(v):
    """Cross-product (skew-symmetric) matrix, so skew(v) @ x == v x x."""
    x, y, z = v
    return np.array([
        [0.0, -z,  y],
        [z,  0.0, -x],
        [-y,  x, 0.0],
    ])


def _small_angle_quat(dtheta):
    """First-order small-angle quaternion for an error rotation vector."""
    q = np.array([1.0, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]])
    return quat.normalize(q)


class MEKF:
    def __init__(self, q0, dt, bias0=None):
        self.q    = q0.copy()
        self.bias = bias0 if bias0 is not None else np.zeros(3)
        self.dt   = dt

        # Error-state covariance, P = diag(P_theta, P_bias)
        self.P = np.diag([1e-2]*3 + [1e-6]*3)

        # Continuous-time process noise spectral densities (discretised
        # by *dt below): gyro angle-random-walk drives delta_theta,
        # bias random-walk drives delta_bias.
        self.Q = np.diag([1e-8]*3 + [1e-12]*3)

        self.R_sun = np.eye(3) * np.radians(0.5)**2
        self.R_mag = np.eye(3) * (1e-7)**2

    # ------------------------------------------------------------------
    # Time update
    # ------------------------------------------------------------------
    def predict(self, omega_meas):
        """Propagate attitude and covariance using the bias-corrected gyro."""
        omega = omega_meas - self.bias

        # Nonlinear quaternion propagation.
        q_dot = 0.5 * quat.omega_matrix(omega) @ self.q
        self.q = quat.normalize(self.q + self.dt * q_dot)

        # Linearised error-state transition:
        #   delta_theta_dot = -[omega x] delta_theta - delta_bias
        #   delta_bias_dot  = 0                          (random walk)
        F = np.zeros((6, 6))
        F[0:3, 0:3] = -_skew(omega)
        F[0:3, 3:6] = -np.eye(3)

        Phi = np.eye(6) + F * self.dt          # first-order state transition
        Qd  = self.Q * self.dt                  # discretised process noise

        self.P = Phi @ self.P @ Phi.T + Qd

    # ------------------------------------------------------------------
    # Measurement updates (vector observations: sun direction, B-field)
    # ------------------------------------------------------------------
    def _vector_update(self, meas_body, ref_eci, R):
        """
        Shared Kalman update for any unit/known-magnitude vector sensor.

        meas_body : measured vector in the body frame
        ref_eci   : the same physical vector's known/modelled direction
                    in ECI (sun direction, modelled B-field, ...)
        R         : measurement noise covariance (3x3)
        """
        C = quat.to_dcm(self.q)          # ECI -> body, current estimate
        pred_body = C @ ref_eci

        # Residual (innovation)
        y = meas_body - pred_body

        # Sensitivity of the predicted body-frame vector to a small
        # attitude error: pred_true = pred_hat - [delta_theta x] pred_hat
        #               => y ~= [pred_hat x] delta_theta + noise
        H = np.zeros((3, 6))
        H[:, 0:3] = _skew(pred_body)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(3))

        dx = K @ y
        dtheta = dx[0:3]
        dbias  = dx[3:6]

        # Reset step: fold the error state into the full state, then
        # implicitly zero it (delta_theta = 0 exactly, by construction).
        #
        # Composition order matters: this module's quaternion convention
        # gives C(p (x) q) = C(q) @ C(p) (verified against quaternion.to_dcm
        # / quaternion.multiply directly), and the local attitude error is
        # defined by C_true = C_err @ C_hat. Achieving that composition
        # requires q_new = q_hat (x) delta_q, NOT delta_q (x) q_hat.
        self.q    = quat.normalize(quat.multiply(self.q, _small_angle_quat(dtheta)))
        self.bias = self.bias + dbias

        # Joseph-form covariance update -- numerically robust (stays
        # symmetric PSD even with an imperfect gain), matching the
        # covariance-update approach used in the reentry_V EKF project.
        I6 = np.eye(6)
        IKH = I6 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        return y

    def update_sun(self, sun_meas_body, sun_ref_eci):
        """Sun-sensor measurement update. No-op if in eclipse (meas=None)."""
        if sun_meas_body is None:
            return None
        return self._vector_update(sun_meas_body, sun_ref_eci, self.R_sun)

    def update_mag(self, mag_meas_body, mag_ref_eci):
        """Magnetometer measurement update."""
        return self._vector_update(mag_meas_body, mag_ref_eci, self.R_mag)

    # ------------------------------------------------------------------
    def get_estimate(self):
        return self.q.copy(), self.bias.copy()
