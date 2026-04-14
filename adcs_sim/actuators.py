"""
Actuator models: reaction wheels and magnetorquers.
"""

import numpy as np


class ReactionWheels:
    """
    N reaction wheels with torque and speed saturation.

    For 3 wheels aligned with body axes the distribution matrix is identity.
    The code generalises to arbitrary axis orientations.
    """

    def __init__(self, config):
        self.n     = config.n_wheels
        self.J_w   = config.J_wheel
        self.t_max = config.max_torque
        self.w_max = config.max_speed
        self.A     = config.axes          # (n, 3): row i = wheel-i axis

    def apply(self, tau_commanded, wheel_speeds):
        """
        Distribute a commanded body torque to the wheels.

        Returns
        -------
        tau_on_body     : ndarray (3,)  — actual torque delivered to the body
        wheel_torques   : ndarray (n,)  — motor torque on each wheel
        """
        # Solve  −A^T τ_w = τ_cmd  →  τ_w = −(A^T)^{-1} τ_cmd
        tau_w = -np.linalg.solve(self.A.T, tau_commanded)

        # Torque saturation
        tau_w = np.clip(tau_w, -self.t_max, self.t_max)

        # Speed saturation: prevent further spin-up beyond max
        for i in range(self.n):
            if abs(wheel_speeds[i]) >= self.w_max:
                if np.sign(tau_w[i]) == np.sign(wheel_speeds[i]):
                    tau_w[i] = 0.0

        tau_body = -(self.A.T @ tau_w)
        return tau_body, tau_w

    def angular_momentum(self, wheel_speeds):
        """Total wheel angular momentum in body frame."""
        return self.J_w * (self.A.T @ wheel_speeds)


class Magnetorquers:
    """B-cross momentum-dumping law."""

    def __init__(self, config):
        self.m_max = config.max_dipole

    def momentum_dump(self, h_wheels, B_body, k=1e4):
        """
        Compute magnetic dipole and resulting torque for desaturation.

        m = −k (B × h_rw) / |B|²      (cross-product law)
        τ = m × B
        """
        B2 = np.dot(B_body, B_body)
        if B2 < 1e-20:
            return np.zeros(3), np.zeros(3)

        m = -k * np.cross(B_body, h_wheels) / B2
        m_norm = np.linalg.norm(m)
        if m_norm > self.m_max:
            m *= self.m_max / m_norm

        return np.cross(m, B_body), m