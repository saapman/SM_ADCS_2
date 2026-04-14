"""
Sensor models  (Phase 5 — structured stubs).

Each sensor turns a truth value into a noisy measurement.
"""

import numpy as np


class Gyroscope:
    """Three-axis rate gyroscope with bias instability and ARW."""

    def __init__(self, arw=0.005, bias_instability=1e-4,
                 bias_walk=1e-6, dt=0.1):
        self.sigma_n = arw / np.sqrt(dt)       # white-noise component
        self.sigma_b = bias_walk * np.sqrt(dt)  # bias random walk per step
        self.bias    = np.zeros(3)

    def measure(self, omega_true):
        self.bias += np.random.randn(3) * self.sigma_b
        return omega_true + self.bias + np.random.randn(3) * self.sigma_n


class SunSensor:
    """Coarse sun sensor (returns unit sun vector or None in eclipse)."""

    def __init__(self, sigma=np.radians(0.5)):
        self.sigma = sigma

    def measure(self, sun_body_true, in_eclipse):
        if in_eclipse:
            return None
        s = sun_body_true + np.random.randn(3) * self.sigma
        return s / np.linalg.norm(s)


class Magnetometer:
    """Three-axis magnetometer."""

    def __init__(self, sigma=1e-7):
        self.sigma = sigma

    def measure(self, B_body_true):
        return B_body_true + np.random.randn(3) * self.sigma