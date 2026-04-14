"""
Keplerian circular orbit propagation (analytical).
"""

import numpy as np


class CircularOrbit:
    def __init__(self, config):
        self.a    = config.a
        self.inc  = config.inclination
        self.raan = config.raan
        self.n    = config.n           # mean motion

        ci, si = np.cos(self.inc), np.sin(self.inc)
        cO, sO = np.cos(self.raan), np.sin(self.raan)

        # Perifocal → ECI rotation (argument of perigee = 0)
        self._R = np.array([
            [ cO, -sO * ci,  sO * si],
            [ sO,  cO * ci, -cO * si],
            [ 0,   si,       ci     ],
        ])

    def propagate(self, t):
        """Return (r_eci, v_eci) at time t.  True anomaly = 0 at t = 0."""
        theta = self.n * t

        r_peri = self.a * np.array([np.cos(theta), np.sin(theta), 0.0])
        v_peri = self.a * self.n * np.array([-np.sin(theta), np.cos(theta), 0.0])

        return self._R @ r_peri, self._R @ v_peri