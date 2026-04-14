"""
Environmental models: magnetic field, sun direction, atmospheric density.
"""

import numpy as np

# Constants
EARTH_DIPOLE_MOMENT = 7.94e22          # A·m²
MAGNETIC_TILT       = np.radians(11.5)
OMEGA_EARTH         = 7.2921159e-5     # rad/s
SOLAR_FLUX          = 1361.0           # W/m²
C_LIGHT             = 2.998e8          # m/s
OBLIQUITY           = np.radians(23.44)
R_EARTH             = 6.371e6          # m


def magnetic_field_eci(r_eci, t):
    """
    Tilted-dipole magnetic field model.
    Returns B in Tesla, ECI frame.
    """
    r     = np.linalg.norm(r_eci)
    r_hat = r_eci / r

    phi = OMEGA_EARTH * t       # Earth rotation
    m_hat = np.array([
        -np.sin(MAGNETIC_TILT) * np.cos(phi),
        -np.sin(MAGNETIC_TILT) * np.sin(phi),
        -np.cos(MAGNETIC_TILT),
    ])

    coeff = 1e-7 * EARTH_DIPOLE_MOMENT / r**3
    return coeff * (3.0 * np.dot(m_hat, r_hat) * r_hat - m_hat)


def sun_direction_eci(t):
    """
    Approximate sun unit-vector in ECI.  t = 0 at vernal equinox.
    """
    lam = 2.0 * np.pi * t / (365.25 * 86400.0)
    return np.array([
        np.cos(lam),
        np.sin(lam) * np.cos(OBLIQUITY),
        np.sin(lam) * np.sin(OBLIQUITY),
    ])


def is_in_eclipse(r_eci, sun_eci):
    """Cylindrical shadow model."""
    s = sun_eci / np.linalg.norm(sun_eci)
    proj = np.dot(r_eci, s)
    if proj > 0:
        return False
    perp = np.linalg.norm(r_eci - proj * s)
    return perp < R_EARTH


def atmospheric_density(altitude):
    """Piecewise-exponential model.  altitude in metres, returns kg/m³."""
    table = [
        (200e3, 2.79e-10, 37.1e3),
        (300e3, 2.42e-11, 45.5e3),
        (400e3, 3.73e-12, 53.6e3),
        (500e3, 6.97e-13, 62.2e3),
        (600e3, 1.45e-13, 71.8e3),
        (700e3, 3.61e-14, 81.4e3),
        (800e3, 1.17e-14, 88.7e3),
    ]
    for h0, rho0, H in table:
        if altitude < h0 + 100e3:
            return rho0 * np.exp(-(altitude - h0) / H)
    h0, rho0, H = table[-1]
    return rho0 * np.exp(-(altitude - h0) / H)