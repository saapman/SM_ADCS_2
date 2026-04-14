"""
Environmental disturbance torques (all in body frame).
"""

import numpy as np
from . import quaternion as quat
from . import environment as env


def gravity_gradient(q, r_eci, J, mu):
    """τ_gg = (3μ/R³)  r̂_b × (J r̂_b)."""
    R = np.linalg.norm(r_eci)
    C = quat.to_dcm(q)
    r_hat_b = (C @ r_eci) / R
    return (3.0 * mu / R**3) * np.cross(r_hat_b, J @ r_hat_b)


def magnetic_residual(residual_dipole, B_body):
    """τ_mag = m_res × B."""
    return np.cross(residual_dipole, B_body)


def aerodynamic(v_body, rho, Cd, area, cp_offset):
    """Aerodynamic drag torque.  τ = r_cp × F_drag."""
    v = np.linalg.norm(v_body)
    if v < 1.0:
        return np.zeros(3)
    F = -0.5 * rho * v**2 * Cd * area * (v_body / v)
    return np.cross(cp_offset, F)


def solar_radiation_pressure(sun_body, area, cp_offset, reflectivity, in_eclipse):
    """SRP torque.  τ = r_cp × F_srp."""
    if in_eclipse:
        return np.zeros(3)
    P = env.SOLAR_FLUX / env.C_LIGHT
    s_norm = np.linalg.norm(sun_body)
    if s_norm < 1e-10:
        return np.zeros(3)
    F = -P * (1.0 + reflectivity) * area * (sun_body / s_norm)
    return np.cross(cp_offset, F)


def total(q, r_eci, v_eci, t, sc, orb):
    """
    Sum of all disturbance torques.

    Returns
    -------
    tau_total : ndarray (3,)
    info      : dict   (individual torques + B_body for actuator use)
    """
    C = quat.to_dcm(q)

    B_body   = C @ env.magnetic_field_eci(r_eci, t)
    v_body   = C @ v_eci
    sun_eci  = env.sun_direction_eci(t)
    sun_body = C @ sun_eci
    eclipse  = env.is_in_eclipse(r_eci, sun_eci)

    alt = np.linalg.norm(r_eci) - env.R_EARTH
    rho = env.atmospheric_density(alt)

    tau_gg   = gravity_gradient(q, r_eci, sc.J, orb.mu)
    tau_mag  = magnetic_residual(sc.residual_dipole, B_body)
    tau_aero = aerodynamic(v_body, rho, sc.Cd, sc.drag_area, sc.cp_offset)
    tau_srp  = solar_radiation_pressure(
        sun_body, sc.srp_area, sc.srp_cp_offset, sc.reflectivity, eclipse
    )

    return tau_gg + tau_mag + tau_aero + tau_srp, {
        "gravity_gradient": tau_gg,
        "magnetic":         tau_mag,
        "aerodynamic":      tau_aero,
        "srp":              tau_srp,
        "B_body":           B_body,
    }