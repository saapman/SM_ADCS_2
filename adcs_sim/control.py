"""
Attitude control laws.
"""

import numpy as np
from . import quaternion as quat
from . import frames


def desired_state(r_eci, v_eci, n):
    """
    Nadir-pointing reference: body frame aligned with LVLH.

    Returns
    -------
    q_des        : desired quaternion (ECI → body)
    omega_des_eci: desired angular velocity in ECI frame
    """
    q_des = frames.lvlh_quaternion(r_eci, v_eci)

    # LVLH rotates at [0, −n, 0] in its own frame
    C_li = frames.eci_to_lvlh_dcm(r_eci, v_eci)
    omega_des_eci = C_li.T @ np.array([0.0, -n, 0.0])

    return q_des, omega_des_eci


def pd(q, omega, q_des, omega_des_eci, Kp, Kd):
    """
    PD controller on quaternion error.

    τ = −Kp q_err_vec − Kd ω_err       (body frame)
    """
    q_err = quat.error(q, q_des)

    C = quat.to_dcm(q)
    omega_err = omega - C @ omega_des_eci

    return -Kp * q_err[1:4] - Kd * omega_err


def off(*_args, **_kwargs):
    """No control."""
    return np.zeros(3)