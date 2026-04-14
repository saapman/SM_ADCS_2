"""
Equations of motion and RK4 integrator.

State vector layout:  [q(4), ω(3), Ω_wheels(n_wheels)]
"""

import numpy as np
from . import quaternion as quat


def _deriv(state, J, J_inv, tau_body, wheel_torques, J_w, axes):
    """
    Time derivative of the state vector.

    tau_body includes *all* torques on the rigid body
    (external disturbances + wheel reaction + magnetorquer).
    wheel_torques are the motor torques *on each wheel*.
    """
    q     = state[0:4]
    omega = state[4:7]
    w_spd = state[7:]

    # Wheel angular momentum
    h_rw = J_w * (axes.T @ w_spd)          # (3,)

    # Total system angular momentum
    H = J @ omega + h_rw

    # Quaternion kinematics
    q_dot = 0.5 * quat.omega_matrix(omega) @ q

    # Euler's equation  (tau_body already contains wheel reaction)
    omega_dot = J_inv @ (tau_body - np.cross(omega, H))

    # Wheel dynamics
    w_spd_dot = wheel_torques / J_w

    return np.concatenate([q_dot, omega_dot, w_spd_dot])


def rk4_step(state, dt, J, J_inv, tau_body, wheel_torques, J_w, axes):
    """
    Single RK4 step with zero-order-hold torques
    (control command is constant over the step, as in a real
    digital controller).
    """
    args = (J, J_inv, tau_body, wheel_torques, J_w, axes)

    k1 = _deriv(state,                   *args)
    k2 = _deriv(state + 0.5 * dt * k1,   *args)
    k3 = _deriv(state + 0.5 * dt * k2,   *args)
    k4 = _deriv(state + dt * k3,          *args)

    state_new = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    # Re-normalise quaternion to stay on the unit sphere
    state_new[0:4] = quat.normalize(state_new[0:4])

    return state_new