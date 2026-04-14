"""
Main simulation loop.
"""

import numpy as np
from . import quaternion as quat
from . import frames, control
from .orbit        import CircularOrbit
from .disturbances import total as disturbance_total
from .actuators    import ReactionWheels, Magnetorquers
from .dynamics     import rk4_step


class History:
    """Growable storage for time-history data."""

    def __init__(self):
        self._lists = dict(
            t=[], q=[], omega=[], wheel_speeds=[],
            tau_ctrl=[], tau_dist=[],
            pointing_error_deg=[], euler_error_deg=[],
            r_eci=[],
        )

    def record(self, t, q, omega, wheel_speeds, tau_ctrl, tau_dist,
               q_desired):
        d = self._lists
        d["t"].append(t)
        d["q"].append(q.copy())
        d["omega"].append(omega.copy())
        d["wheel_speeds"].append(wheel_speeds.copy())
        d["tau_ctrl"].append(tau_ctrl.copy())
        d["tau_dist"].append(tau_dist.copy())

        q_err = quat.error(q, q_desired)
        angle = 2.0 * np.arcsin(
            np.clip(np.linalg.norm(q_err[1:4]), 0.0, 1.0)
        )
        d["pointing_error_deg"].append(np.degrees(angle))
        d["euler_error_deg"].append(np.degrees(2.0 * q_err[1:4]))

    def finalise(self):
        """Convert every list to a NumPy array and return self."""
        for k, v in self._lists.items():
            setattr(self, k, np.array(v))
        return self


def run(cfg):
    """
    Run the simulation described by *cfg* (a SimConfig instance).
    Returns a History object with NumPy arrays.
    """
    orbit = CircularOrbit(cfg.orbit)
    wheels = ReactionWheels(cfg.wheels)
    mag_tq = Magnetorquers(cfg.magnetorquer)

    J     = cfg.sc.J
    J_inv = np.linalg.inv(J)

    state = np.concatenate([
        cfg.initial_quaternion,
        cfg.initial_omega,
        cfg.initial_wheel_speeds,
    ])

    history = History()
    n_steps = int(cfg.duration / cfg.dt)
    report_interval = max(n_steps // 20, 1)

    for step in range(n_steps):
        t = step * cfg.dt

        # --- unpack ---
        q            = quat.normalize(state[0:4])
        omega        = state[4:7]
        wheel_speeds = state[7:]

        # --- orbit ---
        r_eci, v_eci = orbit.propagate(t)

        # --- disturbances ---
        tau_dist, info = disturbance_total(
            q, r_eci, v_eci, t, cfg.sc, cfg.orbit
        )

        # --- reference ---
        q_des, omega_des_eci = control.desired_state(
            r_eci, v_eci, cfg.orbit.n
        )

        # --- control law ---
        if cfg.control.mode == "pd":
            tau_cmd = control.pd(
                q, omega, q_des, omega_des_eci,
                cfg.control.Kp, cfg.control.Kd,
            )
        else:
            tau_cmd = control.off()

        # --- actuators ---
        tau_rw, whl_torques = wheels.apply(tau_cmd, wheel_speeds)

        tau_mt = np.zeros(3)
        if cfg.enable_momentum_dumping:
            tau_mt, _ = mag_tq.momentum_dump(
                wheels.angular_momentum(wheel_speeds),
                info["B_body"],
            )

        # --- total body torque ---
        tau_body = tau_dist + tau_rw + tau_mt

        # --- log ---
        history.record(t, q, omega, wheel_speeds, tau_rw, tau_dist, q_des)

        # --- integrate ---
        state = rk4_step(
            state, cfg.dt, J, J_inv,
            tau_body, whl_torques,
            cfg.wheels.J_wheel, cfg.wheels.axes,
        )

        # --- progress ---
        if step % report_interval == 0:
            pct = 100.0 * step / n_steps
            err = history._lists["pointing_error_deg"][-1]
            print(f"  t={t:8.1f}s  ({pct:5.1f}%)  "
                  f"pointing err = {err:8.3f}°")

    err = history._lists["pointing_error_deg"][-1]
    print(f"  Done.  Final pointing error = {err:.4f}°")
    return history.finalise()