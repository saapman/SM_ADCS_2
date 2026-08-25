"""
Truth-vs-estimate demo for the MEKF (adcs_sim.estimation).

Runs a free-tumbling truth trajectory (same rigid-body dynamics used
elsewhere in the package) alongside noisy gyro/sun/magnetometer
measurements, feeds them into the MEKF, and records the true attitude
estimation error and gyro-bias estimation error over time. This is the
verification counterpart to the control-loop scenarios in main.py: it
shows the *estimator*, not just the plant, is doing something correct
by starting the filter off-attitude and checking that the error
converges rather than diverges.
"""

import numpy as np

from . import quaternion as quat
from . import environment as env
from .orbit import CircularOrbit
from .disturbances import total as disturbance_total
from .dynamics import rk4_step
from .sensors import Gyroscope, SunSensor, Magnetometer
from .estimation import MEKF


class EstimationHistory:
    def __init__(self):
        self._lists = dict(
            t=[], attitude_error_deg=[], bias_error_deg_s=[],
            sigma_theta_deg=[],
        )

    def record(self, t, angle_err_deg, bias_err, sigma_theta_deg):
        d = self._lists
        d["t"].append(t)
        d["attitude_error_deg"].append(angle_err_deg)
        d["bias_error_deg_s"].append(np.degrees(bias_err))
        d["sigma_theta_deg"].append(sigma_theta_deg)

    def finalise(self):
        for k, v in self._lists.items():
            setattr(self, k, np.array(v))
        return self


def run(cfg, initial_attitude_error_deg=15.0, true_gyro_bias=None,
        update_every_n_steps=10):
    """
    Propagate truth (uncontrolled rigid body under disturbance torques)
    and an MEKF running off noisy sensors, and return an
    EstimationHistory of the estimation error over time.

    initial_attitude_error_deg : how far off-attitude the filter starts,
        to demonstrate convergence rather than trivial agreement.
    true_gyro_bias : constant gyro bias truth (rad/s); the filter does
        not know this and must estimate it. Defaults to a small
        representative MEMS-gyro bias.
    update_every_n_steps : sun/mag measurement update cadence relative
        to the integration/gyro step -- vector sensors are typically
        slower than the gyro/IMU rate.
    """
    if true_gyro_bias is None:
        true_gyro_bias = np.radians(np.array([0.02, -0.015, 0.01]))  # rad/s

    orbit = CircularOrbit(cfg.orbit)
    J     = cfg.sc.J
    J_inv = np.linalg.inv(J)

    q_true     = quat.normalize(cfg.initial_quaternion.copy())
    omega_true = cfg.initial_omega.copy()

    # Filter starts off-attitude by a known rotation about an arbitrary
    # axis, with zero (wrong) bias estimate and correspondingly large
    # initial covariance.
    axis = np.array([1.0, 1.0, 0.0]) / np.linalg.norm([1.0, 1.0, 0.0])
    dtheta0 = np.radians(initial_attitude_error_deg) * axis
    q_est0 = quat.normalize(quat.multiply(
        np.array([np.cos(np.linalg.norm(dtheta0) / 2.0),
                  *(np.sin(np.linalg.norm(dtheta0) / 2.0) * axis)]),
        q_true,
    ))

    mekf = MEKF(q0=q_est0, dt=cfg.dt, bias0=np.zeros(3))
    mekf.P[0:3, 0:3] = np.eye(3) * np.radians(initial_attitude_error_deg / 3.0)**2

    gyro = Gyroscope(dt=cfg.dt)
    sun_sensor = SunSensor()
    magnetometer = Magnetometer()

    history = EstimationHistory()
    n_steps = int(cfg.duration / cfg.dt)
    report_interval = max(n_steps // 20, 1)

    for step in range(n_steps):
        t = step * cfg.dt

        r_eci, v_eci = orbit.propagate(t)

        # --- truth dynamics (disturbance torques only, no control) ---
        tau_dist, info = disturbance_total(q_true, r_eci, v_eci, t, cfg.sc, cfg.orbit)
        state = np.concatenate([q_true, omega_true, np.zeros(cfg.wheels.n_wheels)])
        state = rk4_step(
            state, cfg.dt, J, J_inv, tau_dist,
            np.zeros(cfg.wheels.n_wheels), cfg.wheels.J_wheel, cfg.wheels.axes,
        )
        q_true     = quat.normalize(state[0:4])
        omega_true = state[4:7]

        # --- truth-referenced environment vectors ---
        C_true   = quat.to_dcm(q_true)
        sun_eci  = env.sun_direction_eci(t)
        sun_body_true = C_true @ sun_eci
        eclipse  = env.is_in_eclipse(r_eci, sun_eci)
        B_eci    = env.magnetic_field_eci(r_eci, t)
        B_body_true = C_true @ B_eci

        # --- sensor measurements ---
        omega_meas = gyro.measure(omega_true + true_gyro_bias)
        # gyro.measure() also injects its own internal bias-walk noise
        # on top; from the filter's perspective this is simply "the
        # gyro reading", bias unknown.

        # --- MEKF time update ---
        mekf.predict(omega_meas)

        # --- MEKF measurement updates (slower cadence) ---
        if step % update_every_n_steps == 0:
            sun_meas = sun_sensor.measure(sun_body_true, eclipse)
            mekf.update_sun(sun_meas, sun_eci)

            mag_meas = magnetometer.measure(B_body_true)
            mekf.update_mag(mag_meas, B_eci)

        # --- scoring: true attitude error of the estimate ---
        q_hat, bias_hat = mekf.get_estimate()
        q_err = quat.error(q_true, q_hat)
        angle_err_deg = np.degrees(
            2.0 * np.arcsin(np.clip(np.linalg.norm(q_err[1:4]), 0.0, 1.0))
        )
        bias_err = np.linalg.norm(bias_hat - true_gyro_bias)
        sigma_theta_deg = np.degrees(np.sqrt(np.trace(mekf.P[0:3, 0:3]) / 3.0))

        history.record(t, angle_err_deg, bias_err, sigma_theta_deg)

        if step % report_interval == 0:
            pct = 100.0 * step / n_steps
            print(f"  t={t:8.1f}s  ({pct:5.1f}%)  "
                  f"attitude est. error = {angle_err_deg:7.3f}°  "
                  f"1sigma = {sigma_theta_deg:7.3f}°")

    print(f"  Done.  Final attitude est. error = "
          f"{history._lists['attitude_error_deg'][-1]:.4f}°")
    return history.finalise()
