# Satellite ADCS Simulation

Python simulation of a 3U-CubeSat attitude determination and control
system (ADCS): rigid-body attitude dynamics with reaction wheels and
magnetorquers, a nadir-pointing PD attitude controller, environmental
disturbance-torque modelling, and a multiplicative extended Kalman
filter (MEKF) for gyro/sun/magnetometer-based attitude and gyro-bias
estimation. Intended as a GNC learning project, built up in stages from
dynamics through control to estimation.

## Goals

- Model rigid-body attitude dynamics with reaction-wheel actuation on a
  circular low-Earth orbit.
- Simulate realistic disturbance torques (gravity-gradient, residual
  magnetic dipole, aerodynamic drag, solar radiation pressure).
- Close the loop with a quaternion-based PD controller for nadir
  pointing, including reaction-wheel torque/speed saturation and
  magnetorquer momentum dumping.
- Estimate attitude and gyro bias from noisy, blackout-prone sensors
  (gyroscope, sun sensor, magnetometer) with an MEKF, and verify
  convergence against the simulated truth trajectory.

## State and Frames

Attitude state is a scalar-first quaternion `q = [w, x, y, z]` (Hamilton
convention) representing the rotation from ECI to body, plus body-frame
angular velocity `ω` and reaction-wheel spin rates. See
[`adcs_sim/quaternion.py`](adcs_sim/quaternion.py) for the exact
convention and [`adcs_sim/frames.py`](adcs_sim/frames.py) for the
ECI/LVLH definitions used by the nadir-pointing reference.

## Dynamics

Euler's rigid-body equation with reaction-wheel angular-momentum
coupling, integrated with a fixed-step RK4 that re-normalises the
quaternion each step. Control torque is held constant over each step
(zero-order hold), matching how a real digital controller would command
the actuators.

Implementation: [`adcs_sim/dynamics.py`](adcs_sim/dynamics.py)

## Disturbance Torques

Four environmental torque sources are modelled in body frame and summed
each step: gravity-gradient, residual magnetic dipole, aerodynamic drag
(exponential atmosphere), and solar radiation pressure (with a
cylindrical eclipse model).

Implementation: [`adcs_sim/disturbances.py`](adcs_sim/disturbances.py),
[`adcs_sim/environment.py`](adcs_sim/environment.py)

## Control

Quaternion-error PD control (`τ = −Kp·q_err_vec − Kd·ω_err`) against a
nadir-pointing LVLH reference, with reaction-wheel torque/speed
saturation and a B-cross magnetorquer law for momentum dumping when
wheel speeds build up.

Implementation: [`adcs_sim/control.py`](adcs_sim/control.py),
[`adcs_sim/actuators.py`](adcs_sim/actuators.py)

## Estimation

A multiplicative extended Kalman filter (MEKF) estimates attitude and
gyro bias from a gyroscope (propagated every step) and slower sun-sensor
/ magnetometer vector updates, including sun-sensor blackout during
eclipse. The filter uses an additive error state `[δθ, δbias]` in the
tangent space of SO(3):

```text
Predict (every gyro sample):
    ω = ω_meas − bias
    q  <- normalize(q + dt · 0.5 · Ω(ω) q)      (nonlinear quaternion propagation)
    Φ  = I + dt · [[-[ω×], -I], [0, 0]]          (error-state transition)
    P  <- Φ P Φ^T + Q·dt

Update (per vector sensor — sun direction or magnetic field):
    b_hat = C(q) v_ref                            (predicted body-frame vector)
    y     = z − b_hat                             (innovation)
    H     = [skew(b_hat), 0]
    S     = H P H^T + R
    K     = P H^T S^-1
    [δθ, δbias] = K y
    q     <- normalize(q ⊗ δq(δθ))                (reset: fold error into full state)
    bias  <- bias + δbias
    P     <- (I − KH) P (I − KH)^T + K R K^T      (Joseph form, as in the reentry_V EKF)
```

The reset-step quaternion composition order (`q ⊗ δq`, not `δq ⊗ q`) is
dictated by this module's Hamilton-product convention — verified
directly (`C(p ⊗ q) = C(q) C(p)` under `quaternion.multiply`/`to_dcm`)
rather than assumed, since getting it backwards silently produces a
filter that appears to run but diverges instead of converging.

**Verification:** [`adcs_sim/estimation_demo.py`](adcs_sim/estimation_demo.py)
runs a free-tumbling truth trajectory alongside the filter, started
15° off the true attitude with zero bias estimate against a nonzero
true gyro bias, and scores the true attitude/bias estimation error over
a full orbit. Convergence from a large initial offset down to a
sub-degree steady-state error (rather than divergence) is the pass/fail
signal for the filter's math, not just that it runs without crashing.

Implementation: [`adcs_sim/estimation.py`](adcs_sim/estimation.py)

## Scenarios

Run from the project root:

```bash
pip install -r requirements.txt   # numpy, matplotlib
python main.py
```

| # | Scenario | What it shows |
|---|---|---|
| 1 | Free tumble | Open-loop rigid-body dynamics under disturbance torques only |
| 2 | PD detumble → nadir point | Closed-loop attitude control from a tumbling initial state |
| 3 | PD + magnetorquer desaturation | Momentum dumping under sustained disturbance torque (disabled by default in `main.py`; wire in `enable_momentum_dumping` when ready to exercise it) |
| 4 | MEKF estimation | Truth-vs-estimate convergence for the attitude/gyro-bias filter |

Each scenario prints progress and produces a `matplotlib` figure
(pointing error, body rates, control torque, wheel speeds, disturbance
magnitude for 1–3; attitude/bias estimation error vs. a 3σ bound for 4).

## Repo Structure

```
SM_ADCS_2/
├── main.py                    # scenario entry points
├── adcs_sim/
│   ├── quaternion.py          # quaternion algebra, DCM conversions, kinematics
│   ├── frames.py               # ECI/LVLH frame utilities, nadir-pointing reference
│   ├── dynamics.py             # rigid-body EOM + reaction wheels, RK4 integrator
│   ├── disturbances.py         # gravity-gradient, magnetic, aero, SRP torques
│   ├── environment.py          # magnetic field, sun direction, eclipse, atmosphere
│   ├── control.py              # quaternion-error PD attitude control law
│   ├── actuators.py            # reaction-wheel and magnetorquer models
│   ├── sensors.py              # gyro/sun-sensor/magnetometer noise models
│   ├── estimation.py           # multiplicative EKF (attitude + gyro bias)
│   ├── estimation_demo.py      # truth-vs-estimate MEKF verification run
│   ├── sim_runner.py           # dynamics + control simulation loop
│   ├── config.py               # orbit/spacecraft/wheel/control parameters
│   └── plotting.py             # result plots
└── requirements.txt
```

## Notes

This is an AI-assisted learning project. AI tools were used to scaffold
structure, generate the initial estimator stub, implement the MEKF
measurement-update math and the truth-vs-estimate verification harness,
and write this documentation, while working through the dynamics,
control, and estimation concepts. It is intended to demonstrate ability
to build, reason about, and validate GNC simulation and estimation
tools — not as a flight-qualified ADCS design. Omitted effects include
elliptical/perturbed orbits, flexible-body dynamics, actuator
misalignment, and sensor calibration errors.
