"""
Entry point — run from the project root:   python main.py
"""
import sys
print(sys.executable)

import numpy as np
from adcs_sim.config   import SimConfig
from adcs_sim          import sim_runner
from adcs_sim.plotting import plot_results


def scenario_free_tumble():
    print("=" * 60)
    print("SCENARIO 1 — Free tumble with disturbance torques")
    print("=" * 60)
    cfg = SimConfig()
    cfg.control.mode = "off"
    cfg.duration = 6000.0           # ≈ 1 orbit
    hist = sim_runner.run(cfg)
    plot_results(hist, title="Free Tumble (no control)")
    return hist


def scenario_pd_detumble():
    print("=" * 60)
    print("SCENARIO 2 — PD detumble → nadir pointing")
    print("=" * 60)
    cfg = SimConfig()
    cfg.control.mode = "pd"
    cfg.control.Kp   = 5.0e-3
    cfg.control.Kd   = 8.0e-3
    cfg.duration      = 12000.0     # ≈ 2 orbits
    hist = sim_runner.run(cfg)
    plot_results(hist, title="PD Controller — Detumble & Nadir Point")
    return hist


def scenario_pd_with_desaturation():
    print("=" * 60)
    print("SCENARIO 3 — PD + magnetorquer momentum dumping")
    print("=" * 60)
    cfg = SimConfig()
    cfg.control.mode = "pd"
    cfg.control.Kp   = 5.0e-3
    cfg.control.Kd   = 8.0e-3
    cfg.enable_momentum_dumping = True
    cfg.duration      = 18000.0     # ≈ 3 orbits
    hist = sim_runner.run(cfg)
    plot_results(hist, title="PD + Momentum Dumping")
    return hist


if __name__ == "__main__":
    h1 = scenario_free_tumble()
    h2 = scenario_pd_detumble()
    # h3 = scenario_pd_with_desaturation()      # uncomment when ready