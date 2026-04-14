"""
Standard result plots.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_results(hist, title="ADCS Simulation"):
    t = hist.t / 60.0                        # minutes

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=14, y=0.98)

    # --- pointing error ---
    ax = axes[0, 0]
    ax.plot(t, hist.pointing_error_deg, linewidth=0.6)
    ax.set_ylabel("Pointing error (°)")
    ax.set_title("Total pointing error")
    ax.grid(True, alpha=0.3)

    # --- per-axis attitude error ---
    ax = axes[0, 1]
    for i, lbl in enumerate(["Roll", "Pitch", "Yaw"]):
        ax.plot(t, hist.euler_error_deg[:, i], linewidth=0.6, label=lbl)
    ax.set_ylabel("Attitude error (°)")
    ax.set_title("Per-axis error (small-angle approx)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- angular velocity ---
    ax = axes[1, 0]
    omega_dps = np.degrees(hist.omega)
    for i, lbl in enumerate(["ωx", "ωy", "ωz"]):
        ax.plot(t, omega_dps[:, i], linewidth=0.6, label=lbl)
    ax.set_ylabel("Angular velocity (°/s)")
    ax.set_title("Body rates")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- control torque ---
    ax = axes[1, 1]
    for i, lbl in enumerate(["τx", "τy", "τz"]):
        ax.plot(t, hist.tau_ctrl[:, i] * 1e6, linewidth=0.6, label=lbl)
    ax.set_ylabel("Control torque (μN·m)")
    ax.set_title("Reaction-wheel torque")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- wheel speeds ---
    ax = axes[2, 0]
    rpm = hist.wheel_speeds * 60.0 / (2.0 * np.pi)
    for i in range(rpm.shape[1]):
        ax.plot(t, rpm[:, i], linewidth=0.6, label=f"Wheel {i+1}")
    ax.set_ylabel("Wheel speed (RPM)")
    ax.set_title("Reaction-wheel momentum")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- disturbance magnitude ---
    ax = axes[2, 1]
    mag = np.linalg.norm(hist.tau_dist, axis=1) * 1e6
    ax.plot(t, mag, linewidth=0.6)
    ax.set_ylabel("|τ_dist| (μN·m)")
    ax.set_title("Disturbance torque magnitude")
    ax.grid(True, alpha=0.3)

    for a in axes.flat:
        a.set_xlabel("Time (min)")

    plt.tight_layout()
    plt.show()