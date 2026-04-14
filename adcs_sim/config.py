"""
Simulation configuration.

All parameters in SI units unless noted otherwise.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class OrbitalConfig:
    altitude: float = 550e3                             # m above surface
    inclination: float = np.radians(51.6)               # rad (ISS-like)
    raan: float = 0.0                                   # rad
    mu: float = 3.986004418e14                          # m^3/s^2
    R_earth: float = 6.371e6                            # m

    @property
    def a(self):
        return self.R_earth + self.altitude

    @property
    def n(self):
        return np.sqrt(self.mu / self.a**3)

    @property
    def period(self):
        return 2.0 * np.pi / self.n

    @property
    def v_circular(self):
        return np.sqrt(self.mu / self.a)


@dataclass
class SpacecraftConfig:
    mass: float = 4.0                                   # kg (3U CubeSat)
    J: np.ndarray = field(default_factory=lambda: np.array([
        [0.0270,  0.0005,  0.0002],
        [0.0005,  0.0260,  0.0003],
        [0.0002,  0.0003,  0.0090],
    ]))
    residual_dipole: np.ndarray = field(
        default_factory=lambda: np.array([0.005, 0.005, 0.005])    # A·m²
    )
    drag_area: float = 0.03                             # m²
    Cd: float = 2.2
    cp_offset: np.ndarray = field(
        default_factory=lambda: np.array([0.002, 0.001, 0.0])      # m
    )
    srp_area: float = 0.03                              # m²
    srp_cp_offset: np.ndarray = field(
        default_factory=lambda: np.array([0.001, 0.0, 0.001])
    )
    reflectivity: float = 0.6


@dataclass
class WheelConfig:
    n_wheels: int = 3
    J_wheel: float = 2.0e-5                             # kg·m² per wheel
    max_torque: float = 1.0e-3                           # N·m
    max_speed: float = 6000.0 * 2.0 * np.pi / 60.0      # rad/s
    axes: np.ndarray = field(default_factory=lambda: np.eye(3))


@dataclass
class MagnetorquerConfig:
    max_dipole: float = 0.2                              # A·m²


@dataclass
class ControlConfig:
    mode: str = "pd"            # "off", "pd"
    Kp: float = 5.0e-3
    Kd: float = 8.0e-3


@dataclass
class SimConfig:
    dt: float = 0.1                                      # s
    duration: float = 6000.0                             # s
    initial_quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    initial_omega: np.ndarray = field(
        default_factory=lambda: np.radians(np.array([5.0, -3.0, 4.0]))
    )
    initial_wheel_speeds: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    enable_momentum_dumping: bool = False

    orbit: OrbitalConfig = field(default_factory=OrbitalConfig)
    sc: SpacecraftConfig = field(default_factory=SpacecraftConfig)
    wheels: WheelConfig = field(default_factory=WheelConfig)
    magnetorquer: MagnetorquerConfig = field(default_factory=MagnetorquerConfig)
    control: ControlConfig = field(default_factory=ControlConfig)