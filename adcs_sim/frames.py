"""
Reference frame utilities.

Frames
------
ECI  (J2000) : x → vernal equinox, z → north pole.
LVLH         : x → along velocity (circular orbit),
               y → −orbit normal,
               z → nadir (toward Earth centre).

For a circular orbit the LVLH frame rotates at the orbital rate n
about its own y-axis, so  ω_LVLH/ECI expressed in LVLH = [0, −n, 0].
"""

import numpy as np
from . import quaternion as quat


def eci_to_lvlh_dcm(r_eci, v_eci):
    """
    DCM from ECI to LVLH:  v_lvlh = C @ v_eci.
    """
    r_hat = r_eci / np.linalg.norm(r_eci)

    h     = np.cross(r_eci, v_eci)
    h_hat = h / np.linalg.norm(h)

    z_lvlh = -r_hat                           # nadir
    y_lvlh = -h_hat                           # −orbit normal
    x_lvlh = np.cross(y_lvlh, z_lvlh)        # ≈ velocity direction

    return np.array([x_lvlh, y_lvlh, z_lvlh])


def lvlh_quaternion(r_eci, v_eci):
    """
    Quaternion representing LVLH orientation w.r.t. ECI.
    This is the desired attitude for nadir-pointing.
    """
    C = eci_to_lvlh_dcm(r_eci, v_eci)
    return quat.from_dcm(C)