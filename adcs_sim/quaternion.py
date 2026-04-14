"""
Quaternion utilities.

Convention
----------
Scalar-first: q = [w, x, y, z].
Hamilton multiplication.
q encodes the body orientation relative to ECI.

    R(q)  maps body vectors to ECI   (v_eci  = R(q) @ v_body).
    C(q)  maps ECI vectors to body   (v_body = C(q) @ v_eci), where C = R^T.

Kinematic equation:  qdot = 0.5 * Omega(omega_body) @ q.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def multiply(p, q):
    """Hamilton product p ⊗ q."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def conjugate(q):
    """Quaternion conjugate (= inverse for unit quaternion)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def normalize(q):
    """Return unit quaternion."""
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def to_dcm(q):
    """
    Direction cosine matrix C such that  v_body = C @ v_eci.
    This is R(q)^T where R is the standard rotation matrix.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y + w*z),      2*(x*z - w*y)     ],
        [2*(x*y - w*z),      1 - 2*(x*x + z*z),  2*(y*z + w*x)     ],
        [2*(x*z + w*y),      2*(y*z - w*x),       1 - 2*(x*x + y*y)],
    ])


def from_dcm(C):
    """
    Extract scalar-first quaternion from a DCM that maps ECI to body.
    Uses Shepperd's method for numerical stability.
    """
    R = C.T                           # R maps body to ECI (standard rotation matrix)
    tr = np.trace(R)

    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    # Enforce positive scalar part (canonical form)
    if q[0] < 0:
        q = -q
    return normalize(q)


def to_euler_321(q):
    """
    Convert to Euler angles (roll, pitch, yaw) via 3-2-1 sequence.
    For human-readable output only — do NOT use for dynamics.
    """
    C = to_dcm(q)
    pitch = np.arcsin(np.clip(-C[0, 2], -1.0, 1.0))
    roll  = np.arctan2(C[1, 2], C[2, 2])
    yaw   = np.arctan2(C[0, 1], C[0, 0])
    return np.array([roll, pitch, yaw])


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def omega_matrix(omega):
    """
    4×4 matrix Ω such that  qdot = 0.5 * Ω(ω) @ q.
    omega is the body-frame angular velocity.
    """
    wx, wy, wz = omega
    return np.array([
        [ 0,  -wx, -wy, -wz],
        [wx,   0,   wz, -wy],
        [wy,  -wz,  0,   wx],
        [wz,   wy, -wx,  0 ],
    ])


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

def error(q_current, q_desired):
    """
    Error quaternion:  q_err = q_current ⊗ q_desired*.

    For small errors  q_err ≈ [1, δθ_x/2, δθ_y/2, δθ_z/2]
    so the vector part is proportional to the attitude error.
    The sign of the scalar part is enforced positive to ensure
    shortest-path rotation.
    """
    q_err = multiply(q_current, conjugate(q_desired))
    if q_err[0] < 0:
        q_err = -q_err
    return q_err