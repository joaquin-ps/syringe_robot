"""
SyringeBot — 5-bar planar parallel manipulator kinematics.

A five-bar (2-DOF) closed-loop linkage with two actuated revolute joints
at the base.

    A ——————————————— B          (base, length L0)
    |                 |
   L1 (θ₁)          L3 (θ₂)     ← actuated
    |                 |
    C                 D
     \               /
     L2             L4           ← passive
       \           /
        \         /
          P (end-effector)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


# ─── Data structures ──────────────────────────────────────────────────────────

class JointPositions(NamedTuple):
    """All five joint/node positions of the linkage."""
    A: np.ndarray   # left  base  (fixed)
    B: np.ndarray   # right base  (fixed)
    C: np.ndarray   # left  elbow (end of L1)
    D: np.ndarray   # right elbow (end of L3)
    P: np.ndarray | None  # end-effector (None when mechanism can't close)


# ─── SyringeBot ──────────────────────────────────────────────────────────────

@dataclass
class SyringeBot:
    """
    Kinematic model of a 5-bar planar parallel manipulator.

    Parameters
    ----------
    L0 : float   Base distance between the two fixed ground joints.
    L1 : float   Left  proximal link  (A → C).
    L2 : float   Left  distal   link  (C → P).
    L3 : float   Right proximal link  (B → D).
    L4 : float   Right distal   link  (D → P).
    """
    L0: float = 3.0
    L1: float = 2.0
    L2: float = 2.5
    L3: float = 2.0
    L4: float = 2.5

    # actuated joint angles (radians)
    theta1: float = field(default=None)
    theta2: float = field(default=None)

    # +1 / -1  selects which of the two assembly modes to use
    elbow_sign: int = +1

    def __post_init__(self):
        if self.theta1 is None:
            self.theta1 = np.radians(70.0)
        if self.theta2 is None:
            self.theta2 = np.radians(110.0)

    # ── properties ─────────────────────────────────────────────────────────

    @property
    def link_lengths(self) -> dict[str, float]:
        return dict(L0=self.L0, L1=self.L1, L2=self.L2,
                    L3=self.L3, L4=self.L4)

    @link_lengths.setter
    def link_lengths(self, lengths: dict[str, float]):
        for k, v in lengths.items():
            if hasattr(self, k):
                setattr(self, k, float(v))

    @property
    def angles_deg(self) -> tuple[float, float]:
        return np.degrees(self.theta1), np.degrees(self.theta2)

    # ── base joint positions ─────────────────────────────────────────────

    def _base_joints(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the two fixed base joints A and B, centered at the origin."""
        half = self.L0 / 2.0
        return np.array([-half, 0.0]), np.array([half, 0.0])

    # ── forward kinematics ─────────────────────────────────────────────────

    def forward_kinematics(self) -> JointPositions:
        """
        Compute all joint positions from the current actuated angles.
        Returns a ``JointPositions`` named-tuple.
        ``P`` is *None* when the distal links cannot close.
        """
        A, B = self._base_joints()
        C = A + self.L1 * np.array([np.cos(self.theta1),
                                     np.sin(self.theta1)])
        D = B + self.L3 * np.array([np.cos(self.theta2),
                                     np.sin(self.theta2)])
        P = self._circle_intersection(C, self.L2, D, self.L4,
                                       self.elbow_sign)
        return JointPositions(A, B, C, D, P)

    def singularity_clearance(self) -> float:
        """
        Dimensionless margin in (0, ∞); larger is farther from closure singularity.

        Uses the distance between elbow joints C and D relative to the distal
        link lengths: near full extension or folded (distal circles nearly
        tangent), the mechanism loses leverage. Also uses the triangle height
        *h* of the closure construction (small *h* means near singular).

        Returns ``0.0`` if FK does not close (``P`` is ``None``).
        """
        j = self.forward_kinematics()
        if j.P is None:
            return 0.0
        C, D = j.C, j.D
        d = float(np.linalg.norm(D - C))
        r1, r2 = float(self.L2), float(self.L4)
        if d < 1e-12:
            return 0.0
        margin_dist = min(d - abs(r1 - r2), (r1 + r2) - d)
        margin_dist = max(margin_dist, 0.0)
        a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
        h = float(np.sqrt(max(r1 * r1 - a * a, 0.0)))
        scale = max(min(r1, r2), 1e-9)
        return max(min(margin_dist, h) / scale, 0.0)

    def is_near_singularity(self, threshold: float = 0.07) -> bool:
        """True if :meth:`singularity_clearance` is below *threshold* (tunable)."""
        return self.singularity_clearance() < float(threshold)

    # ── inverse kinematics ─────────────────────────────────────────────────

    def inverse_kinematics(self, px: float, py: float,
                           apply: bool = False) -> tuple[float | None,
                                                          float | None]:
        """
        Solve for actuated angles that place the end-effector at (px, py).

        Both IK solutions are computed for each chain; the one nearest to
        the current joint angle is chosen so that small Cartesian moves
        produce smooth, continuous joint motion.

        Parameters
        ----------
        px, py : target end-effector position.
        apply  : if *True*, store the solved angles and re-align
                 ``elbow_sign`` so FK reproduces the target.

        Returns (theta1, theta2) in radians, or (None, None) if unreachable.
        """
        A, B = self._base_joints()
        P = np.array([px, py])

        t1_pair = self._two_link_ik_both(A, self.L1, self.L2, P)
        t2_pair = self._two_link_ik_both(B, self.L3, self.L4, P)

        if t1_pair is None or t2_pair is None:
            return None, None

        t1 = min(t1_pair, key=lambda t: self._angle_dist(t, self.theta1))
        t2 = min(t2_pair, key=lambda t: self._angle_dist(t, self.theta2))

        if apply:
            self.theta1 = t1
            self.theta2 = t2
            self._align_elbow_sign(px, py)

        return t1, t2

    def _align_elbow_sign(self, px: float, py: float):
        """Pick the elbow_sign that makes FK reproduce the IK target."""
        target = np.array([px, py])
        for sign in [self.elbow_sign, -self.elbow_sign]:
            self.elbow_sign = sign
            P = self.forward_kinematics().P
            if P is not None and np.linalg.norm(P - target) < 1e-3:
                return

    def toggle_elbow(self):
        """Flip the assembly-mode sign."""
        self.elbow_sign *= -1

    # ── workspace ──────────────────────────────────────────────────────────

    def compute_workspace(self, n_samples: int = 200) -> np.ndarray:
        """
        Sweep both actuated angles and return all reachable end-effector
        positions as an (N, 2) array.  Fully vectorised — fast even for
        large *n_samples*.
        """
        th1 = np.linspace(-np.pi, np.pi, n_samples)
        th2 = np.linspace(-np.pi, np.pi, n_samples)
        T1, T2 = np.meshgrid(th1, th2)
        T1, T2 = T1.ravel(), T2.ravel()

        half = self.L0 / 2.0
        Cx = -half + self.L1 * np.cos(T1)
        Cy = self.L1 * np.sin(T1)
        Dx = half + self.L3 * np.cos(T2)
        Dy = self.L3 * np.sin(T2)

        dx, dy = Dx - Cx, Dy - Cy
        d = np.hypot(dx, dy)

        r1, r2 = self.L2, self.L4
        valid = ((d <= r1 + r2 + 1e-9)
                 & (d >= abs(r1 - r2) - 1e-9)
                 & (d > 1e-12))

        d_v = d[valid]
        dx_v, dy_v = dx[valid], dy[valid]
        Cx_v, Cy_v = Cx[valid], Cy[valid]

        a = (r1**2 - r2**2 + d_v**2) / (2.0 * d_v)
        h = np.sqrt(np.maximum(r1**2 - a**2, 0.0))
        ux, uy = dx_v / d_v, dy_v / d_v
        mx, my = Cx_v + a * ux, Cy_v + a * uy

        s = self.elbow_sign
        return np.column_stack([mx - s * h * uy, my + s * h * ux])

    # ── private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _circle_intersection(c1, r1, c2, r2, sign=+1):
        """Intersection of two circles; *sign* picks the solution."""
        d = np.hypot(c2[0] - c1[0], c2[1] - c1[1])
        if d > r1 + r2 + 1e-9 or d < abs(r1 - r2) - 1e-9 or d < 1e-12:
            return None
        a = (r1**2 - r2**2 + d**2) / (2.0 * d)
        h = np.sqrt(max(r1**2 - a**2, 0.0))
        ux, uy = (c2[0] - c1[0]) / d, (c2[1] - c1[1]) / d
        mx, my = c1[0] + a * ux, c1[1] + a * uy
        return np.array([mx - sign * h * uy, my + sign * h * ux])

    @staticmethod
    def _two_link_ik_both(base, la, lb, target):
        """Return both IK solutions for a 2-link chain, or None."""
        d = np.linalg.norm(target - base)
        if d > la + lb + 1e-9 or d < abs(la - lb) - 1e-9:
            return None
        ca = np.clip((la**2 + d**2 - lb**2) / (2 * la * d), -1.0, 1.0)
        alpha = np.arccos(ca)
        beta = np.arctan2(target[1] - base[1], target[0] - base[0])
        return beta + alpha, beta - alpha

    @staticmethod
    def _angle_dist(a: float, b: float) -> float:
        """Shortest unsigned angular distance."""
        return abs(np.arctan2(np.sin(a - b), np.cos(a - b)))
