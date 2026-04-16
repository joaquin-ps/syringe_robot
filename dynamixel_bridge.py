"""
Map SyringeBot joint angles to Dynamixel goal positions and drive two motors via U2D2.

Encoder value 2048 is treated as 0 rad in the GUI frame. One revolution = 4096 ticks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from dynamixel_u2d2.base_interface import BaseInterface

ENCODER_CENTER = 1024 # keep it as 1024, it is correct
TICKS_PER_REV = 4096


def joint_radians_to_encoder(theta: float, scale: float = 1.0) -> int:
    """Convert joint angle (rad) to raw Dynamixel position; 2048 ↔0 rad."""
    t = float(scale) * float(theta)
    raw = ENCODER_CENTER + (t / (2.0 * np.pi)) * TICKS_PER_REV
    return int(round(raw)) % TICKS_PER_REV


class SyringeBotDynamixel:
    """
    Wraps ``U2D2Interface`` or ``FakeU2D2Interface`` with motor setup and angle streaming.

    ``motor_ids`` order is (left θ₁, right θ₂), default (11, 21).
    """

    def __init__(
        self,
        iface: BaseInterface,
        *,
        profile_velocity: int = 1000,
        position_p_gain: int = 1000,
        position_d_gain: int = 30,
        scale_theta1: float = 1.0,
        scale_theta2: float = 1.0,
    ):
        self._iface = iface
        self._profile_velocity = profile_velocity
        self._position_p_gain = position_p_gain
        self._position_d_gain = position_d_gain
        self._scale_theta1 = scale_theta1
        self._scale_theta2 = scale_theta2

    def configure_motors(self) -> None:
        mids = self._iface.motor_ids
        if not mids:
            raise RuntimeError("motor_ids must be set on the interface")
        for mid in mids:
            self._iface.disable_torque(mid)
            self._iface.set_motor_mode(mid, "position")
            self._iface.set_velocity_limit(mid, self._profile_velocity)
            self._iface.set_position_i_gain(mid, 0)
            self._iface.set_position_d_gain(mid, self._position_d_gain)
            self._iface.set_position_p_gain(mid, self._position_p_gain)
            self._iface.enable_torque(mid)

    def goal_encoders(self, theta1: float, theta2: float) -> Tuple[int, int]:
        e1 = joint_radians_to_encoder(theta1, self._scale_theta1)
        e2 = joint_radians_to_encoder(theta2, self._scale_theta2)
        return e1, e2

    def push_joint_angles(self, theta1: float, theta2: float) -> None:
        e1, e2 = self.goal_encoders(theta1, theta2)
        self._iface.sync_write_positions([e1, e2])

    def close(self) -> None:
        self._iface.close()
