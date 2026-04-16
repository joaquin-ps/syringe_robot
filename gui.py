#!/usr/bin/env python3
"""
Interactive GUI for the SyringeBot 5-bar parallel manipulator.

Run:
    conda activate syringe_robot
    python gui.py

With U2D2 hardware (Protocol 2, 4M baud, IDs 11 / 21):

    python gui.py --dynamixel u2d2 --port /dev/ttyUSB0 --baud 4000000
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button

from syringe_bot import SyringeBot

try:
    from dynamixel_bridge import SyringeBotDynamixel
except ImportError:  # optional until env is installed
    SyringeBotDynamixel = None  # type: ignore[misc, assignment]


# ─── Colour palette ──────────────────────────────────────────────────────────

COLORS = {
    "L1": "#5b9bd5",
    "L2": "#6ab04c",
    "L3": "#c0504d",
    "L4": "#e8a838",
    "base": "#555555",
    "ee": "#e74c3c",
    "bg": "#f4f4f4",
    "ws": "#b3cde3",
}


# ─── GUI ──────────────────────────────────────────────────────────────────────

class SyringeBotGUI:
    """Matplotlib-based interactive GUI wrapping a :class:`SyringeBot`."""

    def __init__(
        self,
        bot: SyringeBot | None = None,
        *,
        dynamixel: SyringeBotDynamixel | None = None,
        singularity_threshold: float = 0.07,
    ):
        self.bot = bot or SyringeBot(
            L0=15.0,
            L1=15.0,
            L2=15.0,
            L3=15.0,
            L4=15.0,
            theta1=0.0,
            theta2=0.0,
        )
        self._suppress = False
        self._mode = "joint"  # "joint" or "cartesian"
        self._dxl = dynamixel
        self._dxl_status = ""
        # Dimensionless clearance (see SyringeBot.singularity_clearance); <0 disables blocking.
        self._singularity_threshold = float(singularity_threshold)
        self._singularity_msg = ""
        self._last_safe_theta = (float(self.bot.theta1), float(self.bot.theta2))

        self._ws_points: np.ndarray | None = None
        self._xlim: tuple[float, float] = (-15, 15)
        self._ylim: tuple[float, float] = (-5, 25)

        self._build_ui()
        self._recompute_workspace()
        self._apply_mode()          # disable the inactive group on start
        self._sync_xy_sliders()
        self.fig.canvas.mpl_connect("close_event", self._on_figure_close)
        if self._dxl is not None:
            self._dxl.configure_motors()
            self._sync_dynamixels()
        self._redraw()
        plt.show()

    # ── workspace & view limits ────────────────────────────────────────────

    def _recompute_workspace(self):
        self._ws_points = self.bot.compute_workspace(n_samples=250)

        L = self.bot.link_lengths
        reach_left = L["L1"] + L["L2"]
        reach_right = L["L3"] + L["L4"]
        max_reach = max(reach_left, reach_right)

        x_lo = min(0, -max_reach) - 0.5
        x_hi = max(L["L0"], L["L0"] + max_reach) + 0.5
        y_lo = -max_reach - 0.5
        y_hi = max_reach + 0.5

        if self._ws_points is not None and len(self._ws_points) > 0:
            ws = self._ws_points
            x_lo = min(x_lo, ws[:, 0].min()) - 0.5
            x_hi = max(x_hi, ws[:, 0].max()) + 0.5
            y_lo = min(y_lo, ws[:, 1].min()) - 0.5
            y_hi = max(y_hi, ws[:, 1].max()) + 0.5

        cx, cy = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
        half = max(x_hi - x_lo, y_hi - y_lo) / 2
        self._xlim = (cx - half, cx + half)
        self._ylim = (cy - half, cy + half)

        self._update_xy_slider_range()

    def _update_xy_slider_range(self):
        """Keep the X / Y slider ranges in sync with the workspace."""
        if not hasattr(self, "sl_x"):
            return
        for sl, (lo, hi) in [(self.sl_x, self._xlim),
                              (self.sl_y, self._ylim)]:
            sl.valmin, sl.valmax = lo, hi
            sl.ax.set_xlim(lo, hi)

    # ── layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.fig = plt.figure(figsize=(15, 9.6), facecolor=COLORS["bg"])
        self.fig.canvas.manager.set_window_title("SyringeBot")

        self.ax = self.fig.add_axes([0.05, 0.06, 0.55, 0.88])
        self.ax.set_aspect("equal")

        rx, rw, sh = 0.66, 0.30, 0.022

        def _sax(y):
            return self.fig.add_axes([rx, y, rw, sh])

        # ── joint-angle sliders ──
        self._lbl_joint = self.fig.text(
            rx + rw / 2, 0.965, "Joint Angles  ◄ active",
            ha="center", fontsize=10.5, fontweight="bold", color="#333")

        t1_deg, t2_deg = self.bot.angles_deg
        self.sl_t1 = Slider(_sax(0.93), "θ₁  °", -180, 180,
                            valinit=t1_deg, valstep=0.5,
                            color=COLORS["L1"])
        self.sl_t2 = Slider(_sax(0.895), "θ₂  °", -180, 180,
                            valinit=t2_deg, valstep=0.5,
                            color=COLORS["L3"])

        # ── cartesian sliders ──
        self._lbl_cart = self.fig.text(
            rx + rw / 2, 0.87, "Cartesian Position",
            ha="center", fontsize=10.5, fontweight="bold", color="#999")

        joints = self.bot.forward_kinematics()
        ix = joints.P[0] if joints.P is not None else 1.5
        iy = joints.P[1] if joints.P is not None else 2.0

        self.sl_x = Slider(_sax(0.835), "X", -8, 8,
                           valinit=ix, valstep=0.01,
                           color=COLORS["ee"])
        self.sl_y = Slider(_sax(0.80), "Y", -8, 8,
                           valinit=iy, valstep=0.01,
                           color=COLORS["ee"])

        # ── mode + elbow buttons ──
        self.btn_mode = Button(
            self.fig.add_axes([rx + 0.02, 0.76, 0.12, 0.032]),
            "Mode: Joint", color="#7986cb", hovercolor="#9fa8da")
        self.btn_mode.label.set_color("white")
        self.btn_mode.label.set_fontweight("bold")

        self.btn_elbow = Button(
            self.fig.add_axes([rx + 0.17, 0.76, 0.12, 0.032]),
            f'Elbow: {"+" if self.bot.elbow_sign == 1 else "−"}',
            color=COLORS["L4"], hovercolor="#f5d78e")
        self.btn_elbow.label.set_fontweight("bold")

        # ── link-length sliders ──
        self.fig.text(rx + rw / 2, 0.74, "Link Lengths",
                      ha="center", fontsize=10.5, fontweight="bold",
                      color="#333")

        link_cfg = [
            ("L₀ base", "L0", "#888888", 0.705),
            ("L₁",      "L1", COLORS["L1"], 0.67),
            ("L₂",      "L2", COLORS["L2"], 0.635),
            ("L₃",      "L3", COLORS["L3"], 0.60),
            ("L₄",      "L4", COLORS["L4"], 0.565),
        ]
        self.sl_L = {}
        lengths = self.bot.link_lengths
        for label, key, col, y in link_cfg:
            self.sl_L[key] = Slider(_sax(y), label, 0.2, 25.0,
                                    valinit=lengths[key],
                                    valstep=0.05, color=col)

        # ── end-effector text input (IK) ──
        self.fig.text(rx + rw / 2, 0.535,
                      "End-Effector  (type + Go)",
                      ha="center", fontsize=9.5, fontweight="bold",
                      color="#333")

        bx = rx + 0.01
        ty = 0.50
        self.fig.text(bx, ty + 0.008, "X:", fontsize=10,
                      va="center", fontweight="bold")
        self.tb_x = TextBox(
            self.fig.add_axes([bx + 0.035, ty - 0.005, 0.09, 0.026]),
            "", initial="")
        self.fig.text(bx + 0.15, ty + 0.008, "Y:", fontsize=10,
                      va="center", fontweight="bold")
        self.tb_y = TextBox(
            self.fig.add_axes([bx + 0.185, ty - 0.005, 0.09, 0.026]),
            "", initial="")

        self.btn_go = Button(
            self.fig.add_axes([rx + 0.06, 0.455, 0.18, 0.032]),
            "Go  (IK)", color=COLORS["L1"], hovercolor="#7ec8e3")
        self.btn_go.label.set_color("white")
        self.btn_go.label.set_fontweight("bold")

        # ── info panel ──
        self.info = self.fig.text(
            rx + rw / 2, 0.41, "", ha="center", va="top",
            fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#ccc"))

        # ── callbacks ──
        self.sl_t1.on_changed(self._on_angle)
        self.sl_t2.on_changed(self._on_angle)
        self.sl_x.on_changed(self._on_cartesian)
        self.sl_y.on_changed(self._on_cartesian)
        for s in self.sl_L.values():
            s.on_changed(self._on_link)
        self.btn_go.on_clicked(self._on_ik_text)
        self.btn_mode.on_clicked(self._on_mode_toggle)
        self.btn_elbow.on_clicked(self._on_elbow)

    # ── slider enable / disable ──────────────────────────────────────────

    @staticmethod
    def _set_sliders_enabled(sliders, enabled):
        """Enable or disable (grey-out) a list of Slider widgets."""
        for sl in sliders:
            sl.set_active(enabled)
            sl.poly.set_alpha(1.0 if enabled else 0.2)
            sl.valtext.set_alpha(1.0 if enabled else 0.4)

    def _apply_mode(self):
        """Enable the active slider group and disable the other."""
        joint_sl = [self.sl_t1, self.sl_t2]
        cart_sl = [self.sl_x, self.sl_y]

        if self._mode == "joint":
            self._set_sliders_enabled(joint_sl, True)
            self._set_sliders_enabled(cart_sl, False)
            self._lbl_joint.set_text("Joint Angles  ◄ active")
            self._lbl_joint.set_color("#333")
            self._lbl_cart.set_text("Cartesian Position")
            self._lbl_cart.set_color("#999")
            self.btn_mode.label.set_text("Mode: Joint")
        else:
            self._set_sliders_enabled(joint_sl, False)
            self._set_sliders_enabled(cart_sl, True)
            self._lbl_joint.set_text("Joint Angles")
            self._lbl_joint.set_color("#999")
            self._lbl_cart.set_text("Cartesian Position  ◄ active")
            self._lbl_cart.set_color("#333")
            self.btn_mode.label.set_text("Mode: Cartesian")

    # ── slider sync helpers ────────────────────────────────────────────────

    def _sync_xy_sliders(self):
        """Push current end-effector position into the X / Y sliders."""
        joints = self.bot.forward_kinematics()
        if joints.P is None:
            return
        self._suppress = True
        x, y = float(joints.P[0]), float(joints.P[1])
        x = np.clip(x, self.sl_x.valmin, self.sl_x.valmax)
        y = np.clip(y, self.sl_y.valmin, self.sl_y.valmax)
        self.sl_x.set_val(x)
        self.sl_y.set_val(y)
        self._suppress = False

    def _sync_angle_sliders(self):
        """Push current joint angles into the θ sliders."""
        self._suppress = True
        self.sl_t1.set_val(np.degrees(self.bot.theta1))
        self.sl_t2.set_val(np.degrees(self.bot.theta2))
        self._suppress = False

    def _sync_dynamixels(self):
        """Stream current θ₁, θ₂ to the left/right motors (IDs set on the interface)."""
        if self._dxl is None:
            return
        try:
            self._dxl.push_joint_angles(self.bot.theta1, self.bot.theta2)
            self._dxl_status = ""
        except Exception as exc:
            self._dxl_status = f"Dynamixel: {exc}"

    def _on_figure_close(self, _event):
        if self._dxl is not None:
            try:
                self._dxl.close()
            except Exception:
                pass
            self._dxl = None

    def _singularity_guard_enabled(self) -> bool:
        return self._singularity_threshold >= 0.0

    def _guard_apply_joint_angles(self, t1: float, t2: float) -> bool:
        """Apply θ₁,θ₂ if closure is away from singularity; else revert sliders."""
        if not self._singularity_guard_enabled():
            self.bot.theta1, self.bot.theta2 = t1, t2
            self._last_safe_theta = (t1, t2)
            self._singularity_msg = ""
            return True
        prev = self._last_safe_theta
        self.bot.theta1, self.bot.theta2 = t1, t2
        bad = (
            self.bot.forward_kinematics().P is None
            or self.bot.is_near_singularity(self._singularity_threshold)
        )
        if bad:
            self.bot.theta1, self.bot.theta2 = prev
            self._suppress = True
            self.sl_t1.set_val(np.degrees(self.bot.theta1))
            self.sl_t2.set_val(np.degrees(self.bot.theta2))
            self._suppress = False
            c = self.bot.singularity_clearance()
            self._singularity_msg = (
                f"Near singularity (margin={c:.3f}, "
                f"need ≥{self._singularity_threshold:.3f}) — move blocked"
            )
            return False
        self._last_safe_theta = (t1, t2)
        self._singularity_msg = ""
        return True

    def _set_cartesian_sliders(self, px: float, py: float) -> None:
        """Set XY sliders without triggering callbacks."""
        self._suppress = True
        self.sl_x.set_val(float(np.clip(px, self.sl_x.valmin, self.sl_x.valmax)))
        self.sl_y.set_val(float(np.clip(py, self.sl_y.valmin, self.sl_y.valmax)))
        self._suppress = False

    def _restore_safe_pose(
        self,
        safe_t: tuple[float, float],
        safe_p: np.ndarray | None,
        message: str,
    ) -> None:
        self.bot.theta1, self.bot.theta2 = safe_t
        self._last_safe_theta = safe_t
        self._suppress = True
        self.sl_t1.set_val(np.degrees(safe_t[0]))
        self.sl_t2.set_val(np.degrees(safe_t[1]))
        self._suppress = False
        if safe_p is not None:
            self._set_cartesian_sliders(float(safe_p[0]), float(safe_p[1]))
        self._singularity_msg = message

    def _follow_cartesian_path(self, px: float, py: float) -> bool:
        """
        Track the Cartesian slider path incrementally to avoid IK branch jumps.

        If a singular/unreachable point is encountered, clamp at last safe point.
        """
        start_p = self.bot.forward_kinematics().P
        if start_p is None:
            self._singularity_msg = "Current pose is invalid; cannot move in Cartesian mode"
            return False

        safe_t = (float(self.bot.theta1), float(self.bot.theta2))
        safe_p = np.array(start_p, dtype=float)
        dx, dy = float(px - start_p[0]), float(py - start_p[1])
        dist = float(np.hypot(dx, dy))
        n_steps = max(1, min(120, int(np.ceil(dist / 0.08))))

        for i in range(1, n_steps + 1):
            a = i / n_steps
            tx = float(start_p[0] + a * dx)
            ty = float(start_p[1] + a * dy)
            t1, t2 = self.bot.inverse_kinematics(tx, ty, apply=True)
            if t1 is None or t2 is None:
                self._restore_safe_pose(
                    safe_t,
                    safe_p,
                    "Target path left workspace — clamped at last safe point",
                )
                return False
            if self._singularity_guard_enabled() and self.bot.is_near_singularity(
                self._singularity_threshold
            ):
                c = self.bot.singularity_clearance()
                self._restore_safe_pose(
                    safe_t,
                    safe_p,
                    (
                        f"Near singularity (margin={c:.3f}, "
                        f"need ≥{self._singularity_threshold:.3f}) — clamped"
                    ),
                )
                return False
            safe_t = (float(self.bot.theta1), float(self.bot.theta2))
            cur_p = self.bot.forward_kinematics().P
            if cur_p is not None:
                safe_p = np.array(cur_p, dtype=float)

        self._last_safe_theta = safe_t
        self._singularity_msg = ""
        return True

    # ── callbacks ──────────────────────────────────────────────────────────

    def _on_angle(self, _=None):
        if self._suppress:
            return
        t1, t2 = np.radians(self.sl_t1.val), np.radians(self.sl_t2.val)
        if not self._guard_apply_joint_angles(t1, t2):
            self._sync_xy_sliders()
            self._sync_dynamixels()
            self._redraw()
            return
        self._sync_xy_sliders()
        self._sync_dynamixels()
        self._redraw()

    def _on_cartesian(self, _=None):
        if self._suppress:
            return
        px, py = self.sl_x.val, self.sl_y.val
        self._follow_cartesian_path(px, py)
        self._sync_angle_sliders()
        self._sync_dynamixels()
        self._redraw()

    def _on_link(self, _=None):
        prev_L = dict(self.bot.link_lengths)
        self.bot.link_lengths = {k: s.val for k, s in self.sl_L.items()}
        self._recompute_workspace()
        if self._singularity_guard_enabled() and (
            self.bot.forward_kinematics().P is None
            or self.bot.is_near_singularity(self._singularity_threshold)
        ):
            self.bot.link_lengths = prev_L
            self._suppress = True
            for k, s in self.sl_L.items():
                s.set_val(prev_L[k])
            self._suppress = False
            self._recompute_workspace()
            self._singularity_msg = (
                "Link change would reach a singular pose — reverted"
            )
            self._redraw()
            return
        self._singularity_msg = ""
        self._sync_xy_sliders()
        self._redraw()

    def _on_ik_text(self, _=None):
        try:
            px, py = float(self.tb_x.text), float(self.tb_y.text)
        except ValueError:
            self._set_info("  Enter valid numeric X and Y.  ")
            self.fig.canvas.draw_idle()
            return

        self._follow_cartesian_path(px, py)
        self._sync_angle_sliders()
        self._sync_xy_sliders()
        self._sync_dynamixels()
        self._redraw()

    def _on_mode_toggle(self, _=None):
        self._mode = "cartesian" if self._mode == "joint" else "joint"
        self._apply_mode()
        self.fig.canvas.draw_idle()

    def _on_elbow(self, _=None):
        prev_sign = self.bot.elbow_sign
        self.bot.toggle_elbow()
        sign_ch = "+" if self.bot.elbow_sign == 1 else "−"
        self.btn_elbow.label.set_text(f"Elbow: {sign_ch}")
        self._recompute_workspace()
        if self._singularity_guard_enabled() and (
            self.bot.forward_kinematics().P is None
            or self.bot.is_near_singularity(self._singularity_threshold)
        ):
            self.bot.elbow_sign = prev_sign
            sign_ch = "+" if self.bot.elbow_sign == 1 else "−"
            self.btn_elbow.label.set_text(f"Elbow: {sign_ch}")
            self._recompute_workspace()
            self._singularity_msg = "Elbow mode would be singular — reverted"
            self._redraw()
            return
        self._singularity_msg = ""
        self._sync_xy_sliders()
        self._sync_dynamixels()
        self._redraw()

    # ── drawing ────────────────────────────────────────────────────────────

    def _redraw(self):
        joints = self.bot.forward_kinematics()
        A, B, C, D, P = joints
        L = self.bot.link_lengths

        ax = self.ax
        ax.cla()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_title("SyringeBot — 5-Bar Parallel Manipulator",
                      fontsize=13, fontweight="bold", pad=10)

        if self._ws_points is not None and len(self._ws_points) > 0:
            ax.scatter(self._ws_points[:, 0], self._ws_points[:, 1],
                       s=0.4, c=COLORS["ws"], alpha=0.35,
                       edgecolors="none", zorder=1, label="Workspace")

        ax.plot([A[0], B[0]], [A[1], B[1]],
                color=COLORS["base"], linewidth=7,
                solid_capstyle="butt", zorder=2)
        gw = L["L0"] * 0.05
        for base_pt in (A, B):
            for k in np.linspace(-3, 3, 7):
                x0 = base_pt[0] + k * gw
                ax.plot([x0 - gw * 0.5, x0 + gw * 0.5],
                        [base_pt[1] - gw * 2.0, base_pt[1] - gw * 0.4],
                        color=COLORS["base"], linewidth=0.7, zorder=2)
            ax.plot(*base_pt, "s", color="black", markersize=10, zorder=5)

        if P is not None:
            self._draw_valid(ax, A, B, C, D, P)
        else:
            self._draw_invalid(ax, A, B, C, D, L)

        ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
        ax.set_xlim(self._xlim)
        ax.set_ylim(self._ylim)

        self.fig.canvas.draw_idle()

    def _draw_valid(self, ax, A, B, C, D, P):
        for p1, p2, col, lbl in [
            (A, C, COLORS["L1"], "L₁"),
            (C, P, COLORS["L2"], "L₂"),
            (B, D, COLORS["L3"], "L₃"),
            (D, P, COLORS["L4"], "L₄"),
        ]:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-",
                    color=col, linewidth=4, solid_capstyle="round",
                    label=lbl, zorder=3)

        for pt, col in [(C, COLORS["L1"]), (D, COLORS["L3"])]:
            ax.plot(*pt, "o", color=col, markersize=10,
                    mec="white", mew=1.5, zorder=5)

        ax.plot(*P, "o", color=COLORS["ee"], markersize=15,
                mec="white", mew=2, zorder=6)
        ax.annotate(f"  ({P[0]:.2f}, {P[1]:.2f})",
                    xy=P, fontsize=9, fontweight="bold",
                    color="#c0392b", va="bottom")

        t1d, t2d = self.bot.angles_deg
        sign_ch = "+" if self.bot.elbow_sign == 1 else "−"
        msg = (
            f"End-Effector  X={P[0]:+.3f}  Y={P[1]:+.3f}\n"
            f"θ₁ = {t1d:+8.2f}°\n"
            f"θ₂ = {t2d:+8.2f}°\n"
            f"Elbow mode: {sign_ch}")
        if self._dxl_status:
            msg += f"\n{self._dxl_status}"
        if self._singularity_msg:
            msg += f"\n{self._singularity_msg}"
        self._set_info(msg)

    def _draw_invalid(self, ax, A, B, C, D, L):
        ax.plot([A[0], C[0]], [A[1], C[1]], "--",
                color=COLORS["L1"], linewidth=2, zorder=3, label="L₁")
        ax.plot([B[0], D[0]], [B[1], D[1]], "--",
                color=COLORS["L3"], linewidth=2, zorder=3, label="L₃")

        for center, r, col in [(C, L["L2"], COLORS["L2"]),
                                (D, L["L4"], COLORS["L4"])]:
            ax.add_patch(plt.Circle(
                center, r, fill=False, linestyle=":",
                color=col, linewidth=1.2, zorder=3))

        ax.text(0.5, 0.5, "No closure — links cannot reach",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="red", fontweight="bold", alpha=0.7)
        msg = "Configuration impossible\n(distal links cannot close)"
        if self._dxl_status:
            msg += f"\n{self._dxl_status}"
        if self._singularity_msg:
            msg += f"\n{self._singularity_msg}"
        self._set_info(msg)

    def _set_info(self, text):
        self.info.set_text(text)


# ─── entry point ──────────────────────────────────────────────────────────────


def _build_dynamixel_backend(
    mode: str,
    port: str,
    baudrate: int,
    *,
    profile_velocity: int,
    p_gain: int,
    scale_theta1: float,
    scale_theta2: float,
):
    if SyringeBotDynamixel is None:
        raise SystemExit(
            "dynamixel_bridge / dynamixel_u2d2 not installed. "
            "Install pip deps from environment.yaml."
        )
    motor_ids = [11, 21]  # θ₁ left, θ₂ right
    if mode == "fake":
        from dynamixel_u2d2 import FakeU2D2Interface

        iface = FakeU2D2Interface(
            usb_port=port,
            baudrate=baudrate,
            motor_ids=motor_ids,
            protocol_version=2.0,
        )
    else:
        from dynamixel_u2d2 import U2D2Interface

        iface = U2D2Interface(
            port,
            baudrate=baudrate,
            motor_ids=motor_ids,
            protocol_version=2.0,
        )
    return SyringeBotDynamixel(
        iface,
        profile_velocity=profile_velocity,
        position_p_gain=p_gain,
        scale_theta1=scale_theta1,
        scale_theta2=scale_theta2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SyringeBot GUI with optional Dynamixel sync.")
    parser.add_argument(
        "--dynamixel",
        choices=("off", "u2d2", "fake"),
        default="off",
        help="Stream joint angles to hardware (U2D2) or a fake bus for testing.",
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial device for U2D2 (ignored for --dynamixel fake).",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=4_000_000,
        help="Dynamixel bus baud rate (Protocol 2; use 4000000 for your setup).",
    )
    parser.add_argument(
        "--dxl-profile-velocity",
        type=int,
        default=400,
        help="Profile velocity register (model-specific units; lower = slower moves).",
    )
    parser.add_argument(
        "--dxl-p-gain",
        type=int,
        default=800,
        help="Position P gain after torque is disabled for mode setup.",
    )
    parser.add_argument(
        "--dxl-scale1",
        type=float,
        default=1.0,
        help="Multiply θ₁ (motor 11) before encoder mapping; use -1 if mounted inverted.",
    )
    parser.add_argument(
        "--dxl-scale2",
        type=float,
        default=1.0,
        help="Multiply θ₂ (motor 21) before encoder mapping; use -1 if mounted inverted.",
    )
    parser.add_argument(
        "--singularity-threshold",
        type=float,
        default=0.07,
        help="Block poses when singularity_clearance() falls below this (dimensionless). "
        "Use a negative value to disable (e.g. -1).",
    )
    args = parser.parse_args()

    dxl = None
    if args.dynamixel != "off":
        dxl = _build_dynamixel_backend(
            args.dynamixel,
            args.port,
            args.baud,
            profile_velocity=args.dxl_profile_velocity,
            p_gain=args.dxl_p_gain,
            scale_theta1=args.dxl_scale1,
            scale_theta2=args.dxl_scale2,
        )

    bot = SyringeBot(
        L0=15.0,
        L1=15.0,
        L2=15.0,
        L3=15.0,
        L4=15.0,
        theta1=np.radians(90.0),
        theta2=np.radians(90.0),
    )
    SyringeBotGUI(
        bot,
        dynamixel=dxl,
        singularity_threshold=args.singularity_threshold,
    )
