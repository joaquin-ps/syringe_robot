#!/usr/bin/env python3
"""
Draw and replay trajectories for SyringeBot (sim first, robot optional).
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
try:
    from scipy.interpolate import CubicSpline, splprep, splev
except Exception:  # scipy is optional; fall back to linear resampling
    CubicSpline = None
    splprep = None
    splev = None

from syringe_bot import SyringeBot

try:
    from dynamixel_bridge import SyringeBotDynamixel
except ImportError:  # optional if user only wants simulation
    SyringeBotDynamixel = None  # type: ignore[misc, assignment]


COLORS = {
    "L1": "#5b9bd5",
    "L2": "#6ab04c",
    "L3": "#c0504d",
    "L4": "#e8a838",
    "base": "#555555",
    "ee": "#e74c3c",
    "bg": "#f4f4f4",
    "ws": "#b3cde3",
    "path": "#9b59b6",
    "trace": "#e74c3c",
}


def minimum_jerk(t: float) -> float:
    """Minimum-jerk blend curve with zero velocity/acceleration at endpoints."""
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def smooth_joint_trajectory(
    joint_traj: list[tuple[float, float]],
    num_samples: int = 2000,
) -> list[tuple[float, float]]:
    """Globally smooth a joint trajectory by dense time re-sampling."""
    if len(joint_traj) < 2:
        return list(joint_traj)
    traj = np.array(joint_traj, dtype=float)
    t = np.linspace(0.0, 1.0, len(traj))
    t_new = np.linspace(0.0, 1.0, max(num_samples, len(traj)))
    if CubicSpline is not None and len(joint_traj) >= 4:
        # C2-continuous joint profile avoids piecewise-linear cornering.
        cs1 = CubicSpline(t, traj[:, 0], bc_type="natural")
        cs2 = CubicSpline(t, traj[:, 1], bc_type="natural")
        t1 = cs1(t_new)
        t2 = cs2(t_new)
    else:
        t1 = np.interp(t_new, t, traj[:, 0])
        t2 = np.interp(t_new, t, traj[:, 1])
    return list(zip(t1, t2))


class DrawPathGUI:
    def __init__(
        self,
        bot: SyringeBot | None = None,
        *,
        dynamixel: SyringeBotDynamixel | None = None,
        singularity_threshold: float = 0.07,
        robot_time_scale: float = 2.5,
        path_smoothing: str = "spline",
    ):
        self.bot = bot or SyringeBot(
            L0=15.0,
            L1=15.0,
            L2=15.0,
            L3=15.0,
            L4=15.0,
            theta1=np.radians(90.0),
            theta2=np.radians(90.0),
        )
        self._dxl = dynamixel
        self._singularity_threshold = float(singularity_threshold)
        # >1.0 means robot executes faster than sim replay.
        self._robot_time_scale = max(0.1, float(robot_time_scale))
        self._path_smoothing = path_smoothing

        self._ws_points = self.bot.compute_workspace(n_samples=250)
        self._compute_limits()

        self._path: list[tuple[float, float]] = []
        self._path_linear: list[tuple[float, float]] = []
        self._trace: list[tuple[float, float]] = []
        self._speed = 2.0
        self._drawing = False
        self._anim: FuncAnimation | None = None
        self._sim_played_once = False
        self._joint_traj: list[tuple[float, float]] = []
        self._path_exec: list[tuple[float, float]] = []
        self._sim_indices: list[int] = []
        self._sim_interval_s = 0.016
        self._sim_duration_s = 0.0
        self._sim_started_at_s: float | None = None
        self._status = "Draw a path, then click Play Sim."

        self._build_ui()
        if self._dxl is not None:
            self._dxl.configure_motors()
            self._status = "Robot connected. Please run Play Sim before Play Robot."
        self._draw_scene()
        plt.show()

    def _compute_limits(self):
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
        self._xlim = (-half+10, half-10)
        # Clamp the lower Y bound so empty space below -3 is not shown.
        self._ylim = (max(cy - half, -3.0), cy + half)

    def _build_ui(self):
        self.fig = plt.figure(figsize=(14, 8.5), facecolor=COLORS["bg"])
        self.fig.canvas.manager.set_window_title("SyringeBot — Draw Path")
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self.ax = self.fig.add_axes([0.06, 0.08, 0.58, 0.86])
        self.ax.set_aspect("equal")

        # Right-hand control panel: centred vertically beside the plot.
        rx = 0.70            # left edge of controls
        rw = 0.25            # control width
        bh = 0.055           # button height
        gap = 0.018          # gap between buttons
        # Stack from top to bottom, vertically centred around 0.5.
        n_items = 5          # 4 buttons + 1 slider
        block_h = n_items * bh + (n_items - 1) * gap
        top = 0.5 + block_h / 2

        def _btn_y(i):
            return top - i * (bh + gap) - bh

        self.btn_play_sim = Button(
            self.fig.add_axes([rx, _btn_y(0), rw, bh]),
            "Play Sim",
            color=COLORS["L2"],
            hovercolor="#82c97e",
        )
        self.btn_play_sim.label.set_fontweight("bold")
        self.btn_play_sim.label.set_color("white")

        self.btn_play_robot = Button(
            self.fig.add_axes([rx, _btn_y(1), rw, bh]),
            "Play Robot",
            color=COLORS["L1"],
            hovercolor="#7ec8e3",
        )
        self.btn_play_robot.label.set_fontweight("bold")
        self.btn_play_robot.label.set_color("white")

        self.btn_clear = Button(
            self.fig.add_axes([rx, _btn_y(2), rw, bh]),
            "Clear",
            color=COLORS["L3"],
            hovercolor="#d97a7a",
        )
        self.btn_clear.label.set_fontweight("bold")
        self.btn_clear.label.set_color("white")

        self.btn_stop = Button(
            self.fig.add_axes([rx, _btn_y(3), rw, bh]),
            "Stop",
            color="#7986cb",
            hovercolor="#9fa8da",
        )
        self.btn_stop.label.set_fontweight("bold")
        self.btn_stop.label.set_color("white")

        sl_y = _btn_y(4) + bh * 0.25
        self.sl_speed = Slider(
            self.fig.add_axes([rx, sl_y, rw, bh * 0.5]),
            "Speed",
            0.2,
            20.0,
            valinit=2.0,
            valstep=0.1,
            color=COLORS["L4"],
        )

        info_y = sl_y - gap - 0.02
        self.info = self.fig.text(
            rx + rw / 2,
            info_y,
            "",
            ha="center",
            va="top",
            fontsize=9.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#ccc"),
        )

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.btn_play_sim.on_clicked(self._on_play_sim)
        self.btn_play_robot.on_clicked(self._on_play_robot)
        self.btn_clear.on_clicked(self._on_clear)
        self.btn_stop.on_clicked(self._on_stop)
        self.sl_speed.on_changed(self._on_speed)

    def _set_status(self, text: str):
        self._status = text
        self.info.set_text(text)

    def _draw_scene(self):
        ax = self.ax
        ax.cla()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.axhline(17.5, color="#666666", linewidth=1.0, alpha=0.25, linestyle="--", zorder=0)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_title("Draw path (left drag) → Play Sim → Play Robot", fontsize=12, fontweight="bold")

        if self._ws_points is not None and len(self._ws_points) > 0:
            ax.scatter(
                self._ws_points[:, 0],
                self._ws_points[:, 1],
                s=0.6,
                c=COLORS["ws"],
                alpha=0.4,
                edgecolors="none",
                zorder=1,
            )

        A, B, C, D, P = self.bot.forward_kinematics()
        ax.plot([A[0], B[0]], [A[1], B[1]], color=COLORS["base"], linewidth=7, solid_capstyle="butt", zorder=2)
        for pt in (A, B):
            ax.plot(*pt, "s", color="black", markersize=9, zorder=4)

        if len(self._path) > 1:
            xs, ys = zip(*self._path)
            ax.plot(xs, ys, "-", color=COLORS["path"], linewidth=2.5, zorder=3, label="Drawn path")
            ax.plot(xs[0], ys[0], "o", color=COLORS["path"], markersize=7, zorder=4)

        if len(self._trace) > 1:
            xs, ys = zip(*self._trace)
            ax.plot(xs, ys, "-", color=COLORS["trace"], linewidth=2, alpha=0.8, zorder=4, label="Trace")

        if P is not None:
            self._draw_robot_links(A, B, C, D, P)

        ax.set_xlim(self._xlim)
        ax.set_ylim(self._ylim)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
        self.info.set_text(self._status)
        self.fig.canvas.draw_idle()

    def _draw_robot_links(self, A, B, C, D, P):
        for p1, p2, col in [
            (A, C, COLORS["L1"]),
            (C, P, COLORS["L2"]),
            (B, D, COLORS["L3"]),
            (D, P, COLORS["L4"]),
        ]:
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color=col, linewidth=4, solid_capstyle="round", zorder=6)
        for pt, col in [(C, COLORS["L1"]), (D, COLORS["L3"])]:
            self.ax.plot(*pt, "o", color=col, markersize=9, mec="white", mew=1.5, zorder=7)
        self.ax.plot(*P, "o", color=COLORS["ee"], markersize=14, mec="white", mew=2, zorder=8)

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1 or self._anim is not None:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._drawing = True
        self._path = [(float(event.xdata), float(event.ydata))]
        self._trace = []
        self._joint_traj = []
        self._path_exec = []
        self._sim_played_once = False
        self._set_status("Drawing path...")
        self._draw_scene()

    def _on_motion(self, event):
        if not self._drawing or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._path.append((float(event.xdata), float(event.ydata)))
        self._draw_scene()

    def _on_release(self, event):
        if event.button != 1:
            return
        self._drawing = False
        self._resample_path()
        self._set_status("Path captured. Click Play Sim.")
        self._draw_scene()

    def _resample_path(self, spacing: float = 0.08):
        if len(self._path) < 2:
            return
        pts = np.array(self._path, dtype=float)
        diffs = np.diff(pts, axis=0)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        cum = np.concatenate([[0.0], np.cumsum(dists)])
        total = float(cum[-1])
        if total < 1e-6:
            return
        n_pts = max(int(total / spacing), 2)
        s = np.linspace(0.0, total, n_pts)
        x_lin = np.interp(s, cum, pts[:, 0])
        y_lin = np.interp(s, cum, pts[:, 1])
        linear_path = list(zip(x_lin, y_lin))
        # Optional XY smoothing mode controlled by CLI flag.
        if self._path_smoothing == "raw":
            self._path_linear = linear_path
            return
        if self._path_smoothing == "linear":
            self._path = linear_path
            self._path_linear = linear_path
            return
        # Default: spline smoothing when scipy is present; otherwise keep linear resampling.
        if splprep is not None and splev is not None and len(self._path) >= 4:
            try:
                # Small positive smoothing removes hand jitter while preserving shape.
                tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0.5)
                u = np.linspace(0.0, 1.0, n_pts)
                x, y = splev(u, tck)
                self._path = list(zip(x, y))
            except Exception:
                self._path = linear_path
        else:
            self._path = linear_path
        self._path_linear = linear_path

    def _build_joint_trajectory(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
        if len(self._path) < 2:
            self._set_status("Need at least 2 path points.")
            return None

        def _try(path_pts: list[tuple[float, float]]) -> tuple[
            list[tuple[float, float]] | None,
            list[tuple[float, float]] | None,
            str | None,
        ]:
            start_theta = (float(self.bot.theta1), float(self.bot.theta2))
            traj: list[tuple[float, float]] = []
            accepted_path: list[tuple[float, float]] = []
            for px, py in path_pts:
                t1, t2 = self.bot.inverse_kinematics(px, py, apply=True)
                if t1 is None or t2 is None:
                    self.bot.theta1, self.bot.theta2 = start_theta
                    return None, None, "Path leaves workspace. Redraw or move path."
                if self._singularity_threshold >= 0.0 and self.bot.is_near_singularity(
                    self._singularity_threshold
                ):
                    self.bot.theta1, self.bot.theta2 = start_theta
                    return None, None, "Path crosses singularity threshold. Redraw path."
                traj.append((float(t1), float(t2)))
                accepted_path.append((float(px), float(py)))
            self.bot.theta1, self.bot.theta2 = start_theta
            return traj, accepted_path, None

        traj, accepted, err = _try(self._path)
        if traj is not None and accepted is not None:
            return traj, accepted

        # If spline-smoothed path is invalid, retry with the linear resample.
        if self._path_linear and self._path_linear != self._path:
            traj, accepted, err_lin = _try(self._path_linear)
            if traj is not None and accepted is not None:
                self._path = list(self._path_linear)
                self._set_status("Spline path invalid; fell back to linear path for simulation.")
                return traj, accepted
            err = err_lin or err

        self._set_status(err or "Unable to build trajectory.")
        return None

    def _on_play_sim(self, _=None):
        if self._anim is not None:
            self._anim.event_source.stop()
            self._anim = None
        self._sim_played_once = False
        built = self._build_joint_trajectory()
        if built is None:
            self._draw_scene()
            return
        self._joint_traj, self._path_exec = built
        if not self._joint_traj:
            self._set_status("No valid trajectory points to simulate.")
            self._draw_scene()
            return
        self._trace = []
        self._set_status("Playing simulation trajectory...")
        # Sim preview uses adaptive decimation so long/smoothed paths remain responsive.
        # Robot execution still uses the full computed trajectory.
        n = len(self._joint_traj)
        speed = max(self._speed, 0.1)
        # Higher speed should shorten preview (fewer rendered frames).
        target_frames = max(20, int(700 / speed))
        step = max(1, int(np.ceil(n / target_frames)))
        self._sim_indices = list(range(0, n, step))
        if not self._sim_indices:
            self._sim_indices = [0]
        if self._sim_indices[-1] != n - 1:
            self._sim_indices.append(n - 1)
        interval = max(1, int(10 / speed))
        self._sim_interval_s = interval / 1000.0
        self._sim_duration_s = len(self._sim_indices) * self._sim_interval_s
        self._sim_started_at_s = time.perf_counter()
        self._anim = FuncAnimation(
            self.fig,
            self._animate_step,
            frames=len(self._sim_indices),
            interval=interval,
            repeat=False,
            blit=False,
        )
        self.fig.canvas.draw_idle()

    def _animate_step(self, frame: int):
        if frame >= len(self._sim_indices):
            return
        idx = self._sim_indices[frame]
        t1, t2 = self._joint_traj[idx]
        self.bot.theta1, self.bot.theta2 = t1, t2
        self._trace.append(self._path_exec[idx])
        self._draw_scene()
        if frame == len(self._sim_indices) - 1:
            self._sim_played_once = True
            self._anim = None
            if self._sim_started_at_s is not None:
                self._sim_duration_s = max(
                    0.01, time.perf_counter() - self._sim_started_at_s
                )
            self._sim_started_at_s = None
            self._set_status("Simulation done. Click Play Robot to execute hardware.")
            self._draw_scene()

    def _build_robot_command_stream(self) -> list[tuple[float, float, float, float, float]]:
        """
        Build a globally smoothed, constant-rate robot command stream.

        Returns tuples: (t1, t2, px, py, dt_seconds)
        """
        if not self._joint_traj:
            return []
        # Keep command rate high but still practical on U2D2 link.
        stream_hz = 260.0
        # Robot timing is derived from sim duration, then scaled for faster execution.
        if self._sim_duration_s > 0.0:
            target_duration = self._sim_duration_s / self._robot_time_scale
            num_samples = int(round(target_duration * stream_hz))
        else:
            num_samples = max(600, len(self._joint_traj) * 2)
        # Avoid over-dense command streams that slow effective execution.
        num_samples = max(300, min(num_samples, 2400))
        smooth_traj = smooth_joint_trajectory(self._joint_traj, num_samples=num_samples)
        if not smooth_traj:
            return []

        dt = 1.0 / stream_hz
        n_path = max(1, len(self._path_exec))
        cmds: list[tuple[float, float, float, float, float]] = []
        n = len(smooth_traj)
        for i, (t1, t2) in enumerate(smooth_traj):
            if n == 1:
                alpha = 0.0
            else:
                alpha = minimum_jerk(i / (n - 1))
            idx = min(int(alpha * (n_path - 1)), n_path - 1)
            px, py = self._path_exec[idx]
            cmds.append((float(t1), float(t2), float(px), float(py), dt))
        return cmds

    def _on_play_robot(self, _=None):
        if self._dxl is None:
            self._set_status("Robot backend is off. Start with --dynamixel u2d2 or fake.")
            self._draw_scene()
            return
        if not self._sim_played_once:
            self._set_status("Run Play Sim first, then Play Robot.")
            self._draw_scene()
            return
        if not self._joint_traj:
            self._set_status("No prepared joint trajectory.")
            self._draw_scene()
            return
        if not self._sim_indices:
            self._set_status("No sim pacing data. Run Play Sim first.")
            self._draw_scene()
            return

        first_t1, first_t2 = self._joint_traj[0]
        self.bot.theta1, self.bot.theta2 = first_t1, first_t2
        self._trace = [self._path_exec[0]]
        self._draw_scene()
        self._set_status("Commanded first point on robot. Confirm in terminal.")
        self._dxl.push_joint_angles(first_t1, first_t2)
        self._draw_scene()

        input("Robot moved to first point. Press Enter to execute drawing trajectory...")

        self._set_status("Executing trajectory on robot...")
        self._trace = []
        cmd_stream = self._build_robot_command_stream()
        next_tick = time.perf_counter()
        for i, (t1, t2, px, py, cmd_dt) in enumerate(cmd_stream):
            self.bot.theta1, self.bot.theta2 = t1, t2
            self._dxl.push_joint_angles(t1, t2)
            self._trace.append((px, py))
            # Avoid render cost on every micro-step; keeps execution speed aligned with sim.
            if i % 6 == 0 or i == len(cmd_stream) - 1:
                self._draw_scene()
                plt.pause(0.001)

            next_tick += cmd_dt
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        self._set_status("Robot trajectory complete.")
        self._draw_scene()

    def _on_stop(self, _=None):
        if self._anim is not None:
            self._anim.event_source.stop()
            self._anim = None
        self._set_status("Stopped.")
        self._draw_scene()

    def _on_clear(self, _=None):
        if self._anim is not None:
            self._anim.event_source.stop()
            self._anim = None
        self._path = []
        self._trace = []
        self._joint_traj = []
        self._path_exec = []
        self._sim_played_once = False
        self._set_status("Cleared. Draw a new path.")
        self._draw_scene()

    def _on_speed(self, val):
        self._speed = float(val)

    def _on_close(self, _event):
        if self._dxl is not None:
            try:
                self._dxl.close()
            except Exception:
                pass
            self._dxl = None


def _build_dynamixel_backend(
    mode: str,
    port: str,
    baudrate: int,
    *,
    profile_velocity: int,
    p_gain: int,
    scale_theta1: float,
    scale_theta2: float,
) -> SyringeBotDynamixel:
    if SyringeBotDynamixel is None:
        raise SystemExit(
            "dynamixel_bridge / dynamixel_u2d2 not installed. "
            "Install pip deps from environment.yaml."
        )
    motor_ids = [11, 21]
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
    parser = argparse.ArgumentParser(description="Draw and replay SyringeBot trajectories.")
    parser.add_argument("--dynamixel", choices=("off", "u2d2", "fake"), default="off")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=4_000_000)
    parser.add_argument("--dxl-profile-velocity", type=int, default=400)
    parser.add_argument("--dxl-p-gain", type=int, default=800)
    parser.add_argument("--dxl-scale1", type=float, default=1.0)
    parser.add_argument("--dxl-scale2", type=float, default=1.0)
    parser.add_argument("--singularity-threshold", type=float, default=0.07)
    parser.add_argument(
        "--robot-time-scale",
        type=float,
        default=2.5,
        help="Robot speed multiplier vs sim replay duration (>1 faster, <1 slower).",
    )
    parser.add_argument(
        "--path-smoothing",
        choices=("spline", "linear", "raw"),
        default="spline",
        help="Path preprocessing after drawing: spline (smooth), linear (straight segments), raw (exact points).",
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
    DrawPathGUI(
        bot,
        dynamixel=dxl,
        singularity_threshold=args.singularity_threshold,
        robot_time_scale=args.robot_time_scale,
        path_smoothing=args.path_smoothing,
    )
