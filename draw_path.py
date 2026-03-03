#!/usr/bin/env python3
"""
Draw a path inside the SyringeBot workspace and watch the robot follow it.

Usage:
    conda activate syringe_robot
    python draw_path.py

Controls:
    • Click and drag inside the workspace to draw a path
    • Press "Play" to animate the robot along the path
    • Press "Clear" to erase and draw a new path
    • Adjust playback speed with the slider
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation

from syringe_bot import SyringeBot


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


class DrawPathGUI:
    def __init__(self, bot: SyringeBot | None = None):
        self.bot = bot or SyringeBot()

        self._ws_points = self.bot.compute_workspace(n_samples=250)
        self._compute_limits()

        self._path: list[tuple[float, float]] = []
        self._drawing = False
        self._anim: FuncAnimation | None = None
        self._anim_idx = 0
        self._trace: list[tuple[float, float]] = []
        self._speed = 1.0

        self._build_ui()
        self._draw_static()
        plt.show()

    def _compute_limits(self):
        L = self.bot.link_lengths
        reach = max(L["L1"] + L["L2"], L["L3"] + L["L4"])

        x_lo, x_hi = -reach - 0.5, L["L0"] + reach + 0.5
        y_lo, y_hi = -reach - 0.5, reach + 0.5

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

    def _build_ui(self):
        self.fig = plt.figure(figsize=(12, 9), facecolor=COLORS["bg"])
        self.fig.canvas.manager.set_window_title("SyringeBot — Draw Path")

        self.ax = self.fig.add_axes([0.08, 0.12, 0.84, 0.82])
        self.ax.set_aspect("equal")

        # buttons
        self.btn_play = Button(
            self.fig.add_axes([0.15, 0.03, 0.12, 0.05]),
            "Play", color=COLORS["L2"], hovercolor="#82c97e")
        self.btn_play.label.set_fontweight("bold")
        self.btn_play.label.set_color("white")

        self.btn_clear = Button(
            self.fig.add_axes([0.30, 0.03, 0.12, 0.05]),
            "Clear", color=COLORS["L3"], hovercolor="#d97a7a")
        self.btn_clear.label.set_fontweight("bold")
        self.btn_clear.label.set_color("white")

        self.btn_stop = Button(
            self.fig.add_axes([0.45, 0.03, 0.12, 0.05]),
            "Stop", color="#7986cb", hovercolor="#9fa8da")
        self.btn_stop.label.set_fontweight("bold")
        self.btn_stop.label.set_color("white")

        # speed slider
        self.sl_speed = Slider(
            self.fig.add_axes([0.65, 0.03, 0.25, 0.03]),
            "Speed", 0.1, 5.0, valinit=1.0, valstep=0.1,
            color=COLORS["L4"])

        # connect events
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.btn_play.on_clicked(self._on_play)
        self.btn_clear.on_clicked(self._on_clear)
        self.btn_stop.on_clicked(self._on_stop)
        self.sl_speed.on_changed(self._on_speed)

    def _draw_static(self):
        """Draw workspace, base, and current path (no robot yet)."""
        ax = self.ax
        ax.cla()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_title("Draw a path inside the workspace, then press Play",
                     fontsize=12, fontweight="bold", pad=8)

        # workspace
        if self._ws_points is not None and len(self._ws_points) > 0:
            ax.scatter(self._ws_points[:, 0], self._ws_points[:, 1],
                       s=0.6, c=COLORS["ws"], alpha=0.4,
                       edgecolors="none", zorder=1)

        # base bar
        L = self.bot.link_lengths
        A, B = np.array([0.0, 0.0]), np.array([L["L0"], 0.0])
        ax.plot([A[0], B[0]], [A[1], B[1]],
                color=COLORS["base"], linewidth=6,
                solid_capstyle="butt", zorder=2)
        for pt in (A, B):
            ax.plot(*pt, "s", color="black", markersize=9, zorder=5)

        # user-drawn path
        if len(self._path) > 1:
            xs, ys = zip(*self._path)
            ax.plot(xs, ys, "-", color=COLORS["path"], linewidth=2.5,
                    solid_capstyle="round", zorder=3, label="Drawn path")
            ax.plot(xs[0], ys[0], "o", color=COLORS["path"],
                    markersize=8, zorder=4)

        ax.set_xlim(self._xlim)
        ax.set_ylim(self._ylim)
        self.fig.canvas.draw_idle()

    def _draw_robot(self, px: float, py: float):
        """Draw the robot at the given end-effector position."""
        t1, t2 = self.bot.inverse_kinematics(px, py, apply=True)
        if t1 is None or t2 is None:
            return

        joints = self.bot.forward_kinematics()
        A, B, C, D, P = joints
        if P is None:
            return

        ax = self.ax

        # links
        for p1, p2, col in [
            (A, C, COLORS["L1"]),
            (C, P, COLORS["L2"]),
            (B, D, COLORS["L3"]),
            (D, P, COLORS["L4"]),
        ]:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-",
                    color=col, linewidth=4, solid_capstyle="round", zorder=6)

        # elbow joints
        for pt, col in [(C, COLORS["L1"]), (D, COLORS["L3"])]:
            ax.plot(*pt, "o", color=col, markersize=9,
                    mec="white", mew=1.5, zorder=7)

        # end-effector
        ax.plot(*P, "o", color=COLORS["ee"], markersize=14,
                mec="white", mew=2, zorder=8)

    # ── mouse events ───────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if self._anim is not None:
            return
        self._drawing = True
        self._path = [(event.xdata, event.ydata)]

    def _on_motion(self, event):
        if not self._drawing or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._path.append((event.xdata, event.ydata))
        self._draw_static()

    def _on_release(self, event):
        if event.button != 1:
            return
        self._drawing = False
        self._resample_path()
        self._draw_static()

    def _resample_path(self, spacing: float = 0.05):
        """Resample the drawn path to have roughly uniform point spacing."""
        if len(self._path) < 2:
            return
        pts = np.array(self._path)
        diffs = np.diff(pts, axis=0)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        cum = np.concatenate([[0], np.cumsum(dists)])
        total = cum[-1]
        if total < 1e-6:
            return
        n_pts = max(int(total / spacing), 2)
        new_s = np.linspace(0, total, n_pts)
        new_x = np.interp(new_s, cum, pts[:, 0])
        new_y = np.interp(new_s, cum, pts[:, 1])
        self._path = list(zip(new_x, new_y))

    # ── button callbacks ───────────────────────────────────────────────────

    def _on_play(self, _=None):
        if len(self._path) < 2:
            return
        if self._anim is not None:
            self._anim.event_source.stop()
            self._anim = None

        self._anim_idx = 0
        self._trace = []

        interval = max(10, int(30 / self._speed))
        self._anim = FuncAnimation(
            self.fig, self._animate_step,
            frames=len(self._path),
            interval=interval,
            repeat=False,
            blit=False)
        self.fig.canvas.draw_idle()

    def _on_stop(self, _=None):
        if self._anim is not None:
            self._anim.event_source.stop()
            self._anim = None
        self._draw_static()

    def _on_clear(self, _=None):
        if self._anim is not None:
            self._anim.event_source.stop()
            self._anim = None
        self._path = []
        self._trace = []
        self._draw_static()

    def _on_speed(self, val):
        self._speed = val

    # ── animation ──────────────────────────────────────────────────────────

    def _animate_step(self, frame):
        if frame >= len(self._path):
            return

        px, py = self._path[frame]
        self._trace.append((px, py))

        self._draw_static()

        # draw trace so far
        if len(self._trace) > 1:
            xs, ys = zip(*self._trace)
            self.ax.plot(xs, ys, "-", color=COLORS["trace"], linewidth=2,
                         alpha=0.7, zorder=4)

        self._draw_robot(px, py)
        self.ax.set_xlim(self._xlim)
        self.ax.set_ylim(self._ylim)


if __name__ == "__main__":
    bot = SyringeBot(L0=3.0, L1=2.0, L2=2.5, L3=2.0, L4=2.5)
    DrawPathGUI(bot)
