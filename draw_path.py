#!/usr/bin/env python3
"""
Draw and replay trajectories for SyringeBot (sim first, robot optional).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

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

try:
    from svg.path import parse_path
except Exception:  # pip install svg.path
    parse_path = None  # type: ignore[misc, assignment]


# Horizontal guide in the plot (must match ``ax.axhline`` in ``_draw_scene``).
SVG_GUIDE_LINE_Y = 17.5

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


def _svg_local_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _parse_svg_points_attr(raw: str) -> list[tuple[float, float]]:
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", raw or "")]
    if len(nums) < 4:
        return []
    return list(zip(nums[0::2], nums[1::2]))


def _parse_svg_matrix_transform(transform: str | None) -> np.ndarray | None:
    """Parse first ``matrix(a,b,c,d,e,f)`` in ``transform``; returns 3×3 homogeneous matrix."""
    if not transform or not transform.strip():
        return None
    m = re.search(r"matrix\s*\(\s*([^)]+)\)", transform.strip(), re.I)
    if not m:
        return None
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", m.group(1))]
    if len(nums) < 6:
        return None
    a, b, c, d, e, f = nums[:6]
    return np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]], dtype=float)


def _apply_svg_matrix_points(
    pts: list[tuple[float, float]], mat: np.ndarray
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for x, y in pts:
        v = mat @ np.array([x, y, 1.0], dtype=float)
        out.append((float(v[0]), float(v[1])))
    return out


def _stroke_from_path_element(
    el: ET.Element, *, segment_point_cap: int = 400
) -> list[tuple[float, float]]:
    """Sample ``<path>`` ``d=`` and apply element ``transform`` (``matrix(...)``) if present."""
    d = el.get("d", "")
    stroke = _sample_path_d(d, segment_point_cap=segment_point_cap)
    if len(stroke) < 2:
        return []
    M = _parse_svg_matrix_transform(el.get("transform"))
    if M is not None:
        stroke = _apply_svg_matrix_points(stroke, M)
    return stroke


def _sample_path_d(d: str, *, segment_point_cap: int = 400) -> list[tuple[float, float]]:
    if parse_path is None or not (d or "").strip():
        return []
    cap = max(8, min(800, int(segment_point_cap)))
    try:
        path = parse_path(d.strip())
    except Exception:
        return []
    pts: list[tuple[float, float]] = []
    for seg in path:
        try:
            ell = float(seg.length(error=1e-4))
        except TypeError:
            try:
                ell = float(seg.length())
            except Exception:
                ell = 0.0
        except Exception:
            ell = 0.0
        n = max(2, min(cap, int(8 + ell * 0.28)))
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.0
            z = seg.point(t)
            pts.append((float(z.real), float(z.imag)))
    return pts


def _join_strokes_with_bridge(
    strokes: list[list[tuple[float, float]]], bridge_steps: int = 12
) -> list[tuple[float, float]]:
    strokes = [s for s in strokes if len(s) >= 2]
    if not strokes:
        return []
    out = list(strokes[0])
    for stroke in strokes[1:]:
        a = out[-1]
        b = stroke[0]
        for k in range(1, bridge_steps + 1):
            t = k / (bridge_steps + 1)
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        out.extend(stroke[1:])
    return out


def extract_svg_strokes(
    svg_text: str, *, segment_point_cap: int = 400
) -> list[list[tuple[float, float]]]:
    """Polylines in document order: <path d>, <polyline>, <polygon>."""
    root = ET.fromstring(svg_text)
    strokes: list[list[tuple[float, float]]] = []
    for el in root.iter():
        tag = _svg_local_tag(el.tag)
        if tag == "path" and el.get("d"):
            stroke = _stroke_from_path_element(el, segment_point_cap=segment_point_cap)
            if len(stroke) >= 2:
                strokes.append(stroke)
        elif tag in ("polyline", "polygon") and el.get("points"):
            stroke = _parse_svg_points_attr(el.get("points", ""))
            if len(stroke) >= 2:
                M = _parse_svg_matrix_transform(el.get("transform"))
                if M is not None:
                    stroke = _apply_svg_matrix_points(stroke, M)
                if tag == "polygon":
                    x0, y0 = stroke[0]
                    x1, y1 = stroke[-1]
                    if (x0 - x1) ** 2 + (y0 - y1) ** 2 > 1e-12:
                        stroke = stroke + [stroke[0]]
                strokes.append(stroke)
    return strokes


def workspace_fit_box(
    ws: np.ndarray | None,
    plot_xlim: tuple[float, float],
    plot_ylim: tuple[float, float],
    *,
    shrink: float = 0.04,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Rectangle inside the reachable workspace (clamped to plot limits) for SVG fitting."""
    if ws is None or len(ws) < 3:
        return plot_xlim, plot_ylim
    wx0, wx1 = float(ws[:, 0].min()), float(ws[:, 0].max())
    wy0, wy1 = float(ws[:, 1].min()), float(ws[:, 1].max())
    dx = wx1 - wx0
    dy = wy1 - wy0
    if dx < 1e-9 or dy < 1e-9:
        return plot_xlim, plot_ylim
    mx = shrink * dx
    my = shrink * dy
    tx0 = max(plot_xlim[0], wx0 + mx)
    tx1 = min(plot_xlim[1], wx1 - mx)
    ty0 = max(plot_ylim[0], wy0 + my)
    ty1 = min(plot_ylim[1], wy1 - my)
    if tx1 <= tx0 + 1e-6 or ty1 <= ty0 + 1e-6:
        return plot_xlim, plot_ylim
    return (tx0, tx1), (ty0, ty1)


def _pick_svg_filepath() -> str | None:
    """Prefer ``zenity`` on Linux (plays well with Matplotlib); else Tk file dialog."""
    if sys.platform.startswith("linux") and shutil.which("zenity"):
        try:
            r = subprocess.run(
                ["zenity", "--file-selection", "--title=Select SVG"],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            p = (r.stdout or "").strip()
            if r.returncode == 0 and p:
                return p
        except Exception:
            pass
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk._default_root
    created = False
    if root is None:
        root = tk.Tk()
        root.withdraw()
        created = True
    try:
        root.lift()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        root.update_idletasks()
        root.update()
        fp = filedialog.askopenfilename(
            master=root,
            title="Select SVG",
            filetypes=[("SVG", "*.svg"), ("All files", "*")],
        )
    finally:
        if created:
            root.destroy()
    return fp if fp else None


def fit_stroke_points_to_axes(
    points: list[tuple[float, float]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    margin_frac: float = 0.06,
    anchor: str = "center",
) -> list[tuple[float, float]]:
    """Map SVG coordinates (y down) into a target rectangle with uniform scale.

    ``anchor``: ``"center"`` (default), ``"top_left"``, ``"top_right"``, or
    ``"top_center"`` (centered in ``x``, top-aligned in ``y`` within the target rect).
    """
    if len(points) < 2:
        return []
    arr = np.asarray(points, dtype=float)
    arr[:, 1] = -arr[:, 1]
    mnx, mxx = float(arr[:, 0].min()), float(arr[:, 0].max())
    mny, mxy = float(arr[:, 1].min()), float(arr[:, 1].max())
    dx = mxx - mnx or 1.0
    dy = mxy - mny or 1.0
    xw = xlim[1] - xlim[0]
    yh = ylim[1] - ylim[0]
    scale = (1.0 - 2.0 * margin_frac) * min(xw / dx, yh / dy)
    cx = 0.5 * (mnx + mxx)
    cy = 0.5 * (mny + mxy)
    tcx = 0.5 * (xlim[0] + xlim[1])
    tcy = 0.5 * (ylim[0] + ylim[1])
    xs0 = (arr[:, 0] - cx) * scale
    ys0 = (arr[:, 1] - cy) * scale
    px_min = float(xs0.min())
    px_max = float(xs0.max())
    py_min = float(ys0.min())
    py_max = float(ys0.max())
    mx = margin_frac * xw
    my = margin_frac * yh
    if anchor == "top_left":
        ox = (xlim[0] + mx) - px_min
        oy = (ylim[1] - my) - py_max
    elif anchor == "top_right":
        ox = (xlim[1] - mx) - px_max
        oy = (ylim[1] - my) - py_max
    elif anchor == "top_center":
        ox = tcx - 0.5 * (px_min + px_max)
        oy = (ylim[1] - my) - py_max
    else:
        ox = tcx - 0.5 * (px_min + px_max)
        oy = tcy - 0.5 * (py_min + py_max)
    xs = xs0 + ox
    ys = ys0 + oy
    return [(float(xs[i]), float(ys[i])) for i in range(len(xs))]


class DrawPathGUI:
    def __init__(
        self,
        bot: SyringeBot | None = None,
        *,
        dynamixel: SyringeBotDynamixel | None = None,
        singularity_threshold: float = 0.07,
        robot_time_scale: float = 2.5,
        path_smoothing: str = "spline",
        initial_svg: str | None = None,
        svg_resample_spacing: float = 0.024,
        svg_sim_frame_cap: int = 4000,
        svg_segment_point_cap: int = 400,
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
        self._svg_resample_spacing = float(np.clip(svg_resample_spacing, 1e-4, 5.0))
        self._svg_sim_frame_cap = int(np.clip(svg_sim_frame_cap, 50, 20_000))
        self._svg_segment_point_cap = int(np.clip(svg_segment_point_cap, 8, 800))

        self._ws_points = self.bot.compute_workspace(n_samples=250)
        self._compute_limits()

        self._path: list[tuple[float, float]] = []
        self._path_linear: list[tuple[float, float]] = []
        self._trace: list[tuple[float, float]] = []
        self._speed = 20.0
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
        # Two-step hardware replay: first click moves to start ("Reset Robot"), second runs trajectory.
        self._robot_armed_for_execute = False
        self._path_is_svg_loaded = False

        self._build_ui()
        if self._dxl is not None:
            self._dxl.configure_motors()
            self._status = "Robot connected. Please run Play Sim before Play Robot."
        if initial_svg:
            try:
                self._load_svg_from_file(initial_svg)
            except Exception as e:
                self._set_status(f"--svg failed: {e}")
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
        n_items = 6          # 5 buttons + 1 slider
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

        self.btn_load_svg = Button(
            self.fig.add_axes([rx, _btn_y(3), rw, bh]),
            "Load SVG",
            color="#8d6e63",
            hovercolor="#bcaaa4",
        )
        self.btn_load_svg.label.set_fontweight("bold")
        self.btn_load_svg.label.set_color("white")

        self.btn_stop = Button(
            self.fig.add_axes([rx, _btn_y(4), rw, bh]),
            "Stop",
            color="#7986cb",
            hovercolor="#9fa8da",
        )
        self.btn_stop.label.set_fontweight("bold")
        self.btn_stop.label.set_color("white")

        sl_y = _btn_y(5) + bh * 0.25
        self.sl_speed = Slider(
            self.fig.add_axes([rx, sl_y, rw, bh * 0.5]),
            "Speed",
            0.2,
            50.0,
            valinit=20.0,
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
        self.btn_load_svg.on_clicked(self._on_load_svg)
        self.btn_stop.on_clicked(self._on_stop)
        self.sl_speed.on_changed(self._on_speed)
        self._sync_robot_play_button()

    def _sync_robot_play_button(self):
        """Label reflects two-step robot flow after a successful sim."""
        if self._dxl is None:
            self.btn_play_robot.label.set_text("Play Robot")
            return
        if not self._sim_played_once:
            self.btn_play_robot.label.set_text("Play Robot")
            return
        if self._robot_armed_for_execute:
            self.btn_play_robot.label.set_text("Play Robot")
        else:
            self.btn_play_robot.label.set_text("Reset Robot")

    def _load_svg_from_file(self, filepath: str) -> None:
        if parse_path is None:
            raise RuntimeError("Install svg.path (pip install svg.path, or conda env from environment.yaml).")
        p = Path(filepath).expanduser()
        if not p.is_file():
            raise RuntimeError(f"Not a file: {p}")
        svg_text = p.read_text(encoding="utf-8", errors="replace")
        strokes = extract_svg_strokes(
            svg_text, segment_point_cap=self._svg_segment_point_cap
        )
        if not strokes:
            raise RuntimeError("No usable <path d>, <polyline>, or <polygon> in SVG.")
        joined = _join_strokes_with_bridge(strokes)
        fit_xlim, fit_ylim = workspace_fit_box(self._ws_points, self._xlim, self._ylim)
        # Keep SVG entirely above the dashed guide line; place in the upper-left corner of that band.
        gap = 0.15
        y_lo, y_hi = fit_ylim
        y_lo = max(y_lo, SVG_GUIDE_LINE_Y + gap)
        if y_hi <= y_lo + 1e-6:
            y_lo = SVG_GUIDE_LINE_Y + gap
        fit_ylim = (y_lo, y_hi)
        fitted = fit_stroke_points_to_axes(
            joined,
            fit_xlim,
            fit_ylim,
            margin_frac=0.05,
            anchor="top_center",
        )
        if len(fitted) < 2:
            raise RuntimeError("SVG produced too few points after fitting to axes.")
        self._path = fitted
        self._path_is_svg_loaded = True
        self._path_linear = []
        self._trace = []
        self._joint_traj = []
        self._path_exec = []
        self._sim_played_once = False
        self._robot_armed_for_execute = False
        self._sync_robot_play_button()
        self._resample_path()
        self._set_status(f"Loaded SVG ({len(self._path)} pts). Click Play Sim.")

    def _on_load_svg(self, _=None):
        if self._anim is not None:
            self._set_status("Stop simulation before loading SVG.")
            self._draw_scene()
            return
        if parse_path is None:
            self._set_status("Install svg.path (see environment.yaml).")
            self._draw_scene()
            return
        fp = _pick_svg_filepath()
        if not fp:
            self._set_status("SVG load cancelled (or no file dialog). Try: draw_path.py --svg file.svg")
            self._draw_scene()
            return
        try:
            self._load_svg_from_file(fp)
        except Exception as e:
            self._set_status(f"SVG load failed: {e}")
        self.fig.canvas.flush_events()
        self._draw_scene()

    def _set_status(self, text: str):
        self._status = text
        self.info.set_text(text)

    def _draw_scene(self):
        ax = self.ax
        ax.cla()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.axhline(
            SVG_GUIDE_LINE_Y,
            color="#666666",
            linewidth=1.0,
            alpha=0.25,
            linestyle="--",
            zorder=0,
        )
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
        self._path_is_svg_loaded = False
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

    def _resample_path(self, spacing: float | None = None):
        if len(self._path) < 2:
            return
        if spacing is None:
            spacing = self._svg_resample_spacing if self._path_is_svg_loaded else 0.08
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
            self._path_linear = [(float(pts[i, 0]), float(pts[i, 1])) for i in range(len(pts))]
            return
        # Imported SVG: skip XY spline so Play Sim follows the outline; keep dense polyline.
        if self._path_is_svg_loaded:
            self._path = linear_path
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
        self._robot_armed_for_execute = False
        self._sync_robot_play_button()
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
        # SVG / long paths: many more frames so the red trace follows the geometry accurately.
        if self._path_is_svg_loaded:
            cap = self._svg_sim_frame_cap
            target_frames = min(n, cap, max(320, int(3600 / speed)))
        else:
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
            self._set_status(
                "Simulation done. Click Reset Robot to move to start, then Play Robot to run."
            )
            self._sync_robot_play_button()
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
        if not self._path_exec:
            self._set_status("No execution path. Run Play Sim first.")
            self._draw_scene()
            return
        if not self._sim_indices:
            self._set_status("No sim pacing data. Run Play Sim first.")
            self._draw_scene()
            return

        if not self._robot_armed_for_execute:
            first_t1, first_t2 = self._joint_traj[0]
            self.bot.theta1, self.bot.theta2 = first_t1, first_t2
            self._trace = [self._path_exec[0]]
            self._dxl.push_joint_angles(first_t1, first_t2)
            self._robot_armed_for_execute = True
            self._set_status("Robot at start pose. Click Play Robot again to run trajectory.")
            self._sync_robot_play_button()
            self._draw_scene()
            return

        self._set_status("Executing trajectory on robot...")
        self._robot_armed_for_execute = False
        self._sync_robot_play_button()
        self._trace = []
        cmd_stream = self._build_robot_command_stream()
        if not cmd_stream:
            self._set_status("No robot command stream generated.")
            self._draw_scene()
            return
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

        self._set_status("Robot trajectory complete. Click Reset Robot to go to start again.")
        self._sync_robot_play_button()
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
        self._path_is_svg_loaded = False
        self._path_linear = []
        self._trace = []
        self._joint_traj = []
        self._path_exec = []
        self._sim_played_once = False
        self._robot_armed_for_execute = False
        self._sync_robot_play_button()
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
        default=6.0,
        help="Robot speed multiplier vs sim replay duration (>1 faster, <1 slower).",
    )
    parser.add_argument(
        "--path-smoothing",
        choices=("spline", "linear", "raw"),
        default="spline",
        help="Path preprocessing after drawing: spline (smooth), linear (straight segments), raw (exact points).",
    )
    parser.add_argument(
        "--svg",
        type=str,
        default=None,
        metavar="FILE.svg",
        help="Load path geometry from an SVG file at startup (<path>, <polyline>, <polygon>).",
    )
    parser.add_argument(
        "--svg-resample-spacing",
        type=float,
        default=0.20,
        metavar="DX",
        help="After SVG load: arc-length spacing along path (larger = fewer IK points). Typical 0.02–0.15.",
    )
    parser.add_argument(
        "--svg-sim-frame-cap",
        type=int,
        default=4000,
        metavar="N",
        help="Max Play Sim animation frames for SVG (lower = coarser preview, faster).",
    )
    parser.add_argument(
        "--svg-segment-point-cap",
        type=int,
        default=400,
        metavar="K",
        help="Max samples per SVG <path> segment when parsing curves (lower = fewer raw points).",
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
        initial_svg=args.svg,
        svg_resample_spacing=args.svg_resample_spacing,
        svg_sim_frame_cap=args.svg_sim_frame_cap,
        svg_segment_point_cap=args.svg_segment_point_cap,
    )
