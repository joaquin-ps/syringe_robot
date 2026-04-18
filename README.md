# Syringe Robot

Tools for simulating and running a 5-bar `SyringeBot` both from an interactive joint/cartesian GUI and from a draw-and-replay GUI, with optional Dynamixel U2D2 hardware control.

## Requirements

- Python environment with dependencies from `environment.yaml`
- Dynamixel setup (for hardware mode):
  - Protocol `2.0`
  - Baud `4000000`
  - Motor IDs: `11` (left), `21` (right)
  - Encoder center: `1024` == `0 rad`

## Run Main GUI (`gui.py`)

Simulation only:

```bash
python gui.py
```

Hardware with U2D2:

```bash
python gui.py --dynamixel u2d2 
```

Useful options:

- `--port /dev/ttyUSB0 --baud 4000000` dynamixel settings
- `--singularity-threshold 0.07` controls how aggressively singularity proximity is blocked
- `--dxl-profile-velocity` and `--dxl-p-gain` tune Dynamixel response

## Run Draw + Replay GUI (`draw_path.py`)

Simulation only:

```bash
python draw_path.py
```

Hardware with U2D2 (fast robot replay + linear path smoothing):

```bash
python draw_path.py --dynamixel u2d2 --path-smoothing linear
```

Flow:

1. Draw path with left mouse drag.
2. Click **Play Sim**.
3. After sim, the robot button shows **Reset Robot**: click it once to move the hardware to the trajectory start.
4. The button then shows **Play Robot**: click again to execute the full drawing on hardware (no terminal prompt).

Path smoothing modes:

- `--path-smoothing spline` (default): smooth XY curve
- `--path-smoothing linear`: straight segments (less smoothing)
- `--path-smoothing raw`: exact drawn points

### SVG paths

`draw_path.py` can build a trace from vector shapes (requires `svg.path` from `environment.yaml`):

```bash
python draw_path.py --svg drawings/dog.svg --dynamixel u2d2
```

- **Button:** **Load SVG** opens a file picker (on Linux, `zenity` is used when available so the dialog appears above Matplotlib; otherwise Tk). The outline is **uniformly scaled to fit in the workspace band above the dashed guide line** (`y ≈ 17.5`), **centered horizontally** on the workspace fit box and **top-aligned** in that band (SVG Y-down is flipped to match the plot).
- **CLI:** `python draw_path.py --svg path/to/file.svg` (e.g. `dog.svg` in this repo)

Supported elements (in order): `<path d="...">`, `<polyline points="...">`, `<polygon points="...">`. Each element’s `transform="matrix(...)"` is applied to the sampled points (needed for many exported SVGs). Multiple shapes are joined with a short straight bridge so playback stays one continuous path. Transforms on parent `<g>` groups are not composed yet.

Loaded SVGs skip extra XY spline smoothing (so **Play Sim** follows the outline closely), use a denser polyline resample, and request more animation frames during preview. Hand-drawn paths still follow `--path-smoothing` (`spline` / `linear` / `raw`).

**Tuning SVG point count** (restart `draw_path.py` with flags; affects **Load SVG** and `--svg`):

| Flag | Default | Effect |
|------|---------|--------|
| `--svg-resample-spacing` | `0.024` | Arc-length step along the path after load (**larger → fewer** IK / sim points). Try `0.06`–`0.12` to thin a heavy path. |
| `--svg-sim-frame-cap` | `4000` | Upper bound on **Play Sim** preview frames for SVG (**lower → coarser** red trace, faster UI). |
| `--svg-segment-point-cap` | `400` | Max samples **per Bézier segment** while parsing `<path d>` (**lower → fewer** vertices before fitting). |

Example (fewer points, lighter sim):

```bash
python draw_path.py --svg drawings/dog.svg --svg-resample-spacing 0.08 --svg-sim-frame-cap 900 --svg-segment-point-cap 120
```

Speed option:

- `--robot-time-scale <value>`: robot speed multiplier vs sim duration (`>1` faster, `<1` slower)
