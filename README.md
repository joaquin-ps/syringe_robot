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
python gui.py --dynamixel u2d2 --port /dev/ttyUSB0 --baud 4000000
```

Useful options:

- `--singularity-threshold 0.07` controls how aggressively singularity proximity is blocked
- `--dxl-profile-velocity` and `--dxl-p-gain` tune Dynamixel response

## Run Draw + Replay GUI (`draw_path.py`)

Simulation only:

```bash
python draw_path.py
```

Hardware with U2D2 (fast robot replay + linear path smoothing):

```bash
python draw_path.py --dynamixel u2d2 --robot-time-scale 6 --path-smoothing linear
```

Flow:

1. Draw path with left mouse drag.
2. Click **Play Sim**.
3. Click **Play Robot** (robot moves to first point, waits for Enter in terminal, then executes).

Path smoothing modes:

- `--path-smoothing spline` (default): smooth XY curve
- `--path-smoothing linear`: straight segments (less smoothing)
- `--path-smoothing raw`: exact drawn points
- `--path-smoothing off`: alias for `linear`

Speed option:

- `--robot-time-scale <value>`: robot speed multiplier vs sim duration (`>1` faster, `<1` slower)
