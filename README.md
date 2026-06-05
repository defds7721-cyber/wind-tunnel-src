# Wind Tunnel

Educational wind-tunnel simulator with a real semi-Lagrangian (Stam-style)
Navier–Stokes fluid solver and rigid-body physics. Draw a shape (or load a
PNG silhouette), watch the airflow, read live Cd / Cl / Reynolds, then drop
the shape under gravity and let strong wind tip it over.

Created by vvas2002.

## Run from source

```
pip install pygame pillow
python wind_tunnel.py
```

## Controls

- **DRAW** mode: drag on the canvas to draw a closed shape.
- **IMAGE** mode: load a PNG and a silhouette obstacle is traced from it.
- Sliders: wind speed, density, wind angle, particle size, viscosity.
- **ВКЛЮЧИТЬ ФИЗИКУ**: the shape becomes a rigid body — it falls, lands,
  and the wind tries to overturn it.

## Builds

Windows and macOS binaries are built automatically by GitHub Actions on
every `v*` tag and attached to the matching Release.