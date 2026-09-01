# Project Status — Gravity Simulator

Last updated: 2026-09-01

This is the handoff file for resuming development later.

## Recommended executable

Run:

`python run_gravity.py`

`run_gravity.py` launches the current application in `gravity_app.py`.

The older `planetary_sim.py` remains the base implementation reused by the current app.

## What currently works

- Live Newtonian N-body gravity simulation in a Gradio web UI.
- S and O object groups with arbitrary positive mass vectors.
- Physical spherical radii with optional radius vector.
- Perfectly elastic, frictionless sphere-sphere collisions.
- Off-center impacts ricochet naturally.
- Collision-time search inside drift intervals to prevent tunneling.
- Symplectic Euler / kick-drift: velocity first, position second.
- Configurable outer `dt` and internal substeps.
- Three boundary modes: Huge, Box, Toroid.
- Continuous, fixed-step, simulated-duration, and real-time-hour run modes.
- Live kinetic, potential, total-energy, energy-error, and momentum plots.
- Collision counter and diagnostics table.
- Fading, Permanent, and None trail modes.
- Permanent trail uses fixed-memory raster storage with a decimated browser preview so Energy/Momentum plots remain responsive.
- Google Colab and local Conda/Jupyter operation.

## Boundary modes

### Huge

Unbounded 3-D Newtonian system. The XY window is only a camera; bodies may leave the visible region and continue interacting.

### Box

3-D reflecting box. Spherical bodies bounce elastically from the walls. Wall impulses mean body-system momentum alone is not conserved through wall collisions.

### Toroid

A square **2-D periodic XY universe**.

- The displayed square represents coordinates `[-L,+L)` in X and Y, where `L = Box / Toroid half-size`.
- A body leaving one edge immediately re-enters from the opposite edge with unchanged velocity.
- Toroid is deliberately 2-D: `z = vz = az = 0`.
- Pair gravity uses the shortest periodic displacement only (minimum-image convention).
- The simulator does **not** sum gravity from an infinite lattice of periodic copies.
- Sphere-sphere collision detection and collision normals also use periodic geometry, so bodies can collide across opposite edges.
- Fading and Permanent trails break at a periodic edge and continue from the antipodal edge; they do not draw a false line across the screen.

For width `W = 2L`, the shortest X displacement is conceptually:

`dx = dx - W * round(dx/W)`

and similarly for Y.

Toroid potential energy diagnostics use the same minimum-image pair distance as the force calculation.

A simple Cartesian center of mass is not globally meaningful on a periodic torus, so Toroid COM X/Y diagnostics are intentionally not treated as physical COM coordinates.

## Agreed collision model

1. Bodies are rigid spheres for contact purposes.
2. No deformation, friction, or spin.
3. Coefficient of restitution `e = 1`.
4. Collision impulse acts along the contact normal / center-to-center line.
5. Tangential relative velocity is unchanged.
6. Individual bodies exchange kinetic energy and momentum.
7. Total kinetic energy and total linear momentum are conserved across an isolated elastic collision up to numerical error.
8. Under gravity, K and U exchange while total mechanical energy should remain approximately constant numerically.

## Numerical behavior

- Outer `dt` is the user-visible time step.
- Each outer step is divided into `internal_substeps`; default is 100.
- Higher substep counts improve close-approach and energy behavior but cost CPU time.
- Relative energy error `(E-E0)/|E0|` is the main health check.
- Large monotonic energy drift means the numerical resolution is too coarse.

## Trails

### Fading

Bounded recent-history deque. Old path segments disappear.

### Permanent

Fixed-resolution XY raster. Runtime memory does not grow with trajectory length.

- Configurable raster resolution.
- Configurable record-every-N-refreshes decimation.
- Configurable opacity and visibility.
- Browser preview is downsampled independently from the full internal raster to keep live diagnostic plots responsive.

### None

No trail history.

## Primary research direction

Use the simulator to explore whether small and medium gravitational systems:

- remain stable,
- remain bounded but become chaotic,
- repeatedly collide and ricochet,
- redistribute energy and momentum,
- eject bodies,
- or diverge spatially.

## Strong next feature

Add a quantitative system-size / boundedness diagnostic. For Huge/Box this can be based on center of mass, for example:

`Rmax(t) = max_i |r_i - R_COM|`

For Toroid, use a periodic-geometry equivalent rather than a naive Cartesian COM distance.

## Other useful next steps

- Angular momentum diagnostics.
- Interactive 3-D view for Huge/Box.
- Escape/event logging.
- Configuration save/load.
- CSV diagnostics export.
- Final-state export.
- Leapfrog / velocity-Verlet option.
- Initial-condition presets.
- Dynamic hide/show/clear of the permanent trail during a running simulation.

## Repository

`https://github.com/leodman/gravity`

Current app:

- `run_gravity.py` — recommended launcher.
- `gravity_app.py` — current UI and Toroid implementation.
- `planetary_sim.py` — base physics/UI implementation reused by the current app.
- `README.md` — user documentation.
- `requirements.txt` — dependencies.
