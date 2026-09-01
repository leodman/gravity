# Gravity — Planetary / N-body Simulation

Interactive Newtonian N-body gravity simulator with a Gradio web UI. It runs in Google Colab, standard Python, or Conda/Jupyter.

Repository: `https://github.com/leodman/gravity`

## Recommended launcher

Use:

```bash
python run_gravity.py
```

The current app is implemented in `gravity_app.py` and reuses the proven base code in `planetary_sim.py`.

## Physics integration

For every body, pairwise Newtonian gravity is calculated and summed. The integration order is intentionally:

`v(n+1) = v(n) + a(n) dt`

then:

`x(n+1) = x(n) + v(n+1) dt`

This is symplectic Euler / kick-drift: velocity first, position second.

The user-selected outer `dt` is subdivided into configurable internal physics substeps. The default is 100. More substeps improve close-encounter and energy accuracy at the cost of CPU time.

## Objects

Bodies are named:

- `S0`, `S1`, ... for heavy/star-like objects.
- `O0`, `O1`, ... for ordinary objects.

Default masses are `S = 1000` and `O = 1`, but arbitrary positive mass vectors may be supplied.

Each body also has a physical radius. Default S/O radii or an explicit radius vector may be used.

## Spherical elastic collisions

Bodies are rigid, smooth spheres for contact purposes.

Two bodies contact when:

`distance = R_i + R_j`

Collisions are perfectly elastic and frictionless:

- coefficient of restitution `e = 1`
- impulse acts along the contact normal / center-to-center line
- tangential relative velocity is unchanged
- off-center impacts naturally ricochet
- kinetic energy and momentum transfer between bodies
- total kinetic energy and total linear momentum are conserved across the instantaneous collision up to numerical error

The code searches for contact inside each drift interval so fast bodies do not simply tunnel through one another between sampled positions.

## Energy

The diagnostics show:

- kinetic energy `K`
- gravitational potential energy `U`
- total mechanical energy `E = K + U`
- initial `K0`, `U0`, `E0`
- relative energy error `(E-E0)/|E0|`

For a well-resolved isolated run, total mechanical energy should remain approximately constant. Large monotonic energy drift means the numerical resolution is too coarse.

## Momentum

The simulator tracks `Px`, `Py`, `Pz`, and `|P|`.

In Huge mode momentum should remain approximately constant. In Box mode the walls apply external impulses to the body system, so body-system momentum changes during wall impacts.

## Boundary modes

### Huge

Unbounded 3-D system. The displayed XY window is only a camera. Bodies may leave it while remaining fully active in the calculation.

### Box

A 3-D reflecting box. Bodies bounce elastically from the walls.

### Toroid

A **2-D periodic XY universe** represented by the same square display.

If the Toroid half-size is `L`, the visible universe is `[-L,+L)` in X and Y. A body crossing one edge re-enters from the opposite edge with unchanged velocity.

Toroid mode deliberately disables the third dimension:

`z = vz = az = 0`

Gravity uses only the **shortest periodic link** between each pair of bodies (minimum-image convention). If the full width is `W = 2L`:

`dx = dx - W * round(dx/W)`

and similarly for Y.

The code deliberately does **not** add gravitational forces from the infinite set of farther periodic copies.

Collisions use the same periodic geometry, so two bodies near opposite edges can attract and collide across that edge.

Toroid potential energy uses the same minimum-image pair distance as the force calculation.

A normal Cartesian center of mass is not globally meaningful on a periodic torus, so Toroid COM X/Y should not be interpreted as ordinary physical COM coordinates.

## Run modes

- `Continuous` — runs until Stop.
- `Fixed steps` — stops after a selected number of outer steps.
- `Simulated duration` — stops at a selected simulation time.
- `Real-time hours` — computes for a selected amount of wall-clock time.

`Physics steps per screen refresh` controls how much physics is calculated between browser updates.

## Trail modes

### Fading

Default rolling trail. Only recent points are kept and old segments disappear.

### Permanent

Fixed-memory XY raster intended for long stability/chaos experiments.

Controls include:

- visible / hidden
- opacity
- raster resolution
- record every N screen refreshes

The full raster stays in memory at fixed size, while only a downsampled preview is sent to Plotly. This prevents Permanent mode from overwhelming the Energy and Momentum plot updates.

Repeated visits accumulate in the raster, making frequently occupied regions stronger.

In Toroid mode, trails break at an edge and continue on the antipodal edge; they do not draw a false diagonal line across the whole universe.

### None

No trail history is maintained.

## Useful radial sanity test

Use:

- `S0 mass = 1000`
- `O0 mass = 1`
- `S0 radius = 0.5`
- `O0 radius = 0.1`
- positions:

```text
0,0,0
7,0,0
```

- velocities:

```text
0,0,0
0,0,0
```

- `dt = 0.001`
- internal substeps `= 100`

Expected behavior: O0 falls radially toward S0, contacts it at separation `0.6`, bounces elastically, and climbs back near its original turning point. Increasing internal substeps should improve return-point and energy accuracy.

## Google Colab

Fresh clone:

```python
!git clone https://github.com/leodman/gravity.git
%cd gravity
!pip install -r requirements.txt
!python run_gravity.py
```

Existing clone:

```python
%cd /content/gravity
!git pull
!python run_gravity.py
```

## Conda / local Python

```bash
git clone https://github.com/leodman/gravity.git
cd gravity
conda create -n gravity python=3.11
conda activate gravity
pip install -r requirements.txt
python run_gravity.py
```

For long experiments, a local Conda machine is often preferable to a temporary Colab runtime.

## Jupyter

From the repository directory after installing dependencies:

```python
%run run_gravity.py
```

## Current research goal

Use the simulator to explore systems that may:

- remain stable
- remain bounded but become chaotic
- repeatedly collide and ricochet
- redistribute energy and momentum
- eject bodies
- diverge spatially
- develop periodic or complicated structures in the 2-D Toroid universe

See `PROJECT_STATUS.md` for the current handoff and next-development ideas.
