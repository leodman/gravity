# Gravity — Planetary / N-body Simulation

Interactive Newtonian N-body gravity simulator written in Python with a Gradio web UI. The same project can run in Google Colab, a normal Python environment, or a Conda/Jupyter setup.

Repository: `https://github.com/leodman/gravity`

## Current physics model

For every body `i`, the simulator computes all pairwise Newtonian gravitational interactions:

`F_ij = G m_i m_j (r_j-r_i) / |r_j-r_i|^3`

The total acceleration is obtained from the sum of all pairwise forces.

The integration order is intentionally:

`v_i(n+1) = v_i(n) + a_i(n) dt`

then

`x_i(n+1) = x_i(n) + v_i(n+1) dt`

This is symplectic Euler / kick-drift: velocity is updated first, then position is advanced using the new velocity.

The selected outer `dt` can be subdivided into many internal physics substeps. The default is 100 internal substeps per outer step. Increasing this improves close-encounter and energy accuracy at the cost of computation time.

## Object groups

Bodies are named:

- `S0`, `S1`, ... for heavy / star-like bodies.
- `O0`, `O1`, ... for ordinary bodies.

Default system:

- `S = 1`
- `O = 1`
- default `mass(S) = 1000`
- default `mass(O) = 1`
- `S0` starts at `(0,0,0)` with zero velocity.
- in the default two-body case, `O0` also starts with zero velocity unless manually overridden.

Masses may be supplied through an arbitrary positive mass vector, so the S/O labels do not impose any physical mass relationship.

## Physical radii and spherical collisions

Bodies are treated as spherical rigid objects for collision purposes while their gravitational field is computed from their center of mass.

Each body has a physical radius `R_i`. You can use default radii for S and O bodies or provide an explicit radius vector.

Two bodies contact when their center-to-center distance reaches:

`d = R_i + R_j`

The simulator uses perfectly elastic, frictionless sphere collisions:

- coefficient of restitution `e = 1`
- impulse acts along the center-to-center normal at contact
- tangential relative velocity is unchanged
- off-center collisions naturally produce ricochets
- linear momentum is transferred between bodies but conserved for the system
- kinetic energy is transferred between bodies but conserved across the instantaneous collision, apart from numerical error

The program searches for the actual contact time inside each drift interval so a fast-moving body cannot simply jump through another sphere between two sampled positions.

## Gravity and energy

During ordinary gravitational motion, kinetic and gravitational potential energy exchange:

`K <-> U`

where

`K = sum(1/2 m_i |v_i|^2)`

and

`U = -sum_{i<j}(G m_i m_j / r_ij)`

Total mechanical energy is:

`E = K + U`

For the ideal Newtonian system with elastic collisions, total mechanical energy is conserved. In the numerical simulation, `E` should remain approximately constant; any drift is numerical error.

The UI explicitly shows the initial values `K0`, `U0`, and `E0`, and plots the relative energy error:

`(E - E0) / |E0|`

This should remain close to zero in a well-resolved run.

## Momentum and center of mass

The simulator tracks the total vector momentum:

`P = sum(m_i v_i)`

and displays:

- `Px`
- `Py`
- `Pz`
- `|P|`

It also records the center of mass coordinates.

In Huge mode, total momentum should remain approximately constant except for numerical error. In Box mode, wall collisions transfer momentum between the simulated bodies and the external box, so body-system momentum alone is not expected to remain constant through wall impacts.

## Initial conditions

Positions may be:

- randomly generated
- entered manually as `x,y,z` rows

Velocities may be:

- zero
- random vectors
- entered manually as `vx,vy,vz` rows

The internal simulation state is always three-dimensional. The current live visualization is an XY projection.

## Boundary modes

### Huge

The visible plotting window is only a camera. A body can leave the visible region and still remains fully part of the gravitational calculation.

This is useful for studying whether bodies escape, the system diverges, or the system remains bounded.

### Box

The simulator uses a cubic reflecting boundary. Spherical bodies bounce elastically from the walls, with the body center constrained by its own physical radius.

## Run modes

The simulator can run in four ways:

- `Continuous` — evolves until the user presses Stop.
- `Fixed steps` — stops after a selected number of outer integration steps.
- `Simulated duration` — stops when the physics clock reaches a selected simulated time.
- `Real-time hours` — continues computing for a selected amount of wall-clock time.

`Physics steps per screen refresh` controls how many outer physics steps are calculated before the browser is updated.

`Delay between screen refreshes` can be used to control animation pacing.

## Trail modes

The live XY display has three trajectory modes.

### Fading

Default mode. A rolling deque keeps only the most recent trail points. Old trail segments disappear automatically.

This is useful for seeing current motion without filling the screen permanently.

### Permanent

Permanent mode uses a fixed-size XY raster trail layer instead of storing an ever-growing list of coordinates.

This is intended for long stability and chaos experiments.

Controls include:

- permanent trail visibility
- trail opacity
- raster resolution
- decimation / record-every-N-refreshes

The raster is updated from decimated screen-refresh positions, not from every internal physics substep. Successive sampled positions are connected so the accumulated path still looks continuous.

Because the permanent trail is stored as a fixed-resolution raster, memory use does not increase with simulation duration.

Repeated visits to the same spatial region accumulate on the raster, which can make frequently occupied trajectories or chaotic regions visually stronger.

### None

No trajectory history is stored or displayed.

## Diagnostics tabs

The web UI currently includes:

- Live simulation
- Energy
- Relative energy error
- Momentum
- Diagnostics table
- Initial state table

Status output includes current step, simulated time, wall time, K, U, E, relative energy error, total momentum magnitude, and collision count.

## Recommended radial collision sanity test

A useful two-body verification is:

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
- `internal substeps = 100`

Expected behavior: O0 falls approximately radially toward S0, contacts it at center separation `0.6`, bounces elastically, and climbs back toward approximately its original distance. S0 recoils slightly because its mass is finite rather than infinite.

The exact return point depends on numerical resolution. Increasing internal substeps should reduce the energy error and improve the return-point accuracy.

## Run in Google Colab

```python
!git clone https://github.com/leodman/gravity.git
%cd gravity
!pip install -r requirements.txt
!python planetary_sim.py
```

For an existing clone:

```python
%cd /content/gravity
!git pull
!python planetary_sim.py
```

Gradio will provide a browser link.

## Run with Conda

```bash
git clone https://github.com/leodman/gravity.git
cd gravity
conda create -n gravity python=3.11
conda activate gravity
pip install -r requirements.txt
python planetary_sim.py
```

The same Gradio interface will open locally.

## Run from Jupyter

After installing the dependencies and entering the repository directory:

```python
%run planetary_sim.py
```

A Conda/local runtime is a good choice for very long experiments because it is not dependent on a temporary Colab session remaining alive.

## Numerical interpretation

The physical laws being modeled conserve total energy and momentum for an isolated system, but the simulator is discrete and finite precision.

Important controls:

- smaller outer `dt` generally improves accuracy
- more internal substeps improve close-encounter resolution
- very close approaches create strong gravitational accelerations and require greater numerical resolution
- gravity softening prevents numerical singular behavior at almost-zero separation, although spherical contact normally prevents physical overlap before reaching zero distance
- large long-term energy drift is a warning that the numerical resolution is insufficient

The relative-energy-error graph is the primary numerical health check.

## Current project goal

The immediate purpose is to experiment with small and medium N-body systems and observe whether they:

- remain stable
- remain bounded but become chaotic
- repeatedly collide and ricochet
- exchange energy and momentum between bodies
- eject one or more bodies
- diverge spatially over long times

The permanent raster trail is specifically intended to make long-term bounded/chaotic structure visible without unbounded memory growth.

## Possible next additions

Useful future extensions include:

- system spatial extent versus time, such as `Rmax(t)` relative to the center of mass
- historical maximum system extent
- angular momentum diagnostics
- 3-D interactive visualization
- selectable higher-order or leapfrog/Verlet integrators for comparison
- automated stability/escape detection
- presets for binary systems, three-body systems, circular orbits, slingshots, and random clusters
- saving/loading full experiment configurations
- exporting diagnostics and final state for later analysis
