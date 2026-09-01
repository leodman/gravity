# Project Status — Gravity Simulator

Last updated: 2026-09-01

This file is a compact handoff for continuing development later.

## What currently works

- Live Newtonian N-body gravity simulation in a Gradio web UI.
- 3-D internal vectors with XY visualization.
- S and O object groups with arbitrary positive mass vectors.
- Physical spherical radii with optional radius vector.
- Perfectly elastic, frictionless sphere-sphere collisions.
- Off-center impacts ricochet naturally.
- Collision-time search inside each drift interval to prevent tunneling.
- Symplectic Euler / kick-drift integration: update velocity first, then position.
- Configurable outer `dt` plus configurable internal physics substeps.
- Huge and Box boundary modes.
- Continuous, fixed-step, simulated-duration, and real-time-hour run modes.
- Start/Restart and Stop controls.
- Live kinetic, potential, and total energy diagnostics.
- Explicit initial K0, U0, E0 values.
- Relative energy error `(E-E0)/|E0|` graph.
- Total vector momentum and momentum magnitude.
- Center-of-mass diagnostics.
- Collision counter.
- Fading trail mode.
- Permanent fixed-memory raster trail mode with decimation, opacity, visibility, and selectable raster resolution.
- No-trail mode.
- Google Colab operation.
- Local/Conda/Jupyter operation.

## Agreed physical assumptions

1. Newtonian gravity between all body centers.
2. Bodies are rigid spheres for contact/collision purposes.
3. No deformation.
4. No friction at contact.
5. No spin/rotation is currently modeled.
6. Coefficient of restitution is 1: collisions are perfectly elastic.
7. Collision impulse is normal to the two surfaces, along the center-to-center line at contact.
8. Tangential velocity is unchanged by the collision impulse.
9. Individual kinetic energies and momenta may change during collision, while total kinetic energy and total linear momentum are conserved for an isolated collision up to numerical error.
10. Under gravity, kinetic and potential energy exchange while total mechanical energy should remain approximately constant numerically.

## Important numerical behavior

The physics is conservative, but the numerical integration is approximate.

- Outer `dt` is the user-visible simulation step.
- Each outer step is divided into `internal_substeps`.
- Default internal substeps: 100.
- Higher substep counts improve close-encounter and energy behavior but consume more CPU.
- Relative energy error should stay close to zero.
- Large monotonic energy error means the selected numerical resolution is too coarse.

A known good radial sanity test is described in README.md.

## Trail behavior

### Fading

Uses a bounded recent-history deque and automatically erases old path segments.

### Permanent

Uses a fixed-resolution raster layer instead of an unbounded point list.

The permanent layer is updated only every selected N browser refreshes and connects sampled positions. Memory use therefore stays bounded over long runs.

It is intended for experiments where the simulation may be left running long enough to reveal stable, bounded-chaotic, or divergent behavior.

### None

No trail data is maintained.

## Primary research direction

Use the simulator to explore long-term behavior of gravitational many-body systems with elastic physical collisions:

- stable configurations
- bounded chaotic configurations
- repeated collision systems
- redistribution of energy and momentum
- body ejection
- spatial divergence

## Strong candidate for the next feature

Add a quantitative system-size diagnostic based on center of mass:

`Rmax(t) = max_i |r_i - R_COM|`

Also record the historical maximum.

This would provide an objective bounded/diverging measurement in addition to visual trail inspection.

## Other useful next steps

- Angular momentum vector and magnitude.
- Interactive 3-D plot.
- Escape detection and escape-event logging.
- Configuration save/load.
- Export diagnostics to CSV.
- Export final positions/velocities.
- Leapfrog / velocity-Verlet option and energy-conservation comparison.
- Initial-condition presets.
- A reproducible experiment ID built from configuration + random seed.
- Dynamic hide/show/clear of the permanent trail while a simulation is already running.

## Repository

`https://github.com/leodman/gravity`

Main executable:

`planetary_sim.py`

Documentation:

- `README.md`
- `PROJECT_STATUS.md`

Dependencies:

`requirements.txt`
