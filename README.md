# Gravity — Planetary / N-body Simulation

Interactive Newtonian N-body gravity simulator written in Python with a Gradio web UI. It is designed to run easily in Google Colab and to evolve into a larger planetary-simulation project.

## Physics model

For every body `i` the simulator computes all pairwise Newtonian gravitational interactions:

`F_ij = G m_i m_j (r_j-r_i) / |r_j-r_i|^3`

The total acceleration is obtained from the sum of all pairwise forces. The time integration deliberately updates velocity first and position second:

`v_i(n+1) = v_i(n) + a_i(n) dt`

`x_i(n+1) = x_i(n) + v_i(n+1) dt`

This is the symplectic-Euler / kick-drift integrator. It is simple, but generally behaves better for orbital mechanics than ordinary forward Euler.

## Object groups

Bodies are named:

- `S0`, `S1`, ... for heavy / star-like bodies.
- `O0`, `O1`, ... for ordinary bodies.

Default system:

- `S = 1`
- `O = 1`
- `mass(S) = 1000`
- `mass(O) = 1`
- `S0` starts at `(0,0,0)` with zero velocity.
- With the default two-body system, `O0` also starts with zero velocity unless manually overridden.

Any positive mass vector can be supplied, so there is no required physical relationship between S and O bodies.

## Initial conditions

Positions can be randomly generated or manually entered. Velocities can be zero, random vectors, or manually entered. The internal state is always three-dimensional:

- position `(x,y,z)`
- velocity `(vx,vy,vz)`
- acceleration `(ax,ay,az)`
- momentum `(px,py,pz)`

The first visualization is an XY projection of the full 3-D state.

## Collisions

Bodies are gravitational point masses but use a selectable effective collision radius for contact detection. When two bodies overlap that radius, a frictionless perfectly elastic collision is applied. Momentum and kinetic energy are conserved up to numerical error.

## Boundary modes

### Huge

The visible window is only a camera. A body can leave the visible region and remains part of the gravitational system.

### Box

The simulation uses a cubic reflecting boundary. Bodies bounce elastically from the walls.

## Diagnostics

The web application displays:

- kinetic energy
- gravitational potential energy
- total mechanical energy
- total momentum vector
- momentum magnitude
- center of mass
- relative total-energy drift

## Run in Google Colab

```python
!git clone https://github.com/leodman/gravity.git
%cd gravity
!pip install -r requirements.txt
!python planetary_sim.py
```

Gradio will provide a browser link.

## Numerical notes

- Smaller `dt` generally improves accuracy.
- Close gravitational encounters can require a much smaller `dt`.
- A small selectable gravity softening term prevents numerical singularities at almost-zero separation.
- A future version should add velocity-Verlet/leapfrog, adaptive stepping, angular momentum, orbital trails, and 3-D visualization.
