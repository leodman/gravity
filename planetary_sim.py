from __future__ import annotations

import math
import time
from collections import deque

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def parse_vector(text):
    if text is None or not str(text).strip():
        return None
    cleaned = str(text).replace(";", ",").replace("\n", ",")
    values = []
    for chunk in cleaned.split(","):
        for token in chunk.split():
            if token:
                values.append(float(token))
    return np.asarray(values, dtype=float)


def parse_xyz_rows(text, n):
    if text is None or not str(text).strip():
        return None
    rows = [r.strip() for r in str(text).replace(";", "\n").splitlines() if r.strip()]
    if len(rows) != n:
        raise ValueError(f"Expected {n} XYZ rows, got {len(rows)}")
    out = []
    for row in rows:
        vals = [float(v) for v in row.replace(",", " ").split()]
        if len(vals) != 3:
            raise ValueError("Each manual position/velocity row must contain exactly 3 values")
        out.append(vals)
    return np.asarray(out, dtype=float)


def gravitational_accelerations(pos, mass, G, softening):
    n = len(mass)
    acc = np.zeros_like(pos, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[j] - pos[i]
            r2 = float(np.dot(rij, rij) + softening**2)
            inv_r3 = 1.0 / (r2 * math.sqrt(r2))
            acc[i] += G * mass[j] * rij * inv_r3
            acc[j] -= G * mass[i] * rij * inv_r3
    return acc


def resolve_body_collisions(pos, vel, mass, radius):
    if radius <= 0:
        return
    contact = 2.0 * radius
    n = len(mass)
    for i in range(n):
        for j in range(i + 1, n):
            delta = pos[j] - pos[i]
            dist = float(np.linalg.norm(delta))
            if dist >= contact:
                continue
            normal = np.array([1.0, 0.0, 0.0]) if dist < 1e-14 else delta / dist
            rel_v = vel[j] - vel[i]
            closing_speed = float(np.dot(rel_v, normal))
            if closing_speed < 0:
                impulse_mag = -2.0 * closing_speed / (1.0 / mass[i] + 1.0 / mass[j])
                impulse = impulse_mag * normal
                vel[i] -= impulse / mass[i]
                vel[j] += impulse / mass[j]
            overlap = contact - dist
            if overlap > 0:
                total = mass[i] + mass[j]
                pos[i] -= normal * overlap * mass[j] / total
                pos[j] += normal * overlap * mass[i] / total


def reflect_box(pos, vel, half_size, radius):
    limit = max(float(half_size) - max(float(radius), 0.0), 1e-12)
    for i in range(len(pos)):
        for axis in range(3):
            while pos[i, axis] > limit or pos[i, axis] < -limit:
                if pos[i, axis] > limit:
                    pos[i, axis] = 2.0 * limit - pos[i, axis]
                    vel[i, axis] *= -1.0
                elif pos[i, axis] < -limit:
                    pos[i, axis] = -2.0 * limit - pos[i, axis]
                    vel[i, axis] *= -1.0


def diagnostics(pos, vel, mass, G, softening):
    kinetic = float(0.5 * np.sum(mass[:, None] * vel * vel))
    potential = 0.0
    for i in range(len(mass)):
        for j in range(i + 1, len(mass)):
            rij = pos[j] - pos[i]
            r = math.sqrt(float(np.dot(rij, rij) + softening**2))
            potential -= G * mass[i] * mass[j] / r
    momentum = np.sum(mass[:, None] * vel, axis=0)
    com = np.sum(mass[:, None] * pos, axis=0) / np.sum(mass)
    return kinetic, potential, kinetic + potential, momentum, com


def initialize_system(n_s, n_o, mass_vector_text, heavy_mass, ordinary_mass,
                      init_extent, velocity_mode, random_speed,
                      position_rows_text, velocity_rows_text, seed):
    n_s, n_o = int(n_s), int(n_o)
    n = n_s + n_o
    if n < 1:
        raise ValueError("At least one body is required")

    names = [f"S{i}" for i in range(n_s)] + [f"O{i}" for i in range(n_o)]
    rng = np.random.default_rng(int(seed))

    mass = parse_vector(mass_vector_text)
    if mass is None:
        mass = np.array([float(heavy_mass)] * n_s + [float(ordinary_mass)] * n_o)
    if len(mass) != n:
        raise ValueError(f"Mass vector contains {len(mass)} values but S+O={n}")
    if np.any(mass <= 0):
        raise ValueError("All masses must be > 0")

    pos = parse_xyz_rows(position_rows_text, n)
    if pos is None:
        pos = rng.uniform(-float(init_extent), float(init_extent), size=(n, 3))
        if n_s:
            pos[0] = 0.0
        if n == 2 and np.linalg.norm(pos[1] - pos[0]) < 0.2 * float(init_extent):
            pos[1] = np.array([0.7 * float(init_extent), 0.0, 0.0])

    vel = parse_xyz_rows(velocity_rows_text, n)
    if vel is None:
        vel = np.zeros((n, 3), dtype=float)
        if velocity_mode == "Random":
            direction = rng.normal(size=(n, 3))
            norm = np.linalg.norm(direction, axis=1)
            norm[norm == 0] = 1.0
            direction /= norm[:, None]
            speed = rng.uniform(0.0, float(random_speed), size=n)
            vel = direction * speed[:, None]
        if n_s:
            vel[0] = 0.0
        if n_s == 1 and n_o == 1:
            vel[1] = 0.0

    return names, mass, pos, vel


def marker_sizes(mass):
    lm = np.log10(np.maximum(mass, 1e-300))
    if np.allclose(lm.max(), lm.min()):
        return np.full(len(mass), 11.0)
    return 8.0 + 18.0 * (lm - lm.min()) / (lm.max() - lm.min())


def live_figure(pos, trails, names, mass, n_s, limit, boundary_mode, box_half_size, sim_time):
    fig = go.Figure()

    # Trails first, so objects stay visible above them.
    for i, name in enumerate(names):
        if trails and len(trails[i]) > 1:
            arr = np.asarray(trails[i])
            fig.add_trace(go.Scatter(
                x=arr[:, 0], y=arr[:, 1], mode="lines",
                line=dict(width=1), name=f"{name} trail",
                showlegend=False, hoverinfo="skip", opacity=0.45,
            ))

    groups = np.array(["S" if i < n_s else "O" for i in range(len(names))], dtype=object)
    custom = np.column_stack([mass, groups, pos[:, 2]])
    fig.add_trace(go.Scatter(
        x=pos[:, 0], y=pos[:, 1], mode="markers+text", text=names,
        textposition="top center", marker=dict(size=marker_sizes(mass)),
        customdata=custom,
        hovertemplate=(
            "%{text}<br>x=%{x:.6g}<br>y=%{y:.6g}<br>z=%{customdata[2]:.6g}"
            "<br>mass=%{customdata[0]:.6g}<br>group=%{customdata[1]}<extra></extra>"
        ),
        name="Bodies",
    ))

    shapes = []
    if boundary_mode == "Box":
        b = float(box_half_size)
        shapes.append(dict(type="rect", x0=-b, x1=b, y0=-b, y1=b, line=dict(width=2)))

    fig.update_layout(
        title=f"Live N-body gravity — simulated time t = {sim_time:.6g}",
        height=650,
        shapes=shapes,
        showlegend=False,
        margin=dict(l=45, r=25, t=60, b=45),
        xaxis=dict(range=[-limit, limit], title="x", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-limit, limit], title="y"),
        uirevision="gravity-live",
    )
    return fig


def energy_figure(history):
    df = pd.DataFrame(history)
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(x=df["time"], y=df["K"], name="Kinetic K"))
        fig.add_trace(go.Scatter(x=df["time"], y=df["U"], name="Potential U"))
        fig.add_trace(go.Scatter(x=df["time"], y=df["E"], name="Total E"))
    fig.update_layout(title="Energy", xaxis_title="simulated time", yaxis_title="energy", height=380)
    return fig


def momentum_figure(history):
    df = pd.DataFrame(history)
    fig = go.Figure()
    if not df.empty:
        for c in ["Px", "Py", "Pz", "Pmag"]:
            fig.add_trace(go.Scatter(x=df["time"], y=df[c], name=("|P|" if c == "Pmag" else c)))
    fig.update_layout(title="Total momentum", xaxis_title="simulated time", yaxis_title="momentum", height=380)
    return fig


def live_simulation(n_s, n_o, run_mode, step_limit, sim_duration, real_hours,
                    dt, calculations_per_frame, frame_delay, G,
                    heavy_mass, ordinary_mass, mass_vector_text,
                    init_extent, velocity_mode, random_speed,
                    position_rows_text, velocity_rows_text,
                    collision_radius, softening, boundary_mode,
                    box_half_size, view_half_size, auto_view,
                    trail_points, diagnostic_points, seed):
    """Generator: every yield updates the browser while integration continues."""
    try:
        dt = float(dt)
        G = float(G)
        if dt <= 0 or G <= 0:
            raise ValueError("Δt and G must be > 0")

        calculations_per_frame = max(1, int(calculations_per_frame))
        step_limit = max(1, int(step_limit))
        sim_duration = max(0.0, float(sim_duration))
        real_hours = max(0.0, float(real_hours))
        frame_delay = max(0.0, float(frame_delay))
        trail_points = max(0, int(trail_points))
        diagnostic_points = max(10, int(diagnostic_points))

        names, mass, pos, vel = initialize_system(
            n_s, n_o, mass_vector_text, heavy_mass, ordinary_mass,
            init_extent, velocity_mode, random_speed,
            position_rows_text, velocity_rows_text, seed,
        )

        if auto_view:
            limit = max(float(box_half_size) if boundary_mode == "Box" else float(view_half_size),
                        float(init_extent), 1.0)
        else:
            limit = max(float(view_half_size), 1e-9)

        trails = [deque(maxlen=trail_points or 1) for _ in names]
        for i in range(len(names)):
            trails[i].append(pos[i].copy())

        history = deque(maxlen=diagnostic_points)
        initial_state = pd.DataFrame({
            "name": names, "mass": mass,
            "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
            "vx": vel[:, 0], "vy": vel[:, 1], "vz": vel[:, 2],
        })

        step = 0
        sim_time = 0.0
        wall_start = time.monotonic()

        ke, pe, te, p, com = diagnostics(pos, vel, mass, G, float(softening))
        e0 = te
        history.append({"step": step, "time": sim_time, "K": ke, "U": pe, "E": te,
                        "Px": p[0], "Py": p[1], "Pz": p[2], "Pmag": np.linalg.norm(p)})

        status = "Running — press Stop to end a Continuous or long-duration run."
        yield (live_figure(pos, trails if trail_points else None, names, mass, int(n_s), limit,
                           boundary_mode, box_half_size, sim_time),
               energy_figure(history), momentum_figure(history),
               pd.DataFrame(history), initial_state, status)

        while True:
            # Stop criteria are evaluated before each visible batch.
            elapsed_wall = time.monotonic() - wall_start
            if run_mode == "Fixed steps" and step >= step_limit:
                break
            if run_mode == "Simulated duration" and sim_time >= sim_duration:
                break
            if run_mode == "Real-time hours" and elapsed_wall >= real_hours * 3600.0:
                break
            # Continuous has no automatic termination.

            batch = calculations_per_frame
            if run_mode == "Fixed steps":
                batch = min(batch, step_limit - step)
            elif run_mode == "Simulated duration" and dt > 0:
                remaining = max(0.0, sim_duration - sim_time)
                batch = min(batch, max(1, int(math.ceil(remaining / dt))))

            for _ in range(batch):
                acc = gravitational_accelerations(pos, mass, G, float(softening))
                vel += acc * dt                 # 1) NEW VELOCITY FIRST
                pos += vel * dt                 # 2) POSITION FROM NEW VELOCITY
                resolve_body_collisions(pos, vel, mass, float(collision_radius))
                if boundary_mode == "Box":
                    reflect_box(pos, vel, float(box_half_size), float(collision_radius))
                step += 1
                sim_time += dt

            for i in range(len(names)):
                trails[i].append(pos[i].copy())

            ke, pe, te, p, com = diagnostics(pos, vel, mass, G, float(softening))
            history.append({"step": step, "time": sim_time, "K": ke, "U": pe, "E": te,
                            "Px": p[0], "Py": p[1], "Pz": p[2], "Pmag": np.linalg.norm(p)})

            drift = (te - e0) / max(abs(e0), 1e-15)
            elapsed_wall = time.monotonic() - wall_start
            status = (f"RUNNING | step={step:,} | simulated t={sim_time:.6g} | "
                      f"wall={elapsed_wall:.1f}s | ΔE/E0={drift:+.3e} | |P|={np.linalg.norm(p):.6g}")

            yield (live_figure(pos, trails if trail_points else None, names, mass, int(n_s), limit,
                               boundary_mode, box_half_size, sim_time),
                   energy_figure(history), momentum_figure(history),
                   pd.DataFrame(history), initial_state, status)

            if frame_delay > 0:
                time.sleep(frame_delay)

        elapsed_wall = time.monotonic() - wall_start
        drift = (te - e0) / max(abs(e0), 1e-15)
        status = (f"FINISHED | step={step:,} | simulated t={sim_time:.6g} | "
                  f"wall={elapsed_wall:.1f}s | ΔE/E0={drift:+.3e} | |P|={np.linalg.norm(p):.6g}")
        yield (live_figure(pos, trails if trail_points else None, names, mass, int(n_s), limit,
                           boundary_mode, box_half_size, sim_time),
               energy_figure(history), momentum_figure(history),
               pd.DataFrame(history), initial_state, status)

    except Exception as exc:
        empty = go.Figure()
        yield empty, empty, empty, pd.DataFrame(), pd.DataFrame(), f"ERROR: {exc}"


DESCRIPTION = """
# Gravity — Live Planetary / N-body Simulator

The simulation now **evolves visibly while it is being calculated**.

Integrator: **symplectic Euler / kick-drift**

`v(n+1) = v(n) + a(n) Δt`  
`x(n+1) = x(n) + v(n+1) Δt`

Choose **Continuous** to run indefinitely until you press **Stop**, or choose a fixed number of steps, a simulated duration, or a number of real-world hours.
"""

with gr.Blocks(title="Gravity — Live Planetary Simulation") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Objects")
            n_s = gr.Number(value=1, precision=0, label="Number of S bodies")
            n_o = gr.Number(value=1, precision=0, label="Number of O bodies")
            heavy_mass = gr.Number(value=1000.0, label="Default S mass")
            ordinary_mass = gr.Number(value=1.0, label="Default O mass")
            mass_vector = gr.Textbox(label="Optional mass vector", placeholder="1000, 500, 1")

            gr.Markdown("## Initial state")
            init_extent = gr.Number(value=10.0, label="Random position half-extent")
            velocity_mode = gr.Radio(["Zero", "Random"], value="Zero", label="Generated initial velocities")
            random_speed = gr.Number(value=0.2, label="Maximum random speed magnitude")
            seed = gr.Number(value=1, precision=0, label="Random seed")
            position_rows = gr.Textbox(label="Optional manual positions: x,y,z per body",
                                       placeholder="0,0,0\n7,0,0", lines=5)
            velocity_rows = gr.Textbox(label="Optional manual velocities: vx,vy,vz per body",
                                       placeholder="0,0,0\n0,3,0", lines=5)

        with gr.Column():
            gr.Markdown("## Run control")
            run_mode = gr.Radio(
                ["Continuous", "Fixed steps", "Simulated duration", "Real-time hours"],
                value="Continuous", label="Run mode")
            step_limit = gr.Number(value=100000, precision=0, label="Steps (Fixed steps mode)")
            sim_duration = gr.Number(value=100.0, label="Simulated time (Simulated duration mode)")
            real_hours = gr.Number(value=1.0, label="Real hours to run (Real-time hours mode)")

            gr.Markdown("## Physics / live refresh")
            G = gr.Number(value=1.0, label="Gravitational constant G")
            dt = gr.Number(value=0.001, label="Physics Δt")
            calculations_per_frame = gr.Number(value=20, precision=0,
                                                label="Physics steps per screen refresh")
            frame_delay = gr.Number(value=0.03, label="Delay between screen refreshes (seconds)")
            collision_radius = gr.Number(value=0.05, label="Effective collision radius")
            softening = gr.Number(value=1e-6, label="Gravity softening ε")

            gr.Markdown("## Boundary / view")
            boundary_mode = gr.Radio(["Huge", "Box"], value="Huge", label="Boundary mode")
            box_half_size = gr.Number(value=12.0, label="Box half-size")
            view_half_size = gr.Number(value=12.0, label="Visible XY half-size")
            auto_view = gr.Checkbox(value=True, label="Automatic initial view size")
            trail_points = gr.Number(value=150, precision=0, label="Trail points per object (0 = off)")
            diagnostic_points = gr.Number(value=500, precision=0, label="Diagnostic history points kept")

    with gr.Row():
        start_btn = gr.Button("▶ Start / Restart", variant="primary")
        stop_btn = gr.Button("■ Stop", variant="stop")

    status = gr.Textbox(label="Simulation status", interactive=False)

    with gr.Tab("Live simulation"):
        sim_plot = gr.Plot()
    with gr.Tab("Energy"):
        energy_plot = gr.Plot()
    with gr.Tab("Momentum"):
        momentum_plot = gr.Plot()
    with gr.Tab("Diagnostics"):
        diagnostics_table = gr.Dataframe(interactive=False)
    with gr.Tab("Initial state"):
        state_table = gr.Dataframe(interactive=False)

    inputs = [
        n_s, n_o, run_mode, step_limit, sim_duration, real_hours,
        dt, calculations_per_frame, frame_delay, G,
        heavy_mass, ordinary_mass, mass_vector,
        init_extent, velocity_mode, random_speed,
        position_rows, velocity_rows,
        collision_radius, softening, boundary_mode,
        box_half_size, view_half_size, auto_view,
        trail_points, diagnostic_points, seed,
    ]
    outputs = [sim_plot, energy_plot, momentum_plot, diagnostics_table, state_table, status]

    run_event = start_btn.click(
        fn=live_simulation,
        inputs=inputs,
        outputs=outputs,
        concurrency_limit=1,
        show_progress="hidden",
    )

    # Gradio cancellation interrupts a Continuous/long generator immediately.
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[run_event], queue=False)

if __name__ == "__main__":
    demo.queue().launch(share=True)
