from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr


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
            raise ValueError("Each manual position/velocity row must contain x,y,z")
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
    limit = max(float(half_size) - max(radius, 0.0), 1e-12)
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
            r = math.sqrt(float(np.dot(pos[j] - pos[i], pos[j] - pos[i]) + softening**2))
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
            speed = rng.uniform(0, float(random_speed), size=n)
            vel = direction * speed[:, None]
        if n_s:
            vel[0] = 0.0
        if n_s == 1 and n_o == 1:
            vel[1] = 0.0
    return names, mass, pos, vel


def marker_sizes(mass):
    lm = np.log10(np.maximum(mass, 1e-300))
    if np.allclose(lm.max(), lm.min()):
        return np.full(len(mass), 10.0)
    return 7.0 + 15.0 * (lm - lm.min()) / (lm.max() - lm.min())


def animation_figure(frames_pos, times, names, mass, n_s, limit, boundary_mode, box_half_size):
    sizes = marker_sizes(mass)
    groups = ["S" if i < n_s else "O" for i in range(len(names))]

    def trace(frame):
        custom = np.column_stack([mass, groups])
        return go.Scatter(
            x=frame[:, 0], y=frame[:, 1], mode="markers+text", text=names,
            textposition="top center", marker=dict(size=sizes), customdata=custom,
            hovertemplate="%{text}<br>x=%{x:.6g}<br>y=%{y:.6g}<br>mass=%{customdata[0]}<br>group=%{customdata[1]}<extra></extra>"
        )

    fig = go.Figure(data=[trace(frames_pos[0])])
    fig.frames = [go.Frame(data=[trace(f)], name=str(k),
                           layout=go.Layout(title_text=f"N-body gravity — t={times[k]:.6g}"))
                  for k, f in enumerate(frames_pos)]
    shapes = []
    if boundary_mode == "Box":
        b = float(box_half_size)
        shapes.append(dict(type="rect", x0=-b, x1=b, y0=-b, y1=b, line=dict(width=2)))
    fig.update_layout(
        title=f"N-body gravity — t={times[0]:.6g}", height=650, shapes=shapes,
        xaxis=dict(range=[-limit, limit], title="x", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-limit, limit], title="y"),
        updatemenus=[dict(type="buttons", showactive=False, buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), transition=dict(duration=0), fromcurrent=True)]),
            dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])
        ])],
        sliders=[dict(currentvalue=dict(prefix="time: "), steps=[
            dict(method="animate", args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))], label=f"{times[k]:.3g}")
            for k in range(len(frames_pos))
        ])]
    )
    return fig


def line_figure(df, columns, title, ytitle):
    fig = go.Figure()
    for col, label in columns:
        fig.add_trace(go.Scatter(x=df["time"], y=df[col], name=label))
    fig.update_layout(title=title, xaxis_title="time", yaxis_title=ytitle, height=420)
    return fig


def simulate(n_s, n_o, steps, dt, G, heavy_mass, ordinary_mass, mass_vector_text,
             init_extent, velocity_mode, random_speed, position_rows_text,
             velocity_rows_text, collision_radius, softening, boundary_mode,
             box_half_size, view_half_size, auto_view, record_every, seed):
    try:
        steps = int(steps)
        dt = float(dt)
        G = float(G)
        record_every = max(1, int(record_every))
        if steps < 1 or dt <= 0 or G <= 0:
            raise ValueError("steps, dt and G must be positive")

        names, mass, pos, vel = initialize_system(
            n_s, n_o, mass_vector_text, heavy_mass, ordinary_mass, init_extent,
            velocity_mode, random_speed, position_rows_text, velocity_rows_text, seed
        )
        initial_pos = pos.copy()
        initial_vel = vel.copy()
        frames_pos = [pos.copy()]
        times = [0.0]
        hist = []

        def record(step):
            ke, pe, te, p, com = diagnostics(pos, vel, mass, G, float(softening))
            hist.append([step, step * dt, ke, pe, te, p[0], p[1], p[2], np.linalg.norm(p), com[0], com[1], com[2]])

        record(0)
        for step in range(1, steps + 1):
            acc = gravitational_accelerations(pos, mass, G, float(softening))
            vel += acc * dt                         # new velocity FIRST
            pos += vel * dt                         # new position from new velocity
            resolve_body_collisions(pos, vel, mass, float(collision_radius))
            if boundary_mode == "Box":
                reflect_box(pos, vel, float(box_half_size), float(collision_radius))
            if step % record_every == 0 or step == steps:
                frames_pos.append(pos.copy())
                times.append(step * dt)
                record(step)

        df = pd.DataFrame(hist, columns=["step", "time", "kinetic_energy", "potential_energy", "total_energy", "Px", "Py", "Pz", "|P|", "COM_x", "COM_y", "COM_z"])
        state = pd.DataFrame({
            "name": names, "mass": mass,
            "x": initial_pos[:, 0], "y": initial_pos[:, 1], "z": initial_pos[:, 2],
            "vx": initial_vel[:, 0], "vy": initial_vel[:, 1], "vz": initial_vel[:, 2]
        })

        if auto_view:
            if boundary_mode == "Huge":
                limit = max(float(init_extent), float(view_half_size), 1.0)
            else:
                limit = max(float(box_half_size), 1.0)
        else:
            limit = max(float(view_half_size), 1e-6)

        sim_fig = animation_figure(frames_pos, times, names, mass, int(n_s), limit, boundary_mode, box_half_size)
        e_fig = line_figure(df, [("kinetic_energy", "Kinetic K"), ("potential_energy", "Potential U"), ("total_energy", "Total E")], "Energy", "energy")
        p_fig = line_figure(df, [("Px", "Px"), ("Py", "Py"), ("Pz", "Pz"), ("|P|", "|P|")], "Total momentum", "momentum")

        e0, ef = df.iloc[0]["total_energy"], df.iloc[-1]["total_energy"]
        drift = (ef - e0) / max(abs(e0), 1e-15)
        status = f"Simulated {len(names)} bodies for {steps} steps; recorded {len(frames_pos)} frames. Relative energy drift={drift:+.3e}; final |P|={df.iloc[-1]['|P|']:.6g}."
        return sim_fig, e_fig, p_fig, df, state, status
    except Exception as exc:
        empty = go.Figure()
        return empty, empty, empty, pd.DataFrame(), pd.DataFrame(), f"ERROR: {exc}"


DESCRIPTION = """
# Gravity — Planetary / N-body Simulator

Classical Newtonian N-body simulation with 3-D vectors and an animated XY view.

Integrator: **symplectic Euler / kick-drift**

`v(n+1) = v(n) + a(n) Δt`

`x(n+1) = x(n) + v(n+1) Δt`

The velocity is deliberately computed first, then the new position.
"""

with gr.Blocks(title="Gravity — Planetary Simulation") as demo:
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
            position_rows = gr.Textbox(label="Optional manual positions: one x,y,z row per body", placeholder="0,0,0\n7,0,0", lines=5)
            velocity_rows = gr.Textbox(label="Optional manual velocities: one vx,vy,vz row per body", placeholder="0,0,0\n0,3,0", lines=5)
        with gr.Column():
            gr.Markdown("## Physics")
            G = gr.Number(value=1.0, label="Gravitational constant G")
            dt = gr.Number(value=0.001, label="Δt")
            steps = gr.Number(value=5000, precision=0, label="Integration steps")
            collision_radius = gr.Number(value=0.05, label="Effective collision radius")
            softening = gr.Number(value=1e-6, label="Gravity softening ε")
            gr.Markdown("## Boundary / view")
            boundary_mode = gr.Radio(["Huge", "Box"], value="Huge", label="Boundary mode")
            box_half_size = gr.Number(value=12.0, label="Box half-size")
            view_half_size = gr.Number(value=12.0, label="Visible XY half-size")
            auto_view = gr.Checkbox(value=True, label="Automatic visible-window size")
            record_every = gr.Number(value=20, precision=0, label="Record every N steps")
            run_btn = gr.Button("Run simulation", variant="primary")

    status = gr.Textbox(label="Simulation status", interactive=False)
    with gr.Tab("Simulation"):
        sim_plot = gr.Plot()
    with gr.Tab("Energy"):
        energy_plot = gr.Plot()
    with gr.Tab("Momentum"):
        momentum_plot = gr.Plot()
    with gr.Tab("Diagnostics"):
        diagnostics_table = gr.Dataframe(interactive=False)
    with gr.Tab("Initial state"):
        state_table = gr.Dataframe(interactive=False)

    run_btn.click(
        simulate,
        inputs=[n_s, n_o, steps, dt, G, heavy_mass, ordinary_mass, mass_vector,
                init_extent, velocity_mode, random_speed, position_rows, velocity_rows,
                collision_radius, softening, boundary_mode, box_half_size,
                view_half_size, auto_view, record_every, seed],
        outputs=[sim_plot, energy_plot, momentum_plot, diagnostics_table, state_table, status]
    )

if __name__ == "__main__":
    demo.launch(share=True)
