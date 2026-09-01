from __future__ import annotations

import math
import time
from collections import deque

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import planetary_sim as base


def minimum_image_xy(delta, half_size):
    """Shortest XY displacement on a square 2-D torus of side 2*half_size."""
    d = np.asarray(delta, dtype=float).copy()
    width = 2.0 * float(half_size)
    d[0] -= width * np.round(d[0] / width)
    d[1] -= width * np.round(d[1] / width)
    d[2] = 0.0
    return d


def wrap_toroid(pos, half_size):
    """Wrap XY to [-L, L); Toroid is strictly 2-D, so z is forced to zero."""
    L = float(half_size)
    width = 2.0 * L
    pos[:, 0] = ((pos[:, 0] + L) % width) - L
    pos[:, 1] = ((pos[:, 1] + L) % width) - L
    pos[:, 2] = 0.0


def toroid_accelerations(pos, mass, G, softening, half_size):
    acc = np.zeros_like(pos, dtype=float)
    for i in range(len(mass)):
        for j in range(i + 1, len(mass)):
            rij = minimum_image_xy(pos[j] - pos[i], half_size)
            r2 = float(np.dot(rij, rij) + softening**2)
            inv_r3 = 1.0 / (r2 * math.sqrt(r2))
            acc[i] += G * mass[j] * rij * inv_r3
            acc[j] -= G * mass[i] * rij * inv_r3
    acc[:, 2] = 0.0
    return acc


def mode_diagnostics(pos, vel, mass, G, softening, boundary_mode, half_size):
    if boundary_mode != "Toroid":
        return base.diagnostics(pos, vel, mass, G, softening)

    kinetic = float(0.5 * np.sum(mass[:, None] * vel * vel))
    potential = 0.0
    for i in range(len(mass)):
        for j in range(i + 1, len(mass)):
            rij = minimum_image_xy(pos[j] - pos[i], half_size)
            r = math.sqrt(float(np.dot(rij, rij) + softening**2))
            potential -= G * mass[i] * mass[j] / r

    momentum = np.sum(mass[:, None] * vel, axis=0)
    momentum[2] = 0.0

    # A simple Cartesian COM is not globally meaningful on a periodic torus.
    com = np.array([np.nan, np.nan, 0.0])
    return kinetic, potential, kinetic + potential, momentum, com


def toroid_initial_overlap(pos, radius, half_size):
    for i in range(len(radius)):
        for j in range(i + 1, len(radius)):
            d = minimum_image_xy(pos[j] - pos[i], half_size)
            if np.linalg.norm(d) < radius[i] + radius[j]:
                return i, j
    return None


def earliest_toroid_collision(pos, vel, radius, max_dt, half_size):
    """Earliest contact using the nearest relevant periodic images in XY."""
    W = 2.0 * float(half_size)
    best_t = None
    best = None
    shifts = (-W, 0.0, W)

    for i in range(len(radius)):
        for j in range(i + 1, len(radius)):
            v = (vel[j] - vel[i]).copy()
            v[2] = 0.0
            R = float(radius[i] + radius[j])
            raw = (pos[j] - pos[i]).copy()
            raw[2] = 0.0

            for sx in shifts:
                for sy in shifts:
                    r = raw + np.array([sx, sy, 0.0])
                    c = float(np.dot(r, r) - R * R)

                    if c <= 1e-12:
                        dist = float(np.linalg.norm(r))
                        if dist > 1e-15 and np.dot(v, r / dist) < 0.0:
                            t = 0.0
                        else:
                            continue
                    else:
                        a = float(np.dot(v, v))
                        if a <= 1e-30:
                            continue
                        b = 2.0 * float(np.dot(r, v))
                        disc = b*b - 4.0*a*c
                        if disc < 0.0:
                            continue
                        root = math.sqrt(max(0.0, disc))
                        vals = [q for q in ((-b-root)/(2*a), (-b+root)/(2*a))
                                if -1e-12 <= q <= max_dt + 1e-12]
                        if not vals:
                            continue
                        t = max(0.0, min(vals))

                    if best_t is None or t < best_t:
                        best_t = t
                        best = (i, j, np.array([sx, sy, 0.0]))
    return best_t, best


def elastic_impulse_from_delta(vel, mass, i, j, delta):
    dist = float(np.linalg.norm(delta))
    if dist < 1e-15:
        return False
    n = delta / dist
    vn = float(np.dot(vel[j] - vel[i], n))
    if vn >= 0.0:
        return False
    J = -2.0 * vn / (1.0/mass[i] + 1.0/mass[j])
    impulse = J * n
    vel[i] -= impulse / mass[i]
    vel[j] += impulse / mass[j]
    vel[:, 2] = 0.0
    return True


def drift_toroid(pos, vel, mass, radius, dt, half_size, max_collisions=100):
    remaining = float(dt)
    collisions = 0
    eps_time = max(1e-12, abs(dt) * 1e-10)

    while remaining > eps_time:
        t_hit, hit = earliest_toroid_collision(pos, vel, radius, remaining, half_size)
        if hit is None:
            pos += vel * remaining
            remaining = 0.0
            break

        if t_hit > eps_time:
            pos += vel * t_hit
            remaining -= t_hit

        i, j, shift = hit
        delta = (pos[j] - pos[i]) + shift
        delta[2] = 0.0
        collisions += int(elastic_impulse_from_delta(vel, mass, i, j, delta))

        tiny = min(remaining, eps_time)
        if tiny > 0:
            pos += vel * tiny
            remaining -= tiny

        if collisions >= max_collisions:
            pos += vel * remaining
            remaining = 0.0
            break

    wrap_toroid(pos, half_size)
    return collisions


def physics_step(pos, vel, mass, radius, G, softening, dt,
                 boundary_mode, half_size, internal_substeps):
    substeps = max(1, int(internal_substeps))
    h = float(dt) / substeps
    collisions = 0

    for _ in range(substeps):
        if boundary_mode == "Toroid":
            vel[:, 2] = 0.0
            vel += toroid_accelerations(pos, mass, G, softening, half_size) * h
            vel[:, 2] = 0.0
            collisions += drift_toroid(pos, vel, mass, radius, h, half_size)
        else:
            vel += base.gravitational_accelerations(pos, mass, G, softening) * h
            collisions += base.drift_with_exact_collisions(pos, vel, mass, radius, h)
            if boundary_mode == "Box":
                base.reflect_box(pos, vel, half_size, radius)

    return collisions


class RasterTrail(base.RasterTrail):
    """Fixed-memory raster with lightweight browser preview and toroid-safe edges."""

    def paint(self, positions, toroid=False):
        px, py, valid = self._xy_to_pixel(positions[:, :2])
        current = [(int(px[i]), int(py[i])) if valid[i] else None for i in range(len(px))]
        if self.previous_pixels is None:
            self.previous_pixels = current
            return

        for prev, cur in zip(self.previous_pixels, current):
            if prev is None or cur is None:
                continue
            if toroid:
                if (abs(cur[0] - prev[0]) > self.resolution // 2 or
                        abs(cur[1] - prev[1]) > self.resolution // 2):
                    continue
            self._paint_line(prev[0], prev[1], cur[0], cur[1])
        self.previous_pixels = current

    def image_trace(self, opacity, max_preview=250):
        if not np.any(self.buffer):
            return None
        stride = max(1, int(np.ceil(self.resolution / float(max_preview))))
        view = self.buffer[::stride, ::stride]
        z = np.log1p(view.astype(np.float32))
        zmax = float(np.max(z))
        if zmax > 0:
            z /= zmax
        return go.Heatmap(
            z=z,
            x=np.linspace(-self.limit, self.limit, view.shape[1]),
            y=np.linspace(-self.limit, self.limit, view.shape[0]),
            zmin=0, zmax=1, showscale=False, hoverinfo="skip",
            opacity=max(0.0, min(1.0, float(opacity))),
            colorscale=[[0.0, "rgba(0,0,0,0)"], [1.0, "rgba(120,120,120,1)"]],
        )


def make_fading_trails(names, pos, fading_points):
    trails = [deque(maxlen=max(2, int(fading_points))) for _ in names]
    for i in range(len(names)):
        trails[i].append(pos[i].copy())
    return trails


def fading_trace_xy(points, limit, toroid):
    """Return x/y with None breaks at periodic edge crossings."""
    if len(points) < 2:
        return [], []
    arr = np.asarray(points)
    if not toroid:
        return arr[:, 0].tolist(), arr[:, 1].tolist()

    xs = [float(arr[0, 0])]
    ys = [float(arr[0, 1])]
    threshold = float(limit)
    for k in range(1, len(arr)):
        if abs(arr[k,0]-arr[k-1,0]) > threshold or abs(arr[k,1]-arr[k-1,1]) > threshold:
            xs.append(None); ys.append(None)
        xs.append(float(arr[k,0])); ys.append(float(arr[k,1]))
    return xs, ys


def live_figure(pos, fading_trails, raster_trail, names, mass, radius, n_s, limit,
                boundary_mode, sim_time, trail_mode, permanent_visible, permanent_opacity):
    fig = go.Figure()

    if trail_mode == "Permanent" and permanent_visible and raster_trail is not None:
        raster = raster_trail.image_trace(permanent_opacity)
        if raster is not None:
            fig.add_trace(raster)

    if trail_mode == "Fading" and fading_trails is not None:
        for points in fading_trails:
            x, y = fading_trace_xy(points, limit, boundary_mode == "Toroid")
            if len(x) > 1:
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines", line=dict(width=1),
                    showlegend=False, hoverinfo="skip", opacity=0.45
                ))

    groups = np.array(["S" if i < n_s else "O" for i in range(len(names))], dtype=object)
    custom = np.column_stack([mass, radius, groups, pos[:,2]])
    fig.add_trace(go.Scatter(
        x=pos[:,0], y=pos[:,1], mode="markers+text", text=names,
        textposition="top center", marker=dict(size=base.marker_sizes(radius, mass)),
        customdata=custom,
        hovertemplate=("%{text}<br>x=%{x:.6g}<br>y=%{y:.6g}<br>z=%{customdata[3]:.6g}"
                       "<br>mass=%{customdata[0]:.6g}<br>radius=%{customdata[1]:.6g}<extra></extra>")
    ))

    shapes = []
    if boundary_mode in ("Box", "Toroid"):
        b = float(limit)
        shapes = [dict(type="rect", x0=-b, x1=b, y0=-b, y1=b, line=dict(width=2))]

    subtitle = "2-D minimum-image periodic universe" if boundary_mode == "Toroid" else boundary_mode
    fig.update_layout(
        title=f"Live N-body gravity — t={sim_time:.6g} — {subtitle} — trail: {trail_mode}",
        height=650, shapes=shapes, showlegend=False,
        margin=dict(l=45, r=25, t=60, b=45),
        xaxis=dict(range=[-limit, limit], title="x", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-limit, limit], title="y"),
        uirevision="gravity-live",
    )
    return fig


def live_simulation(n_s,n_o,run_mode,step_limit,sim_duration,real_hours,dt,internal_substeps,
                    calculations_per_frame,frame_delay,G,heavy_mass,ordinary_mass,mass_vector_text,
                    default_s_radius,default_o_radius,radius_vector_text,init_extent,velocity_mode,
                    random_speed,position_rows_text,velocity_rows_text,softening,boundary_mode,
                    box_half_size,view_half_size,auto_view,trail_mode,fading_trail_points,
                    permanent_visible,permanent_opacity,permanent_decimation,permanent_resolution,
                    diagnostic_points,seed):
    try:
        dt=float(dt); G=float(G)
        if dt<=0 or G<=0:
            raise ValueError("Δt and G must be > 0")

        internal_substeps=max(1,int(internal_substeps))
        calculations_per_frame=max(1,int(calculations_per_frame))
        step_limit=max(1,int(step_limit))
        sim_duration=max(0,float(sim_duration))
        real_hours=max(0,float(real_hours))
        frame_delay=max(0,float(frame_delay))
        diagnostic_points=max(10,int(diagnostic_points))
        permanent_decimation=max(1,int(permanent_decimation))
        permanent_resolution=max(200,int(permanent_resolution))

        names,mass,radius,pos,vel=base.initialize_system(
            n_s,n_o,mass_vector_text,heavy_mass,ordinary_mass,
            default_s_radius,default_o_radius,radius_vector_text,
            init_extent,velocity_mode,random_speed,
            position_rows_text,velocity_rows_text,seed
        )

        if boundary_mode in ("Box","Toroid") and np.any(radius >= float(box_half_size)):
            raise ValueError("Every body radius must be smaller than the Box/Toroid half-size")

        if boundary_mode == "Toroid":
            pos[:,2] = 0.0
            vel[:,2] = 0.0
            wrap_toroid(pos, box_half_size)
            overlap = toroid_initial_overlap(pos, radius, box_half_size)
            if overlap is not None:
                raise ValueError(f"Initial toroidal overlap: {names[overlap[0]]} and {names[overlap[1]]}")
            limit = float(box_half_size)
        elif auto_view:
            limit=max(float(box_half_size) if boundary_mode=="Box" else float(view_half_size),
                      float(init_extent),1.0)
        else:
            limit=max(float(view_half_size),1e-9)

        fading_trails=None
        raster_trail=None
        if trail_mode=="Fading":
            fading_trails=make_fading_trails(names,pos,fading_trail_points)
        elif trail_mode=="Permanent":
            raster_trail=RasterTrail(permanent_resolution,limit)
            raster_trail.paint(pos, toroid=(boundary_mode=="Toroid"))

        history=deque(maxlen=diagnostic_points)
        initial_state=pd.DataFrame({
            "name":names,"mass":mass,"radius":radius,
            "x":pos[:,0],"y":pos[:,1],"z":pos[:,2],
            "vx":vel[:,0],"vy":vel[:,1],"vz":vel[:,2]
        })

        step=0
        sim_time=0.0
        wall_start=time.monotonic()
        collision_count=0
        visible_frame=0

        ke,pe,te,p,com=mode_diagnostics(
            pos,vel,mass,G,float(softening),boundary_mode,float(box_half_size)
        )
        e0=te

        def append_history():
            ke_,pe_,te_,p_,com_=mode_diagnostics(
                pos,vel,mass,G,float(softening),boundary_mode,float(box_half_size)
            )
            erel=(te_-e0)/max(abs(e0),1e-15)
            history.append({
                "step":step,"time":sim_time,"K":ke_,"U":pe_,"E":te_,"Erel":erel,
                "Px":p_[0],"Py":p_[1],"Pz":p_[2],"Pmag":float(np.linalg.norm(p_)),
                "COM_x":com_[0],"COM_y":com_[1],"COM_z":com_[2],
                "collisions":collision_count
            })
            return ke_,pe_,te_,p_

        append_history()

        def outputs(status):
            return (
                live_figure(pos,fading_trails,raster_trail,names,mass,radius,int(n_s),
                            limit,boundary_mode,sim_time,trail_mode,
                            permanent_visible,permanent_opacity),
                base.energy_figure(history,e0),
                base.energy_error_figure(history),
                base.momentum_figure(history),
                pd.DataFrame(history),
                initial_state,
                status
            )

        mode_note = " | Toroid: z=vz=0, minimum-image XY gravity" if boundary_mode=="Toroid" else ""
        yield outputs(f"RUNNING | t=0 | K₀={ke:.8g} | U₀={pe:.8g} | E₀={te:.8g}{mode_note}")

        while True:
            elapsed=time.monotonic()-wall_start
            if run_mode=="Fixed steps" and step>=step_limit: break
            if run_mode=="Simulated duration" and sim_time>=sim_duration: break
            if run_mode=="Real-time hours" and elapsed>=real_hours*3600.0: break

            batch=calculations_per_frame
            if run_mode=="Fixed steps":
                batch=min(batch,step_limit-step)
            elif run_mode=="Simulated duration":
                batch=min(batch,max(1,int(math.ceil(max(0,sim_duration-sim_time)/dt))))

            for _ in range(batch):
                collision_count += physics_step(
                    pos,vel,mass,radius,G,float(softening),dt,
                    boundary_mode,float(box_half_size),internal_substeps
                )
                step+=1
                sim_time+=dt

            visible_frame += 1
            if trail_mode=="Fading":
                for i in range(len(names)):
                    fading_trails[i].append(pos[i].copy())
            elif trail_mode=="Permanent" and visible_frame % permanent_decimation == 0:
                raster_trail.paint(pos, toroid=(boundary_mode=="Toroid"))

            ke,pe,te,p=append_history()
            drift=(te-e0)/max(abs(e0),1e-15)
            elapsed=time.monotonic()-wall_start

            trail_note=""
            if trail_mode=="Permanent":
                trail_note=(f" | raster={raster_trail.resolution}² "
                            f"({raster_trail.buffer.nbytes/(1024*1024):.1f} MiB), "
                            f"every {permanent_decimation} frame(s)")
            status=(f"RUNNING | step={step:,} | t={sim_time:.6g} | wall={elapsed:.1f}s | "
                    f"K={ke:.8g} | U={pe:.8g} | E={te:.8g} | ΔE/E₀={drift:+.3e} | "
                    f"|P|={np.linalg.norm(p):.6g} | collisions={collision_count}{trail_note}{mode_note}")
            yield outputs(status)
            if frame_delay>0:
                time.sleep(frame_delay)

        elapsed=time.monotonic()-wall_start
        drift=(te-e0)/max(abs(e0),1e-15)
        yield outputs(f"FINISHED | step={step:,} | t={sim_time:.6g} | wall={elapsed:.1f}s | "
                      f"ΔE/E₀={drift:+.3e} | collisions={collision_count}{mode_note}")

    except Exception as exc:
        empty=go.Figure()
        yield empty,empty,empty,empty,pd.DataFrame(),pd.DataFrame(),f"ERROR: {exc}"


DESCRIPTION = """
# Gravity — Live Planetary / N-body Simulator

Boundary modes:
- **Huge** — unbounded system; the displayed window is only a camera.
- **Box** — hard reflecting walls with elastic wall collisions.
- **Toroid** — a **2-D periodic XY universe**. Crossing one edge re-enters from the opposite edge.
  Pair gravity and collisions use only the **shortest periodic link** (minimum-image convention);
  distant periodic copies are deliberately ignored. In Toroid mode `z = vz = az = 0`.

Trail modes:
- **Fading** — rolling path (default).
- **Permanent** — fixed-memory raster path with decimated browser preview.
- **None** — no trail.

For Toroid, trails break at a periodic edge and continue on the antipodal edge rather than
drawing a false line across the whole displayed universe.
"""


with gr.Blocks(title="Gravity — Live Planetary Simulation") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Objects")
            n_s=gr.Number(value=1,precision=0,label="Number of S bodies")
            n_o=gr.Number(value=1,precision=0,label="Number of O bodies")
            heavy_mass=gr.Number(value=1000.0,label="Default S mass")
            ordinary_mass=gr.Number(value=1.0,label="Default O mass")
            mass_vector=gr.Textbox(label="Optional mass vector",placeholder="1000, 500, 1")
            gr.Markdown("## Physical radii")
            default_s_radius=gr.Number(value=0.5,label="Default S radius")
            default_o_radius=gr.Number(value=0.1,label="Default O radius")
            radius_vector=gr.Textbox(label="Optional radius vector",placeholder="0.5, 0.4, 0.1")
            gr.Markdown("## Initial state")
            init_extent=gr.Number(value=10.0,label="Random position half-extent")
            velocity_mode=gr.Radio(["Zero","Random"],value="Zero",label="Generated initial velocities")
            random_speed=gr.Number(value=0.2,label="Maximum random speed magnitude")
            seed=gr.Number(value=1,precision=0,label="Random seed")
            position_rows=gr.Textbox(label="Optional manual positions: x,y,z per body",
                                     placeholder="0,0,0\n7,0,0",lines=5)
            velocity_rows=gr.Textbox(label="Optional manual velocities: vx,vy,vz per body",
                                     placeholder="0,0,0\n0,0,0",lines=5)
        with gr.Column():
            gr.Markdown("## Run control")
            run_mode=gr.Radio(["Continuous","Fixed steps","Simulated duration","Real-time hours"],
                              value="Continuous",label="Run mode")
            step_limit=gr.Number(value=100000,precision=0,label="Steps")
            sim_duration=gr.Number(value=100.0,label="Simulated time")
            real_hours=gr.Number(value=1.0,label="Real hours to run")
            gr.Markdown("## Physics / live refresh")
            G=gr.Number(value=1.0,label="Gravitational constant G")
            dt=gr.Number(value=0.001,label="Physics Δt (outer step)")
            internal_substeps=gr.Number(value=100,precision=0,label="Internal physics substeps per Δt")
            calculations_per_frame=gr.Number(value=20,precision=0,label="Physics steps per screen refresh")
            frame_delay=gr.Number(value=0.03,label="Delay between screen refreshes (seconds)")
            softening=gr.Number(value=1e-6,label="Gravity softening ε")
            gr.Markdown("## Boundary / view")
            boundary_mode=gr.Radio(["Huge","Box","Toroid"],value="Huge",label="Boundary mode")
            box_half_size=gr.Number(value=12.0,label="Box / Toroid half-size")
            view_half_size=gr.Number(value=12.0,label="Visible XY half-size (Huge mode)")
            auto_view=gr.Checkbox(value=True,label="Automatic initial view size")
            gr.Markdown("## Trails")
            trail_mode=gr.Radio(["Fading","Permanent","None"],value="Fading",label="Trail mode")
            fading_trail_points=gr.Number(value=150,precision=0,label="Fading trail points per object")
            permanent_visible=gr.Checkbox(value=True,label="Permanent trail visible")
            permanent_opacity=gr.Slider(0.0,1.0,value=0.45,step=0.05,label="Permanent trail opacity")
            permanent_decimation=gr.Number(value=5,precision=0,
                                           label="Permanent trail: record every N screen refreshes")
            permanent_resolution=gr.Number(value=1000,precision=0,
                                           label="Permanent raster resolution (pixels per side)")
            diagnostic_points=gr.Number(value=500,precision=0,label="Diagnostic history points kept")

    with gr.Row():
        start_btn=gr.Button("▶ Start / Restart",variant="primary")
        stop_btn=gr.Button("■ Stop",variant="stop")
    status=gr.Textbox(label="Simulation status",interactive=False)

    with gr.Tab("Live simulation"): sim_plot=gr.Plot()
    with gr.Tab("Energy"): energy_plot=gr.Plot()
    with gr.Tab("Energy error"): energy_error_plot=gr.Plot()
    with gr.Tab("Momentum"): momentum_plot=gr.Plot()
    with gr.Tab("Diagnostics"): diagnostics_table=gr.Dataframe(interactive=False)
    with gr.Tab("Initial state"): state_table=gr.Dataframe(interactive=False)

    inputs=[n_s,n_o,run_mode,step_limit,sim_duration,real_hours,dt,internal_substeps,
            calculations_per_frame,frame_delay,G,heavy_mass,ordinary_mass,mass_vector,
            default_s_radius,default_o_radius,radius_vector,init_extent,velocity_mode,
            random_speed,position_rows,velocity_rows,softening,boundary_mode,box_half_size,
            view_half_size,auto_view,trail_mode,fading_trail_points,permanent_visible,
            permanent_opacity,permanent_decimation,permanent_resolution,diagnostic_points,seed]
    outputs=[sim_plot,energy_plot,energy_error_plot,momentum_plot,diagnostics_table,state_table,status]

    run_event=start_btn.click(fn=live_simulation,inputs=inputs,outputs=outputs,
                              concurrency_limit=1,show_progress="hidden")
    stop_btn.click(fn=None,inputs=None,outputs=None,cancels=[run_event],queue=False)


if __name__ == "__main__":
    demo.queue().launch(share=True)
