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


def initialize_system(n_s, n_o, mass_vector_text, heavy_mass, ordinary_mass,
                      default_s_radius, default_o_radius, radius_vector_text,
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
    if len(mass) != n or np.any(mass <= 0):
        raise ValueError("Mass vector must match S+O and all masses must be > 0")

    radius = parse_vector(radius_vector_text)
    if radius is None:
        radius = np.array([float(default_s_radius)] * n_s + [float(default_o_radius)] * n_o)
    if len(radius) != n or np.any(radius < 0):
        raise ValueError("Radius vector must match S+O and all radii must be >= 0")

    pos = parse_xyz_rows(position_rows_text, n)
    if pos is None:
        pos = rng.uniform(-float(init_extent), float(init_extent), size=(n, 3))
        if n_s:
            pos[0] = 0.0
        if n == 2 and np.linalg.norm(pos[1] - pos[0]) < max(0.2 * float(init_extent), radius.sum() * 1.5):
            pos[1] = np.array([max(0.7 * float(init_extent), radius.sum() * 2.0), 0.0, 0.0])

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[j] - pos[i]) < radius[i] + radius[j]:
                raise ValueError(f"Initial overlap: {names[i]} and {names[j]}")

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
    return names, mass, radius, pos, vel


def gravitational_accelerations(pos, mass, G, softening):
    acc = np.zeros_like(pos, dtype=float)
    for i in range(len(mass)):
        for j in range(i + 1, len(mass)):
            rij = pos[j] - pos[i]
            r2 = float(np.dot(rij, rij) + softening**2)
            inv_r3 = 1.0 / (r2 * math.sqrt(r2))
            acc[i] += G * mass[j] * rij * inv_r3
            acc[j] -= G * mass[i] * rij * inv_r3
    return acc


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


def elastic_sphere_impulse(pos, vel, mass, i, j):
    delta = pos[j] - pos[i]
    dist = float(np.linalg.norm(delta))
    if dist < 1e-15:
        return False
    n = delta / dist
    vn = float(np.dot(vel[j] - vel[i], n))
    if vn >= 0.0:
        return False
    J = -2.0 * vn / (1.0 / mass[i] + 1.0 / mass[j])
    impulse = J * n
    vel[i] -= impulse / mass[i]
    vel[j] += impulse / mass[j]
    return True


def earliest_collision_time(pos, vel, radius, max_dt):
    best_t, best_pair = None, None
    for i in range(len(radius)):
        for j in range(i + 1, len(radius)):
            r = pos[j] - pos[i]
            v = vel[j] - vel[i]
            R = radius[i] + radius[j]
            c = float(np.dot(r, r) - R * R)
            if c <= 1e-12:
                dist = float(np.linalg.norm(r))
                if dist > 1e-15 and np.dot(v, r / dist) < 0:
                    t = 0.0
                else:
                    continue
            else:
                a = float(np.dot(v, v))
                if a <= 1e-30:
                    continue
                b = 2.0 * float(np.dot(r, v))
                disc = b * b - 4.0 * a * c
                if disc < 0:
                    continue
                root = math.sqrt(disc)
                candidates = [t for t in ((-b-root)/(2*a), (-b+root)/(2*a)) if -1e-12 <= t <= max_dt + 1e-12]
                if not candidates:
                    continue
                t = max(0.0, min(candidates))
            if best_t is None or t < best_t:
                best_t, best_pair = t, (i, j)
    return best_t, best_pair


def drift_with_exact_collisions(pos, vel, mass, radius, dt, max_collisions=100):
    remaining = float(dt)
    collisions = 0
    eps_time = max(1e-12, abs(dt) * 1e-10)
    while remaining > eps_time:
        t_hit, pair = earliest_collision_time(pos, vel, radius, remaining)
        if pair is None:
            pos += vel * remaining
            break
        if t_hit > eps_time:
            pos += vel * t_hit
            remaining -= t_hit
        i, j = pair
        collisions += int(elastic_sphere_impulse(pos, vel, mass, i, j))
        tiny = min(remaining, eps_time)
        if tiny > 0:
            pos += vel * tiny
            remaining -= tiny
        if collisions >= max_collisions:
            pos += vel * remaining
            break
    return collisions


def reflect_box(pos, vel, half_size, radius):
    for i in range(len(pos)):
        lim = max(float(half_size) - float(radius[i]), 1e-12)
        for axis in range(3):
            while pos[i, axis] > lim or pos[i, axis] < -lim:
                if pos[i, axis] > lim:
                    pos[i, axis] = 2 * lim - pos[i, axis]
                    vel[i, axis] *= -1
                elif pos[i, axis] < -lim:
                    pos[i, axis] = -2 * lim - pos[i, axis]
                    vel[i, axis] *= -1


def physics_step(pos, vel, mass, radius, G, softening, dt, boundary_mode, box_half_size, internal_substeps):
    h = float(dt) / max(1, int(internal_substeps))
    collisions = 0
    for _ in range(max(1, int(internal_substeps))):
        vel += gravitational_accelerations(pos, mass, G, softening) * h
        collisions += drift_with_exact_collisions(pos, vel, mass, radius, h)
        if boundary_mode == "Box":
            reflect_box(pos, vel, box_half_size, radius)
    return collisions


def marker_sizes(radius, mass):
    if np.max(radius) > 0:
        return 8.0 + 22.0 * radius / max(float(np.max(radius)), 1e-15)
    lm = np.log10(np.maximum(mass, 1e-300))
    if np.allclose(lm.max(), lm.min()):
        return np.full(len(mass), 11.0)
    return 8.0 + 18.0 * (lm-lm.min()) / (lm.max()-lm.min())


def make_trails(names, pos, mode, fading_points, permanent_cap):
    if mode == "None":
        return None
    if mode == "Fading":
        trails = [deque(maxlen=max(2, int(fading_points))) for _ in names]
    else:
        cap = max(0, int(permanent_cap))
        trails = [([] if cap == 0 else deque(maxlen=cap)) for _ in names]
    for i in range(len(names)):
        trails[i].append(pos[i].copy())
    return trails


def live_figure(pos, trails, names, mass, radius, n_s, limit, boundary_mode, box_half_size, sim_time, trail_mode):
    fig = go.Figure()
    if trails is not None:
        for i, name in enumerate(names):
            if len(trails[i]) > 1:
                arr = np.asarray(trails[i])
                fig.add_trace(go.Scatter(x=arr[:,0], y=arr[:,1], mode="lines",
                    line=dict(width=1), name=f"{name} trail", showlegend=False,
                    hoverinfo="skip", opacity=0.45 if trail_mode == "Fading" else 0.75))
    groups = np.array(["S" if i < n_s else "O" for i in range(len(names))], dtype=object)
    custom = np.column_stack([mass, radius, groups, pos[:,2]])
    fig.add_trace(go.Scatter(x=pos[:,0], y=pos[:,1], mode="markers+text", text=names,
        textposition="top center", marker=dict(size=marker_sizes(radius,mass)), customdata=custom,
        hovertemplate="%{text}<br>x=%{x:.6g}<br>y=%{y:.6g}<br>z=%{customdata[3]:.6g}<br>mass=%{customdata[0]:.6g}<br>radius=%{customdata[1]:.6g}<extra></extra>"))
    shapes=[]
    if boundary_mode == "Box":
        b=float(box_half_size); shapes=[dict(type="rect",x0=-b,x1=b,y0=-b,y1=b,line=dict(width=2))]
    fig.update_layout(title=f"Live N-body gravity — t={sim_time:.6g} — trail: {trail_mode}", height=650,
        shapes=shapes, showlegend=False, margin=dict(l=45,r=25,t=60,b=45),
        xaxis=dict(range=[-limit,limit],title="x",scaleanchor="y",scaleratio=1),
        yaxis=dict(range=[-limit,limit],title="y"), uirevision="gravity-live")
    return fig


def energy_figure(history, e0):
    df=pd.DataFrame(history); fig=go.Figure()
    if not df.empty:
        for col,label in [("K","Kinetic K"),("U","Potential U"),("E","Total E")]:
            fig.add_trace(go.Scatter(x=df.time,y=df[col],name=label,mode="lines+markers",marker=dict(size=4)))
        fig.add_hline(y=e0,line_dash="dot",annotation_text="E₀")
    fig.update_layout(title=f"Energy — initial total E₀ = {e0:.8g}",xaxis_title="simulated time",yaxis_title="energy",height=420)
    return fig


def energy_error_figure(history):
    df=pd.DataFrame(history); fig=go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(x=df.time,y=df.Erel,name="(E-E₀)/|E₀|",mode="lines+markers",marker=dict(size=4)))
        fig.add_hline(y=0.0,line_dash="dot")
    fig.update_layout(title="Relative total-energy error",xaxis_title="simulated time",yaxis_title="relative error",height=360)
    return fig


def momentum_figure(history):
    df=pd.DataFrame(history); fig=go.Figure()
    if not df.empty:
        for c in ["Px","Py","Pz","Pmag"]:
            fig.add_trace(go.Scatter(x=df.time,y=df[c],name="|P|" if c=="Pmag" else c,mode="lines+markers",marker=dict(size=3)))
    fig.update_layout(title="Total momentum",xaxis_title="simulated time",yaxis_title="momentum",height=380)
    return fig


def live_simulation(n_s,n_o,run_mode,step_limit,sim_duration,real_hours,dt,internal_substeps,
                    calculations_per_frame,frame_delay,G,heavy_mass,ordinary_mass,mass_vector_text,
                    default_s_radius,default_o_radius,radius_vector_text,init_extent,velocity_mode,
                    random_speed,position_rows_text,velocity_rows_text,softening,boundary_mode,
                    box_half_size,view_half_size,auto_view,trail_mode,trail_points,permanent_trail_cap,
                    diagnostic_points,seed):
    try:
        dt=float(dt); G=float(G)
        if dt<=0 or G<=0: raise ValueError("Δt and G must be > 0")
        internal_substeps=max(1,int(internal_substeps)); calculations_per_frame=max(1,int(calculations_per_frame))
        step_limit=max(1,int(step_limit)); sim_duration=max(0,float(sim_duration)); real_hours=max(0,float(real_hours))
        frame_delay=max(0,float(frame_delay)); diagnostic_points=max(10,int(diagnostic_points))
        names,mass,radius,pos,vel=initialize_system(n_s,n_o,mass_vector_text,heavy_mass,ordinary_mass,
            default_s_radius,default_o_radius,radius_vector_text,init_extent,velocity_mode,random_speed,
            position_rows_text,velocity_rows_text,seed)
        if boundary_mode=="Box" and np.any(radius>=float(box_half_size)):
            raise ValueError("Every body radius must be smaller than box half-size")
        limit=max(float(box_half_size) if boundary_mode=="Box" else float(view_half_size),float(init_extent),1.0) if auto_view else max(float(view_half_size),1e-9)
        trails=make_trails(names,pos,trail_mode,trail_points,permanent_trail_cap)
        history=deque(maxlen=diagnostic_points)
        initial_state=pd.DataFrame({"name":names,"mass":mass,"radius":radius,"x":pos[:,0],"y":pos[:,1],"z":pos[:,2],"vx":vel[:,0],"vy":vel[:,1],"vz":vel[:,2]})
        step=0; sim_time=0.0; wall_start=time.monotonic(); collision_count=0
        ke,pe,te,p,com=diagnostics(pos,vel,mass,G,float(softening)); e0=te
        def append_history():
            ke_,pe_,te_,p_,com_=diagnostics(pos,vel,mass,G,float(softening))
            history.append({"step":step,"time":sim_time,"K":ke_,"U":pe_,"E":te_,"Erel":(te_-e0)/max(abs(e0),1e-15),
                "Px":p_[0],"Py":p_[1],"Pz":p_[2],"Pmag":float(np.linalg.norm(p_)),"COM_x":com_[0],"COM_y":com_[1],"COM_z":com_[2],"collisions":collision_count})
            return ke_,pe_,te_,p_
        append_history()
        status=f"RUNNING | t=0 | K₀={ke:.8g} | U₀={pe:.8g} | E₀={te:.8g} | trail={trail_mode}"
        yield live_figure(pos,trails,names,mass,radius,int(n_s),limit,boundary_mode,box_half_size,sim_time,trail_mode),energy_figure(history,e0),energy_error_figure(history),momentum_figure(history),pd.DataFrame(history),initial_state,status
        while True:
            elapsed=time.monotonic()-wall_start
            if run_mode=="Fixed steps" and step>=step_limit: break
            if run_mode=="Simulated duration" and sim_time>=sim_duration: break
            if run_mode=="Real-time hours" and elapsed>=real_hours*3600: break
            batch=calculations_per_frame
            if run_mode=="Fixed steps": batch=min(batch,step_limit-step)
            elif run_mode=="Simulated duration": batch=min(batch,max(1,int(math.ceil(max(0,sim_duration-sim_time)/dt))))
            for _ in range(batch):
                collision_count+=physics_step(pos,vel,mass,radius,G,float(softening),dt,boundary_mode,float(box_half_size),internal_substeps)
                step+=1; sim_time+=dt
            if trails is not None:
                for i in range(len(names)): trails[i].append(pos[i].copy())
            ke,pe,te,p=append_history(); drift=(te-e0)/max(abs(e0),1e-15); elapsed=time.monotonic()-wall_start
            trail_count=0 if trails is None else len(trails[0])
            status=f"RUNNING | step={step:,} | t={sim_time:.6g} | wall={elapsed:.1f}s | E={te:.8g} | ΔE/E₀={drift:+.3e} | |P|={np.linalg.norm(p):.6g} | collisions={collision_count} | trail={trail_mode} ({trail_count:,} pts/body)"
            yield live_figure(pos,trails,names,mass,radius,int(n_s),limit,boundary_mode,box_half_size,sim_time,trail_mode),energy_figure(history,e0),energy_error_figure(history),momentum_figure(history),pd.DataFrame(history),initial_state,status
            if frame_delay>0: time.sleep(frame_delay)
        status=f"FINISHED | step={step:,} | t={sim_time:.6g} | collisions={collision_count} | trail={trail_mode}"
        yield live_figure(pos,trails,names,mass,radius,int(n_s),limit,boundary_mode,box_half_size,sim_time,trail_mode),energy_figure(history,e0),energy_error_figure(history),momentum_figure(history),pd.DataFrame(history),initial_state,status
    except Exception as exc:
        empty=go.Figure(); yield empty,empty,empty,empty,pd.DataFrame(),pd.DataFrame(),f"ERROR: {exc}"


DESCRIPTION="""
# Gravity — Live Planetary / N-body Simulator

Newtonian gravity with rigid spherical bodies and perfectly elastic frictionless collisions.

Trail modes:
- **Fading** (default): rolling trail; old path points disappear.
- **Permanent**: retains the complete displayed trajectory. Set Permanent trail cap to 0 for unlimited.
- **None**: no trajectory is drawn or stored.

Permanent trails record one point per screen refresh, not one point per internal physics substep. This keeps long stability/chaos runs practical while preserving the visible trajectory.
"""

with gr.Blocks(title="Gravity — Live Planetary Simulation") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Objects")
            n_s=gr.Number(value=1,precision=0,label="Number of S bodies"); n_o=gr.Number(value=1,precision=0,label="Number of O bodies")
            heavy_mass=gr.Number(value=1000.0,label="Default S mass"); ordinary_mass=gr.Number(value=1.0,label="Default O mass")
            mass_vector=gr.Textbox(label="Optional mass vector",placeholder="1000, 500, 1")
            gr.Markdown("## Physical radii")
            default_s_radius=gr.Number(value=0.5,label="Default S radius"); default_o_radius=gr.Number(value=0.1,label="Default O radius")
            radius_vector=gr.Textbox(label="Optional radius vector",placeholder="0.5, 0.4, 0.1")
            gr.Markdown("## Initial state")
            init_extent=gr.Number(value=10.0,label="Random position half-extent")
            velocity_mode=gr.Radio(["Zero","Random"],value="Zero",label="Generated initial velocities")
            random_speed=gr.Number(value=0.2,label="Maximum random speed magnitude"); seed=gr.Number(value=1,precision=0,label="Random seed")
            position_rows=gr.Textbox(label="Optional manual positions: x,y,z per body",placeholder="0,0,0\n7,0,0",lines=5)
            velocity_rows=gr.Textbox(label="Optional manual velocities: vx,vy,vz per body",placeholder="0,0,0\n0,0,0",lines=5)
        with gr.Column():
            gr.Markdown("## Run control")
            run_mode=gr.Radio(["Continuous","Fixed steps","Simulated duration","Real-time hours"],value="Continuous",label="Run mode")
            step_limit=gr.Number(value=100000,precision=0,label="Steps (Fixed steps mode)"); sim_duration=gr.Number(value=100.0,label="Simulated time")
            real_hours=gr.Number(value=1.0,label="Real hours to run")
            gr.Markdown("## Physics / live refresh")
            G=gr.Number(value=1.0,label="Gravitational constant G"); dt=gr.Number(value=0.001,label="Physics Δt (outer step)")
            internal_substeps=gr.Number(value=100,precision=0,label="Internal physics substeps per Δt")
            calculations_per_frame=gr.Number(value=20,precision=0,label="Physics steps per screen refresh"); frame_delay=gr.Number(value=0.03,label="Refresh delay (seconds)")
            softening=gr.Number(value=1e-6,label="Gravity softening ε")
            gr.Markdown("## Boundary / view / trails")
            boundary_mode=gr.Radio(["Huge","Box"],value="Huge",label="Boundary mode"); box_half_size=gr.Number(value=12.0,label="Box half-size")
            view_half_size=gr.Number(value=12.0,label="Visible XY half-size"); auto_view=gr.Checkbox(value=True,label="Automatic initial view size")
            trail_mode=gr.Radio(["Fading","Permanent","None"],value="Fading",label="Trail mode")
            trail_points=gr.Number(value=150,precision=0,label="Fading trail points per object")
            permanent_trail_cap=gr.Number(value=0,precision=0,label="Permanent trail cap per object (0 = unlimited)")
            diagnostic_points=gr.Number(value=500,precision=0,label="Diagnostic history points kept")
    with gr.Row():
        start_btn=gr.Button("▶ Start / Restart",variant="primary"); stop_btn=gr.Button("■ Stop",variant="stop")
    status=gr.Textbox(label="Simulation status",interactive=False)
    with gr.Tab("Live simulation"): sim_plot=gr.Plot()
    with gr.Tab("Energy"): energy_plot=gr.Plot()
    with gr.Tab("Energy error"): energy_error_plot=gr.Plot()
    with gr.Tab("Momentum"): momentum_plot=gr.Plot()
    with gr.Tab("Diagnostics"): diagnostics_table=gr.Dataframe(interactive=False)
    with gr.Tab("Initial state"): state_table=gr.Dataframe(interactive=False)
    inputs=[n_s,n_o,run_mode,step_limit,sim_duration,real_hours,dt,internal_substeps,calculations_per_frame,frame_delay,G,
        heavy_mass,ordinary_mass,mass_vector,default_s_radius,default_o_radius,radius_vector,init_extent,velocity_mode,random_speed,
        position_rows,velocity_rows,softening,boundary_mode,box_half_size,view_half_size,auto_view,trail_mode,trail_points,permanent_trail_cap,diagnostic_points,seed]
    outputs=[sim_plot,energy_plot,energy_error_plot,momentum_plot,diagnostics_table,state_table,status]
    run_event=start_btn.click(fn=live_simulation,inputs=inputs,outputs=outputs,concurrency_limit=1,show_progress="hidden")
    stop_btn.click(fn=None,inputs=None,outputs=None,cancels=[run_event],queue=False)

if __name__=="__main__":
    demo.queue().launch(share=True)
