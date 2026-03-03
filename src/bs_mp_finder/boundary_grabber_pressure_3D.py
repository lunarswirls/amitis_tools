#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import plotly.graph_objects as go
import src.bs_mp_finder.mp_pressure_utils as boundary_utils

# ----------------------------
# SETTINGS
# ----------------------------
debug = False

case = "RPN"
mode = "HNHV"
sim_steps = list(range(105000, 350000 + 1, 1000))

out_dir = f"/Users/danywaller/Projects/mercury/extreme/magnetopause_3D_timeseries/{case}_{mode}/"
os.makedirs(out_dir, exist_ok=True)

rmin_rm = 1.0
rmax_rm = 2.5
rel_tol = 0.15
abs_tol_mp_pressure = 0.0

surf_opacity = 0.25
surf_color = "magenta"

ax_range = [-3, 3]
camera = dict(eye=dict(x=1.6, y=1.3, z=0.9), center=dict(x=0, y=0, z=0))

z_geo_rm = 0.0
z_mag_rm = 484.0 / 2440.0

x_min = 0.5
y_min = -0.5
y_max = 0.5
z_tol = 0.05

eq_marker_sz = 7
r_clip = 2

# ----------------------------
# FILE BUILDER
# ----------------------------
def build_file_path(sim_step: int) -> str:
    if sim_step < 115000:
        fmode = "Base"
        input_folder = f"/Volumes/data_backup/mercury/extreme/{case}_Base/plane_product/cube/"
    else:
        fmode = "HNHV"
        input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/plane_product/cube/"
    return os.path.join(input_folder, f"Amitis_{case}_{fmode}_{sim_step:06d}_merged_4RM.nc")

# ----------------------------
# SURFACE FROM BINARY MASK
# ----------------------------
def mesh_from_mask(mp_mask, x, y, z, r_clip=1.9):

    nx, ny, nz = mp_mask.shape
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])

    verts = []
    faces = []
    vert_index = {}

    def add_vertex(v):
        key = tuple(v)
        if key not in vert_index:
            vert_index[key] = len(verts)
            verts.append(v)
        return vert_index[key]

    # Iterate through all voxels
    for i in range(nx-1):
        for j in range(ny-1):
            for k in range(nz-1):

                if not mp_mask[i, j, k]:
                    continue

                xc, yc, zc = x[i], y[j], z[k]
                r = np.sqrt(xc**2 + yc**2 + zc**2)

                # Clip unwanted region BEFORE meshing
                if (xc > 0.0) and (r > r_clip):
                    continue

                # Check 6 directions
                neighbors = [
                    (i+1, j, k),
                    (i-1, j, k),
                    (i, j+1, k),
                    (i, j-1, k),
                    (i, j, k+1),
                    (i, j, k-1),
                ]

                face_dirs = [
                    (+dx/2, 0, 0),
                    (-dx/2, 0, 0),
                    (0, +dy/2, 0),
                    (0, -dy/2, 0),
                    (0, 0, +dz/2),
                    (0, 0, -dz/2),
                ]

                for (ni, nj, nk), (sx, sy, sz) in zip(neighbors, face_dirs):

                    if ni < 0 or nj < 0 or nk < 0 or \
                       ni >= nx or nj >= ny or nk >= nz or \
                       not mp_mask[ni, nj, nk]:

                        # This face is exposed → build quad

                        cx = xc + sx
                        cy = yc + sy
                        cz = zc + sz

                        # Determine face orientation
                        if sx != 0:
                            quad = [
                                (cx, cy-dy/2, cz-dz/2),
                                (cx, cy+dy/2, cz-dz/2),
                                (cx, cy+dy/2, cz+dz/2),
                                (cx, cy-dy/2, cz+dz/2),
                            ]
                        elif sy != 0:
                            quad = [
                                (cx-dx/2, cy, cz-dz/2),
                                (cx+dx/2, cy, cz-dz/2),
                                (cx+dx/2, cy, cz+dz/2),
                                (cx-dx/2, cy, cz+dz/2),
                            ]
                        else:
                            quad = [
                                (cx-dx/2, cy-dy/2, cz),
                                (cx+dx/2, cy-dy/2, cz),
                                (cx+dx/2, cy+dy/2, cz),
                                (cx-dx/2, cy+dy/2, cz),
                            ]

                        ids = [add_vertex(v) for v in quad]

                        # Split quad into 2 triangles
                        faces.append([ids[0], ids[1], ids[2]])
                        faces.append([ids[0], ids[2], ids[3]])

    if len(faces) == 0:
        return None, None

    return np.array(verts), np.array(faces)

# ----------------------------
# MERCURY SPHERE
# ----------------------------
def mercury_sphere_traces():
    theta = np.linspace(0, np.pi, 80)
    phi = np.linspace(0, 2*np.pi, 160)
    theta, phi = np.meshgrid(theta, phi)

    xs = np.sin(theta) * np.cos(phi)
    ys = np.sin(theta) * np.sin(phi)
    zs = np.cos(theta)

    mask_pos = xs >= 0
    mask_neg = xs <= 0

    sphere_day = go.Surface(
        x=np.where(mask_pos, xs, np.nan),
        y=np.where(mask_pos, ys, np.nan),
        z=np.where(mask_pos, zs, np.nan),
        colorscale=[[0,"lightgrey"],[1,"lightgrey"]],
        showscale=False,
        hoverinfo="skip",
        name="Mercury (day)",
        showlegend=False,
    )

    sphere_night = go.Surface(
        x=np.where(mask_neg, xs, np.nan),
        y=np.where(mask_neg, ys, np.nan),
        z=np.where(mask_neg, zs, np.nan),
        colorscale=[[0,"black"],[1,"black"]],
        showscale=False,
        hoverinfo="skip",
        name="Mercury (night)",
        showlegend=False,
    )

    return sphere_day, sphere_night

# ----------------------------
# MAIN LOOP
# ----------------------------
sphere_day, sphere_night = mercury_sphere_traces()

mesh_verts = []
mesh_faces = []
step_labels = []

for sim_step in sim_steps:

    f_3d = build_file_path(sim_step)
    if not os.path.exists(f_3d):
        print("File not found:", f_3d)
        continue

    tsec = sim_step * 0.002
    lab = f"t={tsec:.3f} s"
    print("Processing:", lab)

    x, y, z, PB_pa, Pdyn_pa, mp_mask_da = boundary_utils.compute_mp_mask_pressure_balance(
        f_3d,
        debug=debug,
        r_min_rm=rmin_rm,
        r_max_rm=rmax_rm,
        rel_tol=rel_tol,
        abs_tol_pa=abs_tol_mp_pressure,
    )

    mp_mask = (mp_mask_da.values > 0)

    verts, faces = mesh_from_mask(mp_mask, x, y, z, r_clip=r_clip)

    mesh_verts.append(verts)
    mesh_faces.append(faces)
    step_labels.append(lab)

# ----------------------------
# PLOTLY FIGURE
# ----------------------------
mp_traces = []

for i in range(len(mesh_verts)):

    if mesh_verts[i] is None:
        mp_traces.append(go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[]))
        continue

    mp_traces.append(
        go.Mesh3d(
            x=mesh_verts[i][:,0],
            y=mesh_verts[i][:,1],
            z=mesh_verts[i][:,2],
            i=mesh_faces[i][:,0],
            j=mesh_faces[i][:,1],
            k=mesh_faces[i][:,2],
            color=surf_color,
            opacity=surf_opacity,
            flatshading=True,
            visible=(i==0),
            hoverinfo="skip"
        )
    )

fig = go.Figure(data=[sphere_day, sphere_night] + mp_traces)

steps = []
for i, lab in enumerate(step_labels):

    vis = [True, True] + [False]*len(mp_traces)
    vis[2+i] = True

    steps.append(dict(
        method="update",
        label=lab,
        args=[{"visible": vis}]
    ))

fig.update_layout(
    sliders=[dict(active=0, steps=steps)],
    scene=dict(
        xaxis=dict(range=ax_range),
        yaxis=dict(range=ax_range),
        zaxis=dict(range=ax_range),
        aspectmode="cube",
        camera=camera,
    ),
    height=900,
    width=900,
)

html_path = os.path.join(out_dir, f"{case}_{mode}_mp_mask_surface_slider.html")
fig.write_html(html_path, include_plotlyjs="cdn")
print("Saved:", html_path)