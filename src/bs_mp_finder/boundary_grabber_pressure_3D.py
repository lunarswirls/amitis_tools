#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import plotly.graph_objects as go
from skimage import measure
import bs_mp_finder.mp_pressure_utils as boundary_utils
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------------------
# SETTINGS
# ----------------------------
debug = False

case = "CPS"
mode = "LNHV"
sim_steps = list(range(105000, 350000 + 1, 1000))

out_dir = f"/Users/danywaller/Projects/mercury/extreme/magnetopause_3D_timeseries/{case}_{mode}/"
os.makedirs(out_dir, exist_ok=True)

RMIN_RM = 1.0
RMAX_RM = 4.0
REL_TOL = 0.15
ABS_TOL_PA = 0.0

ISO_LEVEL = 0.0
DECIMATE = 2
MAX_FACES = 250_000

SURF_OPACITY = 0.25
SURF_COLOR = "magenta"

AX_RANGE = [-2, 2]
camera = dict(eye=dict(x=1.6, y=1.3, z=0.9), center=dict(x=0, y=0, z=0))

Z_GEO_RM = 0.0
Z_MAG_RM = 484.0 / 2440.0

X_MIN = 0.5
Y_MIN = -0.5
Y_MAX = 0.5
Z_TOL = 0.05

EQ_MARKER_SIZE = 7

# ----------------------------
# Helpers
# ----------------------------
def build_file_path(sim_step: int) -> str:
    if sim_step < 115000:
        input_folder = f"/Volumes/data_backup/mercury/extreme/{case}_Base/plane_product/object/"
        return os.path.join(input_folder, f"Amitis_{case}_Base_{sim_step:06d}_xz_comp.nc")
    input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/plane_product/object/"
    return os.path.join(input_folder, f"Amitis_{case}_{mode}_{sim_step:06d}_xz_comp.nc")


def mercury_sphere_traces(plot_depth=1.0):
    theta = np.linspace(0, np.pi, 80)
    phi = np.linspace(0, 2*np.pi, 160)
    theta, phi = np.meshgrid(theta, phi)

    xs = plot_depth * np.sin(theta) * np.cos(phi)
    ys = plot_depth * np.sin(theta) * np.sin(phi)
    zs = plot_depth * np.cos(theta)

    eps = 0.0
    mask_pos = xs >= -eps
    mask_neg = xs <= eps

    sphere_day = go.Surface(
        x=np.where(mask_pos, xs, np.nan),
        y=np.where(mask_pos, ys, np.nan),
        z=np.where(mask_pos, zs, np.nan),
        surfacecolor=np.ones_like(xs),
        colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
        cmin=0, cmax=1,
        showscale=False,
        lighting=dict(ambient=1, diffuse=0, specular=0),
        hoverinfo="skip",
        name="Mercury (day)",
        showlegend=False,
    )

    sphere_night = go.Surface(
        x=np.where(mask_neg, xs, np.nan),
        y=np.where(mask_neg, ys, np.nan),
        z=np.where(mask_neg, zs, np.nan),
        surfacecolor=np.zeros_like(xs),
        colorscale=[[0, "black"], [1, "black"]],
        cmin=0, cmax=1,
        showscale=False,
        lighting=dict(ambient=1, diffuse=0, specular=0),
        hoverinfo="skip",
        name="Mercury (night)",
        showlegend=False,
    )
    return sphere_day, sphere_night


def mesh_from_pressures(PB_pa, Pdyn_pa, x, y, z, outside_body):
    """
    Return (verts_xyz, faces) for F=log10(PB/Pdyn)=ISO_LEVEL using marching cubes.
    """
    eps_pa = 1e-30
    F = np.log10((PB_pa + eps_pa) / (Pdyn_pa + eps_pa))
    F = np.where(outside_body, F, np.nan)

    sl = slice(None, None, DECIMATE)
    Fd = F[sl, sl, sl]
    xd = x[sl]; yd = y[sl]; zd = z[sl]
    if len(xd) < 2 or len(yd) < 2 or len(zd) < 2:
        return None, None

    V = np.nan_to_num(Fd, nan=+10.0)

    dx = float(xd[1] - xd[0])
    dy = float(yd[1] - yd[0])
    dz = float(zd[1] - zd[0])

    try:
        verts, faces, _, _ = measure.marching_cubes(V, level=ISO_LEVEL, spacing=(dx, dy, dz))
    except (ValueError, RuntimeError):
        return None, None

    verts[:, 0] += float(xd[0])
    verts[:, 1] += float(yd[0])
    verts[:, 2] += float(zd[0])

    if faces.shape[0] > MAX_FACES:
        sel = np.random.choice(faces.shape[0], MAX_FACES, replace=False)
        faces = faces[sel]

    return verts, faces


def plotly_mesh_trace_from_verts_faces(verts, faces, visible=False, name="Magnetopause"):
    if verts is None:
        return go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], visible=visible, name=name, showlegend=True)
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=SURF_COLOR,
        opacity=SURF_OPACITY,
        flatshading=True,
        name=name,
        visible=visible,
        hoverinfo="skip",
        showlegend=True
    )


def max_point_on_equator(mp_mask, x, y, z, Rg, region, z0, z_tol):
    """
    Return dict with x,y,z,r (or NaNs) for max r within mp_mask & region & |Z-z0|<=z_tol.
    """
    # Find z indices within tolerance without building full Zg each time
    iz = np.where(np.abs(z - z0) <= z_tol)[0]
    if iz.size == 0:
        return dict(x=np.nan, y=np.nan, z=np.nan, r=np.nan)

    # Restrict mask to those z-slices
    sel = mp_mask[:, :, iz] & region[:, :, iz]
    if not np.any(sel):
        return dict(x=np.nan, y=np.nan, z=np.nan, r=np.nan)

    Rsub = Rg[:, :, iz]
    rmax = np.nanmax(np.where(sel, Rsub, np.nan))

    # pick one voxel achieving rmax
    loc = np.argwhere(sel & (np.abs(Rsub - rmax) < 1e-12))
    ix, iy, kiz = loc[0]
    iz0 = int(iz[kiz])

    return dict(x=float(x[ix]), y=float(y[iy]), z=float(z[iz0]), r=float(rmax))


# ----------------------------
# PASS 1: compute once per timestep, store compact results
# ----------------------------
sphere_day, sphere_night = mercury_sphere_traces(plot_depth=1.0)

step_labels = []
valid_filepaths = []

# store meshes for both Plotly+Matplotlib without recomputing marching cubes
mesh_verts = []
mesh_faces = []

geo_pts = []
mag_pts = []

# cached grid products (only if grid is constant)
grid_cached = False
x_cache = y_cache = z_cache = None
Rg_cache = None
outside_body_cache = None
region_cache = None

for sim_step in sim_steps:
    f_3d = build_file_path(sim_step)
    if not os.path.exists(f_3d):
        print(f"[WARN] missing 3D file: {f_3d}")
        continue

    tsec = sim_step * 0.002
    lab = f"t={tsec:.3f} s"
    print(f"Processing timestep: {lab}")

    x, y, z, PB_pa, Pdyn_pa, mp_mask_da = boundary_utils.compute_mp_mask_pressure_balance(
        f_3d,
        debug=debug,
        r_min_rm=RMIN_RM,
        r_max_rm=RMAX_RM,
        rel_tol=REL_TOL,
        abs_tol_pa=ABS_TOL_PA,
    )
    mp_mask = (mp_mask_da.values > 0)

    # cache grid stuff if constant
    if (not grid_cached):
        Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
        Rg_cache = np.sqrt(Xg**2 + Yg**2 + Zg**2)

        outside_body_cache = (Rg_cache >= RMIN_RM) & (Rg_cache < RMAX_RM)
        region_cache = (Xg > X_MIN) & (Yg >= Y_MIN) & (Yg <= Y_MAX) & outside_body_cache

        x_cache, y_cache, z_cache = x, y, z
        grid_cached = True
    else:
        # if grid changed, recompute caches for this step
        if (len(x) != len(x_cache)) or (len(y) != len(y_cache)) or (len(z) != len(z_cache)) \
           or (not np.allclose(x, x_cache)) or (not np.allclose(y, y_cache)) or (not np.allclose(z, z_cache)):
            Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
            Rg_cache = np.sqrt(Xg**2 + Yg**2 + Zg**2)
            outside_body_cache = (Rg_cache >= RMIN_RM) & (Rg_cache < RMAX_RM)
            region_cache = (Xg > X_MIN) & (Yg >= Y_MIN) & (Yg <= Y_MAX) & outside_body_cache
            x_cache, y_cache, z_cache = x, y, z

    # max points (use cached Rg + cached region; avoid creating Xg/Yg/Zg again)
    geo_p = max_point_on_equator(mp_mask, x_cache, y_cache, z_cache, Rg_cache, region_cache, Z_GEO_RM, Z_TOL)
    mag_p = max_point_on_equator(mp_mask, x_cache, y_cache, z_cache, Rg_cache, region_cache, Z_MAG_RM, Z_TOL)
    geo_p.update(t=float(tsec), label=lab)
    mag_p.update(t=float(tsec), label=lab)
    geo_pts.append(geo_p)
    mag_pts.append(mag_p)

    # mesh (store verts/faces only; re-use later for Plotly and Matplotlib)
    verts, faces = mesh_from_pressures(PB_pa, Pdyn_pa, x_cache, y_cache, z_cache, outside_body_cache)
    mesh_verts.append(verts)
    mesh_faces.append(faces)

    step_labels.append(lab)
    valid_filepaths.append(f_3d)

if len(step_labels) == 0:
    raise RuntimeError("No valid timesteps found.")

# ----------------------------
# Build Plotly slider figure
# ----------------------------
mp_traces = []
geo_point_traces = []
mag_point_traces = []

for i, lab in enumerate(step_labels):
    mp_traces.append(plotly_mesh_trace_from_verts_faces(mesh_verts[i], mesh_faces[i], visible=False, name="Magnetopause"))

    gp = geo_pts[i]
    mp = mag_pts[i]
    alt_geo = gp["r"] - 1.0 if np.isfinite(gp["r"]) else np.nan
    alt_mag = mp["r"] - 1.0 if np.isfinite(mp["r"]) else np.nan

    geo_name = f"Geo eq max ΔR={alt_geo:.2f} R<sub>M</sub>" if np.isfinite(alt_geo) else "Geo eq max ΔR=NaN"
    mag_name = f"Mag eq max ΔR={alt_mag:.2f} R<sub>M</sub>" if np.isfinite(alt_mag) else "Mag eq max ΔR=NaN"

    geo_point_traces.append(
        go.Scatter3d(
            x=[] if not np.isfinite(gp["x"]) else [gp["x"]],
            y=[] if not np.isfinite(gp["y"]) else [gp["y"]],
            z=[] if not np.isfinite(gp["z"]) else [gp["z"]],
            mode="markers",
            marker=dict(size=EQ_MARKER_SIZE, color="yellow"),
            name=geo_name,
            visible=False,
            customdata=[] if not np.isfinite(alt_geo) else [alt_geo],
            hovertemplate="Geo eq max<br>X=%{x:.2f}<br>Y=%{y:.2f}<br>Z=%{z:.2f}<br>ΔR=%{customdata:.2f} R<sub>M</sub><extra></extra>",
            showlegend=True,
        )
    )

    mag_point_traces.append(
        go.Scatter3d(
            x=[] if not np.isfinite(mp["x"]) else [mp["x"]],
            y=[] if not np.isfinite(mp["y"]) else [mp["y"]],
            z=[] if not np.isfinite(mp["z"]) else [mp["z"]],
            mode="markers",
            marker=dict(size=EQ_MARKER_SIZE, color="blue"),
            name=mag_name,
            visible=False,
            customdata=[] if not np.isfinite(alt_mag) else [alt_mag],
            hovertemplate="Mag eq max<br>X=%{x:.2f}<br>Y=%{y:.2f}<br>Z=%{z:.2f}<br>ΔR=%{customdata:.2f} R<sub>M</sub><extra></extra>",
            showlegend=True,
        )
    )

# first visible
mp_traces[0].visible = True
geo_point_traces[0].visible = True
mag_point_traces[0].visible = True

steps = []
n_mesh = len(mp_traces)
base = 2

for i, lab in enumerate(step_labels):
    idx_mesh = base + i
    idx_geo = base + n_mesh + i
    idx_mag = base + 2*n_mesh + i

    vis = [True, True] + [False]*n_mesh + [False]*n_mesh + [False]*n_mesh
    vis[idx_mesh] = True
    vis[idx_geo] = True
    vis[idx_mag] = True

    alt_geo = geo_pts[i]["r"] - 1.0 if np.isfinite(geo_pts[i]["r"]) else np.nan
    alt_mag = mag_pts[i]["r"] - 1.0 if np.isfinite(mag_pts[i]["r"]) else np.nan
    geo_name = f"Geo eq max ΔR={alt_geo:.2f} R<sub>M</sub>" if np.isfinite(alt_geo) else "Geo eq max ΔR=NaN"
    mag_name = f"Mag eq max ΔR={alt_mag:.2f} R<sub>M</sub>" if np.isfinite(alt_mag) else "Mag eq max ΔR=NaN"

    steps.append(dict(
        method="update",
        label=lab,
        args=[
            {"visible": vis},
            {f"data[{idx_geo}].name": geo_name,
             f"data[{idx_mag}].name": mag_name},
        ],
    ))

sliders = [dict(active=0, currentvalue=dict(prefix="Time: "), pad=dict(t=50), steps=steps)]

fig = go.Figure(data=[sphere_day, sphere_night] + mp_traces + geo_point_traces + mag_point_traces)
fig.update_layout(
    title=f"{case} {mode} — MP mesh from log10(P<sub>B</sub>/P<sub>dyn</sub>)=0",
    template="plotly_white",
    height=900,
    width=900,
    showlegend=True,
    legend=dict(x=1.02, y=0.95, bgcolor="rgba(255,255,255,0.9)"),
    sliders=sliders,
    scene=dict(
        xaxis=dict(title="X (R<sub>M</sub>)", range=AX_RANGE),
        yaxis=dict(title="Y (R<sub>M</sub>)", range=AX_RANGE),
        zaxis=dict(title="Z (R<sub>M</sub>)", range=AX_RANGE),
        aspectmode="cube",
        camera=camera,
    ),
)

html_path = os.path.join(out_dir, f"{case}_{mode}_mp_pressure_balance_mesh_slider.html")
fig.write_html(html_path, include_plotlyjs="cdn")
print(f"Saved interactive mesh slider plot: {html_path}")

# ----------------------------
# Matplotlib animation export (.mov)
# ----------------------------
FPS = 5
mov_path = os.path.join(out_dir, f"{case}_{mode}_mp_pressure_balance_mesh_fps{FPS}.mov")

X_LIM = (0.5, 1.8)
Z_LIM = (-1.0, 1.0)
Y_LIM = (-1.0, 1.0)

fig_m = plt.figure(figsize=(7, 7), dpi=150)
ax = fig_m.add_subplot(111, projection="3d")
ax.view_init(elev=0, azim=90)

ax.set_xlim(*X_LIM)
ax.set_ylim(*Y_LIM)
ax.set_zlim(*Z_LIM)
ax.set_xlabel("X (R$_M$)")
ax.set_ylabel("Y (R$_M$)")
ax.set_zlabel("Z (R$_M$)")
ax.set_title(f"{case} {mode} — MP mesh from log10(P$_B$/P$_{{dyn}}$)=0 (XZ zoom)")

ang = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(ang), np.zeros_like(ang), np.sin(ang), color="k", lw=1, alpha=0.4)

mesh_artist = None
geo_artist = None
mag_artist = None
txt_artist = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

def update(frame_idx):
    global mesh_artist, geo_artist, mag_artist
    if mesh_artist is not None:
        mesh_artist.remove()
        mesh_artist = None
    if geo_artist is not None:
        geo_artist.remove()
        geo_artist = None
    if mag_artist is not None:
        mag_artist.remove()
        mag_artist = None

    verts = mesh_verts[frame_idx]
    faces = mesh_faces[frame_idx]
    if verts is not None:
        tri = verts[faces]
        mesh_artist = Poly3DCollection(tri, facecolor=SURF_COLOR, edgecolor="none", alpha=SURF_OPACITY)
        ax.add_collection3d(mesh_artist)

    gp = geo_pts[frame_idx]
    mp = mag_pts[frame_idx]
    if np.isfinite(gp["x"]):
        geo_artist = ax.scatter([gp["x"]], [gp["y"]], [gp["z"]], s=50, c="yellow", depthshade=False)
    if np.isfinite(mp["x"]):
        mag_artist = ax.scatter([mp["x"]], [mp["y"]], [mp["z"]], s=50, c="blue", depthshade=False)

    alt_geo = gp["r"] - 1.0 if np.isfinite(gp["r"]) else np.nan
    alt_mag = mp["r"] - 1.0 if np.isfinite(mp["r"]) else np.nan
    txt_artist.set_text(f"{step_labels[frame_idx]}\nGeo ΔR={alt_geo:.2f} R$_M$\nMag ΔR={alt_mag:.2f} R$_M$")

    artists = [txt_artist]
    if mesh_artist is not None: artists.append(mesh_artist)
    if geo_artist is not None: artists.append(geo_artist)
    if mag_artist is not None: artists.append(mag_artist)
    return tuple(artists)

anim = animation.FuncAnimation(fig_m, update, frames=len(step_labels), interval=1000/FPS, blit=False)
writer = animation.FFMpegWriter(fps=FPS, codec="h264")
anim.save(mov_path, writer=writer)
plt.close(fig_m)

print(f"Saved animation: {mov_path}")
