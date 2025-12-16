#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import os
import xarray as xr
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# settings
input_folder = "/Users/danywaller/Projects/mercury/CPS_Base/object/"
outdir = "/Users/danywaller/Projects/mercury/CPS_Base/"
outfile = os.path.join(outdir, "current_spherical_shells.html")
VAR_X, VAR_Y, VAR_Z = "Jx", "Jy", "Jz"
SUB = 5
R_m = 2440.e3  # Mercury radius [m]
sim_steps = list(range(27000, 115000 + 1, 1000))

# calculate median current
Jx_list, Jy_list, Jz_list = [], [], []

for sim_step in sim_steps:
    f = os.path.join(input_folder, f"Amitis_CPS_Base_{sim_step:06d}_xz_comp.nc")
    ds = xr.open_dataset(f)
    Jx, Jy, Jz = ds[VAR_X].values, ds[VAR_Y].values, ds[VAR_Z].values
    while Jx.ndim > 3:
        Jx, Jy, Jz = Jx.squeeze(), Jy.squeeze(), Jz.squeeze()
    Jx_list.append(Jx)
    Jy_list.append(Jy)
    Jz_list.append(Jz)
    ds.close()

Jx_med = np.median(np.stack(Jx_list, axis=0), axis=0)
Jy_med = np.median(np.stack(Jy_list, axis=0), axis=0)
Jz_med = np.median(np.stack(Jz_list, axis=0), axis=0)

Jmag = np.sqrt(Jx_med**2 + Jy_med**2 + Jz_med**2)

# -------------------------------
# GRID COORDINATES
# -------------------------------
ds0 = xr.open_dataset(os.path.join(input_folder, f"Amitis_CPS_Base_{sim_steps[0]:06d}_xz_comp.nc"))
Nx, Ny, Nz = Jx_med.shape
x = np.linspace(float(ds0.full_xmin), float(ds0.full_xmax), Nx) / R_m
y = np.linspace(float(ds0.full_ymin), float(ds0.full_ymax), Ny) / R_m
z = np.linspace(float(ds0.full_zmin), float(ds0.full_zmax), Nz) / R_m
ds0.close()

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

fig, ax = plt.subplots(figsize=(8, 6))
circle = plt.Circle((0, 0), 1, edgecolor='black', facecolor='cornflowerblue', alpha=0.3, linewidth=1, )
plt.pcolormesh(x, z, Jmag[:,54,:], vmin=0, vmax=1200, shading='auto', cmap='gist_heat_r')
ax.add_patch(circle)
plt.xlabel(r"$\text{X (R}_{M}\text{)}$")
plt.ylabel(r"$\text{Z (R}_{M}\text{)}$")
plt.title(r"$\text{CPS Median J}_{mag}$")
plt.colorbar(label=r"$J_{mag}$")
plt.tight_layout()
fig_path = os.path.join(outdir, f"cps_jmag_median.png")
plt.savefig(fig_path, dpi=300)
plt.close()

# ===============================
# INTERPOLATORS
# ===============================
interp_Vx = RegularGridInterpolator((x, y, z), Jx_med, bounds_error=False, fill_value=np.nan)
interp_Vy = RegularGridInterpolator((x, y, z), Jy_med, bounds_error=False, fill_value=np.nan)
interp_Vz = RegularGridInterpolator((x, y, z), Jz_med, bounds_error=False, fill_value=np.nan)

# ===============================
# SPHERICAL GRID
# ===============================
theta = np.linspace(0, 2*np.pi, 80)
phi   = np.linspace(0, np.pi, 40)
TH, PH = np.meshgrid(theta, phi)

# Radii to animate
r_max = min(
    np.max(np.abs(x)),
    np.max(np.abs(y)),
    np.max(np.abs(z))
)

radii = np.linspace(0.75*r_max, 1.5*R_m, 10)

# project onto sphere
def project_to_tangent(Vx, Vy, Vz, Xs, Ys, Zs):
    # Radial unit vector
    rmag = np.sqrt(Xs**2 + Ys**2 + Zs**2)
    rx, ry, rz = Xs/rmag, Ys/rmag, Zs/rmag

    # Remove radial component
    dot = Vx*rx + Vy*ry + Vz*rz
    Vtx = Vx - dot*rx
    Vty = Vy - dot*ry
    Vtz = Vz - dot*rz

    return Vtx, Vty, Vtz

# ===============================
# BUILD FRAMES
# ===============================
frames = []

for r in radii:
    # Sphere coordinates
    Xs = r * np.cos(TH) * np.sin(PH)
    Ys = r * np.sin(TH) * np.sin(PH)
    Zs = r * np.cos(PH)

    pts = np.stack([Xs.ravel(), Ys.ravel(), Zs.ravel()], axis=-1)

    # Interpolate vector field
    Vxs = interp_Vx(pts).reshape(Xs.shape)
    Vys = interp_Vy(pts).reshape(Xs.shape)
    Vzs = interp_Vz(pts).reshape(Xs.shape)

    # Tangential projection
    Vtx, Vty, Vtz = project_to_tangent(Vxs, Vys, Vzs, Xs, Ys, Zs)

    # Vector magnitude (for coloring)
    Vmag = np.sqrt(Vtx**2 + Vty**2 + Vtz**2)

    # Downsample arrows
    skip = (slice(None, None, SUB), slice(None, None, SUB))

    frames.append(go.Frame(
        data=[
            # Sphere surface
            go.Surface(
                x=Xs / R_m,
                y=Ys / R_m,
                z=Zs / R_m,
                surfacecolor=Vmag,
                coloraxis="coloraxis",
                showscale=False  # IMPORTANT
            ),
        ],
        name=f"{r/R_m:.2f} Rm"
    ))

# ===============================
# INITIAL FIGURE
# ===============================
fig = go.Figure(
    data=frames[0].data,
    frames=frames
)

# ===============================
# SLIDER + PLAY BUTTON
# ===============================
fig.update_layout(
    title="Current Projected onto Spherical Shells",
    scene=dict(
        xaxis=dict(title="X (Rₘ)", range=[-3, 3]),
        yaxis=dict(title="Y (Rₘ)", range=[-3, 3]),
        zaxis=dict(title="Z (Rₘ)", range=[-3, 3]),
        aspectmode="cube"
    ),
    updatemenus=[{
        "type": "buttons",
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 600, "redraw": True}}]
        }]
    }],
    sliders=[{
        "steps": [{
            "method": "animate",
            "args": [[f.name], {"mode": "immediate"}],
            "label": f.name
        } for f in frames]
    }],
    width=900,
    height=900,
    coloraxis=dict(
        colorscale="Viridis",
        cmin=900,
        cmax=100000,
        colorbar=dict(
            title=dict(
                text="|J<sub>mag</sub>|",
                side="right"
            ),
            thickness=20
        )
    )
)

fig.write_html(
    outfile,
    include_plotlyjs="embed",
    auto_play=False
)

