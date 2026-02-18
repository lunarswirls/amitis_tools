# %%
# Imports:
import os
from datetime import datetime
import numpy as np
import pandas as pd
import random
import xarray as xr
from scipy.interpolate import griddata
from src.field_topology.topology_utils import *
from scipy.signal import savgol_filter
import plotly.graph_objects as go

# --------------------------
# SETTINGS
# --------------------------
case = "RPS_HNHV"
step = 350000  # 115000 for base or 350000 for hnhv

# Shell limits
rmin = "1.00"
rmax = "1.05"

input_folder = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/10/out/"
ncfile = os.path.join(input_folder, f"Amitis_{case}_{step}.nc")

output_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology_density_seeding_{rmin}-{rmax}_RM/{case}/"
os.makedirs(output_folder, exist_ok=True)

# Planet parameters
RM = 2440.0          # Mercury radius [km]
RC = 2400.0          # depth within conductive layer [km]

rmin = float(rmin) * RM
rmax = float(rmax) * RM

plot_depth = RM

# Seed settings
n_lat = 60
n_lon = n_lat*2
max_steps = 5000
h_step = 50.0
surface_tol = 75.0

max_lines = 500  # downsample trajectory points for plotting

# %%
# --------------------------
# LOAD VECTOR FIELD FROM NETCDF
# --------------------------
def load_field(nc_file):
    ds = xr.open_dataset(nc_file)
    x_vals = ds["Nx"].values
    y_vals = ds["Ny"].values
    z_vals = ds["Nz"].values

    # Extract fields (drop time dimension) and transpose:  Nz, Ny, Nx --> Nx, Ny, Nz
    J_x = np.transpose(ds["Bx"].isel(time=0).values, (2,1,0))
    J_y = np.transpose(ds["By"].isel(time=0).values, (2,1,0))
    J_z = np.transpose(ds["Bz"].isel(time=0).values, (2,1,0))

    # densities in cm^-3
    den_tot = ds["den_tot"].isel(time=0).values

    # Transpose Nz, Ny, Nx → Nx, Ny, Nz
    den_tot = np.transpose(den_tot, (2, 1, 0))

    ds.close()
    return x_vals, y_vals, z_vals, J_x, J_y, J_z, den_tot

x, y, z, Jx, Jy, Jz, tot_den = load_field(ncfile)

jtime = datetime.now()
print(f"Loaded B-vector and density at {str(jtime)}")

# %%
# ------------------------
# CREATE 3D GRID
# ------------------------
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
R = np.sqrt(X**2 + Y**2 + Z**2)

# ------------------------
# SELECT SHELL
# ------------------------
shell_mask = (R >= rmin) & (R <= rmax)
X = X[shell_mask]
Y = Y[shell_mask]
Z = Z[shell_mask]
tot_den_shell = tot_den[shell_mask]

# ------------------------
# SPHERICAL COORDINATES
# ------------------------
R_shell = R[shell_mask]
lat = np.degrees(np.arcsin(Z / R_shell))
lon = np.degrees(np.arctan2(Y, X))

# ------------------------
# TRIANGULATE IRREGULAR POINTS
# ------------------------
points = np.vstack([lon, lat]).T

# Regular grid
lat_grid = np.linspace(-90, 90, 180)
lon_grid = np.linspace(-180, 180, 360)
Lon_grid, Lat_grid = np.meshgrid(lon_grid, lat_grid)

# Interpolate scattered density onto the grid
tot_den_grid = griddata(
    points=(lon, lat),
    values=tot_den_shell,
    xi=(Lon_grid, Lat_grid),
    method='nearest'
)

# %%
# --------------------------
# THRESHOLD ON SHELL (95th percentile)
# --------------------------
p95 = np.nanpercentile(tot_den, 95)   # if any NaNs exist
print("p95 shell density =", p95)

p995 = np.nanpercentile(tot_den, 99.5)   # if any NaNs exist
print("p99.5 shell density =", p995)

# --------------------------
# SEEDS ON SPHERE WHERE local density > p95
# --------------------------
lats_surface = np.linspace(-90, 90, 180)
lons_surface = np.linspace(-180, 180, 360)
Lon_s, Lat_s = np.meshgrid(lons_surface, lats_surface)

seed_den = griddata(
    points=np.vstack([Lon_grid.ravel(), Lat_grid.ravel()]).T,
    values=tot_den_grid.ravel(),
    xi=(Lon_s, Lat_s),
    method="nearest",
)

# density threshold
keep = seed_den >  200 # p995

# dayside filter: X > 0 on the seed sphere
phi_s   = np.radians(Lat_s)
theta_s = np.radians(Lon_s)
X_s = plot_depth * np.cos(phi_s) * np.cos(theta_s)
Y_s = plot_depth * np.cos(phi_s) * np.sin(theta_s)
Z_s = plot_depth * np.sin(phi_s)

dayside = X_s > 0
southern = Z_s < 0

# combine
keep = keep   # & dayside & southern

Lat_keep = Lat_s[keep]
Lon_keep = Lon_s[keep]

seeds = []
for lat, lon in zip(Lat_keep, Lon_keep):
    phi = np.radians(lat)
    theta = np.radians(lon)
    x_s = plot_depth * np.cos(phi) * np.cos(theta)
    y_s = plot_depth * np.cos(phi) * np.sin(theta)
    z_s = plot_depth * np.sin(phi)
    seeds.append([x_s, y_s, z_s])

seeds = np.asarray(seeds)
print("Kept", seeds.shape[0], "dayside seeds out of", Lat_s.size)

# %%
# --------------------------
# TRACE FIELD LINES
# --------------------------
lines_by_topo = {"closed": [], "open": []}

for seed in seeds:
    traj_fwd, exit_fwd_y = trace_field_line_rk(seed, Jx, Jy, Jz, x, y, z, plot_depth, max_steps=max_steps, h=h_step)
    traj_bwd, exit_bwd_y = trace_field_line_rk(seed, Jx, Jy, Jz, x, y, z, plot_depth, max_steps=max_steps, h=-h_step)
    topo = classify(traj_fwd, traj_bwd, plot_depth + surface_tol, exit_fwd_y, exit_bwd_y)
    if topo in ["closed", "open"]:
        lines_by_topo[topo].append(traj_fwd)
        lines_by_topo[topo].append(traj_bwd)

classtime = datetime.now()
print(f"Classified all lines at {str(classtime)}")

# %%
def smooth_traj(traj, k=5, order=2):
    """
    Function to lightly smooth field lines
    """
    if traj.shape[0] < k:
        return traj

    return np.column_stack([
        savgol_filter(traj[:, i], k, order, mode="interp")
        for i in range(3)
    ])

# --------------------------
# PLOT 3D FIELD LINES
# --------------------------
colors = {"closed": "blue", "open": "red"}
fig = go.Figure()

# add planet sphere
theta = np.linspace(0, np.pi, 100)        # colatitude
phi   = np.linspace(0, 2*np.pi, 200)      # longitude
theta, phi = np.meshgrid(theta, phi)

xs = plot_depth * np.sin(theta) * np.cos(phi)
ys = plot_depth * np.sin(theta) * np.sin(phi)
zs = plot_depth * np.cos(theta)

eps = 0
mask_pos = xs >= -eps
mask_neg = xs <=  eps

# light grey hemisphere (X > 0)
fig.add_trace(go.Surface(
    x=np.where(mask_pos, xs, np.nan),
    y=np.where(mask_pos, ys, np.nan),
    z=np.where(mask_pos, zs, np.nan),
    surfacecolor=np.ones_like(xs),
    colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
    cmin=0,
    cmax=1,
    showscale=False,
    lighting=dict(ambient=1, diffuse=0, specular=0),
    hoverinfo='skip'
))

# black hemisphere (X <= 0)
fig.add_trace(go.Surface(
    x=np.where(mask_neg, xs, np.nan),
    y=np.where(mask_neg, ys, np.nan),
    z=np.where(mask_neg, zs, np.nan),
    surfacecolor=np.zeros_like(xs),
    colorscale=[[0, 'black'], [1, 'black']],
    cmin=0,
    cmax=1,
    showscale=False,
    lighting=dict(ambient=1, diffuse=0, specular=0),
    hoverinfo='skip'
))

# --------------------------
# scatter plot seeds (on surface)
# --------------------------
fig.add_trace(go.Scatter3d(
    x=seeds[:, 0],
    y=seeds[:, 1],
    z=seeds[:, 2],
    mode="markers",
    name="seeds",
    marker=dict(
        size=5,
        color="yellow",
        opacity=1.0,
        symbol="circle"
    ),
    hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>"
))

# add field lines
for topo, lines in lines_by_topo.items():
    first = True  # flag to show legend only once per topo

    # Downsample lines if there are too many
    if len(lines) > max_lines:
        lines_to_plot = random.sample(lines, max_lines)
    else:
        lines_to_plot = lines

    for traj in lines_to_plot:
        traj_s = smooth_traj(traj)
        # traj_s = traj

        fig.add_trace(go.Scatter3d(
            x=traj_s[:, 0],
            y=traj_s[:, 1],
            z=traj_s[:, 2],
            mode='lines',
            line=dict(color=colors[topo], width=2),
            name=topo,
            legendgroup=topo,
            showlegend=first
        ))
        first = False  # only first trace per topo shows in legend

fig.update_layout(
    template="plotly",
    width=1200,
    height=900,
    scene=dict(
            xaxis=dict(title='X [km]', range=[-12 * RM, 4.5 * RM]),
            yaxis=dict(title='Y [km]', range=[-4.5 * RM, 4.5 * RM]),
            zaxis=dict(title='Z [km]', range=[-4.5 * RM, 4.5 * RM]),
            aspectmode='manual',
            aspectratio=dict(x=2.9, y=1.8, z=1.8),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
    legend=dict(
        groupclick="togglegroup"
    ),
    title=f"{case} Magnetic Field Line Topology: t = {step*0.002} s<br>Seeded where surface density lat/lon bin > 200 cm^-3"
)

out_html = f"{case}_{step}_B_vector_density_seed_topology_den200cm-3.html"
fig.write_html(os.path.join(output_folder, out_html), include_plotlyjs="cdn")
fig.write_image(os.path.join(output_folder, out_html.replace(".html", ".png")), scale=2)
plottime = datetime.now()
print(f"Saved figure at {str(plottime)}")
# %%
