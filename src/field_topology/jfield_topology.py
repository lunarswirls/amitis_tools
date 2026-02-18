# %%
import os
from datetime import datetime
import numpy as np
import pandas as pd
import random
import xarray as xr
from src.field_topology.topology_utils import *
import plotly.graph_objects as go

# --------------------------
# SETTINGS
# --------------------------
case = "CPN_Base"
step = 115000  # 115000 for base or 350000 for hnhv

# input_folder = f"/Users/danywaller/Projects/mercury/extreme/{case}_Base_largerxdomain_smallergridsize/out/"
# ncfile = os.path.join(input_folder, f"Amitis_{case}_Base_115000.nc")

# input_folder = f"/Volumes/data_backup/mercury/extreme/{case}_Base/plane_product/object/"
# ncfile = os.path.join(input_folder, f"Amitis_{case}_Base_115000_xz_comp.nc")

input_folder = f"/Volumes/data_backup/mercury/extreme/{case}/05/out/"
ncfile = os.path.join(input_folder, f"Amitis_{case}_{step}.nc")

# input_folder = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}_HNHV/plane_product/object/"
# ncfile = os.path.join(input_folder, f"Amitis_{case}_HNHV_350000_xz_comp.nc")

# input_folder = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/10/out/"
# ncfile = os.path.join(input_folder, f"Amitis_{case}_350000.nc")

# output_folder = f"/Users/danywaller/Projects/mercury/extreme/jfield_topology/{case}_Base_largerxdomain_smallergridsize/"
output_folder = f"/Users/danywaller/Projects/mercury/extreme/jfield_topology/{case}/"
os.makedirs(output_folder, exist_ok=True)

# Planet parameters
RM = 2440.0          # Mercury radius [km]
RC = 2400.0          # depth within conductive layer [km]

plot_depth = RM

# Seed settings
n_lat = 60
n_lon = n_lat*2
max_steps = 5000
h_step = 50.0
surface_tol = 75.0

max_lines = 250  # downsample trajectory points for plotting

# %%
# --------------------------
# CREATE SEEDS ON SPHERE
# --------------------------
lats_surface = np.linspace(-90, 90, n_lat)
lons_surface = np.linspace(-180, 180, n_lon)
seeds = []
for lat in lats_surface:
    for lon in lons_surface:
        phi = np.radians(lat)
        theta = np.radians(lon)
        x_s = plot_depth*np.cos(phi)*np.cos(theta)
        y_s = plot_depth*np.cos(phi)*np.sin(theta)
        z_s = plot_depth*np.sin(phi)
        seeds.append(np.array([x_s, y_s, z_s]))
seeds = np.array(seeds)

# %%
# --------------------------
# LOAD VECTOR FIELD FROM NETCDF
# --------------------------
def load_field(ncfile):
    ds = xr.open_dataset(ncfile)
    x = ds["Nx"].values
    y = ds["Ny"].values
    z = ds["Nz"].values

    # Extract fields (drop time dimension) and transpose:  Nz, Ny, Nx --> Nx, Ny, Nz
    Jx = np.transpose(ds["Jx"].isel(time=0).values, (2,1,0))
    Jy = np.transpose(ds["Jy"].isel(time=0).values, (2,1,0))
    Jz = np.transpose(ds["Jz"].isel(time=0).values, (2,1,0))
    ds.close()
    return x, y, z, Jx, Jy, Jz

x, y, z, Jx, Jy, Jz = load_field(ncfile)

start = datetime.now()
print(f"Loaded {ncfile} at {str(start)}")

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
# function to lightly smooth field lines
from scipy.signal import savgol_filter

def smooth_traj(traj, k=5, order=2):
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
    title=f"{case} Current Field Line Topology: t = {step*0.002} s"
)

out_html = f"{case}_{step}_J_vector_topology.html"
fig.write_html(os.path.join(output_folder, out_html), include_plotlyjs="cdn")
fig.write_image(os.path.join(output_folder, out_html.replace(".html", ".png")), scale=2)
plottime = datetime.now()
print(f"Saved figure at {str(plottime)}")
