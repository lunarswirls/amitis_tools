#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines
from src.field_topology.topology_utils import *

# ======================
# USER PARAMETERS
# ======================
case = "CPS_Base_largerxdomain_smallergridsize"
input_folder = f"/Users/danywaller/Projects/mercury/extreme/{case}/out/"
if "larger" in case:
    fname = case.split("_")[0] + "_" + case.split("_")[1]
else:
    fname = case
ncfile = os.path.join(input_folder, f"Amitis_{fname}_115000.nc")

output_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology/{case}/"
os.makedirs(output_folder, exist_ok=True)

plot_lines = True  # True to plot field lines
RM = 2440.0  # Mercury radius [km]
dx = 200.0  # grid spacing
trace_length = 20 * RM
surface_tol = dx

# seed resolution
n_lat = 60
n_lon = n_lat*2

max_steps = 100000  # max RK steps
h_step = 50.0  # integration step size [km]

start = datetime.now()
print(f"Started processing {ncfile} at {str(start)}")
step = int(ncfile.split("/")[-1].split("_")[3].split(".")[0])

# ======================
# LOAD DATA
# ======================
ds = xr.open_dataset(ncfile)
x = ds["Nx"].values
y = ds["Ny"].values
z = ds["Nz"].values

# Extract fields (drop time dimension)
Bx = (ds["Bx"].isel(time=0).values + ds["Bdx"].isel(time=0).values)
By = (ds["By"].isel(time=0).values + ds["Bdy"].isel(time=0).values)
Bz = (ds["Bz"].isel(time=0).values + ds["Bdz"].isel(time=0).values)

#  Transpose: Nz, Ny, Nx --> Nx, Ny, Nz
Bx_plane = np.transpose(Bx, (2, 1, 0))
By_plane = np.transpose(By, (2, 1, 0))
Bz_plane = np.transpose(Bz, (2, 1, 0))

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
zmin, zmax = z.min(), z.max()

# ======================
# CREATE 2D SEEDS IN X–Z PLANE
# ======================
y0 = 0.0  # X–Z plane

seeds = []

# ---- 1. Planet surface seeds (circle in X–Z plane) ----
n_surface = 360
theta = np.linspace(0, 2 * np.pi, n_surface, endpoint=False)

for t in theta:
    x_s = RM * np.cos(t)
    z_s = RM * np.sin(t)
    seeds.append([x_s, y0, z_s])

# ---- 2. Domain border seeds ----
n_border = 200

# Left / Right boundaries
z_vals = np.linspace(zmin, zmax, n_border)
for z_s in z_vals:
    seeds.append([xmin, y0, z_s])
    seeds.append([xmax, y0, z_s])

# Top / Bottom boundaries
x_vals = np.linspace(xmin, xmax, n_border)
for x_s in x_vals:
    seeds.append([x_s, y0, zmin])
    seeds.append([x_s, y0, zmax])

seeds = np.array(seeds)
print(f"Total seeds: {len(seeds)}")

# Compute footprints
footprints = []
footprints_class = []
lines_by_topo = {"closed": [], "open": [], "solar_wind": []}

for seed in seeds:
    traj_fwd, exit_fwd_y = trace_field_line_rk(seed, Bx_plane, By_plane, Bz_plane, x, y, z, RM, max_steps=max_steps, h=h_step)
    traj_bwd, exit_bwd_y = trace_field_line_rk(seed, Bx_plane, By_plane, Bz_plane, x, y, z, RM, max_steps=max_steps, h=-h_step)
    topo = classify(traj_fwd, traj_bwd, RM, exit_fwd_y, exit_bwd_y)
    if topo not in ["TBD"]:
        lines_by_topo[topo].append(traj_fwd[:, [0, 2]])
        lines_by_topo[topo].append(traj_bwd[:, [0, 2]])

    for traj in [traj_fwd, traj_bwd]:
        r_end = traj[-1]
        if np.linalg.norm(r_end) <= RM + surface_tol:
            lat, lon = cartesian_to_latlon(r_end)
            footprints.append((lat, lon))
            footprints_class.append(topo)

df_planet = pd.DataFrame({
    "latitude_deg": [lat for lat, lon in footprints],
    "longitude_deg": [lon for lat, lon in footprints],
    "classification": footprints_class
})

# ======================
# PLOT FIELD LINES (X-Z PLANE)
# ======================
if plot_lines:
    colors = {"closed": "blue", "open": "red", "solar_wind": "gray"}
    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw Mercury surface
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(RM * np.cos(theta), RM * np.sin(theta), "k", lw=2)

    # Plot traced field lines
    for topo, segments in lines_by_topo.items():
        if segments:
            lc = LineCollection(segments, colors=colors[topo], linewidths=0.8, alpha=0.5)
            ax.add_collection(lc)

    # Legend
    legend_handles = [mlines.Line2D([], [], color=c, label=k) for k, c in colors.items() if k in ["closed", "open", "solar_wind"]]
    ax.legend(handles=legend_handles, loc="upper right")

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Z [km]")
    ax.set_aspect("equal")
    ax.set_title(f"{case.replace("_", " ")} Magnetic Field-Line Topology (X-Z Plane)")
    ax.set_xlim(-10 * RM, 4.5 * RM)
    ax.set_ylim(-4.5 * RM, 4.5 * RM)
    plt.tight_layout()
    output_topo = os.path.join(output_folder, "2D_topology/")
    os.makedirs(output_topo, exist_ok=True)
    plt.savefig(os.path.join(output_topo, f"{case}_bfield_topology_{step}.png"), dpi=150,
                bbox_inches="tight")
    print("Saved:\t", os.path.join(output_topo, f"{case}_bfield_topology_{step}.png"))
    plt.close()

# ======================
# LATITUDE STATISTICS
# ======================
def stats(arr):
    return np.min(arr), np.max(arr), np.mean(arr)


for topo in ["closed", "open"]:
    subset = df_planet[df_planet['classification'] == topo]
    if subset.empty:
        print(f"No {topo} footprints found.")
        continue

    lats = subset['latitude_deg'].values
    north = lats[lats >= 0]
    south = lats[lats < 0]

    if len(north) > 0:
        n_min, n_max, n_mean = stats(north)
        print(f"North hemisphere ({topo}): min={n_min:.2f}°, max={n_max:.2f}°, mean={n_mean:.2f}°")
    else:
        print(f"No north hemisphere {topo} footprints.")

    if len(south) > 0:
        s_min, s_max, s_mean = stats(south)
        print(f"South hemisphere ({topo}): min={s_min:.2f}°, max={s_max:.2f}°, mean={s_mean:.2f}°")
    else:
        print(f"No south hemisphere {topo} footprints.")
