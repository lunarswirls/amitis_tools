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
case = "CPS"

input_folder = f"/Volumes/data_backup/extreme_base/{case}_Base/plane_product/all_xz/"
ncfile = os.path.join(input_folder, f"Amitis_{case}_Base_115000_xz_comp.nc")
output_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology/{case}_Base/"
os.makedirs(output_folder, exist_ok=True)

plot_lines = True  # True to plot field lines
RM = 2440.0  # Mercury radius [km]
dx = 75.0  # grid spacing
trace_length = 8 * RM
surface_tol = dx

# seed resolution
n_lat = 60
n_lon = 120

max_steps = 5000  # max RK steps
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

Bx = np.transpose(ds["Bx_tot"].isel(time=0).values, (2, 1, 0))  # Nx,Ny,Nz
By = np.transpose(ds["By_tot"].isel(time=0).values, (2, 1, 0))
Bz = np.transpose(ds["Bz_tot"].isel(time=0).values, (2, 1, 0))

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
zmin, zmax = z.min(), z.max()

# ======================
# CREATE SEED POINTS ON SURFACE
# ======================
lats_surface = np.linspace(-90, 90, n_lat)
lons_surface = np.linspace(-180, 180, n_lon)

seeds = []
for lat in lats_surface:
    for lon in lons_surface:
        phi = np.radians(lat)
        theta = np.radians(lon)
        x_s = RM * np.cos(phi) * np.cos(theta)
        y_s = RM * np.cos(phi) * np.sin(theta)
        z_s = RM * np.sin(phi)
        seeds.append(np.array([x_s, y_s, z_s]))
seeds = np.array(seeds)

# Compute footprints
footprints = []
footprints_class = []
lines_by_topo = {"closed": [], "open": []}

for seed in seeds:
    traj_fwd, exit_fwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=h_step)
    traj_bwd, exit_bwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=-h_step)
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

csv_file = os.path.join(output_folder, f"Amitis_{case}_Base_115000_xz_comp_footprints.csv")
df_planet.to_csv(csv_file, index=False)
print(f"Saved {len(df_planet)} footprints to {csv_file}")

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
    legend_handles = [mlines.Line2D([], [], color=c, label=k) for k, c in colors.items() if k in ["closed", "open"]]
    ax.legend(handles=legend_handles, loc="upper right")

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Z [km]")
    ax.set_aspect("equal")
    ax.set_title("Mercury Magnetic Field-Line Topology (X-Z Plane)")
    ax.set_xlim(-5 * RM, 5 * RM)
    ax.set_ylim(-5 * RM, 5 * RM)
    plt.tight_layout()
    plt.show()

# ======================
# PLOT FOOTPRINTS
# ======================
fig, ax = plt.subplots(figsize=(9, 4))
for topo, color in [("closed", "blue"), ("open", "red")]:
    subset = df_planet[df_planet['classification'] == topo]
    if subset.empty: continue
    ax.scatter(subset['longitude_deg'], subset['latitude_deg'], s=10, color=color, label=topo, alpha=0.7)

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")
ax.set_title("Mercury Magnetic Footprints")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()


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
