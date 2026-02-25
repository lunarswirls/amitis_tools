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
branch = "HNHV"
case = "CPN"
# 115000 (pre) or 142000 (transient) or 174000 (post) or 350000 (new)
step = 350000

input_folder = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}_HNHV/plane_product/object/"
ncfile = os.path.join(input_folder, f"Amitis_{case}_HNHV_{step}_xz_comp.nc")

output_folder = f"/Users/danywaller/Projects/mercury/extreme/jfield_topology/{case}_{branch}/"
os.makedirs(output_folder, exist_ok=True)

plot_lines = True  # True to plot field lines
RM = 2440.0  # Mercury radius [km]
dx = 75.0  # grid spacing
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

Bx = np.transpose(ds["Jx"].isel(time=0).values, (2, 1, 0))  # Nx,Ny,Nz
By = np.transpose(ds["Jy"].isel(time=0).values, (2, 1, 0))
Bz = np.transpose(ds["Jz"].isel(time=0).values, (2, 1, 0))

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

seeds = np.array(seeds)
print(f"Total seeds: {len(seeds)}")

# Compute footprints
footprints = []
footprints_class = []
lines_by_topo = {"closed": [], "open": []}

for seed in seeds:
    traj_fwd, exit_fwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=h_step)
    traj_bwd, exit_bwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=-h_step)
    topo = classify(traj_fwd, traj_bwd, RM, exit_fwd_y, exit_bwd_y)
    if topo in ["closed", "open"]:
        lines_by_topo[topo].append(traj_fwd[:, [0, 2]])
        lines_by_topo[topo].append(traj_bwd[:, [0, 2]])

# ======================
# PLOT FIELD LINES (X-Z PLANE)
# ======================
if plot_lines:
    colors = {"closed": "blue", "open": "red"}
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
    ax.set_title(f"{case.replace("_", " ")} Current Topology (X-Z Plane); t = {step*0.002} s")
    ax.set_xlim(-6 * RM, 4.5 * RM)
    ax.set_ylim(-4.5 * RM, 4.5 * RM)
    plt.tight_layout()
    output_topo = os.path.join(output_folder, "2D_topology/")
    os.makedirs(output_topo, exist_ok=True)
    plt.savefig(os.path.join(output_topo, f"{case}_jfield_topology_{step}.png"), dpi=150,
                bbox_inches="tight")
    print("Saved:\t", os.path.join(output_topo, f"{case}_jfield_topology_{step}.png"))
    plt.close()
