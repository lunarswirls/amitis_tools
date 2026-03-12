#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines
from src.field_topology.topology_utils import trace_field_line_rk, classify

# ======================
# USER PARAMETERS
# ======================
cases = [
    "RPN_HNHV",
    "CPN_HNHV",
    "RPS_HNHV",
    "CPS_HNHV"
]

titles = [
    "(A) RPN",
    "(B) CPN",
    "(C) RPS",
    "(D) CPS"
]

RM = 2440.0  # Mercury radius [km]
dx = 75.0    # grid spacing

max_steps = 50000
h_step = 50.0

plot_lines = True

# Base directories
base_input = "/Volumes/data_backup/mercury/extreme/High_HNHV"
base_output = "/Users/danywaller/Projects/mercury/extreme/bfield_topology/2x2_figure/"
os.makedirs(base_output, exist_ok=True)

# ======================
# TRACE ALL CASES FIRST
# ======================
all_lines_by_case = {}

for case in cases:
    input_folder = f"{base_input}/{case}/plane_product/cube/"
    ncfile = os.path.join(input_folder, f"Amitis_{case}_115000_merged_4RM.nc")
    print(f"Processing {ncfile}")

    ds = xr.open_dataset(ncfile)
    x = ds["Nx"].values
    y = ds["Ny"].values
    z = ds["Nz"].values

    # transform Nz,Ny,Nx --> Nx,Ny,Nz
    Bx = np.transpose(ds["Bx_tot"].isel(time=0).values, (2, 1, 0))  # nT
    By = np.transpose(ds["By_tot"].isel(time=0).values, (2, 1, 0))  # nT
    Bz = np.transpose(ds["Bz_tot"].isel(time=0).values, (2, 1, 0))  # nT

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    zmin, zmax = z.min(), z.max()

    # ----------------------
    # Create seeds
    # ----------------------
    y0 = 0.0  # X–Z plane
    seeds = []

    # Surface seeds
    n_surface = 90
    theta = np.linspace(0, 2 * np.pi, n_surface, endpoint=False)
    for t in theta:
        x_s = RM * np.cos(t)
        z_s = RM * np.sin(t)
        seeds.append([x_s, y0, z_s])

    # Domain border seeds
    n_border = 50
    z_vals = np.linspace(zmin, zmax, n_border)
    for z_s in z_vals:
        seeds.append([xmin, y0, z_s])
        seeds.append([xmax, y0, z_s])
    x_vals = np.linspace(xmin, xmax, n_border)
    for x_s in x_vals:
        seeds.append([x_s, y0, zmin])
        seeds.append([x_s, y0, zmax])

    seeds = np.array(seeds)
    print(f"Total seeds for {case}: {len(seeds)}")

    # ----------------------
    # Trace field lines
    # ----------------------
    lines_by_topo = {"Planet-connected": [], "IMF": []}

    for seed in seeds:
        traj_fwd, exit_fwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=h_step)
        traj_bwd, exit_bwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=-h_step)
        topo = classify(traj_fwd, traj_bwd, RM, exit_fwd_y, exit_bwd_y)

        if topo in ["closed", "open"]:
            lines_by_topo["Planet-connected"].append(traj_fwd[:, [0, 2]])
            lines_by_topo["Planet-connected"].append(traj_bwd[:, [0, 2]])
        elif topo == "solar_wind":
            lines_by_topo["IMF"].append(traj_fwd[:, [0, 2]])
            lines_by_topo["IMF"].append(traj_bwd[:, [0, 2]])

    all_lines_by_case[case] = lines_by_topo
    ds.close()

# ======================
# PLOT 2x2 FIGURE
# ======================
colors = {"Planet-connected": "deepskyblue", "IMF": "gray"}
fig, axes = plt.subplots(2, 2, figsize=(7, 7))
axes = axes.flatten()

for i, case in enumerate(cases):
    ax = axes[i]

    # Draw Mercury surface
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(RM*np.cos(theta), RM*np.sin(theta), "k", lw=2)

    # Plot field lines
    for topo, segments in all_lines_by_case[case].items():
        if segments:
            lc = LineCollection(segments, colors=colors[topo], linewidths=0.8, alpha=0.5)
            ax.add_collection(lc)

    # Axis formatting
    ax.set_aspect("equal")
    ax.set_xlim(-4*RM, 4*RM)
    ax.set_ylim(-4*RM, 4*RM)
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Z [km]")
    ax.set_title(titles[i], fontweight="bold")

# Shared legend
legend_handles = [mlines.Line2D([], [], color=c, label=k) for k, c in colors.items()]
leg = fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=12)
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
# bulk-set the properties of all lines and texts
plt.setp(leg_lines, linewidth=2.5)

plt.suptitle("Magnetic Field Line Topology (t = 230 s)", fontsize=14, fontweight="bold", y=0.99)

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_png = os.path.join(base_output, "all_cases_2x2_topology.png")
plt.savefig(out_png, dpi=300)
plt.show()
print("Saved figure:", out_png)