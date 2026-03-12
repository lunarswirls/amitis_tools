#!/usr/bin/env python3
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
cases = ["RPN_HNHV", "CPN_HNHV", "RPS_HNHV", "CPS_HNHV"]
titles = ["RPN", "CPN", "RPS", "CPS"]  # for labeling only

selected_times = [230, 270, 330, 700]  # seconds

RM = 2440.0  # Mercury radius [km]
max_steps = 100000
h_step = 50.0

# Base directories
base_input = "/Volumes/data_backup/mercury/extreme/High_HNHV"
base_output = "/Users/danywaller/Projects/mercury/extreme/bfield_topology/4x4_figure/"
os.makedirs(base_output, exist_ok=True)

y_tol = 250e-3 * RM  # tolerance for staying in X-Z plane

# Function to truncate trajectory at Y tolerance
def truncate_at_y_tol(traj, y_tol):
    mask = np.abs(traj[:, 1]) <= y_tol
    if not np.any(mask):
        return None  # entire line is outside tolerance
    # Find last valid index before exceeding y_tol
    last_idx = np.where(mask)[0][-1]
    return traj[:last_idx + 1]

# ======================
# TRACE FIELD LINES FOR EACH CASE AND TIME
# ======================
all_lines = {}  # all_lines[case][time] = {"Planet-connected": [...], "IMF": [...]}

for case in cases:
    input_folder = f"{base_input}/{case}/plane_product/cube/"
    ncfiles = sorted([f for f in os.listdir(input_folder) if f.endswith(".nc")])

    all_lines[case] = {}

    for t_sec in selected_times:
        # Convert time to simulation step
        step = int(t_sec / 0.002)
        # Find matching NC file
        ncfile_match = [f for f in ncfiles if f"_{step}_" in f]
        if not ncfile_match:
            print(f"Warning: no file found for {case} at t={t_sec}s (step={step})")
            continue
        ncfile = os.path.join(input_folder, ncfile_match[0])
        print(f"Processing {ncfile} for t={t_sec}s")

        ds = xr.open_dataset(ncfile)
        x = ds["Nx"].values
        y = ds["Ny"].values
        z = ds["Nz"].values

        # Transform to Nx,Ny,Nz
        Bx = np.transpose(ds["Bx_tot"].isel(time=0).values, (2, 1, 0))
        By = np.transpose(ds["By_tot"].isel(time=0).values, (2, 1, 0))
        Bz = np.transpose(ds["Bz_tot"].isel(time=0).values, (2, 1, 0))

        xmin, xmax = x.min(), x.max()
        zmin, zmax = z.min(), z.max()
        y0 = 0.0  # X-Z plane

        # ---- Seed points ----
        seeds = []

        # Planet surface seeds
        n_surface = 60
        theta = np.linspace(0, 2*np.pi, n_surface, endpoint=False)
        for th in theta:
            seeds.append([RM*np.cos(th), y0, RM*np.sin(th)])

        # Domain border seeds
        n_border = 60
        z_vals = np.linspace(zmin, zmax, n_border)
        for z_s in z_vals:
            seeds.append([xmin, y0, z_s])
            seeds.append([xmax, y0, z_s])
        x_vals = np.linspace(xmin, xmax, n_border)
        for x_s in x_vals:
            seeds.append([x_s, y0, zmin])
            seeds.append([x_s, y0, zmax])

        seeds = np.array(seeds)

        # ---- Trace field lines ----
        lines_by_topo = {"Planet-connected": [], "IMF": []}

        for seed in seeds:
            traj_fwd, exit_fwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z,
                                                       RM, max_steps=max_steps, h=h_step)
            traj_bwd, exit_bwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z,
                                                       RM, max_steps=max_steps, h=-h_step)
            topo = classify(traj_fwd, traj_bwd, RM, exit_fwd_y, exit_bwd_y)

            # Truncate lines that exceed Y tolerance
            traj_fwd_trunc = truncate_at_y_tol(traj_fwd, y_tol)
            traj_bwd_trunc = truncate_at_y_tol(traj_bwd, y_tol)

            if traj_fwd_trunc is None or traj_bwd_trunc is None:
                continue  # if the line never stays in-plane at all, skip

            if 0:
                # Reject lines leaving X-Z plane
                if np.any(np.abs(traj_fwd[:, 1]) > y_tol) or np.any(np.abs(traj_bwd[:, 1]) > y_tol):
                    continue  # skip this seed

            if topo in ["closed", "open"]:
                lines_by_topo["Planet-connected"].append(traj_fwd_trunc[:, [0, 2]])
                lines_by_topo["Planet-connected"].append(traj_bwd_trunc[:, [0, 2]])
            elif topo == "solar_wind":
                lines_by_topo["IMF"].append(traj_fwd_trunc[:, [0, 2]])
                lines_by_topo["IMF"].append(traj_bwd_trunc[:, [0, 2]])

        all_lines[case][t_sec] = lines_by_topo
        ds.close()

# ======================
# PLOT 4x4 FIGURE (rows = times, columns = cases)
# ======================
colors = {"Planet-connected": "deepskyblue", "IMF": "gray"}
n_rows = len(selected_times)
n_cols = len(cases)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 2.5*n_rows), sharex=True, sharey=True)
axes = np.atleast_2d(axes)

for row, t_sec in enumerate(selected_times):
    for col, case in enumerate(cases):
        ax = axes[row, col]

        # Mercury surface (in R_M)
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), "k", lw=2)  # radius normalized to 1 R_M

        # Field lines (convert km -> R_M)
        lines_by_topo = all_lines.get(case, {}).get(t_sec, {})
        for topo, segments in lines_by_topo.items():
            if segments:
                # Divide each segment by RM
                segments_rm = [seg / RM for seg in segments]
                lc = LineCollection(segments_rm, colors=colors[topo], linewidths=0.8, alpha=0.5)
                ax.add_collection(lc)

        ax.set_aspect("equal")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

        # Panel label like (a1), (b2)
        col_letter = chr(ord('a') + col)
        panel_label = f"({col_letter}{row+1})"
        ax.set_title(f"{panel_label} {titles[col]} t = {t_sec:.0f} s", fontsize=10, fontweight="bold")

        # Only leftmost axes get Y label
        if col == 0:
            ax.set_ylabel("Z [R$_M$]")
        # Only bottom axes get X label
        if row == n_rows-1:
            ax.set_xlabel("X [R$_M$]")

# Shared legend at bottom
legend_handles = [mlines.Line2D([], [], color=c, label=k) for k, c in colors.items()]
leg = fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=9)
plt.setp(leg.get_lines(), linewidth=2.5)

# plt.suptitle("Magnetic Field-Line Topology (X-Z Plane)", fontsize=16, fontweight="bold", y=0.975)
# plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.tight_layout(rect=[0, 0.035, 1, 0.995], h_pad=0.5, w_pad=0.2)

# Save figure
out_png = os.path.join(base_output, "all_cases_4x4_topology.png")
plt.savefig(out_png, dpi=300)  #), bbox_inches="tight")
plt.show()
print("Saved figure:", out_png)