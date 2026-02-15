#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import os, glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pyamitis.amitis_particle import *

# -----------------------------
# Case selection
# -----------------------------
case = "inert_sunward"

dt = 20  # 1 or 20
nfiles = 28  # 11 or 28

if "inert_sunward" in case:
    main_path = "/Users/danywaller/Projects/mercury/inert_small_body_sunward_IMF/"
    case = "SW_IMF"
elif "inert_planetward" in case:
    main_path = "/Users/danywaller/Projects/mercury/inert_small_body_planetward_IMF/"
    case = "PW_IMF"
else:
    raise ValueError("Unrecognized case! Are you using inert_body files?")

# -----------------------------
# Geometry + grid cell volume
# -----------------------------
R_M = 2440.0e3  # [m]

dx = dy = dz = 75.0e3  # [m]
V_cell = dx * dy * dz  # [m^3]

# Shell thicknesses to test (meters)
dR_values = np.arange(7.5, 75.0e3 + 1.0, 100.0)  # 7.5 m → 75 km in 100 m steps
select_R_values = R_M + dR_values

# -----------------------------
# Particle file paths
# -----------------------------
sub_filepath = main_path + f"particles_{dt}sec_n{nfiles}/"
sub_filename = f"Subset_{case}"

subset_filelist = np.array(sorted(glob.glob(sub_filepath + sub_filename + "*.npz")))
subset_filelist = np.unique([f[:-9] for f in subset_filelist])

# -----------------------------
# Load ALL particles once
# -----------------------------
prx_all, pry_all, prz_all = [], [], []
psid_all = []

start_time = datetime.now()
file_counter = 0

for f in subset_filelist:
    stem = Path(f).stem
    sim_step = stem.split("_")[3]
    print(f"---------- {sim_step} ----------")

    obj = amitis_particle(sub_filepath, sub_filename, int(sim_step))
    obj.load_particle_data(None)

    prx_all.append(obj.rx)
    pry_all.append(obj.ry)
    prz_all.append(obj.rz)
    psid_all.append(obj.sid)

    file_counter += 1

# Concatenate
prx_all = np.concatenate(prx_all)
pry_all = np.concatenate(pry_all)
prz_all = np.concatenate(prz_all)
psid_all = np.concatenate(psid_all)

r_all = np.sqrt(prx_all**2 + pry_all**2 + prz_all**2)

print(f"\nLoaded {len(psid_all):,} total macroparticles across {file_counter} files.")

# -----------------------------
# Count macroparticles vs shell thickness and compute macroparticles/cell
# -----------------------------
N_macro_shell = np.zeros_like(dR_values, dtype=np.int64)
macro_per_cell = np.zeros_like(dR_values, dtype=float)
N_cells_shell = np.zeros_like(dR_values, dtype=float)

# Precompute constant inner volume
V_inner = (4.0 / 3.0) * np.pi * R_M**3

for k, select_R in enumerate(select_R_values):
    mask = (r_all >= R_M) & (r_all <= select_R)
    N = int(np.sum(mask))
    N_macro_shell[k] = N

    # Shell volume and implied number of grid cells
    V_outer = (4.0 / 3.0) * np.pi * select_R**3
    V_shell = max(V_outer - V_inner, 0.0)  # [m^3]
    n_cells = V_shell / V_cell if V_shell > 0 else np.nan
    N_cells_shell[k] = n_cells

    # Macroparticles per grid cell (mean)
    macro_per_cell[k] = (N / n_cells) if (np.isfinite(n_cells) and n_cells > 0) else np.nan

# -----------------------------
# Plot: (1) Total macroparticles vs ΔR
#       (2) Macroparticles per grid cell vs ΔR
# -----------------------------
dR_km = dR_values * 1e-3

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

# Panel 1: Total in shell
axes[0].plot(dR_km, N_macro_shell, c="hotpink", lw=2)
axes[0].set_ylabel("Total macroparticles in shell", fontsize=14)
axes[0].set_title(f"Macroparticle statistics vs shell thickness: dt = {dt} s, N$_{{files}}$ = {nfiles}", fontsize=16)
axes[0].grid(True, alpha=0.3)

# Panel 2: Macroparticles per grid cell
axes[1].plot(dR_km, macro_per_cell, c="mediumorchid", lw=2)
# Horizontal reference line: 10 macroparticles per cell
axes[1].axhline(
    10.0,
    linestyle="--",
    linewidth=2,
    alpha=0.7
)
axes[1].text(
    0.97, 9.95,
    "10 macroparticles / cell",
    ha="right",
    va="top",
    transform=axes[1].get_yaxis_transform(),
    fontsize=12
)
axes[1].set_xlabel(r"Shell thickness $dR$ [km]", fontsize=14)
axes[1].set_ylabel("Macroparticles per grid cell", fontsize=14)
axes[1].grid(True, alpha=0.3)

# Annotate cell volume used
V_cell_km3 = V_cell * 1e-9  # m^3 → km^3

axes[1].text(
    0.02, 0.95,
    (
        rf"$\Delta x=\Delta y=\Delta z={dx/1e3:.0f}\,\mathrm{{km}}$" "\n"
        rf"$V_{{cell}}={V_cell_km3:.0f}\,\mathrm{{km}}^3$"
    ),
    transform=axes[1].transAxes,
    va="top",
    fontsize=12
)

outfile = os.path.join(main_path, f"{case}_macrocount_and_macro_per_cell_vs_dR_{dt}sec_n{nfiles}.png")
plt.tight_layout()
plt.savefig(outfile, dpi=150)
plt.close(fig)

print("Saved plot:", outfile)

fig, ax1 = plt.subplots(1, 1, figsize=(6, 6), sharex=True)

# Panel 2: Macroparticles per grid cell
ax1.plot(dR_km, macro_per_cell, c="mediumorchid", lw=2)
# Horizontal reference line: 10 macroparticles per cell
ax1.axhline(
    10.0,
    linestyle="--",
    linewidth=2,
    alpha=0.7
)
ax1.text(
    0.97, 9.95,
    "10 macroparticles / cell",
    ha="right",
    va="top",
    transform=ax1.get_yaxis_transform(),
    fontsize=12
)
ax1.set_xlabel(r"Shell thickness $dR$ [km]", fontsize=14)
ax1.set_ylabel("Macroparticles per grid cell", fontsize=14)
ax1.set_title(f"Macroparticle count vs shell thickness:\ndt = {dt} s, N$_{{files}}$ = {nfiles}", fontsize=16)
ax1.grid(True, alpha=0.3)

# Annotate cell volume used
V_cell_km3 = V_cell * 1e-9  # m^3 → km^3

ax1.text(
    0.02, 0.95,
    (
        rf"$\Delta x=\Delta y=\Delta z={dx/1e3:.0f}\,\mathrm{{km}}$" "\n"
        rf"$V_{{cell}}={V_cell_km3:.0f}\,\mathrm{{km}}^3$"
    ),
    transform=ax1.transAxes,
    va="top",
    fontsize=12
)

outfile = os.path.join(main_path, f"{case}_macro_per_cell_vs_dR_{dt}sec_n{nfiles}.png")
plt.tight_layout()
plt.savefig(outfile, dpi=150)
plt.close(fig)

print("Saved plot:", outfile)

end_time = datetime.now()
print(f"Finished {case} in {(end_time - start_time).total_seconds():.1f} seconds\n")