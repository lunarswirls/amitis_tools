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
R_m = 2440.e3  # Mercury radius [m]
sim_steps = list(range(27000, 115000 + 1, 1000))

# calculate median current
den01_list = []
den02_list = []

for sim_step in sim_steps:
    f = os.path.join(input_folder, f"Amitis_CPS_Base_{sim_step:06d}_xz_comp.nc")
    ds = xr.open_dataset(f)
    den01, den02 = ds["den01"].values, ds["den02"].values
    while den01.ndim > 3:
        den01, den02 = den01.squeeze(), den02.squeeze()
    den01_list.append(den01)
    den02_list.append(den02)
    ds.close()

den01_med = np.median(np.stack(den01_list, axis=0), axis=0)
den02_med = np.median(np.stack(den02_list, axis=0), axis=0)

den_diff = den01_med - den02_med

# -------------------------------
# GRID COORDINATES
# -------------------------------
ds0 = xr.open_dataset(os.path.join(input_folder, f"Amitis_CPS_Base_{sim_steps[0]:06d}_xz_comp.nc"))
Nx, Ny, Nz = den_diff.shape
x = np.linspace(float(ds0.full_xmin), float(ds0.full_xmax), Nx) / R_m
y = np.linspace(float(ds0.full_ymin), float(ds0.full_ymax), Ny) / R_m
z = np.linspace(float(ds0.full_zmin), float(ds0.full_zmax), Nz) / R_m
ds0.close()

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

fig, ax = plt.subplots(figsize=(8, 6))
plt.pcolormesh(x, z, den_diff[:,54,:], vmin=np.nanmin(den_diff), vmax=200, shading='auto', cmap='winter_r')
circle = plt.Circle((0, 0), 1, edgecolor='black', facecolor='darkorange', alpha=1., linewidth=1, )
ax.add_patch(circle)
plt.xlabel(r"$\text{X (R}_{M}\text{)}$")
plt.ylabel(r"$\text{Z (R}_{M}\text{)}$")
plt.title(r"CPS Median den01-den02")
plt.colorbar(label=r"den01-den02")
ax.grid(which='major', axis='x', color='#000000', linestyle='--', alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(outdir, f"cps_density_median_diff.png")
plt.savefig(fig_path, dpi=300)
plt.close()