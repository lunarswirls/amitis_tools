#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# SETTINGS
input_folder = "/Users/danywaller/Projects/mercury/CPS_Base/fig_xz/"
out_folder = "/Users/danywaller/Projects/mercury/CPS_Base/slice_current/"
os.makedirs(out_folder, exist_ok=True)

# name of variables inside the NetCDF file
VAR_U = "Jx"   # x-component of current
VAR_V = "Jy"   # y-component of current
VAR_W = "Jz"   # z-component of current

# subsampling factor (3D quiver can get heavy)
SUB = 4

# first stable timestamp approx. 25000 for dt=0.002, numsteps=115000
sim_steps = list(range(27000, 115000 + 1, 1000))

for sim_step in sim_steps:
    filename = 'Base_' + "%06d" % sim_step

    f = input_folder + "Amitis_CPS_" + filename + "_xz_comp.nc"

    print(f"Processing Amitis_CPS_{str(filename)}_xz_comp.nc ...")

    ds = xr.open_dataset(f)

    # --------------------------------------------------------
    # Extract physical domain extents
    # --------------------------------------------------------
    xmin = float(ds.full_xmin)
    xmax = float(ds.full_xmax)
    ymin = float(ds.full_ymin)
    ymax = float(ds.full_ymax)
    zmin = float(ds.full_zmin)
    zmax = float(ds.full_zmax)

    dx = float(ds.full_dx)
    dz = float(ds.full_dz)

    # Build coordinate arrays
    x = np.arange(xmin, xmax + dx / 2, dx)  # inclusive upper bound
    z = np.arange(zmin, zmax + dz / 2, dz)

    x = x / 2440.e3  # convert to R_m
    z = z / 2440.e3

    # --------------------------------------------------------
    # Extract velocity arrays
    # --------------------------------------------------------
    Jx = ds[VAR_U].sel(Ny=0, method="nearest")
    Jy = ds[VAR_V].sel(Ny=0, method="nearest")
    Jz = ds[VAR_W].sel(Ny=0, method="nearest")

    j_mag = np.sqrt(Jx**2 + Jy**2 + Jz**2)

    data = Jy.squeeze()

    fig, ax = plt.subplots(figsize=(8, 6))
    circle = plt.Circle((0, 0), 1, edgecolor='black', facecolor='cornflowerblue', alpha=0.3, linewidth=1, )
    plt.pcolormesh(x, z, data, vmin=-150, vmax=150, shading='auto', cmap='RdBu')  # gist_heat_r
    ax.add_patch(circle)
    plt.xlabel(r"$\text{X (R}_{M}\text{)}$")
    plt.ylabel(r"$\text{Z (R}_{M}\text{)}$")
    plt.title(f"Jy at y = 0, t = {sim_step * 0.002} seconds")
    plt.xlim([0, 2.6])
    plt.ylim([-2.6, 0])
    plt.colorbar(label="Jy")
    plt.tight_layout()
    fig_path = os.path.join(out_folder, f"cps_jy_{sim_step}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()