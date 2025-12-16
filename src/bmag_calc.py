#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# SETTINGS
input_folder = "/Users/danywaller/Projects/mercury/RPS_Base/object/"
out_folder = "/Users/danywaller/Projects/mercury/RPS_Base/slice_b_dipole_mag/"
os.makedirs(out_folder, exist_ok=True)

# name of variables inside the NetCDF file
VAR_X = "Bx"   # x-component of B
VAR_Y = "By"   # y-component of B
VAR_Z = "Bz"   # z-component of B

VAR_XT = "Bx_tot"   # x-component of Btotal
VAR_YT = "By_tot"   # y-component of Btotal
VAR_ZT = "Bz_tot"   # z-component of Btotal

# first stable timestamp approx. 25000 for dt=0.002, numsteps=115000
sim_steps = list(range(27000, 115000 + 1, 1000))

for sim_step in sim_steps:
    filename = 'Base_' + "%06d" % sim_step

    f = input_folder + "Amitis_RPS_" + filename + "_xz_comp.nc"

    ds = xr.open_dataset(f)

    # Extract physical domain extents
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

    # Extract arrays
    BX = ds[VAR_X].sel(Ny=0, method="nearest").squeeze()
    BY = ds[VAR_Y].sel(Ny=0, method="nearest").squeeze()
    BZ = ds[VAR_Z].sel(Ny=0, method="nearest").squeeze()

    BXT = ds[VAR_XT].sel(Ny=0, method="nearest").squeeze()
    BYT = ds[VAR_YT].sel(Ny=0, method="nearest").squeeze()
    BZT = ds[VAR_ZT].sel(Ny=0, method="nearest").squeeze()

    # differences
    # dBx = np.abs(BXT - BX)
    # dBy = np.abs(BYT - BY)
    # dBz = np.abs(BZT - BZ)
    dBmag_raw = np.sqrt(BXT ** 2 + BYT ** 2 + BZT ** 2) - np.sqrt(BX ** 2 + BY ** 2 + BZ ** 2)

    threshold = 150.0

    dBx = np.abs(BXT - BX).where(np.abs(BXT - BX) <= threshold, np.nan)
    dBy = np.abs(BYT - BY).where(np.abs(BYT - BY) <= threshold, np.nan)
    dBz = np.abs(BZT - BZ).where(np.abs(BZT - BZ) <= threshold, np.nan)
    dBmag = dBmag_raw.where(np.abs(dBmag_raw) <= threshold, np.nan)

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axs = axes.ravel()

    titles = [r"$|B_x^{tot}-B_x|$", r"$|B_y^{tot}-B_y|$", r"$|B_z^{tot}-B_z|$", r"$|\mathbf{B}_{tot}| - |\mathbf{B}|$"]
    data_list = [dBx, dBy, dBz, dBmag]
    cmaps = ["Reds", "Blues", "Greens", "Purples"]

    for ax, data, title, cmap in zip(axs, data_list, titles, cmaps):
        if cmap == "Purples":
            v_min = 0
            v_max = 100
        else:
            v_min = 0
            v_max = 100
        im = ax.pcolormesh(x, z, data, vmin=v_min, vmax=v_max, shading="auto", cmap=cmap)
        circle = plt.Circle((0, 0), 1, edgecolor="black", facecolor="none", linewidth=2)
        ax.add_patch(circle)
        ax.set_title(title, fontsize=14)
        fig.colorbar(im, ax=ax, orientation="vertical", label=title)

    for ax in axs:
        ax.set_xlabel(r"$X\ (\mathrm{R_M})$")
        ax.set_ylabel(r"$Z\ (\mathrm{R_M})$")
        ax.set_aspect("equal")

    plt.suptitle(f"RPS Magnetic Field Differences at y=0, t={sim_step * 0.002:.2f} s", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    fig_path = os.path.join(out_folder, f"rps_b_diff_{sim_step}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
