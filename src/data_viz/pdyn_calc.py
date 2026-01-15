#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# SETTINGS
input_folder = "/Users/danywaller/Projects/mercury/extreme/CPS_Base/fig_yz/"
out_folder = "/Users/danywaller/Projects/mercury/extreme/CPS_Base/slice_pdyn_yz/"
os.makedirs(out_folder, exist_ok=True)

# proton mass (kg)
M_P = 1.6726e-27

# take the last approx. 10 seconds
sim_steps = list(range(95000, 115000 + 1, 1000))

for sim_step in sim_steps:
    filename = 'Base_' + "%06d" % sim_step

    f = input_folder + "Amitis_CPS_" + filename + "_yz_comp.nc"

    print(f"Processing Amitis_CPS_{str(filename)}_yz_comp.nc ...")

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
    dy = float(ds.full_dy)
    dz = float(ds.full_dz)

    # Build coordinate arrays
    x = np.arange(ymin, ymax + dy / 2, dy)  # inclusive upper bound
    z = np.arange(zmin, zmax + dz / 2, dz)

    x = x / 2440.e3  # convert to R_m
    z = z / 2440.e3

    # velocity in km/s
    Vx = ds["vx01"].sel(Nx=0, method="nearest").squeeze() + ds["vx03"].sel(Nx=0, method="nearest").squeeze()
    Vy = ds["vy01"].sel(Nx=0, method="nearest").squeeze() + ds["vy03"].sel(Nx=0, method="nearest").squeeze()
    Vz = ds["vz01"].sel(Nx=0, method="nearest").squeeze() + ds["vz03"].sel(Nx=0, method="nearest").squeeze()

    v_mag = np.sqrt(Vx**2 + Vy**2 + Vz**2) * 1e3    # km/s -> m/s

    # densities in cm^-3
    den01 = ds["den01"].sel(Nx=0, method="nearest").squeeze()
    den02 = ds["den02"].sel(Nx=0, method="nearest").squeeze()
    den03 = ds["den03"].sel(Nx=0, method="nearest").squeeze()
    den04 = ds["den04"].sel(Nx=0, method="nearest").squeeze()

    # sum all densities to get total density
    den_tot = (den01 + den02 + den03 + den04)  * 1e6     # cm^-3 -> m^-3

    # dynamic pressure (Pa)
    P_pa = den_tot * M_P * v_mag ** 2

    # convert to nPa
    P_npa = P_pa * 1e9

    fig, ax = plt.subplots(figsize=(8, 6))
    circle = plt.Circle((0, 0), 1, edgecolor='black', facecolor='cornflowerblue', alpha=0.3, linewidth=1, )
    plt.pcolormesh(x, z, P_npa, shading='auto', cmap='gist_heat_r', vmin=0, vmax=120)
    ax.add_patch(circle)
    plt.xlabel(r"$\text{Y (R}_{M}\text{)}$")
    plt.ylabel(r"$\text{Z (R}_{M}\text{)}$")
    plt.title(rf"$P_{{dyn}}$ at z = 0, t = {sim_step * 0.002:.3f} s")
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.colorbar(label=r"$P_{dyn}$ [nPa]")
    plt.tight_layout()
    fig_path = os.path.join(out_folder, f"cps_pdyn_{sim_step}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()