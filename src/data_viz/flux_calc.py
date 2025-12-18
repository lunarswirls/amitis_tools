#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt

# change this to wherever you want your output figures to be saved
outdir = "/Users/danywaller/Projects/mercury/rps_cps_comparison/surf_flux/"
os.makedirs(outdir, exist_ok=True)  # creates directory if it doesnt already exist

# change this to your input directory
indir = "/Users/danywaller/Projects/mercury/"

# first stable timestamp approx. 25000 for dt=0.002, numsteps=115000
sim_steps = list(range(27000, 115000 + 1, 1000))

for sim_step in sim_steps:
    filename = 'Base_' + "%06d" % sim_step

    f1 = indir + "RPS_Base/object/Amitis_RPS_" + filename + "_xz_comp.nc"
    f2 = indir + "CPS_Base/object/Amitis_CPS_" + filename + "_xz_comp.nc"

    # --- Load files ---
    ds1 = xr.open_dataset(f1)
    ds2 = xr.open_dataset(f2)

    tot_den_rps = ds1["den01"].sel(Ny=0, method="nearest")
    tot_den_cps = ds2["den01"].sel(Ny=0, method="nearest")

    varnames = ["den02", "den03", "den04"]

    # --- Extract x–z plane at y = 0 and sum ---
    for var in varnames:
        tot_den_rps += ds1[var].sel(Ny=0, method="nearest")
        tot_den_cps += ds2[var].sel(Ny=0, method="nearest")

    # Convert density from cm^-3 to km^-3
    den_rps_km3 = tot_den_rps * 1e15
    den_cps_km3 = tot_den_cps * 1e15

    vx01_rps = ds1["vx01"].sel(Ny=0, method="nearest")
    vx01_cps = ds2["vx01"].sel(Ny=0, method="nearest")
    vy01_rps = ds1["vy01"].sel(Ny=0, method="nearest")
    vy01_cps = ds2["vy01"].sel(Ny=0, method="nearest")
    vz01_rps = ds1["vz01"].sel(Ny=0, method="nearest")
    vz01_cps = ds2["vz01"].sel(Ny=0, method="nearest")

    vx03_rps = ds1["vx03"].sel(Ny=0, method="nearest")
    vx03_cps = ds2["vx03"].sel(Ny=0, method="nearest")
    vy03_rps = ds1["vy03"].sel(Ny=0, method="nearest")
    vy03_cps = ds2["vy03"].sel(Ny=0, method="nearest")
    vz03_rps = ds1["vz03"].sel(Ny=0, method="nearest")
    vz03_cps = ds2["vz03"].sel(Ny=0, method="nearest")

    v01_mag_rps = np.sqrt(vx01_rps**2 + vy01_rps**2 + vz01_rps**2)  # units: km/s
    v01_mag_cps = np.sqrt(vx01_cps**2 + vy01_cps**2 + vz01_cps**2)  # units: km/s

    v03_mag_rps = np.sqrt(vx03_rps**2 + vy03_rps**2 + vz03_rps**2)  # units: km/s
    v03_mag_cps = np.sqrt(vx03_cps**2 + vy03_cps**2 + vz03_cps**2)  # units: km/s

    mean_v_rps = 0.5 * (v01_mag_rps + v03_mag_rps)  # units: km/s
    mean_v_cps = 0.5 * (v01_mag_cps + v03_mag_cps)  # units: km/s

    surf_flux_rps = mean_v_rps * den_rps_km3  # units: 1/s-km^2
    surf_flux_cps = mean_v_cps * den_cps_km3  # units: 1/s-km^2

    # --- Subtract surface flux ---
    difference = surf_flux_rps - surf_flux_cps

    # Extract metadata from either dataset (ds1 and ds2 should have same extent)
    xmin = float(ds1.full_xmin)
    xmax = float(ds1.full_xmax)
    zmin = float(ds1.full_zmin)
    zmax = float(ds1.full_zmax)

    dx = float(ds1.full_dx)
    dz = float(ds1.full_dz)

    # Build coordinate arrays
    x = np.arange(xmin, xmax + dx/2, dx)   # inclusive upper bound
    z = np.arange(zmin, zmax + dz/2, dz)

    x = x/2440.e3  # convert to R_m
    z = z/2440.e3

    vx_sw = -400.0  # units: km/s
    vy_sw = 0.0  # units: km/s
    vz_sw = 0.0  # units: km/s
    den_h_sw = 38.0e16  # units: km^-3
    den_he_sw = 1.0e16  # units: km^-3
    upstream = (0.5 * (
                np.sqrt(vx_sw ** 2 + vy_sw ** 2 + vz_sw ** 2) + np.sqrt(vx_sw ** 2 + vy_sw ** 2 + vz_sw ** 2))) * (
                           den_h_sw + den_he_sw)

    data = difference.squeeze() / upstream

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.pcolormesh(x, z, data, vmin=0, vmax=5000, shading='auto', cmap='gist_heat_r')
    circle = plt.Circle((0, 0), 1, edgecolor='black', facecolor='cornflowerblue', alpha=0.3, linewidth=1,)
    ax.add_patch(circle)
    plt.xlabel(r"$\text{X (R}_{M}\text{)}$")
    plt.ylabel(r"$\text{Z (R}_{M}\text{)}$")
    plt.title(f"Flux Difference at y = 0, t = {sim_step*0.002} seconds")
    plt.colorbar(label="Flux Difference (RPS - CPS)")
    plt.tight_layout()
    fig_path = os.path.join(outdir, f"flux_diff_{sim_step}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()