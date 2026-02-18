#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports:
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# SETTINGS
case = "RPS_HNHV"

sim_end = True

# Planet parameters
RM = 2440.0  # km

# Shell limits
rmin = "1.00"
rmax = "1.05"

if "Base" in case:
    input_folder = f"/Volumes/data_backup/mercury/extreme/{case}/plane_product/object/"
    output_folder = f"/Users/danywaller/Projects/mercury/extreme/density_lonlat/{case}/"
    sim_steps = list(range(105000, 115000 + 1, 1000))
elif "HNHV" in case and not sim_end:
    input_folder = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/object/"
    output_folder = f"/Users/danywaller/Projects/mercury/extreme/density_lonlat/{case}/"
    sim_steps = range(115000, 200000 + 1, 1000)
elif "HNHV" in case and sim_end:
    input_folder = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/object/"
    output_folder = f"/Users/danywaller/Projects/mercury/extreme/density_lonlat/{case}_end/"
    sim_steps = range(115000, 350000 + 1, 1000)
else:
    raise ValueError("Case not recognized")

output_folder += f"/{rmin}-{rmax}_RM/"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

rmin = float(rmin) * RM
rmax = float(rmax) * RM

for step in sim_steps:

    ncfile = os.path.join(input_folder, f"Amitis_{case}_{step}_xz_comp.nc")

    # ------------------------
    # LOAD DENSITY
    # ------------------------
    def load_density(nc_file):
        ds = xr.open_dataset(nc_file)

        x = ds["Nx"].values
        y = ds["Ny"].values
        z = ds["Nz"].values

        # densities in cm^-3
        den01 = ds["den01"].isel(time=0).values
        den02 = ds["den02"].isel(time=0).values
        den03 = ds["den03"].isel(time=0).values
        den04 = ds["den04"].isel(time=0).values

        # Transpose Nz, Ny, Nx → Nx, Ny, Nz
        den01 = np.transpose(den01, (2, 1, 0))
        den02 = np.transpose(den02, (2, 1, 0))
        den03 = np.transpose(den03, (2, 1, 0))
        den04 = np.transpose(den04, (2, 1, 0))

        # sum all densities to get total density
        tot_den = (den01 + den02 + den03 + den04)

        ds.close()
        return x, y, z, tot_den

    x, y, z, tot_den = load_density(ncfile)

    # ------------------------
    # CREATE 3D GRID
    # ------------------------
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)

    # ------------------------
    # SELECT SHELL
    # ------------------------
    shell_mask = (R >= rmin) & (R <= rmax)
    X = X[shell_mask]
    Y = Y[shell_mask]
    Z = Z[shell_mask]
    tot_den = tot_den[shell_mask]

    # ------------------------
    # SPHERICAL COORDINATES
    # ------------------------
    R_shell = R[shell_mask]
    lat = np.degrees(np.arcsin(Z / R_shell))
    lon = np.degrees(np.arctan2(Y, X))

    # ------------------------
    # TRIANGULATE IRREGULAR POINTS
    # ------------------------
    points = np.vstack([lon, lat]).T

    # Regular grid
    lat_grid = np.linspace(-90, 90, 180)
    lon_grid = np.linspace(-180, 180, 360)
    Lon_grid, Lat_grid = np.meshgrid(lon_grid, lat_grid)

    # Interpolate scattered density onto the grid
    tot_den_grid = griddata(
        points=(lon, lat),
        values=tot_den,
        xi=(Lon_grid, Lat_grid),
        method='nearest'
    )

    # Compute log10 density for visualization (handle zeros/negative)
    log_den_grid = np.log10(np.maximum(tot_den_grid, 1e-10))

    # Streamplot works on regular grid
    fig, ax = plt.subplots(figsize=(12,6), subplot_kw={"projection": "hammer"})
    im = ax.pcolormesh(np.deg2rad(lon_grid), np.deg2rad(lat_grid), tot_den_grid, cmap='plasma', shading='auto', vmin=0, vmax=500)

    # plt.colorbar(im, ax=ax, label=r'$\log_{10}$(density) [cm$^{-3}$]')
    plt.colorbar(im, ax=ax, label=r'N [cm$^{-3}$]')

    # Longitude ticks (-170 to 170 every n °)
    lon_ticks_deg = np.arange(-120, 121, 60)
    lon_ticks_rad = np.deg2rad(lon_ticks_deg)

    # Latitude ticks (-90 to 90 every n °)
    lat_ticks_deg = np.arange(-60, 61, 30)
    lat_ticks_rad = np.deg2rad(lat_ticks_deg)

    # Apply to the current axis
    ax.set_xticks(lon_ticks_rad)
    ax.set_yticks(lat_ticks_rad)

    # Label ticks in degrees
    ax.set_xticklabels([f"{int(l)}°" for l in lon_ticks_deg])
    ax.set_yticklabels([f"{int(l)}°" for l in lat_ticks_deg])
    ax.set_title(f'{case.replace("_", " ")} density map, shell {rmin/RM:.2f}-{rmax/RM:.2f} RM at t = {step*0.002} s')
    ax.grid(True)

    outfile = os.path.join(output_folder, f"{case}_density_shell_{rmin / RM:.2f}-{rmax / RM:.2f}_RM_step_{step}.png")
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

print(f"Finished with {case} density plots")
