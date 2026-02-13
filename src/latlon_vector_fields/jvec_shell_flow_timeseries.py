#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports:
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.tri as mtri
from scipy.spatial import Delaunay

# SETTINGS
case = "CPS_HNHV"

filter_jmag = True
jmag_min = 2  # nA/m^2

sim_end = True

# Planet parameters
RM = 2440.0  # km

# Shell limits
rmin = 1.0 * RM
rmax = 1.2 * RM

if "Base" in case:
    input_folder = f"/Volumes/data_backup/mercury/extreme/{case}/plane_product/object/"
    output_folder = f"/Users/danywaller/Projects/mercury/extreme/jfield_lonlat/{case}/"
    sim_steps = list(range(105000, 115000 + 1, 1000))
elif "HNHV" in case and not sim_end:
    input_folder = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/object/"
    output_folder = f"/Users/danywaller/Projects/mercury/extreme/jfield_lonlat/{case}/"
    sim_steps = range(115000, 200000 + 1, 1000)
elif "HNHV" in case and sim_end:
    input_folder = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/object/"
    output_folder = f"/Users/danywaller/Projects/mercury/extreme/jfield_lonlat/{case}_end/"
    sim_steps = range(115000, 350000 + 1, 1000)
else:
    raise ValueError("Case not recognized")

os.makedirs(output_folder, exist_ok=True)

for step in sim_steps:

    ncfile = os.path.join(input_folder, f"Amitis_{case}_{step}_xz_comp.nc")

    # ------------------------
    # LOAD J VECTOR
    # ------------------------
    def load_jvector(nc_file):
        ds = xr.open_dataset(nc_file)

        x = ds["Nx"].values
        y = ds["Ny"].values
        z = ds["Nz"].values

        Jx = ds["Jx"].isel(time=0).values
        Jy = ds["Jy"].isel(time=0).values
        Jz = ds["Jz"].isel(time=0).values

        # Transpose Nz, Ny, Nx → Nx, Ny, Nz
        Jx = np.transpose(Jx, (2, 1, 0))
        Jy = np.transpose(Jy, (2, 1, 0))
        Jz = np.transpose(Jz, (2, 1, 0))

        ds.close()
        return x, y, z, Jx, Jy, Jz

    x, y, z, Jx, Jy, Jz = load_jvector(ncfile)

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
    Jx = Jx[shell_mask]
    Jy = Jy[shell_mask]
    Jz = Jz[shell_mask]

    # ------------------------
    # SPHERICAL COORDINATES
    # ------------------------
    lat = np.degrees(np.arcsin(Z / R[shell_mask]))
    lon = np.degrees(np.arctan2(Y, X))

    # ------------------------
    # TANGENTIAL VECTORS ON SHELL
    # ------------------------
    # Unit vectors
    er = np.vstack((X,Y,Z)).T
    er /= np.linalg.norm(er, axis=1)[:,None]

    etheta = np.vstack((-Z*X, -Z*Y, X**2 + Y**2)).T
    etheta /= np.linalg.norm(etheta, axis=1)[:,None]

    ephi = np.vstack((-Y, X, np.zeros_like(X))).T
    ephi /= np.linalg.norm(ephi, axis=1)[:,None]

    J = np.vstack((Jx,Jy,Jz)).T
    J_theta = np.sum(J * etheta, axis=1)
    J_phi   = np.sum(J * ephi, axis=1)
    Jmag    = np.sqrt(J_theta**2 + J_phi**2)

    if filter_jmag:
        # ------------------------
        # FILTER CURRENTS
        # ------------------------
        mask_strong = Jmag > jmag_min
        lat = lat[mask_strong]
        lon = lon[mask_strong]
        J_theta = J_theta[mask_strong]
        J_phi   = J_phi[mask_strong]

    # ------------------------
    # TRIANGULATE IRREGULAR POINTS
    # ------------------------
    points = np.vstack([lon, lat]).T
    tri = Delaunay(points)
    triang = mtri.Triangulation(lon, lat, triangles=tri.simplices)

    # Regular grid
    lat_grid = np.linspace(-90, 90, 180)
    lon_grid = np.linspace(-180, 180, 360)
    Lon_grid, Lat_grid = np.meshgrid(lon_grid, lat_grid)

    # Interpolate scattered vectors onto the grid
    J_theta_grid = griddata(
        points=(lon, lat),
        values=J_theta,
        xi=(Lon_grid, Lat_grid),
        method='nearest')

    J_phi_grid = griddata(
        points=(lon, lat),
        values=J_phi,
        xi=(Lon_grid, Lat_grid),
        method='nearest')

    # Streamplot works on regular grid
    fig, ax = plt.subplots(figsize=(12,6))
    magnitude = np.sqrt(J_theta_grid**2 + J_phi_grid**2)
    strm = ax.streamplot(lon_grid, lat_grid, J_phi_grid, J_theta_grid, color=magnitude, cmap='plasma', density=2.2, linewidth=1, norm=plt.Normalize(vmin=0, vmax=500))

    plt.colorbar(strm.lines, ax=ax, label='|J|')

    ax.set_xlim(-180,180)
    ax.set_ylim(-90,90)
    ax.set_xlabel('Longitude [°]')
    ax.set_ylabel('Latitude [°]')
    ax.set_title(f'{case.replace("_", " ")} current streamlines, shell {rmin/RM:.2f}-{rmax/RM:.2f} RM at t = {step*0.002} s')
    ax.grid(True)

    outfile = os.path.join(output_folder, f"{case}_current_streamlines_{step}_shell_{rmin / RM:.2f}-{rmax / RM:.2f}_RM.png")
    plt.savefig(outfile, dpi=150)
    plt.close()

print(f"Finished with {case} plots")