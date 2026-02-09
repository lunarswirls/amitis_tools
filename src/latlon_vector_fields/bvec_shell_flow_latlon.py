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
case = "RPN"
step = 115000
input_folder = f"/Volumes/data_backup/mercury/extreme/{case}_Base/plane_product/object/"
ncfile = os.path.join(input_folder, f"Amitis_{case}_Base_{step}_xz_comp.nc")

output_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_lonlat/{case}_Base/"
os.makedirs(output_folder, exist_ok=True)

filter_bmag = False

# Planet parameters
RM = 2440.0  # km

# Shell limits
rmin = 1.4 * RM
rmax = 1.6 * RM

# ------------------------
# LOAD B VECTOR
# ------------------------
def load_bvector(ncfile):
    ds = xr.open_dataset(ncfile)

    x = ds["Nx"].values
    y = ds["Ny"].values
    z = ds["Nz"].values

    Bx = ds["Bx_tot"].isel(time=0).values
    By = ds["By_tot"].isel(time=0).values
    Bz = ds["Bz_tot"].isel(time=0).values

    # Transpose Nz, Ny, Nx → Nx, Ny, Nz
    Bx = np.transpose(Bx, (2, 1, 0))
    By = np.transpose(By, (2, 1, 0))
    Bz = np.transpose(Bz, (2, 1, 0))

    ds.close()
    return x, y, z, Bx, By, Bz


x, y, z, Bx, By, Bz = load_bvector(ncfile)

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
Bx = Bx[shell_mask]
By = By[shell_mask]
Bz = Bz[shell_mask]

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

B = np.vstack((Bx,By,Bz)).T
B_theta = np.sum(B * etheta, axis=1)
B_phi   = np.sum(B * ephi, axis=1)
Bmag    = np.sqrt(B_theta**2 + B_phi**2)

if filter_bmag:
    # ------------------------
    # FILTER STRONG FIELDS
    # ------------------------
    mask_strong = Bmag > 50
    lat = lat[mask_strong]
    lon = lon[mask_strong]
    B_theta = B_theta[mask_strong]
    B_phi   = B_phi[mask_strong]

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
B_theta_grid = griddata(
    points=(lon, lat),
    values=B_theta,
    xi=(Lon_grid, Lat_grid),
    method='nearest')

B_phi_grid = griddata(
    points=(lon, lat),
    values=B_phi,
    xi=(Lon_grid, Lat_grid),
    method='nearest')

# Streamplot works on regular grid
fig, ax = plt.subplots(figsize=(12,6))
magnitude = np.sqrt(B_theta_grid**2 + B_phi_grid**2)
strm = ax.streamplot(lon_grid, lat_grid, B_phi_grid, B_theta_grid, color=magnitude, cmap='plasma', density=2.2, linewidth=1, norm=plt.Normalize(vmin=0, vmax=300))

plt.colorbar(strm.lines, ax=ax, label='$|B|$')

ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.set_xlabel('Longitude [°]')
ax.set_ylabel('Latitude [°]')
ax.set_title(f'{case} B streamlines, shell {rmin/RM:.2f}-{rmax/RM:.2f} RM at t = {step*0.002} s')
ax.grid(True)

outfile = os.path.join(output_folder, f"{case}_bfield_streamlines_{step}_shell_{rmin / RM:.2f}-{rmax / RM:.2f}_RM.png")
plt.savefig(outfile, dpi=150)
plt.close()
print("Saved:", outfile)