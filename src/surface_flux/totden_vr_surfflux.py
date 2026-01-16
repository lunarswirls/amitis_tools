#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
from flux_utils import compute_radial_flux
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# -------------------------------
# Configuration
# -------------------------------
case = "RPS"
input_folder1  = f"/Users/danywaller/Projects/mercury/extreme/{case}_Base/object/"
output_folder = f"/Users/danywaller/Projects/mercury/extreme/surface_flux/"
os.makedirs(output_folder, exist_ok=True)

debug = False

R_M = 2440.0        # Mercury radius [km]
LAT_BINS = 180      # Surface latitude bins
LON_BINS = 360      # Surface longitude bins

# -------------------------------
# Load grid (assume first file is representative)
# -------------------------------
first_file = sorted([f for f in os.listdir(input_folder1) if f.endswith("_xz_comp.nc")])[0]
ds0 = xr.open_dataset(os.path.join(input_folder1, first_file))

x = ds0["Nx"].values
y = ds0["Ny"].values
z = ds0["Nz"].values

nc_file = os.path.join(input_folder1, f"Amitis_{case}_Base_115000_xz_comp.nc")
ds = xr.open_dataset(nc_file)

if debug:
    print(ds["den01"].dims)
    print(ds["den01"].shape)
    print(len(x), len(y), len(z))

# Total density (protons + alphas) [units: cm^-3]
den = (ds["den01"].isel(time=0).values + ds["den02"].isel(time=0).values + ds["den03"].isel(time=0).values + ds["den04"].isel(time=0).values)

# Total velocity [units: km/s]
vx = (ds["vx01"].isel(time=0).values + ds["vx02"].isel(time=0).values + ds["vx03"].isel(time=0).values + ds["vx04"].isel(time=0).values)
vy = (ds["vy01"].isel(time=0).values + ds["vy02"].isel(time=0).values + ds["vy03"].isel(time=0).values + ds["vy04"].isel(time=0).values)
vz = (ds["vz01"].isel(time=0).values + ds["vz02"].isel(time=0).values + ds["vz03"].isel(time=0).values + ds["vz04"].isel(time=0).values)

# Convert velocities from km/s to cm/s
vx_cms, vy_cms, vz_cms = vx * 1e5, vy * 1e5, vz * 1e5

# Radial unit vector at each grid point
Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
r_mag = np.sqrt(Xg ** 2 + Yg ** 2 + Zg ** 2)

# this is defined such that negative flux is 'precipitation'
nx, ny, nz = Xg / r_mag, Yg / r_mag, Zg / r_mag

# Radial flux: n * (v dot r_hat)
unweighted_flux = den * (vx_cms * nx + vy_cms * ny + vz_cms * nz)

weighted_flux, vr = compute_radial_flux(ds, x, y, z)

# -------------------------------
# Interpolate radial flux onto Mercury surface
# -------------------------------
lat = np.linspace(-90, 90, LAT_BINS)
lon = np.linspace(-180, 180, LON_BINS)

lat_r = np.deg2rad(lat)
lon_r = np.deg2rad(lon)
Xs = R_M * np.cos(lat_r[:, None]) * np.cos(lon_r[None, :])
Ys = R_M * np.cos(lat_r[:, None]) * np.sin(lon_r[None, :])
Zs = R_M * np.sin(lat_r[:, None]) * np.ones_like(lon_r[None, :])

points_surface = np.stack((Zs, Ys, Xs), axis=-1).reshape(-1, 3)

# total density
interp0 = RegularGridInterpolator((z, y, x), den, bounds_error=False, fill_value=np.nan)
tot_den = interp0(points_surface).reshape(LAT_BINS, LON_BINS)
tot_den = tot_den[::-1, :]  # flip latitude for plotting

# radial velocity
interp1 = RegularGridInterpolator((z, y, x), vr, bounds_error=False, fill_value=np.nan)
v_r = interp1(points_surface).reshape(LAT_BINS, LON_BINS)
v_r = v_r[::-1, :]  # flip latitude for plotting

# unweighted surface flux
interp2 = RegularGridInterpolator((z, y, x), unweighted_flux, bounds_error=False, fill_value=np.nan)
uw_flux_surface = interp2(points_surface).reshape(LAT_BINS, LON_BINS)
uw_flux_surface = uw_flux_surface[::-1, :]  # flip latitude for plotting
# Mask non-positive values
uw_flux_surface_masked = np.where(uw_flux_surface > 0, uw_flux_surface, np.nan)
# Log10
uw_log_flux_surface = np.log10(uw_flux_surface_masked)

# weighted surface flux
interp3 = RegularGridInterpolator((z, y, x), weighted_flux, bounds_error=False, fill_value=np.nan)
flux_surface = interp3(points_surface).reshape(LAT_BINS, LON_BINS)
flux_surface = flux_surface[::-1, :]  # flip latitude for plotting
# Mask non-positive values
flux_surface_masked = np.where(flux_surface > 0, flux_surface, np.nan)
# Log10
log_flux_surface = np.log10(flux_surface_masked)

# -------------------------------
# Plot
# -------------------------------
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={"projection": "hammer"})

# Plot flux
lon_grid, lat_grid = np.meshgrid(lon_r, lat_r)  # radians
# shift lon to [-pi, pi]
lon_grid = np.where(lon_grid > np.pi, lon_grid - 2*np.pi, lon_grid)

# total density
# ax0 = axes[0]
sc = ax0.pcolormesh(lon_grid, lat_grid, tot_den, cmap="viridis", shading="auto")
cbar = fig.colorbar(sc, ax=ax0, orientation="horizontal", pad=0.05, shrink=0.5)
cbar.set_label(r"N [cm$^{-3}$])")
ax0.set_title(f"Total Density")

# v_r
# ax1 = axes[1]
sc = ax1.pcolormesh(lon_grid, lat_grid, v_r, cmap="viridis", shading="auto")
cbar = fig.colorbar(sc, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.5)
cbar.set_label(r"$V_r$ [km/s])")
ax1.set_title(f"Radial Velocity")

# UNWEIGHTED surface flux
# ax2 = axes[2]
sc = ax2.pcolormesh(lon_grid, lat_grid, uw_log_flux_surface, cmap="viridis", shading="auto")
cbar = fig.colorbar(sc, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.5)
cbar.set_label(r"$\log_{10}$(F [cm$^{-2}$ s$^{-1}$])")
ax2.set_title(f"Unweighted Surface Flux")

# WEIGHTED surface flux
# ax3 = axes[3]
sc = ax3.pcolormesh(lon_grid, lat_grid, log_flux_surface, cmap="viridis", shading="auto")
cbar = fig.colorbar(sc, ax=ax3, orientation="horizontal", pad=0.05, shrink=0.5)
cbar.set_label(r"$\log_{10}$(F [cm$^{-2}$ s$^{-1}$])")
ax3.set_title(f"Weighted Surface Flux")

# Longitude ticks (-170 to 170 every n °)
lon_ticks_deg = np.arange(-120, 121, 60)
lon_ticks_rad = np.deg2rad(lon_ticks_deg)

# Latitude ticks (-90 to 90 every n °)
lat_ticks_deg = np.arange(-60, 61, 30)
lat_ticks_rad = np.deg2rad(lat_ticks_deg)

for ax in [ax0, ax1, ax2, ax3]:
    # Apply to the current axis
    ax.set_xticks(lon_ticks_rad)
    ax.set_yticks(lat_ticks_rad)

    # Label ticks in degrees
    ax.set_xticklabels([f"{int(l)}°" for l in lon_ticks_deg])
    ax.set_yticklabels([f"{int(l)}°" for l in lat_ticks_deg])

    ax.grid(True, alpha=0.3, color="grey")

plt.suptitle(f"{case} t = {115000 * 0.002} s")
# Save figure
plt.tight_layout()
outfile_png = os.path.join(output_folder, f"radial_flux_calc_testing_{case}_115000.png")
plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
print("Saved figure:", outfile_png)