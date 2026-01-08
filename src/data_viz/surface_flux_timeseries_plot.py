#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import os
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# -------------------------------
# Configuration
# -------------------------------

case = "CPS"        # choose case

# input_folder1 = f"/Users/danywaller/Projects/mercury/extreme/{case}_Base/object/"
# output_folder = f"/Users/danywaller/Projects/mercury/extreme/surface_flux/timeseries_{case.lower()}"
input_folder1 = f"/Users/danywaller/Projects/mercury/extreme/High_HNHV/{case}_HNHV/object/"
output_folder = f"/Users/danywaller/Projects/mercury/extreme/High_HNHV_surface_flux/timeseries_{case.lower()}"
os.makedirs(output_folder, exist_ok=True)

R_M = 2440.0        # Mercury radius [km]

LAT_BINS = 180      # [degrees]
LON_BINS = 360      # [degrees]

# -------------------------------
# Load grid (once)
# -------------------------------
ds0 = xr.open_dataset(
    # os.path.join(input_folder1, f"Amitis_{case}_Base_000000_xz_comp.nc")
    # Amitis_{case}_Base_000000_xz_comp.nc

    os.path.join(input_folder1, f"Amitis_{case}_HNHV_115000_xz_comp.nc")
)

x = ds0["Nx"].values  # [units: km]
y = ds0["Ny"].values  # [units: km]
z = ds0["Nz"].values  # [units: km]

# take last 10-ish seconds
# sim_steps = range(98000, 115000 + 1, 1000)
sim_steps = range(115000, 350000 + 1, 1000)

t_index = 0

for step in sim_steps:

    ds1 = xr.open_dataset(
        # os.path.join(input_folder1, f"Amitis_{case}_Base_{step:06d}_xz_comp.nc")
        os.path.join(input_folder1, f"Amitis_{case}_HNHV_{step:06d}_xz_comp.nc")
    )

    # Total density   # [units: cm^-3]
    den = (ds1["den01"].isel(time=t_index).values + ds1["den02"].isel(time=t_index).values  # protons
           + ds1["den03"].isel(time=t_index).values + ds1["den04"].isel(time=t_index).values)  # alphas

    # Velocities   # [units: km/s]
    vx = (ds1["vx01"].isel(time=t_index).values + ds1["vx02"].isel(time=t_index).values  # protons
          + ds1["vx03"].isel(time=t_index).values + ds1["vx04"].isel(time=t_index).values)  # alphas
    vy = (ds1["vy01"].isel(time=t_index).values + ds1["vy02"].isel(time=t_index).values
          + ds1["vy03"].isel(time=t_index).values + ds1["vy04"].isel(time=t_index).values)
    vz = (ds1["vz01"].isel(time=t_index).values + ds1["vz02"].isel(time=t_index).values
          + ds1["vz03"].isel(time=t_index).values + ds1["vz04"].isel(time=t_index).values)

    # Convert density from cm^-3 to km^-3
    den_km = den * 1e15

    # Radial unit vector at each grid point (same shape as den)
    # Assuming grid points x,y,z already loaded from ds0
    Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
    r_mag = np.sqrt(Xg**2 + Yg**2 + Zg**2)
    nx = Xg / r_mag
    ny = Yg / r_mag
    nz = Zg / r_mag

    # Radial flux in km^-2 s^-1
    flux_km = den_km * (vx * nx + vy * ny + vz * nz)

    # convert to cm^-2 s^-1
    flux = flux_km * 1e-10

    # -------------------------------
    # Interpolator (Cartesian space)
    # -------------------------------
    interp = RegularGridInterpolator(
        (z, y, x),
        flux,
        bounds_error=False,
        fill_value=np.nan
    )

    # -------------------------------
    # Surface grid
    # -------------------------------
    lat = np.linspace(-90, 90, LAT_BINS)
    lon = np.linspace(0, 360, LON_BINS)


    def surface_points_from_angles(lat_deg, lon_deg, RM):
        lat_r = np.deg2rad(lat_deg)
        lon_r = np.deg2rad(lon_deg)

        X_s = RM * np.cos(lat_r[:, None]) * np.cos(lon_r[None, :])
        Y_s = RM * np.cos(lat_r[:, None]) * np.sin(lon_r[None, :])
        Z_s = RM * np.sin(lat_r[:, None]) * np.ones_like(lon_r[None, :])

        return X_s, Y_s, Z_s


    lat_flipped = lat[::-1]
    Xs, Ys, Zs = surface_points_from_angles(lat_flipped, lon, R_M)

    # -------------------------------
    # Interpolate onto surface
    # -------------------------------
    points = np.stack((Zs, Ys, Xs), axis=-1).reshape(-1, 3)
    flux_surface = interp(points).reshape(LAT_BINS, LON_BINS)
    flux_surface = flux_surface[::-1, :]

    # -------------------------------
    # Normalize geometry for Plotly
    # -------------------------------
    Xn = Xs / R_M
    Yn = Ys / R_M
    Zn = Zs / R_M

    # -------------------------------
    # Calculate min and max for clims
    # -------------------------------
    c_min = np.nanpercentile(flux_surface, 5)
    c_max = np.nanpercentile(flux_surface, 95)

    # -------------------------------
    # Fine grid interpolation
    # -------------------------------
    LAT_FINE = 360*3
    LON_FINE = 720*3

    lat_fine = np.linspace(-90, 90, LAT_FINE)
    lon_fine = np.linspace(0, 360, LON_FINE)

    interp = RegularGridInterpolator(
        (lat, lon),
        flux_surface,
        bounds_error=False,
        fill_value=np.nan
    )

    lon_grid_fine, lat_grid_fine = np.meshgrid(lon_fine, lat_fine)
    points_fine = np.column_stack((lat_grid_fine.ravel(), lon_grid_fine.ravel()))
    flux_fine = interp(points_fine).reshape(LAT_FINE, LON_FINE)
    flux_fine = flux_fine[::-1, :]

    # -------------------------------
    # Flatten
    # -------------------------------
    x_flat = lon_grid_fine.ravel()
    y_flat = lat_grid_fine.ravel()
    z_flat = flux_fine.ravel()

    # Apply -180° shift
    lon_grid_fine_shifted = (lon_grid_fine - 180) % 360
    x_flat_shifted = lon_grid_fine_shifted.ravel()

    quick_cmax = 10e7
    quick_cmin = -10e7

    # -------------------------------
    # 2D plot of everything
    # -------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # --- Scatter plot ---
    sc = ax.scatter(
        x_flat_shifted,
        y_flat,
        c=z_flat,
        s=2,
        cmap="viridis",
        vmin=quick_cmax,
        vmax=quick_cmin
    )

    # --- Colorbar ---
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Radial flux [cm$^{-2}$ s$^{-1}$]")

    # --- Vertical dashed lines ---
    ax.axvline(90, color="white", linewidth=2, linestyle="--")
    ax.axvline(270, color="white", linewidth=2, linestyle="--")

    # --- Axes settings ---
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title(f"{case} Surface Flux, t = {step * 0.002} seconds")

    # --- Save ---
    outfile_png = os.path.join(output_folder, f"{case}_HNHV_surface_flux_{step}.png")
    plt.tight_layout()
    plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
    print("Saved:\t", outfile_png)
    plt.close()
