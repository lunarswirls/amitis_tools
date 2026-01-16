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
row = 1
col = 1

# overrides following subplot setup and makes 1 simple 2d plot for debugging
quickplot = True

# choose case
cases = ["RPS", "CPS", "RPN", "CPN"]

output_folder = f"/Users/danywaller/Projects/mercury/extreme_base/surface_flux/"
os.makedirs(output_folder, exist_ok=True)

R_M = 2440.0        # Mercury radius [km]

LAT_BINS = 180      # [degrees]
LON_BINS = 360      # [degrees]

# -------------------------------
# 2D plot of everything
# -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 6))

for case in cases:

    input_folder1  = f"/Users/danywaller/Projects/mercury/extreme_base/{case}_Base/object/"

    # take last 10-ish seconds
    sim_steps = range(98000, 115000 + 1, 1000)

    t_index = 0

    # -------------------------------
    # Load grid (once)
    # -------------------------------
    ds0 = xr.open_dataset(
        os.path.join(input_folder1, f"Amitis_{case}_Base_{sim_steps.start:06d}_xz_comp.nc")  # Amitis_{case}_Base_000000_xz_comp.nc
    )

    x = ds0["Nx"].values  # [units: km]
    y = ds0["Ny"].values  # [units: km]
    z = ds0["Nz"].values  # [units: km]

    # -------------------------------
    # Time-averaged total radial flux (full volume)
    # -------------------------------
    flux_sum = None
    count = 0

    for step in sim_steps:
        ds1 = xr.open_dataset(
            os.path.join(input_folder1, f"Amitis_{case}_Base_{step:06d}_xz_comp.nc")
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

        # Convert velocities from km/s to cm/s
        vx_cms = vx * 1e5
        vy_cms = vy * 1e5
        vz_cms = vz * 1e5

        # Radial unit vector at each grid point (same shape as den)
        # Assuming grid points x,y,z already loaded from ds0
        Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
        r_mag = np.sqrt(Xg**2 + Yg**2 + Zg**2)
        nx = Xg / r_mag
        ny = Yg / r_mag
        nz = Zg / r_mag

        # Radial flux in cm^-2 s^-1
        flux = den * (vx_cms * nx + vy_cms * ny + vz_cms * nz)

        if flux_sum is None:
            flux_sum = np.zeros_like(flux, dtype=np.float64)

        flux_sum += flux
        count += 1

    flux_avg = flux_sum / count

    # -------------------------------
    # Interpolator (Cartesian space)
    # -------------------------------
    interp = RegularGridInterpolator(
        (z, y, x),
        flux_avg,
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

    quick_cmax = 100e6
    quick_cmin = -150e6

    # --- Select subplot ---
    if case == "RPS":
        row, col = 0, 0
    elif case == "CPS":
        row, col = 1, 0
    elif case == "RPN":
        row, col = 0, 1
    elif case == "CPN":
        row, col = 1, 1

    # Create figure and axes if not already created
    ax = axs[row, col]

    # --- Scatter plot ---
    sc = ax.scatter(
        x_flat_shifted,
        y_flat,
        c=z_flat,
        s=2,
        cmap="viridis",
        vmin=quick_cmin,
        vmax=quick_cmax
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
    ax.set_title(f"{case}")

    print(f"Plotted: {case}")

print("Saving...")
# --- Save ---
outfile_png = os.path.join(output_folder, f"all_cases_surface_flux_2D.png")
plt.tight_layout()
plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
print("Saved:\t", outfile_png)
