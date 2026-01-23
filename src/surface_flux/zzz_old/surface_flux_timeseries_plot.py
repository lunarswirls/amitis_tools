#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import os
import numpy as np
import xarray as xr
from src.surface_flux.flux_utils import compute_radial_flux
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# -------------------------------
# Configuration
# -------------------------------
# base cases: CPN_Base RPN_Base CPS_Base RPS_Base
# HNHV cases: CPN_HNHV RPN_HNHV CPS_HNHV RPS_HNHV
case = "CPN_Base"

if "Base" in case:
    input_folder1 = f"/Volumes/data_backup/mercury/extreme/{case}/plane_product/object/"
elif "HNHV" in case:
    input_folder1 = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/object/"

# input_folder1 = f"/Users/danywaller/Projects/mercury/extreme/{case}_Base/object/"
# output_folder = f"/Users/danywaller/Projects/mercury/extreme/surface_flux/timeseries_{case.lower()}"

output_folder = f"/Users/danywaller/Projects/mercury/extreme/surface_flux/timeseries_{case.lower()}"
os.makedirs(output_folder, exist_ok=True)

footprints = None  # valid arguments: 'compute', 'add', or None

R_M = 2440.0        # Mercury radius [km]

LAT_BINS = 360      # [degrees]
LON_BINS = 360      # [degrees]

# -------------------------------
# Load grid (once)
# -------------------------------
ds0 = xr.open_dataset(
    # os.path.join(input_folder1, f"Amitis_{case}_Base_000000_xz_comp.nc")
    # Amitis_{case}_Base_000000_xz_comp.nc

    os.path.join(input_folder1, f"Amitis_{case}_115000_xz_comp.nc")
)

x = ds0["Nx"].values  # [units: km]
y = ds0["Ny"].values  # [units: km]
z = ds0["Nz"].values  # [units: km]

ds0.close()

if "Base" in case:
    # take last 10-ish seconds
    sim_steps = range(98000, 115000 + 1, 1000)
elif "HNHV" in case:
    sim_steps = range(115000, 350000 + 1, 1000)

t_index = 0

for step in sim_steps:
    nc_file = os.path.join(input_folder1, f"Amitis_{case}_{step:06d}_xz_comp.nc")
    ds = xr.open_dataset(nc_file)

    flux, vr = compute_radial_flux(ds, x, y, z)

    ds.close()

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
    lon = np.linspace(-180, 180, LON_BINS)


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
    LAT_FINE = 180*3
    LON_FINE = 360*3

    lat_fine = np.linspace(-90, 90, LAT_FINE)
    lon_fine = np.linspace(-180, 180, LON_FINE)

    interp2 = RegularGridInterpolator(
        (lat, lon),
        flux_surface,
        bounds_error=False,
        fill_value=np.nan
    )

    lon_grid_fine, lat_grid_fine = np.meshgrid(lon_fine, lat_fine)
    points_fine = np.column_stack((lat_grid_fine.ravel(), lon_grid_fine.ravel()))
    flux_fine = interp2(points_fine).reshape(LAT_FINE, LON_FINE)
    flux_fine = flux_fine[::-1, :]

    # Mask non-positive values
    flux_surface_masked = np.where(flux_fine > 0, flux_fine, np.nan)

    # Log10
    log_flux_surface = np.log10(flux_surface_masked)

    # -------------------------------
    # Flatten
    # -------------------------------
    x_flat = lon_grid_fine.ravel()
    y_flat = lat_grid_fine.ravel()
    z_flat = log_flux_surface.ravel()

    # c_min = np.nanpercentile(log_flux_surface, 5)
    # c_max = np.nanpercentile(log_flux_surface, 95)
    c_min = 4
    c_max = 10

    # -------------------------------
    # 2D plot of everything
    # -------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # --- Scatter plot ---
    sc = ax.scatter(
        x_flat,
        y_flat,
        c=z_flat,
        s=2,
        cmap="viridis",
        vmin=c_min,
        vmax=c_max
    )

    # --- Colorbar ---
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log$_{10}$(F) [cm$^{-2}$ s$^{-1}$]")

    # --- Vertical dashed lines ---
    ax.axvline(-90, color="white", linewidth=2, linestyle="--")
    ax.axvline(90, color="white", linewidth=2, linestyle="--")

    # --- Axes settings ---
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title(f"{case} Surface Flux, t = {step * 0.002} seconds")

    # --- Save ---
    outfile_png = os.path.join(output_folder, f"{case}_surface_flux_{step}.png")
    plt.tight_layout()
    plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
    print("Saved:\t", outfile_png)
    plt.close()
