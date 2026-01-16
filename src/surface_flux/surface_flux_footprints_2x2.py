#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from src.surface_flux.flux_utils import compute_radial_flux
from src.field_topology.topology_utils import compute_ocb_transition

# -------------------------------
# Configuration
# -------------------------------
cases = ["RPS", "CPS", "RPN", "CPN"]
output_folder = f"/Users/danywaller/Projects/mercury/extreme/surface_flux/"
os.makedirs(output_folder, exist_ok=True)

debug = False

R_M = 2440.0        # Mercury radius [km]
LAT_BINS = 180      # Surface latitude bins
LON_BINS = 360      # Surface longitude bins


def lon_diff(a, b):
    """
    Minimal angular difference in degrees.
    """
    return np.abs(((a - b + 180) % 360) - 180)

# -------------------------------
# Prepare figure
# -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 8), subplot_kw={"projection": "hammer"})

for case in cases:

    input_folder1  = f"/Users/danywaller/Projects/mercury/extreme/{case}_Base/object/"

    input_folder2 = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology/{case}_Base/"
    # csv_file = os.path.join(input_folder2, f"{case}_last_10_footprints_median_class.csv")  # CSV with footprints
    csv_file = os.path.join(input_folder2, f"{case}_115000_footprints_class.csv")  # CSV with footprints

    # -------------------------------
    # Load footprint CSV
    # -------------------------------
    if os.path.exists(csv_file):
        df_footprints = pd.read_csv(csv_file)
        # print(f"Loaded {len(df_footprints)} footprints for {case}")
    else:
        print(f"No footprint CSV found for {case}, skipping footprints")
        df_footprints = pd.DataFrame(columns=["latitude_deg", "longitude_deg", "classification"])

    # -------------------------------
    # Load grid (assume first file is representative)
    # -------------------------------
    first_file = sorted([f for f in os.listdir(input_folder1) if f.endswith("_xz_comp.nc")])[0]
    ds0 = xr.open_dataset(os.path.join(input_folder1, first_file))

    x = ds0["Nx"].values
    y = ds0["Ny"].values
    z = ds0["Nz"].values

    # -------------------------------
    # Time-average total radial flux
    # -------------------------------
    flux_sum = None
    count = 0

    # Consider last N steps (adjust as needed)
    sim_steps = range(115000, 115000 + 1, 1000)

    for step in sim_steps:
        nc_file = os.path.join(input_folder1, f"Amitis_{case}_Base_{step:06d}_xz_comp.nc")
        ds = xr.open_dataset(nc_file)

        flux, vr = compute_radial_flux(ds, x, y, z)

        if flux_sum is None:
            flux_sum = np.zeros_like(flux, dtype=np.float64)
        flux_sum += flux
        count += 1

    flux_avg = flux_sum / count
    print(f"Computed time-averaged radial flux for {case}")

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
    interp = RegularGridInterpolator((z, y, x), flux_avg, bounds_error=False, fill_value=np.nan)
    flux_surface = interp(points_surface).reshape(LAT_BINS, LON_BINS)
    flux_surface = flux_surface[::-1, :]  # flip latitude for plotting

    # Mask non-positive values
    flux_surface_masked = np.where(flux_surface > 0, flux_surface, np.nan)

    # Log10
    log_flux_surface = np.log10(flux_surface_masked)

    # -------------------------------
    # Plot
    # -------------------------------
    if case == "RPN": row, col = 0, 0
    elif case == "CPN": row, col = 1, 0
    elif case == "RPS": row, col = 0, 1
    elif case == "CPS": row, col = 1, 1

    ax = axs[row, col]

    quick_cmin = 6
    quick_cmax = 8

    # Plot flux
    lon_grid, lat_grid = np.meshgrid(lon_r, lat_r)  # radians
    # shift lon to [-pi, pi]
    lon_grid = np.where(lon_grid > np.pi, lon_grid - 2*np.pi, lon_grid)

    # Surface flux
    sc = ax.pcolormesh(lon_grid, lat_grid, log_flux_surface, cmap="viridis", shading="auto", vmin=quick_cmin, vmax=quick_cmax)
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.5)
    cbar.set_label(r"$\log_{10}$(F [cm$^{-2}$ s$^{-1}$])")

    # Open–Closed Boundary (OCB)
    lon_bins = np.linspace(-180, 180, 180)
    lon_n, lat_n = compute_ocb_transition(df_footprints, lon_bins, "north")
    lon_s, lat_s = compute_ocb_transition(df_footprints, lon_bins, "south")

    # Convert to radians for Mollweide
    lon_n_rad = np.deg2rad(lon_n)
    lat_n_rad = np.deg2rad(lat_n)
    lon_s_rad = np.deg2rad(lon_s)
    lat_s_rad = np.deg2rad(lat_s)

    # Mollweide longitude in matplotlib goes from -pi to pi (radians)
    # Latitude stays as is
    ax.plot(lon_n_rad, lat_n_rad, color="magenta", lw=2, label="OCB North")
    ax.plot(lon_s_rad, lat_s_rad, color="magenta", lw=2, ls="--", label="OCB South")

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

    ax.set_title(case)
    ax.grid(True, alpha=0.3, color="grey")

# Save figure
plt.tight_layout()
outfile_png = os.path.join(output_folder, "all_cases_surface_flux_with_footprints_115000.png")
plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
print("Saved figure:", outfile_png)