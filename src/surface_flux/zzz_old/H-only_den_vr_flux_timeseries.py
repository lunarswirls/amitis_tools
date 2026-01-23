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
# base cases: CPN_Base RPN_Base CPS_Base RPS_Base
# HNHV cases: CPN_HNHV RPN_HNHV CPS_HNHV RPS_HNHV
case = "CPN_Base"
input_folder1  = f"/Users/danywaller/Projects/mercury/extreme/{case}/object/"
output_folder = f"/Users/danywaller/Projects/mercury/extreme/timeseries_HvsHe_surface_flux/{case}"
os.makedirs(output_folder, exist_ok=True)

debug = False

# first stable timestamp approx. 25000 for dt=0.002, numsteps=115000
if "Base" in case:
    # take last 10-ish seconds
    sim_steps = range(98000, 115000 + 1, 1000)
elif "HNHV" in case:
    sim_steps = range(115000, 350000 + 1, 1000)

R_M = 2440.0        # Mercury radius [km]
LAT_BINS = 360      # Surface latitude bins
LON_BINS = 360      # Surface longitude bins

# -------------------------------
# Load grid (assume first file is representative)
# -------------------------------
first_file = sorted([f for f in os.listdir(input_folder1) if f.endswith("_xz_comp.nc")])[0]
ds0 = xr.open_dataset(os.path.join(input_folder1, first_file))

x = ds0["Nx"].values
y = ds0["Ny"].values
z = ds0["Nz"].values

for sim_step in sim_steps:
    filetime = f"{sim_step:06d}"

    nc_file = os.path.join(input_folder1, f"Amitis_{case}_{filetime}_xz_comp.nc")
    ds = xr.open_dataset(nc_file)

    if debug:
        print(ds["den01"].dims)
        print(ds["den01"].shape)
        print(len(x), len(y), len(z))

    # Density (protons only) [units: cm^-3]
    h_den = (ds["den01"].isel(time=0).values + ds["den02"].isel(time=0).values)

    # Density (alphas only) [units: cm^-3]
    he_den = (ds["den03"].isel(time=0).values + ds["den04"].isel(time=0).values)

    h_weighted_flux, h_vr = compute_radial_flux(ds, x, y, z, sum_fields="protons")

    he_weighted_flux, he_vr = compute_radial_flux(ds, x, y, z, sum_fields="alphas")

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

    # HYDROGEN CASES
    # total density
    interp0 = RegularGridInterpolator((z, y, x), h_den, bounds_error=False, fill_value=np.nan)
    h_tot_den = interp0(points_surface).reshape(LAT_BINS, LON_BINS)
    h_tot_den = h_tot_den[::-1, :]  # flip latitude for plotting

    # radial velocity
    interp1 = RegularGridInterpolator((z, y, x), h_vr, bounds_error=False, fill_value=np.nan)
    h_v_r = interp1(points_surface).reshape(LAT_BINS, LON_BINS)
    h_v_r = h_v_r[::-1, :]  # flip latitude for plotting

    # weighted surface flux
    interp3 = RegularGridInterpolator((z, y, x), h_weighted_flux, bounds_error=False, fill_value=np.nan)
    flux_surface = interp3(points_surface).reshape(LAT_BINS, LON_BINS)
    flux_surface = flux_surface[::-1, :]  # flip latitude for plotting
    # Mask non-positive values
    flux_surface_masked = np.where(flux_surface > 0, flux_surface, np.nan)
    # Log10
    h_log_flux_surface = np.log10(flux_surface_masked)

    # HELIUM CASES
    # total density
    interp0 = RegularGridInterpolator((z, y, x), he_den, bounds_error=False, fill_value=np.nan)
    he_tot_den = interp0(points_surface).reshape(LAT_BINS, LON_BINS)
    he_tot_den = he_tot_den[::-1, :]  # flip latitude for plotting

    # radial velocity
    interp1 = RegularGridInterpolator((z, y, x), he_vr, bounds_error=False, fill_value=np.nan)
    he_v_r = interp1(points_surface).reshape(LAT_BINS, LON_BINS)
    he_v_r = he_v_r[::-1, :]  # flip latitude for plotting

    # weighted surface flux
    interp3 = RegularGridInterpolator((z, y, x), he_weighted_flux, bounds_error=False, fill_value=np.nan)
    flux_surface = interp3(points_surface).reshape(LAT_BINS, LON_BINS)
    flux_surface = flux_surface[::-1, :]  # flip latitude for plotting
    # Mask non-positive values
    flux_surface_masked = np.where(flux_surface > 0, flux_surface, np.nan)
    # Log10
    he_log_flux_surface = np.log10(flux_surface_masked)

    # -------------------------------
    # Plot
    # -------------------------------
    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, figsize=(16, 10), subplot_kw={"projection": "hammer"})

    # Plot flux
    lon_grid, lat_grid = np.meshgrid(lon_r, lat_r)  # radians
    # shift lon to [-pi, pi]
    lon_grid = np.where(lon_grid > np.pi, lon_grid - 2*np.pi, lon_grid)

    # proton density
    sc = ax0.pcolormesh(lon_grid, lat_grid, h_tot_den, cmap="plasma", shading="auto", vmin=0, vmax=50)
    cbar = fig.colorbar(sc, ax=ax0, orientation="horizontal", pad=0.05, shrink=0.5)
    cbar.set_label(r"N [cm$^{-3}$]")
    ax0.set_title(f"H+ Density")

    # proton v_r
    sc = ax1.pcolormesh(lon_grid, lat_grid, h_v_r, cmap="RdBu_r", shading="auto", vmin=-150, vmax=150)
    cbar = fig.colorbar(sc, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.5)
    cbar.set_label(r"$V_r$ [km/s]")
    ax1.set_title(f"H+ Radial Velocity")

    # proton flux
    sc = ax2.pcolormesh(lon_grid, lat_grid, h_log_flux_surface, cmap="viridis", shading="auto", vmin=6, vmax=8)
    cbar = fig.colorbar(sc, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.5)
    cbar.set_label(r"$\log_{10}$(F [cm$^{-2}$ s$^{-1}$])")
    ax2.set_title(f"H+ Surface Flux")

    # alpha density
    sc = ax3.pcolormesh(lon_grid, lat_grid, he_tot_den, cmap="plasma", shading="auto", vmin=0, vmax=1)
    cbar = fig.colorbar(sc, ax=ax3, orientation="horizontal", pad=0.05, shrink=0.5)
    cbar.set_label(r"N [cm$^{-3}$]")
    ax3.set_title(f"He++ Density")

    # alpha v_r
    sc = ax4.pcolormesh(lon_grid, lat_grid, he_v_r, cmap="RdBu_r", shading="auto", vmin=-250, vmax=250)
    cbar = fig.colorbar(sc, ax=ax4, orientation="horizontal", pad=0.05, shrink=0.5)
    cbar.set_label(r"$V_r$ [km/s]")
    ax4.set_title(f"He++ Radial Velocity")

    # alpha flux
    sc = ax5.pcolormesh(lon_grid, lat_grid, he_log_flux_surface, cmap="viridis", shading="auto", vmin=4, vmax=7)
    cbar = fig.colorbar(sc, ax=ax5, orientation="horizontal", pad=0.05, shrink=0.5)
    cbar.set_label(r"$\log_{10}$(F [cm$^{-2}$ s$^{-1}$])")
    ax5.set_title(f"He++ Surface Flux")

    # Longitude ticks (-170 to 170 every n °)
    lon_ticks_deg = np.arange(-120, 121, 60)
    lon_ticks_rad = np.deg2rad(lon_ticks_deg)

    # Latitude ticks (-90 to 90 every n °)
    lat_ticks_deg = np.arange(-60, 61, 30)
    lat_ticks_rad = np.deg2rad(lat_ticks_deg)

    for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
        ax.set_xticks(lon_ticks_rad)
        ax.set_yticks(lat_ticks_rad)

        ax.set_xticklabels([])  # disable default longitude labels
        ax.set_yticklabels([f"{int(l)}°" for l in lat_ticks_deg])

        ax.grid(True, alpha=0.3, color="grey")

    label_lat_deg = -10
    label_lat_rad = np.deg2rad(label_lat_deg)

    for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
        for lon_deg, lon_rad in zip(lon_ticks_deg, lon_ticks_rad):
            ax.text(
                lon_rad,
                label_lat_rad,
                f"{int(lon_deg)}°",
                ha="center",
                va="top",
                fontsize=10,
                transform=ax.transData,
            )

    plt.suptitle(f"{case.replace("_"," ")} at t = {sim_step * 0.002} s", y=0.85)
    plt.tight_layout()
    # Save figure
    plt.tight_layout()
    outfile_png = os.path.join(output_folder, f"HvsHe_radial_flux_calc_testing_{case}_{filetime}.png")
    plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
    print("Saved figure:", outfile_png)