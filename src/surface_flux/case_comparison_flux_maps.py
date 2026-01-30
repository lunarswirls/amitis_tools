#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.surface_flux.flux_utils as flux_utils
import src.helper_utils as helper_utils

# SETTINGS
cases = ["RPS_Base", "CPS_Base", "RPN_Base", "CPN_Base"]
output_folder = f"/Users/danywaller/Projects/mercury/extreme/surface_flux/"
os.makedirs(output_folder, exist_ok=True)

debug = False
footprints = False

plot_meth = "raw"  # raw, log, lognorm
run_species = "all"  # 'all' or 'protons' or 'alphas'

species = np.array(['H+', 'tbd', 'He++', 'tbd2'])  # The order is important and it should be based on Amitis.inp file
sim_ppc = [24, 0, 11, 0]  # Number of particles per species, based on Amitis.inp
sim_den = [38.0e6, 0, 1.0e6, 0]  # [/m^3]
sim_vel = [400.e3, 0, 400.e3, 0]  # [m/s]

sim_dx = 75.e3  # simulation cell size based on Amitis.inp
sim_dy = 75.e3  # simulation cell size based on Amitis.inp
sim_dz = 75.e3  # simulation cell size based on Amitis.inp
sim_robs = 2440.e3  # obstacle radius based on Amitis.inp

nlat = 90
nlon = 180

select_R = 2480.e3  # the radius of a sphere + 1/2 grid cell above the surface for particle selection

# Prepare figure
fig, axs = plt.subplots(2, 2, figsize=(12, 8), subplot_kw={"projection": "hammer"})

for case in cases:
    main_path = f'/Volumes/data_backup/mercury/extreme/{case}/05/'

    all_particles_directory = main_path + 'precipitation/'
    all_particles_filename = all_particles_directory + "all_particles_at_surface.npz"

    flux_cm, lat_centers, lon_centers, v_r_map, count_map, n_shell_map = \
        flux_utils.compute_radial_flux(
            all_particles_filename=all_particles_filename,
            sim_dx=sim_dx, sim_dy=sim_dy, sim_dz=sim_dz,
            sim_ppc=sim_ppc, sim_den=sim_den, spec_map=species,
            R_M=sim_robs, select_R=select_R,
            species=run_species,
            n_lat=nlat, n_lon=nlon
        )

    n_lat = len(lat_centers)
    n_lon = len(lon_centers)

    # Rebuild bin edges consistent with centers
    lon_edges = np.linspace(-180.0, 180.0, n_lon + 1)
    lat_edges = np.linspace(-90.0, 90.0, n_lat + 1)

    # ========== 2D maps with units ==========
    cnts = count_map.copy()  # [# particles]
    den = n_shell_map.copy()  # [m^-3] shell volume density
    vr = v_r_map.copy()  # [km/s]
    flux = flux_cm.copy()  # [cm^-2 s^-1]

    vr_abs = np.abs(vr)  # [km/s]
    flux_abs = np.abs(flux)  # [cm^-2 s^-1]

    # Set low-count pixels to NaN
    mask = count_map <= 1e-20
    cnts[mask] = np.nan
    den[mask] = np.nan
    vr_abs[mask] = np.nan
    flux_abs[mask] = np.nan

    # Total upstream density [m^-3]
    sim_den_tot = np.sum(sim_den)

    # Upstream velocity [km/s]
    sim_vel_tot = np.mean(sim_vel) * 1e-3  # [m/s] → [km/s]

    # Upstream flux [cm^-2 s^-1]
    sim_flux_upstream = sim_den_tot * np.mean(sim_vel) * 1e-4  # [m^-3 * m/s] → [cm^-2 s^-1]

    log_flx_norm = helper_utils.safe_log10(flux_abs / sim_flux_upstream)  # [cm^-2 s^-1] / [cm^-2 s^-1]

    # Plot
    if "RPN" in case: row, col = 0, 0
    elif "CPN" in case: row, col = 1, 0
    elif "RPS" in case: row, col = 0, 1
    elif "CPS" in case: row, col = 1, 1

    ax = axs[row, col]
    ax.grid(True, alpha=0.3, linestyle="dotted", color="gray")

    # Surface flux
    sc = ax.pcolormesh(
        np.radians(lon_edges),  # X: shape (n_lon+1,)
        np.radians(lat_edges),  # Y: shape (n_lat+1,)
        flux_abs,  # C: shape (n_lat, n_lon)
        cmap="jet",
        shading="flat", vmin=0.5e8, vmax=8.5e8)
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.5)
    # cbar.set_label(r"$\log_{10}$(F [cm$^{-2}$ s$^{-1}$])")
    cbar.set_label(r"F [cm$^{-2}$ s$^{-1}$]")

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

    ax.set_title(case.split("_")[0])

fig.suptitle(f"Pre-Transient", fontsize=18, y=0.98)
# Save figure
plt.tight_layout()
outfile_png = os.path.join(output_folder, f"all_cases_surface_precipitation_pre-transient.png")

plt.savefig(outfile_png, dpi=200, bbox_inches="tight")
print("Saved figure:", outfile_png)