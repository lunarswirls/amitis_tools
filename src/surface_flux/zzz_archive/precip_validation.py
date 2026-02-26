#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.surface_flux.flux_utils_testing as flux_utils
import src.helper_utils as helper_utils

output_folder = f"/Users/danywaller/Projects/mercury/extreme/precipitation_validation/"

debug = True

plot_meth = "raw"  # raw, log, lognorm
run_species = "all"  # 'all' or 'protons' or 'alphas'

outdir = f"/Users/danywaller/Projects/mercury/extreme/precipitation_validation/{run_species}"
os.makedirs(outdir, exist_ok=True)

species = np.array(['H+'])  # The order is important and it should be based on Amitis.inp file
sim_ppc = [16]  # Number of particles per species, based on Amitis.inp
sim_den = [7.0e6]   # [/m^3]
sim_vel = [415.e3]  # [m/s]
species_mass = np.array([1.0])  # [amu] proton1, proton2, alpha1, alpha2
species_charge = np.array([1.0])  # [e] proton1, proton2, alpha1, alpha2

sim_dx = 200.e3  # simulation cell size based on Amitis.inp [m]
sim_dy = 200.e3  # simulation cell size based on Amitis.inp [m]
sim_dz = 200.e3  # simulation cell size based on Amitis.inp [m]
sim_robs = 1500e3  # obstacle radius based on Amitis.inp [m]

nlat = 90
nlon = 180

select_R = 1575e3 # 2480.e3  # the radius of a sphere + 1/2 grid cell above the surface for particle selection [m]

main_path = f"/Volumes/data_backup/2026_01_27_PrecipitationValidation/"

all_particles_directory = main_path + 'subset/'
all_particles_filename = all_particles_directory + f"Subset_prec_valid_020000_G000.npz"

flux_cm, lat_centers, lon_centers, v_r_map, count_map, n_shell_map, mass_flux_map, energy_flux_map = \
    flux_utils.compute_radial_flux(
        all_particles_filename=all_particles_filename,
        sim_dx=sim_dx, sim_dy=sim_dy, sim_dz=sim_dz,
        sim_ppc=sim_ppc, sim_den=sim_den, spec_map=species,
        species_mass=species_mass, species_charge=species_charge,
        R_M=sim_robs, select_R=select_R,
        species=run_species,
        n_lat=nlat, n_lon=nlon, debug=debug,
    )

n_lat = len(lat_centers)
n_lon = len(lon_centers)

# Rebuild bin edges consistent with centers
lon_edges = np.linspace(-180.0, 180.0, n_lon + 1)
lat_edges = np.linspace(-90.0, 90.0, n_lat + 1)

# ========== 2D maps with units ==========
cnts = count_map.copy()  # [# particles]
den = n_shell_map.copy()  # [cm^-3] shell volume density
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

log_flx = helper_utils.safe_log10(flux_abs)

all_mass_den = species_mass[0] * sim_den[0]
all_vel_avg = (species_mass[0] * sim_den[0] * sim_vel[0]) / all_mass_den

# Total number density
all_den_tot = sim_den[0]  # [m^-3]

# Upstream flux [cm^-2 s^-1]
sim_flux_upstream = all_den_tot * all_vel_avg * 1e-4

print(f"Upstream flux:\t{sim_flux_upstream:.2E}")
print("\n")

# Define fields for plotting (6 fields in 3x2 layout)
fields_raw = [
    (cnts, (1, 50), "viridis", "# particles"),
    (den, (1, 50), "cividis", r"$n$ [cm$^{-3}$]"),
    (vr_abs, (1, 400), "plasma", r"$|v_r|$ [km/s]"),
    (log_flx, (10, 15), "jet", r"log10(F [cm$^{-2}$ s$^{-1}$])"),
]

titles = ["Counts", "Density", "Radial velocity", "Flux"]

# ---- 3. Plot in Hammer projection (3x2 layout) ----
fig, axes = plt.subplots(
2, 2, figsize=(14, 13.5),  # Increased height for third row
subplot_kw={"projection": "hammer"}
)

fig.patch.set_facecolor("white")
axes = axes.flatten()

for ax, (data, clim, cmap, cblabel), title in zip(axes, fields_raw, titles):
    ax.set_facecolor("white")
    ax.grid(True, linestyle="dotted", color="gray")

    # IMPORTANT: use edges (length n+1) and data (n_lat, n_lon)
    pcm = ax.pcolormesh(
        np.radians(lon_edges),  # X: shape (n_lon+1,)
        np.radians(lat_edges),  # Y: shape (n_lat+1,)
        data,                   # C: shape (n_lat, n_lon)
        cmap=cmap,
        shading="flat"
    )
    pcm.set_clim(*clim)

    cbar = plt.colorbar(
        pcm,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        shrink=0.85
    )
    cbar.set_label(cblabel, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.set_title(title, fontsize=20)

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
# Save figure
plt.tight_layout()
plt.show()
