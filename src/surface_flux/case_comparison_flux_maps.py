#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.surface_flux.flux_utils as flux_utils
import src.helper_utils as helper_utils

# SETTINGS
# cases = ["RPS_Base", "CPS_Base", "RPN_Base", "CPN_Base"]
cases = ["RPS_HNHV", "CPS_HNHV", "RPN_HNHV", "CPN_HNHV"]

# FOR HNHV - DOUBLE CHECK ONLY ONE IS TRUE!!!!
transient = False  # 280-300 s
post_transient = False  # 330-350 s
new_state = True  # 680-700 s

output_folder = f"/Users/danywaller/Projects/mercury/extreme/surface_precipitation/"

debug = False
footprints = False

plot_meth = "raw"  # raw, log, lognorm
run_species = "all"  # 'all' or 'protons' or 'alphas'

outdir = f"/Users/danywaller/Projects/mercury/extreme/surface_precipitation/{run_species}"
os.makedirs(outdir, exist_ok=True)

species = np.array(['H+', 'H+', 'He++', 'He++'])  # The order is important and it should be based on Amitis.inp file
sim_ppc = [24, 24, 11, 11]  # Number of particles per species, based on Amitis.inp
sim_den = [38.0e6, 76.0e6, 1.0e6, 2.0e6]   # [/m^3]
sim_vel = [400.e3, 700.0e3, 400.e3, 700.0e3]  # [m/s]
species_mass = np.array([1.0, 1.0, 4.0, 4.0])  # [amu] proton1, proton2, alpha1, alpha2
species_charge = np.array([1.0, 1.0, 2.0, 2.0])  # [e] proton1, proton2, alpha1, alpha2

sim_dx = 75.e3  # simulation cell size based on Amitis.inp [m]
sim_dy = 75.e3  # simulation cell size based on Amitis.inp [m]
sim_dz = 75.e3  # simulation cell size based on Amitis.inp [m]
sim_robs = 2440.e3  # obstacle radius based on Amitis.inp [m]

nlat = 90
nlon = 180

select_R = 2480.e3  # the radius of a sphere + 1/2 grid cell above the surface for particle selection [m]

# Prepare figure
fig, axs = plt.subplots(2, 2, figsize=(12, 8), subplot_kw={"projection": "hammer"})

for case in cases:
    print(f"Processing {case}")
    if "Base" in case:
        main_path = f"/Volumes/data_backup/mercury/extreme/{case}/05/"
    elif "HNHV" in case:
        if transient and not post_transient and not new_state:
            main_path = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/02/"
        elif post_transient and not transient and not new_state:
            main_path = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/03/"
        elif new_state and not post_transient and not transient:
            main_path = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/10/"
        else:
            raise ValueError("Too many flags! Set only one of transient, post_transient, or new_state to True")
    else:
        raise ValueError("Unrecognized case! Are you using one of Base or HNHV?")

    all_particles_directory = main_path + 'precipitation/'
    all_particles_filename = all_particles_directory + f"{case}_all_particles_at_surface.npz"

    flux_cm, lat_centers, lon_centers, v_r_map, count_map, n_shell_map, mass_flux_map, energy_flux_map = \
        flux_utils.compute_radial_flux(
            all_particles_filename=all_particles_filename,
            sim_dx=sim_dx, sim_dy=sim_dy, sim_dz=sim_dz,
            sim_ppc=sim_ppc, sim_den=sim_den, spec_map=species,
            species_mass=species_mass, species_charge=species_charge,
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

    if run_species == "all":
        if "Base" in case:
            # Total upstream density from quasi-neutrality: n_e = Σ(Z_i * n_i) [m^-3]
            all_mass_den = species_mass[0] * sim_den[0] + species_mass[2] * sim_den[2]
            all_vel_avg = (species_mass[0] * sim_den[0] * sim_vel[0] + species_mass[2] * sim_den[2] * sim_vel[2]) / all_mass_den

            # Total number density
            all_den_tot = sim_den[0] + sim_den[2]  # [m^-3]

            # Upstream flux [cm^-2 s^-1]
            sim_flux_upstream = all_den_tot * all_vel_avg * 1e-4
            log_flx_norm = helper_utils.safe_log10(flux_abs / sim_flux_upstream)

        elif "HNHV" in case and not new_state:
            # Total upstream density from quasi-neutrality: n_e = Σ(Z_i * n_i) [m^-3]
            sim_den_tot = np.sum(species_charge * sim_den)

            # Mass-weighted upstream velocity [m/s]
            sim_vel_tot_ms = np.sum(species_mass * sim_den * sim_vel) / np.sum(species_mass * sim_den)
            sim_vel_tot = sim_vel_tot_ms * 1e-3  # [km/s]

            # Upstream flux [cm^-2 s^-1]
            sim_flux_upstream = sim_den_tot * sim_vel_tot_ms * 1e-4
            log_flx_norm = helper_utils.safe_log10(flux_abs / sim_flux_upstream)

        elif "HNHV" in case and new_state:
            # Total upstream density from quasi-neutrality: n_e = Σ(Z_i * n_i) [m^-3]
            all_mass_den = species_mass[1] * sim_den[1] + species_mass[3] * sim_den[3]
            all_vel_avg = (species_mass[1] * sim_den[1] * sim_vel[1] + species_mass[3] * sim_den[3] * sim_vel[3]) / all_mass_den

            # Total number density
            all_den_tot = sim_den[1] + sim_den[3]  # [m^-3]

            # Upstream flux [cm^-2 s^-1]
            sim_flux_upstream = all_den_tot * all_vel_avg * 1e-4
            log_flx_norm = helper_utils.safe_log10(flux_abs / sim_flux_upstream)

    elif run_species == "protons":
        if "Base" in case:
            # Single proton species (index 0)
            sim_flux_upstream = sim_den[0] * sim_vel[0] * 1e-4
            log_flx_norm = helper_utils.safe_log10(flux_abs / sim_flux_upstream)

        elif "HNHV" in case and not new_state:
            # THIS SHOULD WORK FOR TRANSIENT AND POST-TRANSIENT TIMES!

            # Two proton species (indices 0, 1) - mass-weighted
            proton_mass_den = species_mass[0] * sim_den[0] + species_mass[1] * sim_den[1]
            proton_vel_avg = (species_mass[0] * sim_den[0] * sim_vel[0] + species_mass[1] * sim_den[1] * sim_vel[1]) / proton_mass_den

            # Total proton number density
            proton_den_tot = sim_den[0] + sim_den[1]

            sim_flux_upstream = proton_den_tot * proton_vel_avg * 1e-4
            log_flx_norm = helper_utils.safe_log10(flux_abs / sim_flux_upstream)

        elif "HNHV" in case and new_state:
            # Single proton species (index 1) - mass-weighted
            proton_mass_den = species_mass[1] * sim_den[1]
            proton_vel_avg = (species_mass[1] * sim_den[1] * sim_vel[1]) / proton_mass_den

            # Total proton number density
            proton_den_tot = sim_den[1]

            sim_flux_upstream = proton_den_tot * proton_vel_avg * 1e-4
            log_flx_norm = helper_utils.safe_log10(flux_abs / sim_flux_upstream)

    elif run_species == "alphas":
        if "Base" in case:
            # Single alpha species (index 2)
            sim_flux_upstream = sim_den[2] * sim_vel[2] * 1e-4
            log_flx_norm = helper_utils.safe_log10(flux_abs / sim_flux_upstream)

        elif "HNHV" in case and not new_state:
            # THIS SHOULD WORK FOR TRANSIENT AND POST-TRANSIENT TIMES!

            # Two alpha species (indices 2, 3) - mass-weighted
            alpha_mass_den = species_mass[2] * sim_den[2] + species_mass[3] * sim_den[3]
            alpha_vel_avg = (species_mass[2] * sim_den[2] * sim_vel[2] + species_mass[3] * sim_den[3] * sim_vel[3]) / alpha_mass_den

            # Total alpha number density
            alpha_den_tot = sim_den[2] + sim_den[3]

            sim_flux_upstream = alpha_den_tot * alpha_vel_avg * 1e-4
            log_flx_norm = helper_utils.safe_log10(flux_abs / sim_flux_upstream)

        elif "HNHV" in case and new_state:
            # Single alpha species (index 3) - mass-weighted
            alpha_mass_den = species_mass[3] * sim_den[3]
            alpha_vel_avg = (species_mass[3] * sim_den[3] * sim_vel[3]) / alpha_mass_den

            # Total alpha number density
            alpha_den_tot = sim_den[3]

            sim_flux_upstream = alpha_den_tot * alpha_vel_avg * 1e-4
            log_flx_norm = helper_utils.safe_log10(flux_abs / sim_flux_upstream)

    # Plot
    if "RPN" in case: row, col = 0, 0
    elif "CPN" in case: row, col = 1, 0
    elif "RPS" in case: row, col = 0, 1
    elif "CPS" in case: row, col = 1, 1

    if "Base" in case:
        if plot_meth == "raw":
            data = flux_abs
            c_min = 0.5e8
            c_max = 8.5e8
            ax_lab = r"F [cm$^{-2}$ s$^{-1}$]"
        elif plot_meth == "log":
            data = log_flx
            c_min = 3.5
            c_max = 9.5
            ax_lab = r"$\log_{10}$(F [cm$^{-2}$ s$^{-1}$])"
        elif plot_meth == "lognorm":
            data = log_flx_norm
            c_min = 0.0
            c_max = 5.0
            ax_lab = r"$\log_{10}$(F/F$_0$)"
    elif "HNHV" in case and not new_state:
        if plot_meth == "raw":
            data = flux_abs
            c_min = np.nanmin(data)
            c_max = np.nanmax(data)
            ax_lab = r"F [cm$^{-2}$ s$^{-1}$]"
        elif plot_meth == "log":
            data = log_flx
            c_min = 3.5
            c_max = 9.5
            ax_lab = r"$\log_{10}$(F [cm$^{-2}$ s$^{-1}$])"
        elif plot_meth == "lognorm":
            data = log_flx_norm
            c_min = 0.0
            c_max = 5.0
            ax_lab = r"$\log_{10}$(F/F$_0$)"

    elif "HNHV" in case and new_state:
        if plot_meth == "raw":
            data = flux_abs
            c_min = np.nanmin(data)
            c_max = np.nanmax(data)
            ax_lab = r"F [cm$^{-2}$ s$^{-1}$]"
        elif plot_meth == "log":
            data = log_flx
            c_min = 3.5
            c_max = 9.5
            ax_lab = r"$\log_{10}$(F [cm$^{-2}$ s$^{-1}$])"
        elif plot_meth == "lognorm":
            data = log_flx_norm
            c_min = 0.0
            c_max = 5.0
            ax_lab = r"$\log_{10}$(F/F$_0$)"

    ax = axs[row, col]
    ax.grid(True, alpha=0.3, linestyle="dotted", color="gray")

    # Surface flux
    sc = ax.pcolormesh(
        np.radians(lon_edges),  # X: shape (n_lon+1,)
        np.radians(lat_edges),  # Y: shape (n_lat+1,)
        data,  # C: shape (n_lat, n_lon)
        cmap="jet",
        shading="flat", vmin=c_min, vmax=c_max)
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.5)
    cbar.set_label(ax_lab)

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

if "Base" in cases[0]:
    outfname = f"all_cases_{run_species}_surface_precipitation_pre-transient_{plot_meth}.png"
    if run_species == "all":
        stitle = f"Pre-Transient (All species)"
    elif run_species == "protons":
        stitle = f"Pre-Transient (H+)"
    elif run_species == "alphas":
        stitle = f"Pre-Transient (He++)"
elif "HNHV" in cases[0] and transient:
    outfname = f"all_cases_{run_species}_surface_precipitation_transient_{plot_meth}.png"
    if run_species == "all":
        stitle = f"Transient (All species)"
    elif run_species == "protons":
        stitle = f"Transient (H+)"
    elif run_species == "alphas":
        stitle = f"Transient (He++)"
elif "HNHV" in cases[0] and post_transient:
    outfname = f"all_cases_{run_species}_surface_precipitation_post-transient_{plot_meth}.png"
    if run_species == "all":
        stitle = f"Post-Transient (All species)"
    elif run_species == "protons":
        stitle = f"Post-Transient (H+)"
    elif run_species == "alphas":
        stitle = f"Post-Transient (He++)"
elif "HNHV" in cases[0] and new_state:
    outfname = f"all_cases_{run_species}_surface_precipitation_newstate_{plot_meth}.png"
    if run_species == "all":
        stitle = f"New state (All species)"
    elif run_species == "protons":
        stitle = f"New state (H+)"
    elif run_species == "alphas":
        stitle = f"New state (He++)"

fig.suptitle(stitle, fontsize=18, y=0.98)
# Save figure
plt.tight_layout()
outfile_png = os.path.join(outdir, outfname)

plt.savefig(outfile_png, dpi=200, bbox_inches="tight")
print("Saved figure:", outfile_png)