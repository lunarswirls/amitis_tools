#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.surface_flux.flux_utils as flux_utils
import src.helper_utils as helper_utils

cases = ["RPN_Base", "CPN_Base", "RPS_Base", "CPS_Base"]
for case in cases:
    main_path = f'/Volumes/data_backup/mercury/extreme/{case}/05/'
    output_folder = f"/Users/danywaller/Projects/mercury/extreme/surface_cnts_den_vr_flux_mass_energy/1degx1deg_binning/"

    plot_meth = "lognorm"  # raw, log, lognorm
    run_species = "alphas"  # 'all' or 'protons' or 'alphas'

    outdir = output_folder + f"{run_species}"
    os.makedirs(outdir, exist_ok=True)

    species = np.array(['H+', 'H+', 'He++', 'He++'])  # The order is important and it should be based on Amitis.inp file
    sim_ppc = np.array([24, 0, 11, 0])  # Number of particles per species, based on Amitis.inp
    sim_den = np.array([38.0e6, 0, 1.0e6, 0])  # [/m^3]
    sim_vel = np.array([400.e3, 0, 400.e3, 0])  # [m/s]

    # Species properties
    species_mass = np.array([1.0, 0.0, 4.0, 0.0])  # [amu] proton1, proton2, alpha1, alpha2
    species_charge = np.array([1.0, 0.0, 2.0, 0.0])  # [e] proton1, proton2, alpha1, alpha2

    sim_dx = 75.e3  # simulation cell size based on Amitis.inp [m]
    sim_dy = 75.e3  # simulation cell size based on Amitis.inp [m]
    sim_dz = 75.e3  # simulation cell size based on Amitis.inp [m]
    sim_robs = 2440.e3  # obstacle radius based on Amitis.inp [m]

    nlat = 180
    nlon = 360

    select_R = 2480.e3  # the radius of a sphere + 1/2 grid cell above the surface for particle selection [m]

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
    den_cm3 = n_shell_map.copy()  # [cm^-3] shell volume density
    vr = v_r_map.copy()  # [km/s]
    flux = flux_cm.copy()  # [cm^-2 s^-1]
    mass_flux = mass_flux_map.copy()  # [amu cm^-2 s^-1]
    energy_flux = energy_flux_map.copy()  # [eV cm^-2 s^-1]

    vr_abs = np.abs(vr)  # [km/s]
    flux_abs = np.abs(flux)  # [cm^-2 s^-1]
    mass_flux_abs = np.abs(mass_flux)  # [amu cm^-2 s^-1]
    energy_flux_abs = np.abs(energy_flux)  # [eV cm^-2 s^-1]

    # Set low-count pixels to NaN
    mask = count_map <= 1e-30
    cnts[mask] = np.nan
    den_cm3[mask] = np.nan
    vr_abs[mask] = np.nan
    flux_abs[mask] = np.nan
    mass_flux_abs[mask] = np.nan
    energy_flux_abs[mask] = np.nan

    # ========== Logarithmic maps ==========
    log_cnts = helper_utils.safe_log10(cnts)
    log_den  = helper_utils.safe_log10(den_cm3)  # log10(cm^-3)
    log_vel  = helper_utils.safe_log10(vr_abs)   # log10(km/s)
    log_flx  = helper_utils.safe_log10(flux_abs) # log10(cm^-2 s^-1)
    log_mass_flux = helper_utils.safe_log10(mass_flux_abs)  # log10(amu cm^-2 s^-1)
    log_energy_flux = helper_utils.safe_log10(energy_flux_abs)  # log10(eV cm^-2 s^-1)

    # ========== Normalized maps ==========
    if run_species == "all":
        if "Base" in case:
            # Total upstream number density from quasi-neutrality Σ(Z_i * n_i) [m^-3]
            sim_den_tot = np.sum(species_charge * sim_den)  # [m^-3]

            # Mass-weighted upstream velocity [km/s]
            # v_avg = Σ(m_i * n_i * v_i) / Σ(m_i * n_i)
            mass_weighted_velocity = np.sum(species_mass * sim_den * sim_vel) / np.sum(species_mass * sim_den)
            sim_vel_tot = mass_weighted_velocity * 1e-3  # [m/s] → [km/s]

            # Upstream flux using mass-weighted velocity [cm^-2 s^-1]
            sim_flux_upstream = sim_den_tot * mass_weighted_velocity * 1e-4  # [m^-3 * m/s] → [cm^-2 s^-1]

            # Upstream mass flux [amu cm^-2 s^-1]
            # Mass flux = Σ(m_i * n_i * v_i) * 1e-4 [amu/m^3 * m/s * (m^2/cm^2)]
            sim_mass_flux_upstream = np.sum(species_mass * sim_den * sim_vel) * 1e-4  # [amu cm^-2 s^-1]

            # Upstream energy flux [eV cm^-2 s^-1]
            # Energy flux = Σ(0.5 * m_i * n_i * v_i^3) * conversion
            AMU_TO_KG = 1.66053906660e-27
            J_TO_EV = 6.241509074e18
            sim_energy_flux_upstream = 0.5 * np.sum(species_mass * AMU_TO_KG * sim_den * sim_vel**3) * J_TO_EV * 1e-4  # [eV cm^-2 s^-1]

            # Normalized quantities
            log_den_norm = helper_utils.safe_log10(den_cm3 / (sim_den_tot * 1e-6))  # [cm^-3] / [cm^-3]
            log_vel_norm = helper_utils.safe_log10(vr_abs / sim_vel_tot)  # [km/s] / [km/s]
            log_flx_norm = helper_utils.safe_log10(flux_abs / sim_flux_upstream)  # [cm^-2 s^-1] / [cm^-2 s^-1]
            log_mass_flux_norm = helper_utils.safe_log10(mass_flux_abs / sim_mass_flux_upstream)
            log_energy_flux_norm = helper_utils.safe_log10(energy_flux_abs / sim_energy_flux_upstream)

    elif run_species == "protons":
        if "Base" in case:
            # Single proton species (index 0)

            # Total upstream number density from quasi-neutrality Σ(Z_i * n_i) [m^-3]
            sim_den_tot = species_charge[0] * sim_den[0]  # [m^-3]

            # Mass-weighted upstream velocity [km/s]
            # v_avg = Σ(m_i * n_i * v_i) / Σ(m_i * n_i)
            mass_weighted_velocity = (species_mass[0] * sim_den[0] * sim_vel[0]) / (species_mass[0] * sim_den[0])
            sim_vel_tot = mass_weighted_velocity * 1e-3  # [m/s] → [km/s]

            # Upstream flux using mass-weighted velocity [cm^-2 s^-1]
            sim_flux_upstream = sim_den_tot * mass_weighted_velocity * 1e-4  # [m^-3 * m/s] → [cm^-2 s^-1]

            # Upstream mass flux [amu cm^-2 s^-1]
            sim_mass_flux_upstream = species_mass[0] * sim_den[0] * sim_vel[0] * 1e-4

            # Upstream energy flux [eV cm^-2 s^-1]
            AMU_TO_KG = 1.66053906660e-27
            J_TO_EV = 6.241509074e18
            sim_energy_flux_upstream = 0.5 * species_mass[0] * AMU_TO_KG * sim_den[0] * sim_vel[0]**3 * J_TO_EV * 1e-4

            # Normalized quantities
            log_den_norm = helper_utils.safe_log10(den_cm3 / (sim_den_tot * 1e-6))  # [cm^-3] / [cm^-3]
            log_vel_norm = helper_utils.safe_log10(vr_abs / sim_vel_tot)  # [km/s] / [km/s]
            log_flx_norm = helper_utils.safe_log10(flux_abs / sim_flux_upstream)  # [cm^-2 s^-1] / [cm^-2 s^-1]
            log_mass_flux_norm = helper_utils.safe_log10(mass_flux_abs / sim_mass_flux_upstream)
            log_energy_flux_norm = helper_utils.safe_log10(energy_flux_abs / sim_energy_flux_upstream)

    elif run_species == "alphas":
        if "Base" in case:
            # Single alpha species (index 2)

            # Total upstream number density from quasi-neutrality Σ(Z_i * n_i) [m^-3]
            sim_den_tot = species_charge[2] * sim_den[2]  # [m^-3]

            # Mass-weighted upstream velocity [km/s]
            # v_avg = Σ(m_i * n_i * v_i) / Σ(m_i * n_i)
            mass_weighted_velocity = (species_mass[2] * sim_den[2] * sim_vel[2]) / (species_mass[2] * sim_den[2])
            sim_vel_tot = mass_weighted_velocity * 1e-3  # [m/s] → [km/s]

            # Upstream flux using mass-weighted velocity [cm^-2 s^-1]
            sim_flux_upstream = sim_den_tot * mass_weighted_velocity * 1e-4  # [m^-3 * m/s] → [cm^-2 s^-1]

            # Upstream mass flux [amu cm^-2 s^-1]
            sim_mass_flux_upstream = species_mass[2] * sim_den[2] * sim_vel[2] * 1e-4

            # Upstream energy flux [eV cm^-2 s^-1]
            AMU_TO_KG = 1.66053906660e-27
            J_TO_EV = 6.241509074e18
            sim_energy_flux_upstream = 0.5 * species_mass[2] * AMU_TO_KG * sim_den[2] * sim_vel[2]**3 * J_TO_EV * 1e-4

            # Normalized quantities
            log_den_norm = helper_utils.safe_log10(den_cm3 / (sim_den_tot * 1e-6))  # [cm^-3] / [cm^-3]
            log_vel_norm = helper_utils.safe_log10(vr_abs / sim_vel_tot)  # [km/s] / [km/s]
            log_flx_norm = helper_utils.safe_log10(flux_abs / sim_flux_upstream)  # [cm^-2 s^-1] / [cm^-2 s^-1]
            log_mass_flux_norm = helper_utils.safe_log10(mass_flux_abs / sim_mass_flux_upstream)
            log_energy_flux_norm = helper_utils.safe_log10(energy_flux_abs / sim_energy_flux_upstream)

    # Debug output
    print(f"Upstream normalization values:")
    print(f"  Total density: {sim_den_tot * 1e-6:.1f} cm^-3")
    print(f"  Mass-weighted velocity: {sim_vel_tot:.1f} km/s")
    print(f"  Upstream flux: {sim_flux_upstream:.2e} cm^-2 s^-1")
    print(f"  Upstream mass flux: {sim_mass_flux_upstream:.2e} amu cm^-2 s^-1")
    print(f"  Upstream energy flux: {sim_energy_flux_upstream:.2e} eV cm^-2 s^-1")

    # Define fields for plotting (6 fields in 3x2 layout)
    fields_raw = [
        (cnts, (1, 200), "viridis", "# particles"),
        (den_cm3, (1, 140), "cividis", r"$n$ [cm$^{-3}$]"),
        (vr_abs, (1, 250), "plasma", r"$|v_r|$ [km/s]"),
        (flux_abs, (0.05e9, 1.e13), "jet", r"$F_r$ [cm$^{-2}$ s$^{-1}$]"),
        (mass_flux_abs, (np.nanmin(mass_flux_abs), 4.e13), "copper", r"$F_{mass}$ [amu cm$^{-2}$ s$^{-1}$]"),
        (energy_flux_abs, (np.nanmin(energy_flux_abs), 1.e16), "inferno", r"$F_{energy}$ [eV cm$^{-2}$ s$^{-1}$]")
    ]

    fields_log = [
        (cnts, (np.nanmin(cnts), np.nanmax(cnts)), "viridis", "# particles"),
        (log_den, (np.nanmin(log_den), np.nanmax(log_den)), "cividis", r"log$_{10}$($n$) [cm$^{-3}$]"),
        (log_vel, (np.nanmin(log_vel), np.nanmax(log_vel)), "plasma", r"log$_{10}$($|v_r|$) [km s$^{-1}$]"),
        (log_flx, (np.nanmin(log_flx), np.nanmax(log_flx)), "jet", r"log$_{10}$($F_r$) [cm$^{-2}$ s$^{-1}$]"),
        (log_mass_flux, (np.nanmin(log_mass_flux), np.nanmax(log_mass_flux)), "copper", r"log$_{10}$($F_{mass}$) [amu cm$^{-2}$ s$^{-1}$]"),
        (log_energy_flux, (np.nanmin(log_energy_flux), np.nanmax(log_energy_flux)), "inferno", r"log$_{10}$($F_{energy}$) [eV cm$^{-2}$ s$^{-1}$]")
    ]

    fields_log_norm = [
        (cnts, (np.nanmin(cnts), np.nanmax(cnts)), "viridis", "# particles"),
        (log_den_norm, (-1, 1), "cividis", r"log$_{10}$($n/n_0$)"),
        (log_vel_norm, (-1.0, 0.0), "plasma", r"log$_{10}$($|v_r|/v_0$)"),
        (log_flx_norm, (0, 4), "jet", r"log$_{10}$($F_r/F_0$)"),
        (log_mass_flux_norm, (0, 5), "winter", r"log$_{10}$($F_{mass}/F_{mass,0}$)"),
        (log_energy_flux_norm, (0, 4), "inferno", r"log$_{10}$($F_{energy}/F_{energy,0}$)")
    ]

    if plot_meth == 'raw':
        use_fields = fields_raw
    elif plot_meth == 'log':
        use_fields = fields_log
    elif plot_meth == 'lognorm':
        use_fields = fields_log_norm

    titles = ["Counts", "Density", "Radial velocity", "Flux", "Mass flux", "Energy flux"]

    # ---- 3. Plot in Hammer projection (3x2 layout) ----
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 13.5),  # Increased height for third row
        subplot_kw={"projection": "hammer"}
    )

    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for ax, (data, clim, cmap, cblabel), title in zip(axes, use_fields, titles):
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

    # Generate title based on species selection
    if run_species == "all":
        stitle = f"{case.replace('_', ' ')}: All species"
        plot_fname = f"{case}_cnts_den_vr_flux_mass_ener_all_species_{plot_meth}vals"
    elif run_species == "protons":
        stitle = f"{case.replace('_', ' ')}: H+"
        plot_fname = f"{case}_cnts_den_vr_flux_mass_ener_H+_{plot_meth}vals"
    elif run_species == "alphas":
        stitle = f"{case.replace('_', ' ')}: He++"
        plot_fname = f"{case}_cnts_den_vr_flux_mass_ener_He++_{plot_meth}vals"

    fig.suptitle(stitle, fontsize=20, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

    outfile_png = os.path.join(outdir, plot_fname)
    plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
    print("Saved figure:", outfile_png)
    # plt.show()
    plt.close(fig)
