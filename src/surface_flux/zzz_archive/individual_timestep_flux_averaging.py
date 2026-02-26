#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.surface_flux.flux_utils_testing as flux_utils
import src.helper_utils as helper_utils

debug = True

species = np.array(['H+'])  # The order is important and it should be based on Amitis.inp file
sim_ppc = np.array([12])  # Number of particles per species, based on Amitis.inp
sim_den = np.array([40.0e6])  # [/m^3]
sim_vel = np.array([400.e3])  # [m/s]

# Species properties
species_mass = np.array([1.0])  # [amu] proton1
species_charge = np.array([1.0])  # [e] proton1

sim_dx = 200.e3  # simulation cell size based on Amitis.inp [m]
sim_dy = 200.e3  # simulation cell size based on Amitis.inp [m]
sim_dz = 200.e3  # simulation cell size based on Amitis.inp [m]
sim_robs = 2440.e3  # obstacle radius based on Amitis.inp [m]

nlat = 90
nlon = 180

select_R = 2480.e3  # the radius of a sphere + 1/2 grid cell above the surface for particle selection [m]

cases = ["inert_sunward"]

for case in cases:
    if "inert_sunward" in case:
        main_path = f"/Users/danywaller/Projects/mercury/inert_small_body_sunward_IMF/"
        case = "SW_IMF"
    elif "inert_planetward" in case:
        main_path = f"/Users/danywaller/Projects/mercury/inert_small_body_planetward_IMF/"
        case = "PW_IMF"
    elif "validation" in case:
        main_path = f"/Volumes/data_backup/2026_02_12_LongPrecipValidation/"
        case = "prec_valid"

    if "inert" in case:
        output_folder = f"/Users/danywaller/Projects/mercury/precipitation_validation_test_cases_1sec_n28_timestep_avg/"
        all_particles_directory = main_path + 'particles_1sec_n28/'
    else:
        output_folder = f"/Users/danywaller/Projects/precipitation_validation_test_case_timestep_avg/"
        all_particles_directory = main_path + 'particles/'

    plot_meth = "raw"  # raw, log, lognorm
    run_species = "all"  # 'all'

    outdir = output_folder + f"{run_species}"
    os.makedirs(outdir, exist_ok=True)

    sim_steps = list(range(30000, 31400 + 1, 50))

    flux_cm_all = np.zeros((nlat, nlon))
    v_r_map_all = np.zeros((nlat, nlon))
    count_map_all = np.zeros((nlat, nlon))
    den_cm3_all = np.zeros((nlat, nlon))
    mass_flux_map_all = np.zeros((nlat, nlon))
    energy_flux_map_all = np.zeros((nlat, nlon))

    valid_counts = np.zeros((nlat, nlon), dtype=np.int32)

    for step in sim_steps:
        print(f"Running step {step}")
        all_particles_filename = all_particles_directory + f"Subset_{case}_{"%06d" % step}_full_domain.npz"

        flux_cm, lat_centers, lon_centers, v_r_map, count_map, n_shell_map, mass_flux_map, energy_flux_map = \
            flux_utils.compute_radial_flux(
                all_particles_filename=all_particles_filename,
                sim_dx=sim_dx, sim_dy=sim_dy, sim_dz=sim_dz,
                sim_ppc=sim_ppc, sim_den=sim_den, spec_map=species,
                species_mass=species_mass, species_charge=species_charge,
                R_M=sim_robs, select_R=select_R,
                species=run_species,
                n_lat=nlat, n_lon=nlon, debug=debug
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

        # Use a real count threshold (macroparticles) for "valid" pixels
        valid = np.isfinite(count_map) & (count_map >= 1)  # >= 1 macro is a sane start

        # Convert NaNs to 0 before accumulation
        flux_cm_all += np.nan_to_num(flux_abs, nan=0.0)
        v_r_map_all += np.nan_to_num(vr_abs, nan=0.0)
        count_map_all += np.nan_to_num(cnts, nan=0.0)
        den_cm3_all += np.nan_to_num(den_cm3, nan=0.0)
        mass_flux_map_all += np.nan_to_num(mass_flux_abs, nan=0.0)
        energy_flux_map_all += np.nan_to_num(energy_flux_abs, nan=0.0)

        # Also accumulate how many timesteps contributed per pixel (for later averaging)
        # (do this once, define at top before loop: valid_counts = np.zeros((nlat,nlon), dtype=int))
        valid_counts += valid.astype(int)

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
    sim_energy_flux_upstream = 0.5 * np.sum(species_mass * AMU_TO_KG * sim_den * sim_vel ** 3) * J_TO_EV * 1e-4  # [eV cm^-2 s^-1]

    # Debug output
    print(f"Upstream normalization values:")
    print(f"  Total density: {sim_den_tot * 1e-6:.1f} cm^-3")
    print(f"  Mass-weighted velocity: {sim_vel_tot:.1f} km/s")
    print(f"  Upstream flux: {sim_flux_upstream:.2e} cm^-2 s^-1")
    print(f"  Upstream mass flux: {sim_mass_flux_upstream:.2e} amu cm^-2 s^-1")
    print(f"  Upstream energy flux: {sim_energy_flux_upstream:.2e} eV cm^-2 s^-1")

    # Avoid divide-by-zero
    denom = np.where(valid_counts > 0, valid_counts, np.nan)

    flux_cm_all_plot = flux_cm_all / denom
    v_r_map_all_plot = v_r_map_all / denom
    count_map_all_plot = count_map_all / denom
    den_cm3_all_plot = den_cm3_all / denom
    mass_flux_map_all_plot = mass_flux_map_all / denom
    energy_flux_map_all_plot = energy_flux_map_all / denom

    plot_mask = (valid_counts < 3)  # e.g., require at least 3 timesteps contributing
    flux_cm_all_plot = np.where(plot_mask, np.nan, flux_cm_all_plot)
    den_cm3_all_plot = np.where(plot_mask, np.nan, den_cm3_all_plot)
    v_r_map_all_plot = np.where(plot_mask, np.nan, v_r_map_all_plot)
    mass_flux_map_all_plot = np.where(plot_mask, np.nan, mass_flux_map_all_plot)
    energy_flux_map_all_plot = np.where(plot_mask, np.nan, energy_flux_map_all_plot)

    print("=" * 60)
    print("FINAL MAPS STATISTICS")
    print("=" * 60)
    print(f"Total physical particles: {count_map_all_plot.sum():.2e}")
    print(f"den_map_cm3:     [{den_cm3_all_plot[den_cm3_all_plot > 0].min():.2e}, {den_cm3_all_plot[den_cm3_all_plot > 0].max():.2e}] cm^-3")
    print(f"flux_map_cm:     [{flux_cm_all_plot[flux_cm_all_plot > 0].min():.2e}, {flux_cm_all_plot[flux_cm_all_plot > 0].max():.2e}] cm^-2 s^-1")
    print(f"v_r_map:         [{np.nanmin(v_r_map_all_plot):.2f}, {np.nanmax(v_r_map_all_plot):.2f}] km/s")
    print(f"mass_flux_map:   [{mass_flux_map_all_plot[mass_flux_map_all_plot > 0].min():.2e}, {mass_flux_map_all_plot[mass_flux_map_all_plot > 0].max():.2e}] amu cm^-2 s^-1")
    print(
        f"energy_flux_map: [{energy_flux_map_all_plot[energy_flux_map_all_plot > 0].min():.2e}, {energy_flux_map_all_plot[energy_flux_map_all_plot > 0].max():.2e}] eV cm^-2 s^-1")
    print("=" * 60 + "\n")

    fields_raw = [
        (count_map_all_plot, (1, 3), "viridis", "# particles"),
        (den_cm3_all_plot, (0, 200), "cividis", r"$n$ [cm$^{-3}$]"),
        (v_r_map_all_plot, (0, 500), "plasma", r"$|v_r|$ [km/s]"),
        (flux_cm_all_plot, (0, 2e14), "jet", r"$F_r$ [cm$^{-2}$ s$^{-1}$]"),
        (mass_flux_map_all_plot, (0, 2e14), "copper",
         r"$F_{mass}$ [amu cm$^{-2}$ s$^{-1}$]"),
        (energy_flux_map_all_plot, (0, 3e18), "inferno",
         r"$F_{energy}$ [eV cm$^{-2}$ s$^{-1}$]")
    ]

    use_fields = fields_raw

    titles = ["Counts", "Density", "Radial velocity", "Surface Flux", "Mass flux", "Energy flux"]

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

    if "SW" in case:
        title_name = "Sunward IMF"
    if "PW" in case:
        title_name = "Planetward IMF"

    # Generate title based on species selection
    stitle = f"{title_name}: One species (H+)"
    plot_fname = f"{case}_cnts_den_vr_volume_flux_mass_ener_one_species_{plot_meth}vals"

    fig.suptitle(stitle, fontsize=20, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

    outfile_png = os.path.join(outdir, plot_fname)
    plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
    print("Saved figure:", outfile_png)
    # plt.show()
    plt.close(fig)
