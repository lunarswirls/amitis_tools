#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import src.surface_flux.flux_utils as flux_utils
import src.helper_utils as helper_utils

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

output_folder = f"/Users/danywaller/Projects/mercury/extreme/surface_flux/"

cases = ["RPN_Base", "CPN_Base", "RPS_Base", "CPS_Base"]
stats_cases_all = []

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

    stats_case = flux_utils.compute_flux_statistics(flux_abs, lat_centers, lon_centers,
                                          R_M=2439.7e3, case_name=case.split("_")[0])

    stats_cases_all.append(stats_case)

outcsv = os.path.join(output_folder, 'base_flux_comparison.csv')
outtex = os.path.join(output_folder, 'base_flux_comparison.tex')

df = flux_utils.create_comparison_table(stats_cases_all, output_csv=outcsv, output_latex=outtex)