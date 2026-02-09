#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import src.surface_flux.flux_utils as flux_utils

run_species = "all"  # 'all' or 'protons' or 'alphas'

species = np.array(['H+', 'H+', 'He++', 'He++'])  # The order is important and it should be based on Amitis.inp file
sim_ppc = [24, 24, 11, 11]  # Number of particles per species, based on Amitis.inp
sim_den = [38.0e6, 76.0e6, 1.0e6, 2.0e6]   # [/m^3]
sim_vel = [400.e3, 700.0e3, 400.e3, 700.0e3]  # [km/s]
species_mass = np.array([1.0, 1.0, 4.0, 4.0])  # [amu] proton1, proton2, alpha1, alpha2
species_charge = np.array([1.0, 1.0, 2.0, 2.0])  # [e] proton1, proton2, alpha1, alpha2

sim_dx = 75.e3  # simulation cell size based on Amitis.inp
sim_dy = 75.e3  # simulation cell size based on Amitis.inp
sim_dz = 75.e3  # simulation cell size based on Amitis.inp
sim_robs = 2440.e3  # obstacle radius based on Amitis.inp

nlat = 180
nlon = 360

select_R = 2480.e3  # the radius of a sphere + 1/2 grid cell above the surface for particle selection

output_folder = f"/Users/danywaller/Projects/mercury/extreme/surface_precipitation/"

# cases = ["RPN_Base", "RPS_Base", "CPN_Base", "CPS_Base"]
cases = ["RPN_HNHV", "RPS_HNHV", "CPN_HNHV", "CPS_HNHV"]

# FOR HNHV - DOUBLE CHECK ONLY ONE IS TRUE!!!!
transient = False  # 280-300 s
post_transient = True  # 330-350 s
new_state = False  # 680-700 s

stats_cases_all = []

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
    den_cm3 = n_shell_map.copy()  # [cm^-3] shell volume density
    vr = v_r_map.copy()  # [km/s]
    flux = flux_cm.copy()  # [cm^-2 s^-1]
    mass_flux = mass_flux_map.copy()  # [amu cm^-2 s^-1]
    energy_flux = energy_flux_map.copy()  # [eV cm^-2 s^-1]

    vr_abs = np.abs(vr)  # [km/s]
    flux_abs = np.abs(flux)  # [cm^-2 s^-1]
    mass_flux_abs = np.abs(mass_flux)  # [amu cm^-2 s^-1]
    energy_flux_abs = np.abs(energy_flux)  # [eV cm^-2 s^-1]

    stats_case = flux_utils.compute_flux_statistics(flux_abs, lat_centers, lon_centers, R_M=2440.0e3, case_name=case.split("_")[0])

    stats_cases_all.append(stats_case)

if "Base" in cases[0]:
    outcsv = os.path.join(output_folder, 'pre-transient_flux_comparison.csv')
    outtex = os.path.join(output_folder, 'pre-transient_flux_comparison.tex')
elif "HNHV" in cases[0] and transient and not post_transient and not new_state:
    outcsv = os.path.join(output_folder, 'transient_flux_comparison.csv')
    outtex = os.path.join(output_folder, 'transient_flux_comparison.tex')
elif "HNHV" in cases[0] and post_transient and not new_state and not transient:
    outcsv = os.path.join(output_folder, 'post-transient_flux_comparison.csv')
    outtex = os.path.join(output_folder, 'post-transient_flux_comparison.tex')
elif "HNHV" in cases[0] and new_state and not post_transient and not transient:
    outcsv = os.path.join(output_folder, 'newstate_flux_comparison.csv')
    outtex = os.path.join(output_folder, 'newstate_flux_comparison.tex')

df = flux_utils.create_comparison_table(stats_cases_all, output_csv=outcsv, output_latex=outtex)