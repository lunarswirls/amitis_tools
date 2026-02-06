#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports:
from datetime import datetime
import os, glob
import numpy as np
from pathlib import Path
from pyamitis.amitis_particle import *

# base cases: CPN_Base RPN_Base CPS_Base RPS_Base
# HNHV cases: CPN_HNHV RPN_HNHV CPS_HNHV RPS_HNHV
# cases = ["CPN_Base", "RPN_Base", "CPS_Base", "RPS_Base"]
cases = ["RPN_HNHV", "RPS_HNHV", "CPN_HNHV", "CPS_HNHV"]

# DOUBLE CHECK ONLY ONE IS TRUE!!!!
transient = False
post_transient = True
new_state = False

for case in cases:
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

    select_R = 2480.e3

    sub_filepath = main_path + 'particles/'
    sub_filename = f'Subset_{case}'

    all_particles_directory = main_path + 'precipitation/'
    os.makedirs(all_particles_directory, exist_ok=True)

    all_particles_filename = all_particles_directory + f"{case}_all_particles_at_surface.npz"

    subset_filelist = np.array(sorted(glob.glob(sub_filepath + sub_filename + "*.npz")))
    subset_filelist = np.unique([f[:-9] for f in subset_filelist])

    prx, pry, prz = [], [], []
    pvx, pvy, pvz = [], [], []
    psid = []

    file_counter = 0

    start_time = datetime.now()

    # Combine all particle files
    for f in subset_filelist:
        stem = Path(f).stem
        sim_step = stem.split("_")[3]
        print(f"---------- {sim_step} ----------")

        obj = amitis_particle(sub_filepath, sub_filename, int(sim_step))
        obj.load_particle_data(None)

        mask = obj.rx**2 + obj.ry**2 + obj.rz**2 <= select_R**2

        prx.append(obj.rx[mask])
        pry.append(obj.ry[mask])
        prz.append(obj.rz[mask])

        pvx.append(obj.vx[mask])
        pvy.append(obj.vy[mask])
        pvz.append(obj.vz[mask])

        psid.append(obj.sid[mask])
        file_counter += 1

    # Concatenate all arrays
    prx_all = np.concatenate(prx)
    pry_all = np.concatenate(pry)
    prz_all = np.concatenate(prz)
    pvx_all = np.concatenate(pvx)
    pvy_all = np.concatenate(pvy)
    pvz_all = np.concatenate(pvz)
    psid_all = np.concatenate(psid)

    # Print unique species IDs
    unique_psid = np.unique(psid_all)
    print("\n" + "="*60)
    print("PARTICLE SPECIES SUMMARY")
    print("="*60)
    print(f"Unique species IDs: {unique_psid}")
    for sid in unique_psid:
        count = np.sum(psid_all == sid)
        print(f"  Species {sid}: {count:,} particles ({100*count/len(psid_all):.1f}%)")
    print(f"Total particles: {len(psid_all):,}")
    print("="*60 + "\n")

    # Save to file
    np.savez(
        all_particles_filename,
        prx=prx_all,
        pry=pry_all,
        prz=prz_all,
        pvx=pvx_all,
        pvy=pvy_all,
        pvz=pvz_all,
        psid=psid_all,
        num_files=file_counter,
        selected_radius=select_R
    )

    print("Wrote:", all_particles_filename)

    end_time = datetime.now()
    print(f"Finished {case} in {(end_time - start_time).total_seconds()} seconds\n")