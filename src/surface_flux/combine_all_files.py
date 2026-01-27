#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob
import numpy as np
from pathlib import Path
from pyamitis.amitis_particle import *

# ============================================================
# Configuration
# ============================================================

case = "RPS_Base"
main_path = f'/Volumes/data_backup/mercury/extreme/{case}/05/'

species = np.array(['H+', 'He++'])
sim_ppc = [24, 11]
sim_den = [38.0e6, 1.0e6]

sim_dx = sim_dy = sim_dz = 75.e3
sim_robs = 2440.e3
select_R = 2480.e3

dphi   = 2.0   # longitude bin [deg]
dtheta = 2.0   # latitude bin [deg]

sub_filepath = main_path + 'particles/'
sub_filename = f'Subset_{case}'

all_particles_directory = main_path + 'precipitation/'
os.makedirs(all_particles_directory, exist_ok=True)

all_particles_filename = all_particles_directory + "all_particles_at_surface.npz"
moments_filename = all_particles_directory + "moments"

# ============================================================
# Combine all particle files
# ============================================================

def combine_all_files():

    subset_filelist = np.array(sorted(glob.glob(sub_filepath + sub_filename + "*.npz")))
    subset_filelist = np.unique([f[:-9] for f in subset_filelist])

    prx, pry, prz = [], [], []
    pvx, pvy, pvz = [], [], []
    psid = []

    file_counter = 0

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

    np.savez(
        all_particles_filename,
        prx=np.concatenate(prx),
        pry=np.concatenate(pry),
        prz=np.concatenate(prz),
        pvx=np.concatenate(pvx),
        pvy=np.concatenate(pvy),
        pvz=np.concatenate(pvz),
        psid=np.concatenate(psid),
        num_files=file_counter,
        selected_radius=select_R
    )

    print("Wrote:", all_particles_filename)

combine_all_files()
