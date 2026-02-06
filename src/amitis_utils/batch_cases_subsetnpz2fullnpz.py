#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime
from pathlib import Path
import os
import numpy as np

# base cases: CPN_Base RPN_Base CPS_Base RPS_Base
# HNHV cases: CPN_HNHV RPN_HNHV CPS_HNHV RPS_HNHV
cases = ["CPN_HNHV", "CPS_HNHV", "RPN_HNHV", "RPS_HNHV"]
# cases = ["CPN_Base", "RPN_Base", "CPS_Base", "RPS_Base"]

dt = 0.002  # simulation dt defined in Amitis.inp [seconds]

# DOUBLE CHECK ONLY ONE IS TRUE!!!!
transient = False
post_transient = True
new_state = False

for case in cases:
    print(f"Processing {case}")

    # take 20 seconds
    if "Base" in case:
        folder = Path(f"/Volumes/data_backup/mercury/extreme/{case}/05/subset/")
        outdir = f"/Volumes/data_backup/mercury/extreme/{case}/05/particles/"
        sim_steps = list(range(105000, 115000 + 1, 1000))
    elif "HNHV" in case:
        if transient and not post_transient and not new_state:
            folder = Path(f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/02/subset/")
            outdir = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/02/particles/"
            sim_steps = range(140000, 150000 + 1, 1000)  # transient
        elif post_transient and not transient and not new_state:
            folder = Path(f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/03/subset/")
            outdir = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/03/particles/"
            sim_steps = range(165000, 175000 + 1, 1000)  # post-transient
        elif new_state and not post_transient and not transient:
            folder = Path(f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/10/subset/")
            outdir = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/10/particles/"
            sim_steps = range(340000, 350000 + 1, 1000)  # end of simulation
        else:
            raise ValueError("Too many flags! Set only one of transient, post_transient, or new_state to True")
    else:
        raise ValueError("Unrecognized case! Are you using one of Base or HNHV?")

    os.makedirs(outdir, exist_ok=True)

    # Initialize cached coordinates
    x = y = z = None

    start_time = datetime.now()
    for sim_step in sim_steps:
        prefix = f"Subset_{case}_{sim_step}"

        # Load all NPZ chunks with memory mapping
        npz_files = sorted(folder.glob(f"{prefix}*.npz"))
        loaded = [np.load(f, mmap_mode="r") for f in npz_files]

        # Compute global domain extents
        xmin = min(d["xmin"][0] for d in loaded)
        xmax = max(d["xmax"][0] for d in loaded)
        ymin = min(d["ymin"][0] for d in loaded)
        ymax = max(d["ymax"][0] for d in loaded)
        zmin = min(d["zmin"][0] for d in loaded)
        zmax = max(d["zmax"][0] for d in loaded)

        # Use chunk size from first file
        nx_c = int(loaded[0]["nx"][0])
        ny_c = int(loaded[0]["ny"][0])
        nz_c = int(loaded[0]["nz"][0])

        # Number of grid points in full domain
        coordx_max = max(d["coordx"][0] for d in loaded)
        coordy_max = max(d["coordy"][0] for d in loaded)
        coordz_max = max(d["coordz"][0] for d in loaded)

        NX = (coordx_max + 1) * nx_c
        NY = (coordy_max + 1) * ny_c
        NZ = (coordz_max + 1) * nz_c

        # Full domain coordinates (cached)
        if x is None:
            x = np.linspace(xmin, xmax, NX)
            y = np.linspace(ymin, ymax, NY)
            z = np.linspace(zmin, zmax, NZ)

        # Preallocate combined particle arrays
        counts = [d["rx"].size for d in loaded]
        n_particles = sum(counts)

        rx = np.empty(n_particles, dtype=np.float32)
        ry = np.empty(n_particles, dtype=np.float32)
        rz = np.empty(n_particles, dtype=np.float32)
        vx = np.empty(n_particles, dtype=np.float32)
        vy = np.empty(n_particles, dtype=np.float32)
        vz = np.empty(n_particles, dtype=np.float32)
        sid = np.empty(n_particles, dtype=np.int32)

        offset = 0
        for d, n in zip(loaded, counts):
            rx[offset:offset+n] = d["rx"]
            ry[offset:offset+n] = d["ry"]
            rz[offset:offset+n] = d["rz"]
            vx[offset:offset+n] = d["vx"]
            vy[offset:offset+n] = d["vy"]
            vz[offset:offset+n] = d["vz"]
            sid[offset:offset+n] = d["sid"]
            offset += n

        # print("Total particles:", n_particles)

        # Time coordinate
        time_value = sim_step * dt
        time = np.array([time_value], dtype=np.float32)

        # Save concatenated output to NPZ
        outfile_npz = os.path.join(outdir, f"{prefix}_full_domain.npz")

        np.savez_compressed(
            outfile_npz,
            # particle positions
            rx=rx,
            ry=ry,
            rz=rz,
            # particle velocities
            vx=vx,
            vy=vy,
            vz=vz,
            # species IDs
            sid=sid,
            # time
            time=time,
            # full domain coordinates
            x=x,
            y=y,
            z=z,
            # grid info
            NX=NX,
            NY=NY,
            NZ=NZ,
            nx_c=nx_c,
            ny_c=ny_c,
            nz_c=nz_c,
            # metadata
            n_particles=n_particles,
            description="Particle positions and velocities in full 3D domain (points)"
        )

        print(f"NPZ saved: {outfile_npz}")

        # Clear memory
        del rx, ry, rz, vx, vy, vz, sid, time
        for d in loaded:
            d.close()  # Close memory-mapped npz
        del loaded

    end_time = datetime.now()
    print(f"Finished {case} in {(end_time-start_time).total_seconds()} seconds\n")
