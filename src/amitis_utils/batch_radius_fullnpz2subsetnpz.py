#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import numpy as np

case = "CPN"
mode = "HNHV"
sim_steps = list(range(275000, 300000 + 1, 1000))
# sim_steps = list(range(193000, 350000 + 1, 1000))
# sim_steps = list(range(325000, 339000 + 1, 1000))
dt = 0.002  # simulation dt defined in Amitis.inp [seconds]
select_R = 2480.e3  # shell within which you want to keep particles [m]

outdir = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/precipitation_timeseries/"
os.makedirs(outdir, exist_ok=True)

def build_file_path(pfix: str) -> list:
    # pick folder based on simulation timestep
    sstep = int(pfix.split("_")[-1])
    if sstep <= 115000:
        input_folder = f"/Volumes/data_backup/mercury/extreme/{case}_Base/05/subset/"
    elif 115000 < sstep <= 140000:
        input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/01/subset/"
    elif 140000 < sstep <= 150000:
        input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/02/subset/"
    elif 150000 < sstep <= 170000:
        input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/03/subset/"
    elif 170000 < sstep <= 195000:
        input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/04/subset/"
    elif 195000 <= sstep <= 220000:
        input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/05/subset/"
    elif 225000 <= sstep <= 250000:
        input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/06/subset/"
    elif 250000 < sstep <= 275000:
        input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/07/subset/"
    elif 275000 < sstep <= 300000:
        input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/08/subset/"
    elif 300000 < sstep <= 325000:
        input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/09/subset/"
    elif 325000 < sstep <= 350000:
        input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/10/subset/"
    else:
        raise ValueError(f"Unknown simulation step: {sstep}\n\tCheck if time {sstep*dt:.3f} exists in simulation")

    # load all NPZ chunks with memory mapping
    npz_files = sorted(Path(input_folder).glob(f"{prefix}*.npz"))
    lded = [np.load(f, mmap_mode="r") for f in npz_files]
    print(f"Loaded {len(lded)} files")
    return lded


# Initialize cached coordinates
x = y = z = None

for sim_step in sim_steps:
    prefix = f"Subset_{case}_*_" + "%06d" % sim_step

    loaded = build_file_path(prefix)

    # play along nicely if no files
    if len(loaded) == 0:
        continue

    tsec = sim_step * dt
    lab = f"t={tsec:.3f} s"
    print(f"Processing timestep: {lab}")

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

    print("Total particles:", n_particles)

    # Time coordinate
    time_value = sim_step * dt
    time = np.array([time_value], dtype=np.float32)

    # Save concatenated output to NPZ
    outfile_npz = os.path.join(outdir, f"{case}_{mode}_{"%06d" % sim_step}_precipitation_shell.npz")

    mask = rx ** 2 + ry ** 2 + rz ** 2 <= select_R ** 2

    prx = rx[mask]
    pry = ry[mask]
    prz = rz[mask]

    pvx = vx[mask]
    pvy = vy[mask]
    pvz = vz[mask]

    psid = sid[mask]

    # Print unique species IDs
    unique_psid = np.unique(psid)
    print("\n" + "=" * 60)
    print("DOWNSELECTED PARTICLE SPECIES SUMMARY")
    print("=" * 60)
    print(f"Unique species IDs: {unique_psid}")
    for sid in unique_psid:
        count = np.sum(psid == sid)
        print(f"  Species {sid}: {count:,} particles ({100 * count / len(psid):.1f}%)")
    print(f"Downselected particles: {len(psid):,}")

    np.savez_compressed(
        outfile_npz,
        # particle positions
        rx=prx,
        ry=pry,
        rz=prz,
        # particle velocities
        vx=pvx,
        vy=pvy,
        vz=pvz,
        # species IDs
        sid=psid,
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
        n_particles=len(psid),
        selected_radius=select_R,
        description="Particle positions, velocities, and species IDs in 3D domain at 1 timestep"
    )

    print(f"Done. Wrote to: {outfile_npz}")
    print("=" * 60 + "\n")

    # Clear memory
    del rx, ry, rz, vx, vy, vz, sid, time
    for d in loaded:
        d.close()  # Close memory-mapped npz
    del loaded
