#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np

# Settings - CHANGE THESE!
case = "CPS_Base"
dt = 0.002
sim_step = "115000"

# Path definitions
folder = Path(f"/Volumes/data_backup/mercury/extreme/{case}/05/subset/")
outdir = f"/Volumes/data_backup/mercury/extreme/{case}/05/particles/"
os.makedirs(outdir, exist_ok=True)

# File prefix
prefix = f"Subset_{case}_{sim_step}"

# Initialize cached coordinates
x = y = z = None

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

print("Total particles:", n_particles)

# Time coordinate
time_value = float(sim_step) * dt
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