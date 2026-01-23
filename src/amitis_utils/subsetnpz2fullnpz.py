#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np

# -----------------------------
# Settings
# -----------------------------
folder = Path("/Volumes/data_backup/mercury/extreme/RPN_Base/05/subset/")
outdir = "/Volumes/data_backup/mercury/extreme/RPN_Base/05/particles/"
os.makedirs(outdir, exist_ok=True)

sim_step = "108000"
prefix = f"Subset_RPN_Base_{sim_step}"

# -----------------------------
# Load all NPZ chunks
# -----------------------------
npz_files = sorted(folder.glob(f"{prefix}*.npz"))
loaded = [np.load(f) for f in npz_files]

# -----------------------------
# Compute global domain extents
# -----------------------------
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
NX = (max(d["coordx"][0] for d in loaded) + 1) * nx_c
NY = (max(d["coordy"][0] for d in loaded) + 1) * ny_c
NZ = (max(d["coordz"][0] for d in loaded) + 1) * nz_c

# Full domain coordinates
x = np.linspace(xmin, xmax, NX)
y = np.linspace(ymin, ymax, NY)
z = np.linspace(zmin, zmax, NZ)

# -----------------------------
# Combine particle arrays
# -----------------------------
rx = np.concatenate([d["rx"] for d in loaded])
ry = np.concatenate([d["ry"] for d in loaded])
rz = np.concatenate([d["rz"] for d in loaded])
vx = np.concatenate([d["vx"] for d in loaded])
vy = np.concatenate([d["vy"] for d in loaded])
vz = np.concatenate([d["vz"] for d in loaded])
sid = np.concatenate([d["sid"] for d in loaded]).astype(np.int32)

n_particles = rx.size
print("Total particles:", n_particles)

# -----------------------------
# Time coordinate
# -----------------------------
time_value = int(prefix.split("_")[-1]) * 0.002
time = np.array([time_value], dtype=np.float32)

# -----------------------------
# Save concatenated output to NPZ
# -----------------------------
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
