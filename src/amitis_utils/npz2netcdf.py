#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np
import xarray as xr

# -----------------------------
# Settings
# -----------------------------
folder = Path("/Volumes/data_backup/mercury/extreme/CPN_Base/05/subset/")
outdir = "/Volumes/data_backup/mercury/extreme/CPN_Base/05/particles/"
os.makedirs(outdir, exist_ok=True)

sim_step = "112000"
prefix = f"Subset_CPN_Base_{sim_step}"

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
# Create xarray Dataset
# -----------------------------
ds = xr.Dataset(
    data_vars={
        "rx": ("particle", rx),
        "ry": ("particle", ry),
        "rz": ("particle", rz),
        "vx": ("particle", vx),
        "vy": ("particle", vy),
        "vz": ("particle", vz),
        "sid": ("particle", sid),
    },
    coords={
        "time": ("time", time, {"units": "s"}),
        "x": ("x", x, {"units": "m"}),
        "y": ("y", y, {"units": "m"}),
        "z": ("z", z, {"units": "m"}),
    },
    attrs={
        "description": "Particle positions and velocities in full 3D domain (points)",
        "NX": NX,
        "NY": NY,
        "NZ": NZ,
        "chunk_size": f"(nx={nx_c}, ny={ny_c}, nz={nz_c})",
        "n_particles": n_particles
    }
)

# -----------------------------
# Save to NetCDF
# -----------------------------
outfile = os.path.join(outdir, f"Subset_CPN_Base_{sim_step}_full_domain.nc")
ds.to_netcdf(
    outfile,
    format="NETCDF4",
    encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
)

print(f"NetCDF saved: {outfile}")

