#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage.measure import marching_cubes
from scipy.spatial import cKDTree

# ----------------------------
# USER SETTINGS
# ----------------------------
nc_file = "/Users/danywaller/Projects/mercury/extreme/magnetopause_3D_timeseries/CPN_HNHV/CPN_HNHV_mp_mask_timeseries.nc"
out_dir = "/Users/danywaller/Projects/mercury/extreme/magnetopause_3D_timeseries/CPN_HNHV/mp_mesh_analysis/"
os.makedirs(out_dir, exist_ok=True)

INTERP_FACTOR = 2        # refine grid
N_THETA = 61
N_PHI = 121
RMAX = 4.0

# Magnetic offset
Z_OFFSET_RM = 484.0 / 2440.0   # 0.1984 RM

# ----------------------------
# LOAD DATA
# ----------------------------
ds = xr.open_dataset(nc_file)

mask_4d = ds["mp_mask"].values
times = ds["time"].values
x = ds["x"].values
y = ds["y"].values
z = ds["z"].values

dx = np.mean(np.diff(x))
dy = np.mean(np.diff(y))
dz = np.mean(np.diff(z))

# ----------------------------
# DAYSIDE ANGULAR GRID
# ----------------------------
theta = np.linspace(1e-6, np.pi-1e-6, N_THETA)
phi = np.linspace(-np.pi/2+1e-6, np.pi/2-1e-6, N_PHI)
TH, PH = np.meshgrid(theta, phi, indexing="ij")

lat_deg = 90.0 - np.degrees(theta)
lon_deg = np.degrees(phi)

sx = np.sin(TH)*np.cos(PH)
sy = np.sin(TH)*np.sin(PH)
sz = np.cos(TH)

# storage
equator_geo_series = []
equator_mag_series = []

# ----------------------------
# LOOP OVER TIME
# ----------------------------
for it, tsec in enumerate(times):

    print(f"Processing {it+1}/{len(times)}  t={tsec:.2f}s")

    mask = mask_4d[it].astype(float)

    # ----------------------------
    # INTERPOLATE MASK
    # ----------------------------
    mask_interp = zoom(mask, INTERP_FACTOR, order=1)

    dx_i = dx / INTERP_FACTOR
    dy_i = dy / INTERP_FACTOR
    dz_i = dz / INTERP_FACTOR

    # ----------------------------
    # MARCHING CUBES
    # ----------------------------
    verts, faces, _, _ = marching_cubes(
        mask_interp,
        level=0.5,
        spacing=(dx_i, dy_i, dz_i)
    )

    verts[:,0] += x.min()
    verts[:,1] += y.min()
    verts[:,2] += z.min()

    # radial distance
    r_mesh = np.linalg.norm(verts, axis=1)

    # build KDTree for fast nearest lookup
    tree = cKDTree(verts)

    # ----------------------------
    # SAMPLE SURFACE ON ANGULAR GRID
    # ----------------------------
    r_map = np.full(TH.shape, np.nan)

    for i in range(N_THETA):
        for j in range(N_PHI):

            ray = np.array([sx[i,j], sy[i,j], sz[i,j]])
            pts = ray[None,:] * np.linspace(1.0, RMAX, 200)[:,None]

            dist, idx_nn = tree.query(pts)
            k = np.argmin(dist)
            if dist[k] < 0.1:
                r_map[i,j] = np.linalg.norm(pts[k])

    standoff = r_map - 1.0

    # ----------------------------
    # LONGITUDE-INTEGRATED DAYSIDE Δr
    # ----------------------------
    lon_integrated = np.nanmean(standoff, axis=1)

    # Geographic equator (lat=0)
    idx_geo = np.argmin(np.abs(lat_deg - 0.0))
    equator_geo_series.append(lon_integrated[idx_geo])

    # Magnetic equator latitude
    # solve cos(theta)=z_offset/r  using subsolar r
    r_subsolar = r_map[np.argmin(np.abs(lat_deg-0.0)), np.argmin(np.abs(lon_deg-0.0))]
    if np.isfinite(r_subsolar):
        cos_theta_mag = Z_OFFSET_RM / r_subsolar
        cos_theta_mag = np.clip(cos_theta_mag, -1, 1)
        theta_mag = np.arccos(cos_theta_mag)
        lat_mag = 90.0 - np.degrees(theta_mag)
        idx_mag = np.argmin(np.abs(lat_deg - lat_mag))
        equator_mag_series.append(lon_integrated[idx_mag])
    else:
        equator_mag_series.append(np.nan)

# convert to arrays
equator_geo_series = np.array(equator_geo_series)
equator_mag_series = np.array(equator_mag_series)

# ----------------------------
# PLOT 2x1 TIMESERIES
# ----------------------------
fig, axes = plt.subplots(2,1, figsize=(8,8), sharex=True)

axes[0].plot(times, equator_geo_series)
axes[0].set_ylabel("Δr (RM)")
axes[0].set_title("Longitude-Integrated Dayside Δr\nGeographic Equator (lat=0°)")

axes[1].plot(times, equator_mag_series)
axes[1].set_ylabel("Δr (RM)")
axes[1].set_xlabel("Time (s)")
axes[1].set_title("Longitude-Integrated Dayside Δr\nMagnetic Equator (dipole offset +Z)")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "equator_standoff_timeseries.png"))
plt.close()

print("Done.")