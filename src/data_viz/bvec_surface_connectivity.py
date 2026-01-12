#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import solve_ivp

# ======================
# USER PARAMETERS
# ======================
input_folder = f"/Users/danywaller/Projects/mercury/extreme/CPN_Base/fig_xz/"
ncfile = os.path.join(input_folder, f"Amitis_CPN_Base_115000_xz_comp.nc")

RM = 2440.0              # Mercury radius [km]
dx = 75.0                # grid spacing [km]
max_step = dx * 0.5      # integration step size [km]
trace_length = 10 * RM   # max integration length

# ======================
# LOAD DATA
# ======================
ds = xr.open_dataset(ncfile)

x = ds["Nx"].values
y = ds["Ny"].values
z = ds["Nz"].values

# 2D slice at y=0
Bx = ds["Bx_tot"].isel(time=0).isel(Ny=0).values
By = ds["By_tot"].isel(time=0).isel(Ny=0).values
Bz = ds["Bz_tot"].isel(time=0).isel(Ny=0).values

# Transpose to (Nx, Nz) for easier indexing
Bx = Bx.T
By = By.T
Bz = Bz.T

# ======================
# NUMBA INTERPOLATION
# ======================
@njit
def trilinear_2d_interp(x_grid, z_grid, B, xi, zi):
    i = np.searchsorted(x_grid, xi) - 1
    k = np.searchsorted(z_grid, zi) - 1
    i = max(0, min(i, len(x_grid)-2))
    k = max(0, min(k, len(z_grid)-2))

    xd = (xi - x_grid[i]) / (x_grid[i+1] - x_grid[i])
    zd = (zi - z_grid[k]) / (z_grid[k+1] - z_grid[k])

    c00 = B[i,k]*(1-xd) + B[i+1,k]*xd
    c01 = B[i,k+1]*(1-xd) + B[i+1,k+1]*xd

    return c00*(1-zd) + c01*zd

@njit
def get_B_2d(r, Bx, By, Bz, x_grid, z_grid):
    bx = trilinear_2d_interp(x_grid, z_grid, Bx, r[0], r[1])
    by = trilinear_2d_interp(x_grid, z_grid, By, r[0], r[1])
    bz = trilinear_2d_interp(x_grid, z_grid, Bz, r[0], r[1])
    B = np.array([bx, by, bz])
    norm = np.linalg.norm(B)
    if norm == 0:
        return np.zeros(2)
    return np.array([bx, bz]) / norm

# ======================
# MERCURY DETECTION
# ======================
@njit
def hits_mercury_2d(r, RM):
    x0, z0 = r
    return np.sqrt(x0**2 + z0**2) <= RM

@njit
def cartesian_to_lat_numba(r, RM):
    x0, z0 = r
    return np.degrees(np.arcsin(z0 / RM))

# ======================
# FIELD-LINE RHS
# ======================
def fieldline_rhs_2d(s, r):
    return get_B_2d(r, Bx, By, Bz, x, z)

# ======================
# TRACE FIELD LINE
# ======================
def trace_field_line_2d(seed):
    sol_fwd = solve_ivp(fieldline_rhs_2d, [0, trace_length], seed, max_step=max_step)
    sol_bwd = solve_ivp(fieldline_rhs_2d, [0, -trace_length], seed, max_step=max_step)
    return sol_fwd, sol_bwd

def endpoint(sol):
    return np.array([sol.y[0,-1], sol.y[1,-1]])

def classify_field_line(sol_fwd, sol_bwd):
    fwd_hit = hits_mercury_2d(endpoint(sol_fwd), RM)
    bwd_hit = hits_mercury_2d(endpoint(sol_bwd), RM)
    if fwd_hit and bwd_hit:
        return "closed"
    elif fwd_hit or bwd_hit:
        return "open"
    else:
        return "solar_wind"

# ======================
# SEEDS ON MERCURY SURFACE
# ======================
n_lat = 20
n_lon = 40
lats_surface = np.linspace(-90, 90, n_lat)
lons_surface = np.linspace(-180, 180, n_lon)

seeds = []
for lat in lats_surface:
    for lon in lons_surface:
        phi = np.radians(lat)
        theta = np.radians(lon)
        x_s = RM * np.cos(phi) * np.cos(theta)
        z_s = RM * np.sin(phi)
        seeds.append(np.array([x_s, z_s]))
seeds = np.array(seeds)

# ======================
# TRACE ALL FIELD LINES
# ======================
results = {"closed": [], "open": [], "solar_wind": []}
footprints = {"closed": [], "open": []}

print(f"Tracing {len(seeds)} field lines from Mercury surface...")

for seed in seeds:
    sol_fwd, sol_bwd = trace_field_line_2d(seed)
    topo = classify_field_line(sol_fwd, sol_bwd)
    results[topo].append((sol_fwd, sol_bwd))

    # Only store planet-connected footprints
    r_fwd = endpoint(sol_fwd)
    r_bwd = endpoint(sol_bwd)
    if hits_mercury_2d(r_fwd, RM):
        footprints[topo].append(r_fwd)
    if hits_mercury_2d(r_bwd, RM):
        footprints[topo].append(r_bwd)

print("Field lines classified:")
for k in results:
    print(f"{k}: {len(results[k])}")

# ======================
# PLOT FIELD LINES
# ======================
colors = {"closed": "blue", "open": "red", "solar_wind": "gray"}
fig, ax = plt.subplots(figsize=(7,7))
theta = np.linspace(0, 2*np.pi, 400)
ax.plot(RM*np.cos(theta), RM*np.sin(theta), "k", lw=2)

for topo in results:
    for sol_fwd, sol_bwd in results[topo]:
        ax.plot(sol_fwd.y[0], sol_fwd.y[1], color=colors[topo], alpha=0.6)
        ax.plot(sol_bwd.y[0], sol_bwd.y[1], color=colors[topo], alpha=0.6)

ax.set_xlabel("X [km]")
ax.set_ylabel("Z [km]")
ax.set_title("Mercury Magnetic Field-Line Topology (2D)")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# ======================
# PLOT FOOTPRINTS
# ======================
fig, ax = plt.subplots(figsize=(7,7))
for topo, color in [("closed", "blue"), ("open", "red")]:
    if len(footprints[topo]) == 0:
        continue
    pts = np.array(footprints[topo])
    ax.scatter(pts[:,0], pts[:,1], s=10, color=color, alpha=0.7, label=topo)

ax.plot(RM*np.cos(theta), RM*np.sin(theta), "k", lw=2)
ax.set_xlabel("X [km]")
ax.set_ylabel("Z [km]")
ax.set_title("Mercury Field-Line Footprints (2D)")
ax.set_aspect("equal")
ax.legend()
plt.tight_layout()
plt.show()

# ======================
# LATITUDE STATISTICS FOR CLOSED LINES
# ======================
closed_footprints = footprints["closed"]
latitudes = np.array([cartesian_to_lat_numba(r, RM) for r in closed_footprints])

north = latitudes[latitudes > 0]
south = latitudes[latitudes < 0]

print("=== Planet-connected Field Line Latitudes ===")
if len(north) > 0:
    print(f"North Hemisphere: min={north.min():.2f}°, max={north.max():.2f}°, mean={north.mean():.2f}°")
else:
    print("North Hemisphere: no points")

if len(south) > 0:
    print(f"South Hemisphere: min={south.min():.2f}°, max={south.max():.2f}°, mean={south.mean():.2f}°")
else:
    print("South Hemisphere: no points")
