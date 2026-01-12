#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import xarray as xr
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ======================
# USER PARAMETERS
# ======================
case = "CPS"

input_folder = f"/Volumes/data_backup/extreme_base/{case}_Base/plane_product/all_xz/"
ncfile = os.path.join(input_folder, f"Amitis_{case}_Base_115000_xz_comp.nc")
output_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology/{case}_Base/"
os.makedirs(output_folder, exist_ok=True)

plot_lines = True  # True to plot field lines
RM = 2440.0  # Mercury radius [km]
dx = 75.0  # grid spacing
trace_length = 8 * RM
surface_tol = dx

# seed resolution
n_lat = 60
n_lon = 120

max_steps = 5000  # max RK steps
h_step = 50.0  # integration step size [km]

# ======================
# LOAD DATA
# ======================
ds = xr.open_dataset(ncfile)
x = ds["Nx"].values
y = ds["Ny"].values
z = ds["Nz"].values

Bx = np.transpose(ds["Bx_tot"].isel(time=0).values, (2, 1, 0))  # Nx,Ny,Nz
By = np.transpose(ds["By_tot"].isel(time=0).values, (2, 1, 0))
Bz = np.transpose(ds["Bz_tot"].isel(time=0).values, (2, 1, 0))

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
zmin, zmax = z.min(), z.max()


# ======================
# NUMBA TRILINEAR INTERPOLATION
# ======================
@njit
def trilinear_interp(x_grid, y_grid, z_grid, B, xi, yi, zi):
    i = np.searchsorted(x_grid, xi) - 1
    j = np.searchsorted(y_grid, yi) - 1
    k = np.searchsorted(z_grid, zi) - 1

    i = max(0, min(i, len(x_grid) - 2))
    j = max(0, min(j, len(y_grid) - 2))
    k = max(0, min(k, len(z_grid) - 2))

    xd = (xi - x_grid[i]) / (x_grid[i + 1] - x_grid[i])
    yd = (yi - y_grid[j]) / (y_grid[j + 1] - y_grid[j])
    zd = (zi - z_grid[k]) / (z_grid[k + 1] - z_grid[k])

    c000 = B[i, j, k]
    c100 = B[i + 1, j, k]
    c010 = B[i, j + 1, k]
    c001 = B[i, j, k + 1]
    c101 = B[i + 1, j, k + 1]
    c011 = B[i, j + 1, k + 1]
    c110 = B[i + 1, j + 1, k]
    c111 = B[i + 1, j + 1, k + 1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    return c0 * (1 - zd) + c1 * zd


@njit
def get_B(r, Bx, By, Bz, x_grid, y_grid, z_grid):
    bx = trilinear_interp(x_grid, y_grid, z_grid, Bx, r[0], r[1], r[2])
    by = trilinear_interp(x_grid, y_grid, z_grid, By, r[0], r[1], r[2])
    bz = trilinear_interp(x_grid, y_grid, z_grid, Bz, r[0], r[1], r[2])
    B = np.array([bx, by, bz])
    norm = np.linalg.norm(B)
    if norm == 0.0:
        return np.zeros(3)
    return B / norm


@njit
def cartesian_to_latlon_numba(r):
    rmag = np.linalg.norm(r)
    lat = np.degrees(np.arcsin(r[2] / rmag))
    lon = np.degrees(np.arctan2(r[1], r[0]))
    return lat, lon


# ======================
# RK4-5 FIELD LINE INTEGRATOR
# ======================
@njit
def rk45_step(f, r, h, Bx, By, Bz, x_grid, y_grid, z_grid):
    """
    Perform a single Runge-Kutta-Fehlberg (RK4-5) step.

    Parameters
    ----------
    f : function
        Function returning the normalized magnetic field vector at a position r.
        Signature: f(r, Bx, By, Bz, x_grid, y_grid, z_grid) -> np.array([Bx, By, Bz])
    r : ndarray, shape (3,)
        Current position vector [x, y, z] in km.
    h : float
        Step size in km (can be negative for backward tracing).
    Bx, By, Bz : ndarray
        3D arrays of magnetic field components.
    x_grid, y_grid, z_grid : ndarray
        Grid coordinates corresponding to Bx, By, Bz.

    Returns
    -------
    r_next : ndarray, shape (3,)
        Updated position after a single RK4-5 step.
    """
    # RK4-5 coefficients
    k1 = f(r, Bx, By, Bz, x_grid, y_grid, z_grid)
    k2 = f(r + h * k1 * 0.25, Bx, By, Bz, x_grid, y_grid, z_grid)
    k3 = f(r + h * (3*k1 + 9*k2)/32, Bx, By, Bz, x_grid, y_grid, z_grid)
    k4 = f(r + h * (1932*k1 - 7200*k2 + 7296*k3)/2197, Bx, By, Bz, x_grid, y_grid, z_grid)
    k5 = f(r + h * (439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104), Bx, By, Bz, x_grid, y_grid, z_grid)
    k6 = f(r + h * (-8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40),
           Bx, By, Bz, x_grid, y_grid, z_grid)

    # Weighted combination for RK4-5
    r_next = r + h * (16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55)
    return r_next


@njit
def trace_field_line_rk(seed, Bx, By, Bz, x_grid, y_grid, z_grid, RM, max_steps=50000, h=50.0, surface_tol=-1.0):
    """
    Trace a magnetic field line from a seed point using RK4-5 integration.

    Parameters
    ----------
    seed : ndarray, shape (3,)
        Starting position [x, y, z] in km (usually on Mercury's surface).
    Bx, By, Bz : ndarray
        3D arrays of magnetic field components.
    x_grid, y_grid, z_grid : ndarray
        Corresponding 1D arrays of grid coordinates.
    RM : float
        Planet radius in km (Mercury).
    max_steps : int
        Maximum number of integration steps.
    h : float
        Step size in km (can be negative for backward tracing).
    surface_tol : float
        Distance tolerance from planet surface for considering a "hit".

    Returns
    -------
    traj : ndarray, shape (N,3)
        Array of traced positions along the field line. N <= max_steps.
        Tracing stops if the line hits Mercury, leaves the domain, or the field goes to zero.
    """
    traj = np.empty((max_steps, 3), dtype=np.float64)
    traj[0] = seed
    r = seed.copy()

    for i in range(1, max_steps):
        # Compute normalized magnetic field vector at current position
        B = get_B(r, Bx, By, Bz, x_grid, y_grid, z_grid)
        if np.all(B == 0.0):
            # Field vanished; stop integration
            return traj[:i]

        # Advance using RK4-5 step
        r_next = rk45_step(get_B, r, h, Bx, By, Bz, x_grid, y_grid, z_grid)
        traj[i] = r_next
        r = r_next

        # Check if field line hits Mercury's surface
        if np.linalg.norm(r) <= RM + surface_tol:
            return traj[:i+1]

        # Check if field line leaves domain boundaries
        if (r[0] < x_grid[0] or r[0] > x_grid[-1] or
            r[1] < y_grid[0] or r[1] > y_grid[-1] or
            r[2] < z_grid[0] or r[2] > z_grid[-1]):
            return traj[:i+1]

    # Maximum steps reached
    return traj

# ======================
# CLASSIFY FIELD LINES
# ======================
def classify(traj_fwd, traj_bwd, RM):
    hit_fwd = np.linalg.norm(traj_fwd[-1]) <= RM
    hit_bwd = np.linalg.norm(traj_bwd[-1]) <= RM
    if hit_fwd and hit_bwd:
        return "closed"
    elif hit_fwd or hit_bwd:
        return "open"
    else:
        return "solar_wind"


# ======================
# CREATE SEED POINTS ON SURFACE
# ======================
lats_surface = np.linspace(-90, 90, n_lat)
lons_surface = np.linspace(-180, 180, n_lon)

seeds = []
for lat in lats_surface:
    for lon in lons_surface:
        phi = np.radians(lat)
        theta = np.radians(lon)
        x_s = RM * np.cos(phi) * np.cos(theta)
        y_s = RM * np.cos(phi) * np.sin(theta)
        z_s = RM * np.sin(phi)
        seeds.append(np.array([x_s, y_s, z_s]))
seeds = np.array(seeds)

# ======================
# TRACE FIELD LINES
# ======================
footprints = []
footprints_class = []

print(f"Tracing {len(seeds)} field lines...")
for seed in seeds:
    traj_fwd = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=h_step)
    traj_bwd = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=-h_step)
    topo = classify(traj_fwd, traj_bwd, RM)

    for traj in [traj_fwd, traj_bwd]:
        r_end = traj[-1]
        if np.linalg.norm(r_end) <= RM + surface_tol:
            lat, lon = cartesian_to_latlon_numba(r_end)
            footprints.append((lat, lon))
            footprints_class.append(topo)

# ======================
# SAVE TO DATAFRAME
# ======================
df_planet = pd.DataFrame({
    "latitude_deg": [lat for lat, lon in footprints],
    "longitude_deg": [lon for lat, lon in footprints],
    "classification": footprints_class
})

csv_file = os.path.join(output_folder, f"Amitis_{case}_Base_115000_xz_comp_footprints.csv")
df_planet.to_csv(csv_file, index=False)
print(f"Saved {len(df_planet)} footprints to {csv_file}")

# ======================
# PLOT FIELD LINES (X-Z PLANE)
# ======================
if plot_lines:
    colors = {"closed": "blue", "open": "red", "solar_wind": "gray"}
    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw Mercury surface
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(RM * np.cos(theta), RM * np.sin(theta), "k", lw=2)

    # Plot traced field lines
    for seed in seeds:
        traj_fwd = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=h_step)
        traj_bwd = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=-h_step)
        topo = classify(traj_fwd, traj_bwd, RM)

        # X-Z projection
        ax.plot(traj_fwd[:, 0], traj_fwd[:, 2], color=colors.get(topo, "gray"), alpha=0.5)
        ax.plot(traj_bwd[:, 0], traj_bwd[:, 2], color=colors.get(topo, "gray"), alpha=0.5)

    # Legend
    legend_handles = [mlines.Line2D([], [], color=c, label=k) for k, c in colors.items() if k in ["closed", "open"]]
    ax.legend(handles=legend_handles, loc="upper right")

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Z [km]")
    ax.set_aspect("equal")
    ax.set_title("Mercury Magnetic Field-Line Topology (X-Z Plane)")
    ax.set_xlim(-5 * RM, 5 * RM)
    ax.set_ylim(-5 * RM, 5 * RM)
    plt.tight_layout()
    plt.show()

# ======================
# PLOT FOOTPRINTS
# ======================
fig, ax = plt.subplots(figsize=(9, 4))
for topo, color in [("closed", "blue"), ("open", "red")]:
    subset = df_planet[df_planet['classification'] == topo]
    if subset.empty: continue
    ax.scatter(subset['longitude_deg'], subset['latitude_deg'], s=10, color=color, label=topo, alpha=0.7)

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")
ax.set_title("Mercury Magnetic Footprints")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()


# ======================
# LATITUDE STATISTICS
# ======================
def stats(arr):
    return np.min(arr), np.max(arr), np.mean(arr)


for topo in ["closed", "open"]:
    subset = df_planet[df_planet['classification'] == topo]
    if subset.empty:
        print(f"No {topo} footprints found.")
        continue

    lats = subset['latitude_deg'].values
    north = lats[lats >= 0]
    south = lats[lats < 0]

    if len(north) > 0:
        n_min, n_max, n_mean = stats(north)
        print(f"North hemisphere ({topo}): min={n_min:.2f}°, max={n_max:.2f}°, mean={n_mean:.2f}°")
    else:
        print(f"No north hemisphere {topo} footprints.")

    if len(south) > 0:
        s_min, s_max, s_mean = stats(south)
        print(f"South hemisphere ({topo}): min={s_min:.2f}°, max={s_max:.2f}°, mean={s_mean:.2f}°")
    else:
        print(f"No south hemisphere {topo} footprints.")
