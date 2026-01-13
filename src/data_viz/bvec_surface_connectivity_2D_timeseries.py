#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import xarray as xr
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines

case = "RPS"
input_folder = f"/Volumes/data_backup/extreme_base/{case}_Base/plane_product/all_xz/"
output_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology/{case}_Base/"
os.makedirs(output_folder, exist_ok=True)

# Parameters
RM = 2440.0          # Mercury radius [km]
dx = 75.0            # grid spacing [km]
trace_length = 10 * RM
surface_tol = dx

n_lat = 60           # seeds along latitude
n_lon = 120          # seeds along longitude
max_steps = 1000     # RK max steps
h_step = 50.0        # RK step size [km]

# Number of last files to process
N_files = 10
all_files = sorted([f for f in os.listdir(input_folder) if f.endswith("_xz_comp.nc")])
last_files = all_files[-N_files:]
print(f"Processing last {len(last_files)} files: {last_files}")

# ======================
# INTERPOLATION & RK FUNCTIONS
# ======================
@njit
def trilinear_interp(x_grid, y_grid, z_grid, B, xi, yi, zi):
    """
    Perform trilinear interpolation of a scalar field B on a regular 3D grid.

    Parameters
    ----------
    x_grid, y_grid, z_grid : 1D ndarray
        Grid coordinates in x, y, z directions (monotonic).
    B : 3D ndarray
        Scalar field defined on the grid with shape (Nx, Ny, Nz).
    xi, yi, zi : float
        Target interpolation coordinates.

    Returns
    -------
    float
        Interpolated value of B at (xi, yi, zi).
    """

    # Find index of grid cell lower corner
    i = np.searchsorted(x_grid, xi) - 1
    j = np.searchsorted(y_grid, yi) - 1
    k = np.searchsorted(z_grid, zi) - 1

    # Clamp indices to valid interpolation range
    i = max(0, min(i, len(x_grid) - 2))
    j = max(0, min(j, len(y_grid) - 2))
    k = max(0, min(k, len(z_grid) - 2))

    # Normalized distances within grid cell
    xd = (xi - x_grid[i]) / (x_grid[i + 1] - x_grid[i])
    yd = (yi - y_grid[j]) / (y_grid[j + 1] - y_grid[j])
    zd = (zi - z_grid[k]) / (z_grid[k + 1] - z_grid[k])

    # Values at the 8 surrounding grid corners
    c000 = B[i,     j,     k]
    c100 = B[i + 1, j,     k]
    c010 = B[i,     j + 1, k]
    c001 = B[i,     j,     k + 1]
    c101 = B[i + 1, j,     k + 1]
    c011 = B[i,     j + 1, k + 1]
    c110 = B[i + 1, j + 1, k]
    c111 = B[i + 1, j + 1, k + 1]

    # Interpolate along x
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    # Interpolate along y
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    # Interpolate along z
    return c0 * (1 - zd) + c1 * zd


@njit
def get_B(r, Bx, By, Bz, x_grid, y_grid, z_grid):
    """
    Interpolate and normalize the magnetic field vector at position r.

    Parameters
    ----------
    r : ndarray, shape (3,)
        Cartesian position [x, y, z] in km.
    Bx, By, Bz : 3D ndarray
        Magnetic field components on the grid.
    x_grid, y_grid, z_grid : 1D ndarray
        Grid coordinates.

    Returns
    -------
    ndarray, shape (3,)
        Unit vector in the direction of the magnetic field.
        Returns zero vector if field magnitude is zero.
    """

    # Interpolate each field component
    bx = trilinear_interp(x_grid, y_grid, z_grid, Bx, r[0], r[1], r[2])
    by = trilinear_interp(x_grid, y_grid, z_grid, By, r[0], r[1], r[2])
    bz = trilinear_interp(x_grid, y_grid, z_grid, Bz, r[0], r[1], r[2])

    B = np.array([bx, by, bz])
    norm = np.linalg.norm(B)

    # Avoid division by zero
    if norm == 0.0:
        return np.zeros(3)

    return B / norm


@njit
def cartesian_to_latlon(r):
    """
    Convert Cartesian coordinates to planetocentric latitude and longitude.

    Parameters
    ----------
    r : ndarray, shape (3,)
        Cartesian position [x, y, z].

    Returns
    -------
    lat, lon : float
        Latitude and longitude in degrees.
    """

    rmag = np.linalg.norm(r)
    lat = np.degrees(np.arcsin(r[2] / rmag))
    lon = np.degrees(np.arctan2(r[1], r[0]))
    return lat, lon


@njit
def rk45_step(f, r, h, Bx, By, Bz, x_grid, y_grid, z_grid):
    """
    Perform a single Runge–Kutta–Fehlberg (RK4-5) integration step.

    Parameters
    ----------
    f : function
        Function returning normalized field direction at r.
    r : ndarray, shape (3,)
        Current position.
    h : float
        Step size (km). Negative for backward tracing.
    Bx, By, Bz : ndarray
        Magnetic field components.
    x_grid, y_grid, z_grid : ndarray
        Grid coordinates.

    Returns
    -------
    ndarray, shape (3,)
        Updated position after one RK step.
    """

    # Fehlberg RK coefficients
    k1 = f(r, Bx, By, Bz, x_grid, y_grid, z_grid)
    k2 = f(r + h * k1 * 0.25, Bx, By, Bz, x_grid, y_grid, z_grid)
    k3 = f(r + h * (3*k1 + 9*k2) / 32, Bx, By, Bz, x_grid, y_grid, z_grid)
    k4 = f(r + h * (1932*k1 - 7200*k2 + 7296*k3) / 2197, Bx, By, Bz, x_grid, y_grid, z_grid)
    k5 = f(r + h * (439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104),
           Bx, By, Bz, x_grid, y_grid, z_grid)
    k6 = f(r + h * (-8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40),
           Bx, By, Bz, x_grid, y_grid, z_grid)

    # 5th-order solution
    r_next = r + h * (
        16*k1/135 +
        6656*k3/12825 +
        28561*k4/56430 -
        9*k5/50 +
        2*k6/55
    )

    return r_next


@njit
def trace_field_line_rk(seed, Bx, By, Bz, x_grid, y_grid, z_grid, RM, max_steps=5000, h=50.0, surface_tol=-1.0):
    """
    Trace a magnetic field line using RK4-5 integration.

    Integration stops if:
    - the field magnitude goes to zero
    - the field line hits the planet surface
    - the field line leaves the simulation domain
    - max_steps is reached

    Parameters
    ----------
    seed : ndarray, shape (3,)
        Starting position.
    RM : float
        Planet radius (km).
    h : float
        Step size (km).

    Returns
    -------
    ndarray, shape (N,3)
        Field line trajectory.
    """

    traj = np.empty((max_steps, 3), dtype=np.float64)
    traj[0] = seed
    r = seed.copy()
    exit_y_boundary = False

    for i in range(1, max_steps):

        B = get_B(r, Bx, By, Bz, x_grid, y_grid, z_grid)

        if np.all(B == 0.0):
            return traj[:i], exit_y_boundary

        r_next = rk45_step(get_B, r, h, Bx, By, Bz, x_grid, y_grid, z_grid)
        traj[i] = r_next
        r = r_next

        # Planet surface hit
        if np.linalg.norm(r) <= RM + surface_tol:
            return traj[:i + 1], exit_y_boundary

        # Domain boundary exit
        if (r[0] < x_grid[0] or r[0] > x_grid[-1] or
            r[2] < z_grid[0] or r[2] > z_grid[-1]):
            return traj[:i + 1], exit_y_boundary

        # Y-boundary exit — mark flag and stop integration
        if r[1] < y_grid[0] or r[1] > y_grid[-1]:
            exit_y_boundary = True
            return traj[:i + 1], exit_y_boundary

    return traj, exit_y_boundary


def classify(traj_fwd, traj_bwd, RM, exit_fwd_y=False, exit_bwd_y=False):
    """
    Classify a magnetic field line topology.
    Skip 'open' if the line exited the y-boundary.

    Parameters
    ----------
    traj_fwd, traj_bwd : ndarray
        Forward and backward field line trajectories.
    RM : float
        Planet radius.

    Returns
    -------
    str
        'closed', 'open', or 'TBD'
    """
    hit_fwd = np.linalg.norm(traj_fwd[-1]) <= RM
    hit_bwd = np.linalg.norm(traj_bwd[-1]) <= RM

    # Skip 'open' if either direction exited via y-boundary
    if exit_fwd_y or exit_bwd_y:
        if hit_fwd and hit_bwd:
            return "closed"
        else:
            return "TBD"

    if hit_fwd and hit_bwd:
        return "closed"
    elif hit_fwd or hit_bwd:
        return "open"
    else:
        return "TBD"

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
        x_s = RM*np.cos(phi)*np.cos(theta)
        y_s = RM*np.cos(phi)*np.sin(theta)
        z_s = RM*np.sin(phi)
        seeds.append(np.array([x_s, y_s, z_s]))
seeds = np.array(seeds)

# ======================
# LOOP OVER FILES AND COMPUTE FOOTPRINTS
# ======================
all_footprints = []

for ncfile in last_files:
    print(f"Processing {ncfile}")
    ds = xr.open_dataset(os.path.join(input_folder, ncfile))
    x = ds["Nx"].values
    y = ds["Ny"].values
    z = ds["Nz"].values

    Bx = np.transpose(ds["Bx_tot"].isel(time=0).values, (2,1,0))
    By = np.transpose(ds["By_tot"].isel(time=0).values, (2,1,0))
    Bz = np.transpose(ds["Bz_tot"].isel(time=0).values, (2,1,0))

    footprints = []
    footprints_class = []

    for seed in seeds:
        traj_fwd, exit_fwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=h_step)
        traj_bwd, exit_bwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=-h_step)

        topo = classify(traj_fwd, traj_bwd, RM, exit_fwd_y, exit_bwd_y)

        for traj in [traj_fwd, traj_bwd]:
            r_end = traj[-1]
            if np.linalg.norm(r_end) <= RM + surface_tol:
                lat, lon = cartesian_to_latlon(r_end)
                footprints.append((lat, lon))
                footprints_class.append(topo)

    df_planet = pd.DataFrame({
        "latitude_deg": [lat for lat, lon in footprints],
        "longitude_deg": [lon for lat, lon in footprints],
        "classification": footprints_class
    })

    colors = {"closed": "blue", "open": "red"}

    lines_by_topo = {
        "closed": [],
        "open": [],
    }

    for seed in seeds:
        traj_fwd, exit_fwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z,
                                                   RM, max_steps=max_steps, h=h_step)
        traj_bwd, exit_bwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z,
                                                   RM, max_steps=max_steps, h=-h_step)

        topo = classify(traj_fwd, traj_bwd, RM, exit_fwd_y, exit_bwd_y)

        # When plotting, use the first element of the tuple (the trajectory array)
        if topo not in ["TBD"]:
            lines_by_topo[topo].append(traj_fwd[:, [0, 2]])
            lines_by_topo[topo].append(traj_bwd[:, [0, 2]])

    fig, ax = plt.subplots(figsize=(7, 7))

    # Mercury surface
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(RM * np.cos(theta), RM * np.sin(theta), "k", lw=2)

    for topo, segments in lines_by_topo.items():
        if not segments:
            continue

        lc = LineCollection(
            segments,
            colors=colors[topo],
            linewidths=0.8,
            alpha=0.5
        )
        ax.add_collection(lc)

    legend_handles = [
        mlines.Line2D([], [], color="blue", label="Closed"),
        mlines.Line2D([], [], color="red", label="Open"),
    ]

    ax.legend(handles=legend_handles, loc="upper right")

    circle = plt.Circle((0, 0), RM, edgecolor='black', facecolor="black", alpha=0.5, linewidth=2, )
    ax.add_patch(circle)

    step = int(ncfile.split("_")[3])

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Z [km]")
    ax.set_aspect("equal")
    ax.set_title(f"{case} Magnetic Field-Line Topology (X–Z Plane), t = {step * 0.002} seconds")
    ax.set_xlim(-5 * RM, 5 * RM)
    ax.set_ylim(-5 * RM, 5 * RM)

    plt.tight_layout()
    # --- Save ---
    output_topo = os.path.join(output_folder, "topology/")
    outfile_png = os.path.join(output_topo, f"{case}_field_topology_{step}.png")
    plt.tight_layout()
    plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
    print("Saved:\t", outfile_png)
    plt.close()

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
    ax.set_title(f"{case} Magnetic Footprints, t = {step * 0.002} seconds")
    ax.grid(True)
    ax.legend('lower right')
    plt.tight_layout()
    output_ftpt = os.path.join(output_folder, "footprints/")
    os.makedirs(output_ftpt, exist_ok=True)
    outfile_png = os.path.join(output_ftpt, f"{case}_field_footprints_{step}.png")
    plt.tight_layout()
    plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
    print("Saved:\t", outfile_png)
    plt.close()

    all_footprints.append(df_planet)

# ======================
# COMBINE ALL FOOTPRINTS AND COMPUTE MEDIAN CLASSIFICATION
# ======================
# Concatenate all footprints
df_all = pd.concat(all_footprints, ignore_index=True)

# Round lat/lon to seed resolution to group properly
df_all["lat_round"] = df_all["latitude_deg"].round(3)
df_all["lon_round"] = df_all["longitude_deg"].round(3)

# Group by rounded lat/lon
grouped = df_all.groupby(["lat_round", "lon_round"])

# Compute median classification as the most frequent classification at each location
median_classification = grouped["classification"].agg(lambda x: x.value_counts().idxmax()).reset_index()
median_classification.rename(columns={"lat_round": "latitude_deg", "lon_round": "longitude_deg",
                                      "classification": "median_classification"}, inplace=True)

# Save to CSV
median_csv = os.path.join(output_folder, f"{case}_last_{N_files}_footprints_median_class.csv")
median_classification.to_csv(median_csv, index=False)
print(f"Saved median classification footprints to {median_csv}")

# Optional: Print some summary
for topo in ["closed", "open"]:
    n = (median_classification["median_classification"] == topo).sum()
    print(f"{topo.capitalize()} median count: {n}")

