#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines

debug = False
make_plots = True

# --------------------------
# SETTINGS
# --------------------------
case = "RPS"
input_folder = f"/Volumes/data_backup/extreme_base/{case}_Base/05/out/"
ncfile = os.path.join(input_folder, f"Amitis_{case}_Base_115000.nc")

output_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology/{case}_Base/"
os.makedirs(output_folder, exist_ok=True)

# Planet parameters
RM = 2440.0          # Mercury radius [km]
dx = 75.0            # grid spacing [km]
trace_length = 15 * RM
surface_tol = 75.0

# Seed settings
n_lat = 75
n_lon = n_lat*2
max_steps = 100000
h_step = 50.0

# --------------------------
# Numba functions
# --------------------------
@njit
def trilinear_interp(x_grid, y_grid, z_grid, B, xi, yi, zi):
    i = np.searchsorted(x_grid, xi) - 1
    j = np.searchsorted(y_grid, yi) - 1
    k = np.searchsorted(z_grid, zi) - 1
    i = max(0, min(i, len(x_grid)-2))
    j = max(0, min(j, len(y_grid)-2))
    k = max(0, min(k, len(z_grid)-2))
    xd = (xi - x_grid[i]) / (x_grid[i+1]-x_grid[i])
    yd = (yi - y_grid[j]) / (y_grid[j+1]-y_grid[j])
    zd = (zi - z_grid[k]) / (z_grid[k+1]-z_grid[k])
    c000 = B[i,j,k]
    c100 = B[i+1,j,k]
    c010 = B[i,j+1,k]
    c001 = B[i,j,k+1]
    c101 = B[i+1,j,k+1]
    c011 = B[i,j+1,k+1]
    c110 = B[i+1,j+1,k]
    c111 = B[i+1,j+1,k+1]
    c00 = c000*(1-xd)+c100*xd
    c01 = c001*(1-xd)+c101*xd
    c10 = c010*(1-xd)+c110*xd
    c11 = c011*(1-xd)+c111*xd
    c0 = c00*(1-yd)+c10*yd
    c1 = c01*(1-yd)+c11*yd
    return c0*(1-zd)+c1*zd

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
def cartesian_to_latlon(r):
    rmag = np.linalg.norm(r)
    lat = np.degrees(np.arcsin(r[2]/rmag))
    lon = np.degrees(np.arctan2(r[1], r[0]))
    return lat, lon

@njit
def rk45_step(f, r, h, Bx, By, Bz, x_grid, y_grid, z_grid):
    k1 = f(r, Bx, By, Bz, x_grid, y_grid, z_grid)
    k2 = f(r + h*k1*0.25, Bx, By, Bz, x_grid, y_grid, z_grid)
    k3 = f(r + h*(3*k1+9*k2)/32, Bx, By, Bz, x_grid, y_grid, z_grid)
    k4 = f(r + h*(1932*k1 - 7200*k2 + 7296*k3)/2197, Bx, By, Bz, x_grid, y_grid, z_grid)
    k5 = f(r + h*(439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104), Bx, By, Bz, x_grid, y_grid, z_grid)
    k6 = f(r + h*(-8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40), Bx, By, Bz, x_grid, y_grid, z_grid)
    r_next = r + h*(16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55)
    return r_next

@njit
def trace_field_line_rk(seed, Bx, By, Bz, x_grid, y_grid, z_grid, RM, max_steps=5000, h=50.0, surface_tol=-1.0):
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
        if np.linalg.norm(r) <= RM + surface_tol:
            return traj[:i+1], exit_y_boundary
        if (r[0]<x_grid[0] or r[0]>x_grid[-1] or
            r[2]<z_grid[0] or r[2]>z_grid[-1]):
            return traj[:i+1], exit_y_boundary
        if r[1]<y_grid[0] or r[1]>y_grid[-1]:
            exit_y_boundary = True
            return traj[:i+1], exit_y_boundary
    return traj, exit_y_boundary

@njit
def classify(traj_fwd, traj_bwd, RM, exit_fwd_y=False, exit_bwd_y=False):
    # check if last point in trajectory is equal to or less than Mercury radius
    hit_fwd = np.linalg.norm(traj_fwd[-1]) <= RM
    hit_bwd = np.linalg.norm(traj_bwd[-1]) <= RM
    if exit_fwd_y or exit_bwd_y:
        # line ran into domain boundary - terminate as unknown
        return "TBD"
    if hit_fwd and hit_bwd:
        # line originated from and returned to planet surface - closed
        return "closed"
    elif hit_fwd or hit_bwd:
        # line connected to planet surface at only ONE end - open
        return "open"
    else:
        return "TBD"

# --------------------------
# CREATE SURFACE SEEDS
# --------------------------
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

# --------------------------
# FUNCTION TO MERGE PLANES FOR ONE FILE
# --------------------------
def load_full_domain(ncfile, xlim=[None, None], ylim=[None, None], zlim=[None, None]):
    """
    Load full 3D magnetic field from a single NetCDF file.

    Can optionally be cropped by xlim, ylim, zlim, defaults are [None, None]

    Returns:
        x, y, z : 1D coordinate arrays
        Bx, By, Bz : 3D arrays with shape (Nz, Ny, Nx)
    """
    xmin, xmax = xlim
    ymin, ymax = ylim
    zmin, zmax = zlim

    ds = xr.open_dataset(ncfile)

    ds = ds.assign_coords(Nx=ds["Nx"].astype(float), Nz=ds["Nz"].astype(float))

    x = ds["Nx"].values
    y = ds["Ny"].values
    z = ds["Nz"].values

    t = ds["time"].values

    if len(t) > 1:
        print("You have more than one time value in a single file wtf")

    if debug:
        print(min(x), max(x))
        print(min(y), max(y))
        print(min(z), max(z))

    # Extract fields (drop time dimension)
    Bx = (ds["Bx"].isel(time=0).values + ds["Bdx"].isel(time=0).values)
    By = (ds["By"].isel(time=0).values + ds["Bdy"].isel(time=0).values)
    Bz = (ds["Bz"].isel(time=0).values + ds["Bdz"].isel(time=0).values)

    if debug:
        # Print shapes before transpose
        print("Before transpose:")
        for var in ["Bx", "By", "Bz"]:
            print(f"{var}: dims={ds[var].dims}, shape={ds[var].shape}")

    #  Transpose: Nz, Ny, Nx --> Nx, Ny, Nz
    Bx_plane = np.transpose(Bx, (2, 1, 0))
    By_plane = np.transpose(By, (2, 1, 0))
    Bz_plane = np.transpose(Bz, (2, 1, 0))

    if debug:
        # Print shapes after transpose
        print("\nAfter transpose:")
        print("Bx shape:", Bx_plane.shape)
        print("By shape:", By_plane.shape)
        print("Bz shape:", Bz_plane.shape)
        print("\n")

    nx_mask = (x >= xmin if xmin is not None else x >= min(x)) & (x <= xmax if xmax is not None else x <= max(x))
    ny_mask = (y >= ymin if ymin is not None else y >= min(y)) & (y <= ymax if ymax is not None else y <= max(y))
    nz_mask = (z >= zmin if zmin is not None else z >= min(z)) & (z <= zmax if zmax is not None else z <= max(z))

    plot_x = x[nx_mask]
    plot_y = y[ny_mask]
    plot_z = z[nz_mask]

    Bx_masked = Bx_plane[nx_mask, :, :][:, ny_mask, :][:, :, nz_mask]
    By_masked = By_plane[nx_mask, :, :][:, ny_mask, :][:, :, nz_mask]
    Bz_masked = Bz_plane[nx_mask, :, :][:, ny_mask, :][:, :, nz_mask]

    ds.close()
    return plot_x, plot_y, plot_z, Bx_masked, By_masked, Bz_masked

start = datetime.now()
print(f"Started processing {ncfile} at {str(start)}")
x, y, z, Bx, By, Bz = load_full_domain(ncfile)
step = int(ncfile.split("/")[-1].split("_")[3].split(".")[0])

# Compute footprints
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

df_csv = os.path.join(output_folder,f"{case}_{step}_footprints_class.csv")
df_planet.to_csv(df_csv,index=False)
csvsave = datetime.now()
print(f"Saved footprints to {df_csv} at {str(csvsave)}")

if make_plots:
    # --------------------------
    # FIELD LINE PLOT (X-Z)
    # --------------------------
    colors = {"closed": "blue", "open": "red"}
    lines_by_topo = {"closed": [], "open": []}

    for seed in seeds:
        traj_fwd, exit_fwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=h_step)
        traj_bwd, exit_bwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=-h_step)
        topo = classify(traj_fwd, traj_bwd, RM, exit_fwd_y, exit_bwd_y)
        if topo not in ["TBD"]:
            lines_by_topo[topo].append(traj_fwd[:, [0,2]])
            lines_by_topo[topo].append(traj_bwd[:, [0,2]])

    fig, ax = plt.subplots(figsize=(7,7))
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(RM*np.cos(theta), RM*np.sin(theta), "k", lw=2)
    for topo, segments in lines_by_topo.items():
        if segments:
            lc = LineCollection(segments, colors=colors[topo], linewidths=0.8, alpha=0.5)
            ax.add_collection(lc)
    ax.add_patch(plt.Circle((0,0), RM, edgecolor='black', facecolor="black", alpha=0.5, linewidth=2))
    legend_handles = [mlines.Line2D([],[],color="blue",label="Closed"),
                      mlines.Line2D([],[],color="red",label="Open")]
    ax.legend(handles=legend_handles, loc="upper right")

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Z [km]")
    ax.set_aspect("equal")
    ax.set_title(f"{case} Magnetic Field-Line Topology, t = {step*0.002} s")
    ax.set_xlim(-10*RM, 10*RM)
    ax.set_ylim(-10*RM, 10*RM)
    plt.tight_layout()
    output_topo = os.path.join(output_folder,"3D_topology/")
    os.makedirs(output_topo, exist_ok=True)
    plt.savefig(os.path.join(output_topo,f"{case}_field_topology_{step}.png"), dpi=150, bbox_inches="tight")
    print("Saved:\t", os.path.join(output_topo,f"{case}_field_topology_{step}.png"))
    plt.close()

    # --------------------------
    # FOOTPRINT PLOT (Hammer)
    # --------------------------
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='hammer')

    for topo, color in [("closed", "blue"), ("open", "red")]:
        subset = df_planet[df_planet['classification'] == topo]
        if subset.empty:
            continue
        # Convert to radians
        lon_rad = np.radians(subset['longitude_deg'].values)
        lat_rad = np.radians(subset['latitude_deg'].values)
        # For this projection, longitude must be shifted: -pi to pi
        lon_rad = np.where(lon_rad > np.pi, lon_rad - 2 * np.pi, lon_rad)
        ax.scatter(lon_rad, lat_rad, s=10, color=color, label=topo, alpha=0.7)

    # Longitude ticks (-170 to 170 every n °)
    lon_ticks_deg = np.arange(-120, 121, 60)
    lon_ticks_rad = np.deg2rad(lon_ticks_deg)

    # Latitude ticks (-90 to 90 every n °)
    lat_ticks_deg = np.arange(-60, 61, 30)
    lat_ticks_rad = np.deg2rad(lat_ticks_deg)

    # Apply to the current axis
    ax.set_xticks(lon_ticks_rad)
    ax.set_yticks(lat_ticks_rad)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{case} Magnetic Footprints, t = {step * 0.002} s")
    ax.grid(True)
    ax.legend(loc='lower left')

    plt.tight_layout()
    output_ftpt = os.path.join(output_folder, "3D_footprints/")
    os.makedirs(output_ftpt, exist_ok=True)
    plt.savefig(os.path.join(output_ftpt, f"{case}_field_footprints_{step}.png"), dpi=150,
                bbox_inches="tight")
    print("Saved:\t", os.path.join(output_ftpt, f"{case}_field_footprints_{step}.png"))
    plt.close()
