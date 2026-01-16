#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines
from src.field_topology.topology_utils import *

debug = False
make_plots = True

# SETTINGS
case = "RPS"
input_folder = f"/Volumes/data_backup/extreme_base/{case}_Base/plane_product/object/"
ncfile = os.path.join(input_folder, f"Amitis_{case}_Base_115000_xz_comp.nc")

output_folder = f"/Users/danywaller/Projects/mercury/extreme/jfield_topology/{case}_Base/"
os.makedirs(output_folder, exist_ok=True)

# Planet parameters
RM = 2440.0          # Mercury radius [km]
RC = 2080.0          # conductive layer depth [km]

if case in ["RPS", "RPN"]:
    plot_depth = RM
elif case in ["CPS", "CPN"]:
    plot_depth = RC
else:
    raise ValueError("not a real case ID, pick one of RPS, RPN, CPS, CPN")

dx = 75.0            # grid spacing [km]
trace_length = 15 * RM
surface_tol = 75.0

# Seed settings
n_lat = 75
n_lon = n_lat*2
max_steps = 100000
h_step = 50.0

# CREATE SURFACE SEEDS
lats_surface = np.linspace(-90, 90, n_lat)
lons_surface = np.linspace(-180, 180, n_lon)
seeds = []
for lat in lats_surface:
    for lon in lons_surface:
        phi = np.radians(lat)
        theta = np.radians(lon)
        x_s = plot_depth*np.cos(phi)*np.cos(theta)
        y_s = plot_depth*np.cos(phi)*np.sin(theta)
        z_s = plot_depth*np.sin(phi)
        seeds.append(np.array([x_s, y_s, z_s]))
seeds = np.array(seeds)


def load_jvector(ncfile, xlim=[None, None], ylim=[None, None], zlim=[None, None]):
    """
    Load 3D current field from a NetCDF file that covers the full domain.

    Can optionally be cropped by xlim, ylim, zlim, defaults are [None, None]

    Returns:
        x, y, z : 1D coordinate arrays
        Jx, Jy, Jz : 3D arrays with shape (Nx, Ny, Nz)
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
    Jx = ds["Jx"].isel(time=0).values
    Jy = ds["Jy"].isel(time=0).values
    Jz = ds["Jz"].isel(time=0).values

    if debug:
        # Print shapes before transpose
        print("Before transpose:")
        for var in ["Jx", "Jy", "Jz"]:
            print(f"{var}: dims={ds[var].dims}, shape={ds[var].shape}")

    #  Transpose: Nz, Ny, Nx --> Nx, Ny, Nz
    Jx_plane = np.transpose(Jx, (2, 1, 0))
    Jy_plane = np.transpose(Jy, (2, 1, 0))
    Jz_plane = np.transpose(Jz, (2, 1, 0))

    if debug:
        # Print shapes after transpose
        print("\nAfter transpose:")
        print("Jx shape:", Jx_plane.shape)
        print("Jy shape:", Jy_plane.shape)
        print("Jz shape:", Jz_plane.shape)
        print("\n")

    nx_mask = (x >= xmin if xmin is not None else x >= min(x)) & (x <= xmax if xmax is not None else x <= max(x))
    ny_mask = (y >= ymin if ymin is not None else y >= min(y)) & (y <= ymax if ymax is not None else y <= max(y))
    nz_mask = (z >= zmin if zmin is not None else z >= min(z)) & (z <= zmax if zmax is not None else z <= max(z))

    plot_x = x[nx_mask]
    plot_y = y[ny_mask]
    plot_z = z[nz_mask]

    Jx_masked = Jx_plane[nx_mask, :, :][:, ny_mask, :][:, :, nz_mask]
    Jy_masked = Jy_plane[nx_mask, :, :][:, ny_mask, :][:, :, nz_mask]
    Jz_masked = Jz_plane[nx_mask, :, :][:, ny_mask, :][:, :, nz_mask]

    ds.close()
    return plot_x, plot_y, plot_z, Jx_masked, Jy_masked, Jz_masked


x, y, z, Jx, Jy, Jz = load_jvector(ncfile)
step = int(ncfile.split("/")[-1].split("_")[3].split(".")[0])

# Compute footprints
footprints = []
footprints_class = []
lines_by_topo = {"closed": [], "open": []}

for seed in seeds:
    traj_fwd, exit_fwd_y = trace_field_line_rk(seed, Jx, Jy, Jz, x, y, z, plot_depth, max_steps=max_steps, h=h_step)
    traj_bwd, exit_bwd_y = trace_field_line_rk(seed, Jx, Jy, Jz, x, y, z, plot_depth, max_steps=max_steps, h=-h_step)
    topo = classify(traj_fwd, traj_bwd, plot_depth, exit_fwd_y, exit_bwd_y)
    if topo in ["closed", "open"]:
        lines_by_topo[topo].append(traj_fwd)
        lines_by_topo[topo].append(traj_bwd)

    for traj in [traj_fwd, traj_bwd]:
        r_end = traj[-1]
        if np.linalg.norm(r_end) <= plot_depth + surface_tol:
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
    fig, ax = plt.subplots(figsize=(7,7))

    # Draw Mercury surface
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(plot_depth*np.cos(theta), plot_depth*np.sin(theta), "k", lw=2)

    # Plot traced field lines
    for topo, segments in lines_by_topo.items():
        if segments:
            lc = LineCollection(segments, colors=colors[topo], linewidths=0.8, alpha=0.5)
            ax.add_collection(lc)

    ax.add_patch(plt.Circle((0,0), RM, edgecolor='black', facecolor=None, alpha=1.0, linewidth=2))
    # Legend
    legend_handles = [mlines.Line2D([], [], color=c, label=k) for k, c in colors.items() if k in ["closed", "open"]]
    ax.legend(handles=legend_handles, loc="upper right")

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Z [km]")
    ax.set_aspect("equal")
    ax.set_title(f"{case} Current Field-Line Topology, t = {step*0.002} s")
    ax.set_xlim(-2*RM, 2*RM)
    ax.set_ylim(-2*RM, 2*RM)
    plt.tight_layout()
    output_topo = os.path.join(output_folder,"3D_topology/")
    os.makedirs(output_topo, exist_ok=True)
    plt.savefig(os.path.join(output_topo,f"{case}_jfield_topology_{step}.png"), dpi=150, bbox_inches="tight")
    print("Saved:\t", os.path.join(output_topo,f"{case}_jfield_topology_{step}.png"))
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
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title(f"{case} Current Footprints, t = {step * 0.002} s")
    ax.grid(True)
    ax.legend(loc='lower left')

    plt.tight_layout()
    output_ftpt = os.path.join(output_folder, "3D_footprints/")
    os.makedirs(output_ftpt, exist_ok=True)
    plt.savefig(os.path.join(output_ftpt, f"{case}_jfield_footprints_{step}.png"), dpi=150,
                bbox_inches="tight")
    print("Saved:\t", os.path.join(output_ftpt, f"{case}_jfield_footprints_{step}.png"))
    plt.close()
