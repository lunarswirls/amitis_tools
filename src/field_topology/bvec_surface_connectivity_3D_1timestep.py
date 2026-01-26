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
make_line_plot = False
make_foot_plot = True

# --------------------------
# SETTINGS
# --------------------------
case = "CPN_Base_largerxdomain_smallergridsize"
input_folder = f"/Users/danywaller/Projects/mercury/extreme/{case}/out/"
if "larger" in case:
    fname = case.split("_")[0] + "_" + case.split("_")[1]
else:
    fname = case
ncfile = os.path.join(input_folder, f"Amitis_{fname}_115000.nc")

output_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology/{case}/"
os.makedirs(output_folder, exist_ok=True)

# Planet parameters
RM = 2440.0

dx = 75.0            # grid spacing [km]
trace_length = 15 * RM
surface_tol = 75.0

# Seed settings
n_lat = 75
n_lon = n_lat*2
max_steps = 100000
h_step = 25.0

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
lines_by_topo = {"closed": [], "open": []}

for seed in seeds:
    traj_fwd, exit_fwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=h_step)
    traj_bwd, exit_bwd_y = trace_field_line_rk(seed, Bx, By, Bz, x, y, z, RM, max_steps=max_steps, h=-h_step)
    topo = classify(traj_fwd, traj_bwd, RM + dx, exit_fwd_y, exit_bwd_y)
    if topo not in ["TBD"]:
        lines_by_topo[topo].append(traj_fwd[:, [0, 2]])
        lines_by_topo[topo].append(traj_bwd[:, [0, 2]])

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

if make_line_plot:
    # --------------------------
    # FIELD LINE PLOT (X-Z)
    # --------------------------
    colors = {"closed": "blue", "open": "red"}

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
    ax.set_title(f"{case.replace("_", " ")} (Larger X Domain) Magnetic Field-Line Topology, t = {step*0.002} s")
    ax.set_xlim(-20*RM, 20*RM)
    ax.set_ylim(-20*RM, 20*RM)
    plt.tight_layout()
    output_topo = os.path.join(output_folder,"3D_topology/")
    os.makedirs(output_topo, exist_ok=True)
    plt.savefig(os.path.join(output_topo,f"{case}_field_topology_{step}.png"), dpi=150, bbox_inches="tight")
    print("Saved:\t", os.path.join(output_topo,f"{case}_field_topology_{step}.png"))
    plt.close()

if make_foot_plot:
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
    if "larger" in case:
        tstring = f"{fname.replace("_", " ")} (Larger X Domain) Magnetic Footprints, t = {step * 0.002} s"
    else:
        tstring = f"{fname.replace("_", " ")} Magnetic Footprints, t = {step * 0.002} s"
    ax.set_title(tstring)
    ax.grid(True)
    ax.legend(loc='lower left')

    plt.tight_layout()
    output_ftpt = os.path.join(output_folder, "3D_footprints/")
    os.makedirs(output_ftpt, exist_ok=True)
    plt.savefig(os.path.join(output_ftpt, f"{case}_field_footprints_{step}.png"), dpi=150,
                bbox_inches="tight")
    print("Saved:\t", os.path.join(output_ftpt, f"{case}_field_footprints_{step}.png"))
    plt.close()

# --------------------------
# CONNECTEDNESS + OCB METRICS
# --------------------------
seed_lon_spacing = 360.0 / n_lon
bin_width = max(5.0, 2.5 * seed_lon_spacing)
n_bins = int(360.0 / bin_width)

lon_bins = np.linspace(-180, 180, n_bins)

df_ocb = ocb_curve_df(df_planet, lon_bins)
ocb_csv = os.path.join(output_folder, f"{case}_{step}_ocb_curve.csv")
df_ocb.to_csv(ocb_csv, index=False)
print("\nSaved OCB curve to:", ocb_csv)

# --------------------------
# Summarize OCB metrics
# --------------------------
df_ocb_summary = summarize_ocb(df_ocb, planet_radius_km=RM)

# --------------------------
# Compute connectedness index
# --------------------------
C_index = compute_connectedness_index(df_planet)

# --------------------------
# Combine into a single metrics DataFrame
# --------------------------
df_metrics = df_ocb_summary.copy()

# Add metadata
df_metrics.insert(0, "case", case)
df_metrics.insert(1, "step", step)
df_metrics.insert(2, "time_s", step * 0.002)

# Add connectedness
df_metrics["connectedness_index"] = C_index

# --------------------------
# Save full metrics CSV
# --------------------------
metrics_csv = os.path.join(output_folder, f"{case}_{step}_connectedness_metrics.csv")
df_metrics.to_csv(metrics_csv, index=False)

print("Saved connectedness metrics to:", metrics_csv)
