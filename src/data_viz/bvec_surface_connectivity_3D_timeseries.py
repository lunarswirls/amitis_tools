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

debug = False

# --------------------------
# SETTINGS
# --------------------------
case = "RPS"
input_folder_xy = f"/Volumes/data_backup/extreme_base/{case}_Base/plane_product/all_xy/"
input_folder_xz = f"/Volumes/data_backup/extreme_base/{case}_Base/plane_product/all_xz/"
input_folder_yz = f"/Volumes/data_backup/extreme_base/{case}_Base/plane_product/all_yz/"
output_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology/{case}_Base/"
os.makedirs(output_folder, exist_ok=True)

# Planet parameters
RM = 2440.0          # Mercury radius [km]
dx = 75.0            # grid spacing [km]
trace_length = 50 * RM
surface_tol = dx

# Seed settings
n_lat = 60
n_lon = 120
max_steps = 1000
h_step = 50.0

# Files
N_files = 10
all_files = sorted([f for f in os.listdir(input_folder_xz) if f.endswith("_xz_comp.nc")])
last_files = all_files[-N_files:]
print(f"Processing last {len(last_files)} files: {last_files}")

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
def load_merged_field(file_basename):
    """
    Load XY, XZ, YZ planes and merge to cover maximum extent.
    """
    file_xy = os.path.join(input_folder_xy, file_basename.replace("_xz", "_xy").replace("_yz","_xy"))
    file_xz = os.path.join(input_folder_xz, file_basename)
    file_yz = os.path.join(input_folder_yz, file_basename.replace("_xz", "_yz"))

    ds_xy = xr.open_dataset(file_xy)
    ds_xz = xr.open_dataset(file_xz)
    ds_yz = xr.open_dataset(file_yz)

    # Merge coordinates
    x_grid = np.unique(np.concatenate([ds_xy["Nx"].values, ds_xz["Nx"].values, ds_yz["Nx"].values]))
    y_grid = np.unique(np.concatenate([ds_xy["Ny"].values, ds_xz["Ny"].values, ds_yz["Ny"].values]))
    z_grid = np.unique(np.concatenate([ds_xy["Nz"].values, ds_xz["Nz"].values, ds_yz["Nz"].values]))

    Bx_full = np.zeros((len(x_grid), len(y_grid), len(z_grid)))
    By_full = np.zeros_like(Bx_full)
    Bz_full = np.zeros_like(Bx_full)

    def insert_plane(ds):
        # Load original arrays
        Bx_orig = ds["Bx_tot"].isel(time=0).values
        By_orig = ds["By_tot"].isel(time=0).values
        Bz_orig = ds["Bz_tot"].isel(time=0).values

        if debug:
            # Print shapes before transpose
            print("Before transpose:")
            for var in ["Bx_tot", "By_tot", "Bz_tot"]:
                print(f"{var}: dims={ds[var].dims}, shape={ds[var].shape}")

        #  Transpose: Nz, Ny, Nx --> Nx, Ny, Nz
        Bx_plane = np.transpose(Bx_orig, (2, 1, 0))
        By_plane = np.transpose(By_orig, (2, 1, 0))
        Bz_plane = np.transpose(Bz_orig, (2, 1, 0))

        if debug:
            # Print shapes after transpose
            print("\nAfter transpose:")
            print("Bx_plane shape:", Bx_plane.shape)
            print("By_plane shape:", By_plane.shape)
            print("Bz_plane shape:", Bz_plane.shape)
            print("\n")

        xi = np.searchsorted(x_grid, ds["Nx"].values)
        yi = np.searchsorted(y_grid, ds["Ny"].values)
        zi = np.searchsorted(z_grid, ds["Nz"].values)
        Bx_full[xi[0]:xi[0]+Bx_plane.shape[0],
                yi[0]:yi[0]+Bx_plane.shape[1],
                zi[0]:zi[0]+Bx_plane.shape[2]] = Bx_plane
        By_full[xi[0]:xi[0]+By_plane.shape[0],
                yi[0]:yi[0]+By_plane.shape[1],
                zi[0]:zi[0]+By_plane.shape[2]] = By_plane
        Bz_full[xi[0]:xi[0]+Bz_plane.shape[0],
                yi[0]:yi[0]+Bz_plane.shape[1],
                zi[0]:zi[0]+Bz_plane.shape[2]] = Bz_plane

    insert_plane(ds_xy)
    insert_plane(ds_xz)
    insert_plane(ds_yz)

    ds_xy.close()
    ds_xz.close()
    ds_yz.close()
    return x_grid, y_grid, z_grid, Bx_full, By_full, Bz_full

# --------------------------
# LOOP OVER FILES
# --------------------------
all_footprints = []

for ncfile in last_files:
    print(f"Processing {ncfile}")
    x, y, z, Bx, By, Bz = load_merged_field(ncfile)

    if debug:
        print("Bx shape:", Bx.shape)
        print("By shape:", By.shape)
        print("Bz shape:", Bz.shape)

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

    all_footprints.append(df_planet)

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
    step = int(ncfile.split("_")[3])
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Z [km]")
    ax.set_aspect("equal")
    ax.set_title(f"{case} Magnetic Field-Line Topology, t = {step*0.002} s")
    ax.set_xlim(-5*RM, 5*RM)
    ax.set_ylim(-5*RM, 5*RM)
    plt.tight_layout()
    output_topo = os.path.join(output_folder,"topology/")
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
    output_ftpt = os.path.join(output_folder, "footprints/")
    os.makedirs(output_ftpt, exist_ok=True)
    plt.savefig(os.path.join(output_ftpt, f"{case}_field_footprints_{step}.png"), dpi=150,
                bbox_inches="tight")
    print("Saved:\t", os.path.join(output_ftpt, f"{case}_field_footprints_{step}.png"))
    plt.close()

# --------------------------
# AGGREGATE ALL FOOTPRINTS
# --------------------------
df_all = pd.concat(all_footprints, ignore_index=True)
df_all["lat_round"] = df_all["latitude_deg"].round(3)
df_all["lon_round"] = df_all["longitude_deg"].round(3)
median_classification = df_all.groupby(["lat_round","lon_round"])["classification"]\
    .agg(lambda x: x.value_counts().idxmax()).reset_index()
median_classification.rename(columns={"lat_round":"latitude_deg",
                                      "lon_round":"longitude_deg",
                                      "classification":"median_classification"}, inplace=True)
median_csv = os.path.join(output_folder,f"{case}_last_{N_files}_footprints_median_class.csv")
median_classification.to_csv(median_csv,index=False)
print(f"Saved median classification footprints to {median_csv}")
for topo in ["closed","open"]:
    n = (median_classification["median_classification"]==topo).sum()
    print(f"{topo.capitalize()} median count: {n}")
