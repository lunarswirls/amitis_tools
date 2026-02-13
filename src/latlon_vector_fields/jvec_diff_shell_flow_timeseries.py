#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports:
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# SETTINGS
case1 = "RPN_HNHV"
case2 = "CPN_HNHV"

filter_jmag = True
jmag_min = 2  # nA/m^2

sim_end = True

# Planet parameters
RM = 2440.0  # km

# Shell limits
rmin = 1.0 * RM
rmax = 1.2 * RM

# Output folder for difference plots
output_folder = f"/Users/danywaller/Projects/mercury/extreme/jfield_lonlat/{case1}_minus_{case2}/"
os.makedirs(output_folder, exist_ok=True)

# Determine simulation steps (assuming both cases have same steps)
if sim_end:
    sim_steps = range(115000, 350000 + 1, 1000)
else:
    sim_steps = range(115000, 200000 + 1, 1000)


# ------------------------
# FUNCTION TO PROCESS ONE CASE
# ------------------------
def process_case(case, stp, r_min, r_max, filt_jmag=True, j_min=2):
    """
    Load and process a single case, returning interpolated J_theta and J_phi on regular grid
    """
    # Set input folder based on case
    input_folder = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/object/"
    ncfile = os.path.join(input_folder, f"Amitis_{case}_{stp}_xz_comp.nc")

    if not os.path.exists(ncfile):
        return None, None, None, None

    # Load J vector
    ds = xr.open_dataset(ncfile)
    x = ds["Nx"].values
    y = ds["Ny"].values
    z = ds["Nz"].values

    Jx = ds["Jx"].isel(time=0).values
    Jy = ds["Jy"].isel(time=0).values
    Jz = ds["Jz"].isel(time=0).values

    # Transpose Nz, Ny, Nx → Nx, Ny, Nz
    Jx = np.transpose(Jx, (2, 1, 0))
    Jy = np.transpose(Jy, (2, 1, 0))
    Jz = np.transpose(Jz, (2, 1, 0))
    ds.close()

    # Create 3D grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    # Select shell
    shell_mask = (R >= r_min) & (R <= r_max)
    X = X[shell_mask]
    Y = Y[shell_mask]
    Z = Z[shell_mask]
    Jx = Jx[shell_mask]
    Jy = Jy[shell_mask]
    Jz = Jz[shell_mask]

    # Spherical coordinates
    R_shell = R[shell_mask]
    lat = np.degrees(np.arcsin(Z / R_shell))
    lon = np.degrees(np.arctan2(Y, X))

    # Tangential vectors on shell
    er = np.vstack((X, Y, Z)).T
    er /= np.linalg.norm(er, axis=1)[:, None]

    etheta = np.vstack((-Z * X, -Z * Y, X ** 2 + Y ** 2)).T
    etheta /= np.linalg.norm(etheta, axis=1)[:, None]

    ephi = np.vstack((-Y, X, np.zeros_like(X))).T
    ephi /= np.linalg.norm(ephi, axis=1)[:, None]

    J = np.vstack((Jx, Jy, Jz)).T
    J_theta = np.sum(J * etheta, axis=1)
    J_phi = np.sum(J * ephi, axis=1)
    Jmag = np.sqrt(J_theta ** 2 + J_phi ** 2)

    if filt_jmag:
        # Filter currents
        mask_strong = Jmag > j_min
        lat = lat[mask_strong]
        lon = lon[mask_strong]
        J_theta = J_theta[mask_strong]
        J_phi = J_phi[mask_strong]

    # Regular grid
    lat_grid = np.linspace(-90, 90, 180)
    lon_grid = np.linspace(-180, 180, 360)
    Lon_grid, Lat_grid = np.meshgrid(lon_grid, lat_grid)

    # Interpolate scattered vectors onto the grid
    points = np.vstack([lon, lat]).T

    J_theta_grid = griddata(
        points=(lon, lat),
        values=J_theta,
        xi=(Lon_grid, Lat_grid),
        method='nearest')

    J_phi_grid = griddata(
        points=(lon, lat),
        values=J_phi,
        xi=(Lon_grid, Lat_grid),
        method='nearest')

    return lon_grid, lat_grid, J_theta_grid, J_phi_grid


# ------------------------
# FIRST PASS: CALCULATE GLOBAL COLOR LIMITS
# ------------------------
print("First pass: calculating global colormap limits...")
all_signed_diffs = []

for step in sim_steps:
    # Process both cases
    lon_grid1, lat_grid1, J_theta1, J_phi1 = process_case(
        case1, step, rmin, rmax, filter_jmag, jmag_min)

    lon_grid2, lat_grid2, J_theta2, J_phi2 = process_case(
        case2, step, rmin, rmax, filter_jmag, jmag_min)

    # Check if both loaded successfully
    if J_theta1 is None or J_theta2 is None:
        continue

    # Calculate magnitude for each case
    magnitude1 = np.sqrt(J_theta1 ** 2 + J_phi1 ** 2)
    magnitude2 = np.sqrt(J_theta2 ** 2 + J_phi2 ** 2)

    # Signed difference in magnitude
    signed_diff = magnitude1 - magnitude2
    all_signed_diffs.append(signed_diff)

# Calculate global limits
all_signed_diffs = np.concatenate([sd.flatten() for sd in all_signed_diffs])
vmax_global = np.percentile(np.abs(all_signed_diffs), 95)
print(f"Global limits: ±{vmax_global:.2f} nA/m²")

# ------------------------
# SECOND PASS: GENERATE PLOTS WITH CONSISTENT LIMITS
# ------------------------
print("\nSecond pass: generating plots...")
for step in sim_steps:
    print(f"Processing step {step}...")

    # Process both cases
    lon_grid1, lat_grid1, J_theta1, J_phi1 = process_case(
        case1, step, rmin, rmax, filter_jmag, jmag_min)

    lon_grid2, lat_grid2, J_theta2, J_phi2 = process_case(
        case2, step, rmin, rmax, filter_jmag, jmag_min)

    # Check if both loaded successfully
    if J_theta1 is None or J_theta2 is None:
        print(f"  Skipping step {step} - data not available for both cases")
        continue

    # Calculate difference (case1 - case2)
    delta_J_theta = J_theta1 - J_theta2
    delta_J_phi = J_phi1 - J_phi2

    # Calculate magnitude for each case
    magnitude1 = np.sqrt(J_theta1 ** 2 + J_phi1 ** 2)
    magnitude2 = np.sqrt(J_theta2 ** 2 + J_phi2 ** 2)

    # Signed difference in magnitude
    signed_diff = magnitude1 - magnitude2

    # Plot difference
    fig, ax = plt.subplots(figsize=(12, 6))

    # Streamplot of difference field with signed difference magnitude as color
    strm = ax.streamplot(x=lon_grid1, y=lat_grid1, u=delta_J_phi, v=delta_J_theta,
                         color=signed_diff,
                         cmap='RdBu_r',  # Red = case1 stronger, Blue = case2 stronger
                         density=2.2, linewidth=1,
                         norm=plt.Normalize(vmin=-vmax_global, vmax=vmax_global))

    cbar = plt.colorbar(strm.lines, ax=ax, label='Δ|J| [nA/m²]', extend='both')

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude [°]')
    ax.set_ylabel('Latitude [°]')
    ax.set_title(
        f'Signed Difference: {case1} - {case2}\n'
        f'Current streamlines, shell {rmin / RM:.2f}-{rmax / RM:.2f} RM at t = {step * 0.002} s'
    )
    ax.grid(True, alpha=0.3)

    outfile = os.path.join(
        output_folder,
        f"diff_{case1}_minus_{case2}_streamlines_{step}_shell_{rmin / RM:.2f}-{rmax / RM:.2f}_RM.png"
    )
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {outfile}")

print(f"\nFinished all difference plots for {case1} - {case2}")
