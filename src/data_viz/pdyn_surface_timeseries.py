#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# -------------------------------
# Configuration
# -------------------------------
case = "CPS_HNHV"
output_folder = f"/Users/danywaller/Projects/mercury/extreme/timeseries_pdyn_surface/{case}/"
os.makedirs(output_folder, exist_ok=True)

R_M = 2440.0  # Mercury radius [km]
LAT_BINS = 180  # Surface latitude bins
LON_BINS = 360  # Surface longitude bins

# proton mass (kg)
M_P = 1.6726e-27

# Time step
dt = 2.0  # seconds (data every 2 seconds = 1000 simulation steps)

# sim_steps = list(range(115000, 200000 + 1, 1000))
sim_steps = list(range(105000, 350000 + 1, 1000))
n_steps = len(sim_steps)

# actual time array based on dt
timestamps = np.array(sim_steps) * 0.002  # [s]

# -------------------------------
# Extract time series at surface and CMB
# -------------------------------
if "CPN" in case:
    print(f"Case: Conductive Core under Planetward/Northward IMF")
elif "CPS" in case:
    print(f"Case: Conductive Core under Planetward/Southward IMF")
print(f"   Number of steps: {n_steps}")
print(f"   Total time span: {timestamps[-1] - timestamps[0]} s = {(timestamps[-1] - timestamps[0]) / 60:.2f} min\n")

# Storage for time series arrays
surface_pdyn_timeseries = []

for i, sim_step in enumerate(sim_steps):
    if sim_step < 115000:
        case = case.replace("HNHV", "Base")
        input_folder1 = f"/Volumes/data_backup/mercury/extreme/{case}/plane_product/object/"
    else:
        case = case.replace("Base", "HNHV")
        input_folder1 = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/object/"

    nc_file = os.path.join(input_folder1, f"Amitis_{case}_{sim_step:06d}_xz_comp.nc")

    if not os.path.exists(nc_file):
        print(f"Warning: {nc_file} not found, skipping...")
        continue

    ds = xr.open_dataset(nc_file)

    x = ds["Nx"].values  # [km]
    y = ds["Ny"].values  # [km]
    z = ds["Nz"].values  # [km]

    # Original order is Z, Y, X -> transpose to X, Y, Z
    Vx = (np.transpose(ds["vx01"].isel(time=0).values, (2, 1, 0))
          + np.transpose(ds["vx02"].isel(time=0).values, (2, 1, 0))
          + np.transpose(ds["vx03"].isel(time=0).values, (2, 1, 0))
          + np.transpose(ds["vx04"].isel(time=0).values, (2, 1, 0)))
    Vy = (np.transpose(ds["vy01"].isel(time=0).values, (2, 1, 0))
          + np.transpose(ds["vy02"].isel(time=0).values, (2, 1, 0))
          + np.transpose(ds["vy03"].isel(time=0).values, (2, 1, 0))
          + np.transpose(ds["vy04"].isel(time=0).values, (2, 1, 0)))
    Vz = (np.transpose(ds["vz01"].isel(time=0).values, (2, 1, 0))
          + np.transpose(ds["vz02"].isel(time=0).values, (2, 1, 0))
          + np.transpose(ds["vz03"].isel(time=0).values, (2, 1, 0))
          + np.transpose(ds["vz04"].isel(time=0).values, (2, 1, 0)))

    v_mag = np.sqrt(Vx ** 2 + Vy ** 2 + Vz ** 2) * 1e3  # km/s -> m/s

    # Total density (sum of all species)
    den = (ds["den01"].isel(time=0).values +
           ds["den02"].isel(time=0).values +
           ds["den03"].isel(time=0).values +
           ds["den04"].isel(time=0).values)
    den = np.transpose(den, (2, 1, 0))  * 1e6     # cm^-3 -> m^-3

    # dynamic pressure (Pa)
    P_pa = den * M_P * v_mag ** 2

    # convert to nPa
    P_npa = P_pa * 1e9

    # Create interpolators
    interp_pdyn = RegularGridInterpolator((x, y, z), P_npa, bounds_error=False, fill_value=np.nan)

    # Sample at surface (R_M) - use spherical grid
    n_sample = 100  # Number of sample points
    theta_sample = np.linspace(0, np.pi, n_sample)
    phi_sample = np.linspace(0, 2 * np.pi, n_sample)

    # Surface sampling
    pdyn_surface = []
    for theta in theta_sample[::1]:  # Subsample
        for phi in phi_sample[::1]:
            x_s = R_M * np.sin(theta) * np.cos(phi)
            y_s = R_M * np.sin(theta) * np.sin(phi)
            z_s = R_M * np.cos(theta)

            pdyn = interp_pdyn([x_s, y_s, z_s])[0]

            if not np.isnan(pdyn):
                pdyn_surface.append(pdyn)

    surface_pdyn_timeseries.append(np.nanmean(pdyn_surface) if pdyn_surface else np.nan)

    print(f"Processed step {sim_step} ({i + 1}/{n_steps}), time = {timestamps[i]:.1f} s")
    ds.close()

surface_pdyn_timeseries = np.array(surface_pdyn_timeseries)  # Shape: (n_steps,)


# -------------------------------
# Create 4x1 time series plot
# -------------------------------
fig, ax1 = plt.subplots(1, 1, figsize=(8, 3))

ax1.plot(timestamps, surface_pdyn_timeseries, color='#e377c2', linewidth=2.0, linestyle='-', marker='D', markersize=2)

# Define time windows and add shaded boxes
windows = [
    (210, 230),
    (280, 300),
    (330, 350),
    (680, 700)
]

# Add shaded boxes for each window
for window_start, window_end in windows:
    ax1.axvspan(window_start, window_end,
                alpha=0.2,
                color='gray')

ax1.set_ylabel(r"P$_{dyn}$ [nPa]", fontsize=14)
ax1.set_xlabel('Time [s]', fontsize=14)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(labelsize=11)
# ax1.legend(loc='best', fontsize=10)

plt.title(rf"{case.replace("_", " ")} P$_{{dyn}}$, t = {timestamps[0]:.3f} - {timestamps[-1]:.3f} s")

plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'{case}_pdyn_timeseries.png'), dpi=300, bbox_inches='tight')
