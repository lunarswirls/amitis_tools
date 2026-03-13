#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# -------------------------------
# Configuration
# -------------------------------

cases = ["RPN_HNHV", "CPN_HNHV", "RPS_HNHV", "CPS_HNHV"]
labels = ["RPN", "CPN", "RPS", "CPS"]

colors = ["firebrick", "darkorange", "goldenrod", "royalblue"]

selected_times = [230, 270, 330, 700]

R_M = 2440.0  # Mercury radius [km]
M_P = 1.6726e-27  # proton mass (kg)

dt = 2.0  # seconds

sim_steps = list(range(105000, 350000 + 1, 1000))
n_steps = len(sim_steps)

timestamps = np.array(sim_steps) * 0.002

# output
output_folder = "/Users/danywaller/Projects/mercury/extreme/timeseries_pdyn_surface/all_cases/"
os.makedirs(output_folder, exist_ok=True)

print("Number of timesteps:", n_steps)
print("Total duration:", timestamps[-1] - timestamps[0], "seconds\n")

# -------------------------------
# Surface sampling grid
# -------------------------------

n_sample = 100

theta_sample = np.linspace(0, np.pi, n_sample)
phi_sample = np.linspace(0, 2*np.pi, n_sample)

surface_points = []

for theta in theta_sample:
    for phi in phi_sample:

        x_s = R_M*np.sin(theta)*np.cos(phi)
        y_s = R_M*np.sin(theta)*np.sin(phi)
        z_s = R_M*np.cos(theta)

        surface_points.append([x_s, y_s, z_s])

surface_points = np.array(surface_points)

# -------------------------------
# Storage
# -------------------------------

surface_timeseries = {case: [] for case in cases}

# -------------------------------
# Main loop (timesteps only once)
# -------------------------------

for i, sim_step in enumerate(sim_steps):

    print(f"\nProcessing timestep {sim_step} ({i+1}/{n_steps})")

    for case_full in cases:

        case, mode = case_full.split("_")

        # choose input folder
        if sim_step < 115000:
            input_folder = f"/Volumes/data_backup/mercury/extreme/High_Base/{case}_Base/plane_product/cube/"
            nc_file = os.path.join(
                input_folder,
                f"Amitis_{case}_Base_{sim_step:06d}_merged_4RM.nc"
            )
        else:
            input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/plane_product/cube/"
            nc_file = os.path.join(
                input_folder,
                f"Amitis_{case}_{mode}_{sim_step:06d}_merged_4RM.nc"
            )

        if not os.path.exists(nc_file):
            print("Missing:", nc_file)
            surface_timeseries[case_full].append(np.nan)
            continue

        ds = xr.open_dataset(nc_file)

        # grid
        x = ds["Nx"].values
        y = ds["Ny"].values
        z = ds["Nz"].values

        # velocity components
        Vx = (
            np.transpose(ds["vx01"].isel(time=0).values,(2,1,0)) +
            np.transpose(ds["vx02"].isel(time=0).values,(2,1,0)) +
            np.transpose(ds["vx03"].isel(time=0).values,(2,1,0)) +
            np.transpose(ds["vx04"].isel(time=0).values,(2,1,0))
        )

        Vy = (
            np.transpose(ds["vy01"].isel(time=0).values,(2,1,0)) +
            np.transpose(ds["vy02"].isel(time=0).values,(2,1,0)) +
            np.transpose(ds["vy03"].isel(time=0).values,(2,1,0)) +
            np.transpose(ds["vy04"].isel(time=0).values,(2,1,0))
        )

        Vz = (
            np.transpose(ds["vz01"].isel(time=0).values,(2,1,0)) +
            np.transpose(ds["vz02"].isel(time=0).values,(2,1,0)) +
            np.transpose(ds["vz03"].isel(time=0).values,(2,1,0)) +
            np.transpose(ds["vz04"].isel(time=0).values,(2,1,0))
        )

        v_mag = np.sqrt(Vx**2 + Vy**2 + Vz**2) * 1e3

        # density
        den = (
            ds["den01"].isel(time=0).values +
            ds["den02"].isel(time=0).values +
            ds["den03"].isel(time=0).values +
            ds["den04"].isel(time=0).values
        )

        den = np.transpose(den,(2,1,0)) * 1e6

        # dynamic pressure
        P_pa = den * M_P * v_mag**2
        P_npa = P_pa * 1e9

        # interpolator
        interp_pdyn = RegularGridInterpolator(
            (x,y,z),
            P_npa,
            bounds_error=False,
            fill_value=np.nan
        )

        # sample surface
        pdyn_surface = interp_pdyn(surface_points)

        mean_surface = np.nanmean(pdyn_surface)

        surface_timeseries[case_full].append(mean_surface)

        ds.close()

# convert lists to arrays
for case in cases:
    surface_timeseries[case] = np.array(surface_timeseries[case])

import pandas as pd

# -------------------------------
# Save timeseries to CSV
# -------------------------------

# build dataframe
data = {"time_s": timestamps}

for case in cases:
    data[case] = surface_timeseries[case]

df = pd.DataFrame(data)

# output file
csv_file = os.path.join(output_folder, "surface_pdyn_timeseries_all_cases.csv")

df.to_csv(csv_file, index=False)

print("\nSaved CSV:", csv_file)

# -------------------------------
# Plot
# -------------------------------

fig, ax = plt.subplots(figsize=(9,4))

for case, label, color in zip(cases, labels, colors):

    ax.plot(
        timestamps,
        surface_timeseries[case],
        linewidth=2,
        label=label,
        color=color
    )

i = 1
# vertical lines
for t in selected_times:
    ax.axvline(
        float(t),
        linestyle="--",
        color="black",
        alpha=0.7
    )
    # Add string at the top of the line
    ylim = ax.get_ylim()
    ax.text(float(t), ylim[1], f'({i})', color='k', fontsize=14, ha='left', va='bottom', fontweight='bold')
    i+=1

ax.set_xlabel("Time [s]", fontsize=14)
ax.set_ylabel(r"P$_{dyn}$ [nPa]", fontsize=14)

ax.grid(True, linestyle="-", alpha=0.3)

ax.legend(ncol=2)

plt.title("Surface Dynamic Pressure")

plt.tight_layout()

plt.savefig(
    os.path.join(output_folder,"pdyn_timeseries_all_cases.png"),
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved plot to:", output_folder)