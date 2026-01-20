#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# SETTINGS
output_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology/"
os.makedirs(output_folder, exist_ok=True)
step = 115000
method = "_newRM"
R_M = 2440.0        # Mercury radius [km]

# Load equirectangular Mercury surface image
mercury_img_path = "/Users/danywaller/Downloads/MDIS_Monochrome_20170512_PDS16_64ppd_equirectangular.png"
img = mpimg.imread(mercury_img_path)

# If grayscale, add a fake RGB dimension
if img.ndim == 2:
    img = np.stack([img]*3, axis=-1)  # shape becomes (ny, nx, 3)

ny, nx, _ = img.shape

# Create longitude/latitude arrays for each pixel
lon = np.linspace(-np.pi, np.pi, nx)        # -180° to 180° in radians
lat = np.linspace(-np.pi/2, np.pi/2, ny)   # -90° to 90° in radians

lon_grid, lat_grid = np.meshgrid(lon, lat)

# Prepare figure
fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={"projection": "hammer"})

# Plot image using pcolormesh
# flip image vertically so latitude aligns
ax.pcolormesh(lon_grid, lat_grid, np.flipud(img[:, :, :3]), shading='auto', zorder=0)

# --------------------------
# Plot OCBs on top of the image
# --------------------------
cases = ["RPS", "CPS", "RPN", "CPN"]
case_linestyles = {"RPS": "--", "CPS": ":", "RPN": "-", "CPN": "-."}

for case in cases:
    # input_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology/{case}_Base/"
    input_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology{method}/"
    csv_file = os.path.join(input_folder, f"{case}_115000_ocb_curve.csv")  # single timestep CSV with OCB curve
    if os.path.exists(csv_file):
        df_footprints = pd.read_csv(csv_file)
        print(f"Loaded {len(df_footprints)} footprints for {case}")
    else:
        print(f"No OCB CSV found for {case}, skipping footprints")
        df_footprints = pd.DataFrame(columns=["latitude_deg", "longitude_deg", "classification"])

    # Split north and south hemispheres
    df_north = df_footprints[df_footprints["hemisphere"] == "north"]
    df_south = df_footprints[df_footprints["hemisphere"] == "south"]

    # Convert to radians for Mollweide/Hammer projection
    lon_n_rad = np.deg2rad(df_north["longitude_deg"])
    lat_n_rad = np.deg2rad(df_north["ocb_latitude_deg"])

    lon_s_rad = np.deg2rad(df_south["longitude_deg"])
    lat_s_rad = np.deg2rad(df_south["ocb_latitude_deg"])

    # Plot
    ls = case_linestyles[case]

    ax.plot(lon_n_rad, lat_n_rad, color="lime", lw=2, ls=ls)
    ax.plot(lon_s_rad, lat_s_rad, color="cyan", lw=2, ls=ls)

# Longitude ticks (-170 to 170 every n °)
lon_ticks_deg = np.arange(-120, 121, 60)
lon_ticks_rad = np.deg2rad(lon_ticks_deg)

# Latitude ticks (-90 to 90 every n °)
lat_ticks_deg = np.arange(-60, 61, 30)
lat_ticks_rad = np.deg2rad(lat_ticks_deg)

# Apply to the current axis
ax.set_xticks(lon_ticks_rad)
ax.set_yticks(lat_ticks_rad)

# Label ticks in degrees
ax.set_xticklabels([f"{int(l)}°" for l in lon_ticks_deg])
ax.set_yticklabels([f"{int(l)}°" for l in lat_ticks_deg])

# Create proxy artists (black lines) for the legend
legend_lines = [Line2D([0], [0], color="black", lw=2, ls=ls, label=case)
                for case, ls in case_linestyles.items()]

# Add the legend to the plot
ax.legend(handles=legend_lines, title="Case", loc="lower right", bbox_to_anchor=(1.05, -0.05), framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.6, color="black")

ax.set_title(f"Mercury OCB ({method.replace("_","")})")

# Save figure
plt.tight_layout()
# outfile_png = os.path.join(output_folder, "all_cases_OCB_comparison_115000.png")
outfile_png = os.path.join(output_folder, f"all_cases_OCB_comparison_115000{method}.png")

plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
print("Saved figure:", outfile_png)