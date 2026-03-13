#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Configuration
# -------------------------------

cases = ["RPN_HNHV", "CPN_HNHV", "RPS_HNHV", "CPS_HNHV"]
labels = ["RPN", "CPN", "RPS", "CPS"]

colors = ["firebrick", "darkorange", "goldenrod", "royalblue"]

selected_times = [230, 270, 330, 700]

# paths
output_folder = "/Users/danywaller/Projects/mercury/extreme/timeseries_pdyn_surface/all_cases/"
csv_file = os.path.join(output_folder, "surface_pdyn_timeseries_all_cases.csv")

# -------------------------------
# Load CSV
# -------------------------------

df = pd.read_csv(csv_file)

timestamps = df["time_s"].values

surface_timeseries = {}
for case in cases:
    surface_timeseries[case] = df[case].values

print("Loaded CSV:", csv_file)
print("Number of timesteps:", len(timestamps))

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

# vertical lines
for i, t in enumerate(selected_times, start=1):

    ax.axvline(
        float(t),
        linestyle="--",
        color="black",
        alpha=0.7
    )

# get ylim AFTER plotting
ylim = ax.get_ylim()

# label the vertical lines
for i, t in enumerate(selected_times, start=1):

    ax.text(
        float(t)-20,
        ylim[1]-0.075*ylim[1],
        f"({i})",
        color="k",
        fontsize=12,
        ha="left",
        va="bottom",
        fontweight="bold"
    )

ax.set_xlabel("Time [s]", fontsize=14)
ax.set_ylabel(r"P$_{dyn}$ [nPa]", fontsize=14)

ax.grid(True, linestyle="-", alpha=0.3)

ax.legend(ncol=2)

plt.title("Integrated Surface Dynamic Pressure", fontsize=14, fontweight="bold")

plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "pdyn_timeseries_all_cases.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved plot to:", output_folder)