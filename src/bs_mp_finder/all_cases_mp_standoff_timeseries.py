#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

cases = ["(A) RPN_HNHV", "(B) CPN_HNHV", "(C) RPS_HNHV", "(D) CPS_HNHV"]

base_dir = "/Users/danywaller/Projects/mercury/extreme/magnetopause_3D_timeseries"

# Mercury magnetic dipole offset
Z_OFFSET_RM = 484.0 / 2440.0
lat_mag_fixed = np.degrees(np.arcsin(Z_OFFSET_RM))

# ============================================================
# FIGURE SETUP
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

# ============================================================
# LOOP OVER CASES
# ============================================================

for i, case in enumerate(cases):

    ax = axes[i]

    csv_file = os.path.join(
        base_dir,
        case.split(" ")[1],
        "mp_mesh_analysis",
        f"{case.split(" ")[1]}_equatorial_standoff_timeseries.csv"
    )

    df = pd.read_csv(csv_file)

    time = df["time_s"].values
    geo = df["geo_equator_mean_RM"].values
    geo_std = df["geo_equator_std_RM"].values

    mag = df["mag_equator_mean_RM"].values
    mag_std = df["mag_equator_std_RM"].values

    # ========================================================
    # PLOT
    # ========================================================

    _, _, geo_bars = ax.errorbar(
        time,
        geo,
        yerr=geo_std,
        ms=4,
        fmt='o-',
        color="mediumorchid",
        capsize=3,
        label="Geographic Equator (0°)"
    )

    _, _, mag_bars = ax.errorbar(
        time,
        mag,
        yerr=mag_std,
        ms=4,
        fmt='o-',
        color="goldenrod",
        capsize=3,
        label=f"Magnetic Equator (+{lat_mag_fixed:.1f}°)"
    )

    # loop through bars and caps and set the alpha value
    [bar.set_alpha(0.5) for bar in geo_bars]
    [bar.set_alpha(0.5) for bar in mag_bars]

    ax.set_title(case.replace("_", " "), fontsize=14, fontweight="bold")
    ax.grid(True)

# ============================================================
# GLOBAL LABELS
# ============================================================

for ax in axes[2:]:
    ax.set_xlabel("Time (s)", fontsize=12)

for ax in axes[::2]:
    ax.set_ylabel("Δr ($R_M$)", fontsize=12)

for ax in axes:
    ax.legend(loc="upper right", fontsize=12)

plt.suptitle(
    "Dayside Longitude-Integrated (±75°) Magnetopause Standoff Distance",
    fontsize=18, fontweight="bold", y=0.975
)

plt.ylim(1.0, 1.75)

plt.tight_layout(rect=[0,0,1,0.98])

# ============================================================
# SAVE FIGURE
# ============================================================

out_file = os.path.join(base_dir, "equatorial_standoff_2x2_timeseries.png")
plt.savefig(out_file, dpi=300)

print("Saved figure:", out_file)