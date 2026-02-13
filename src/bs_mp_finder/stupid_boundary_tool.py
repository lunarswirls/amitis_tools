#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import bs_mp_finder.stupid_boundary_util as boundary_utils
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# SETTINGS
# ----------------------------
debug = False

# base cases: CPN_Base RPN_Base CPS_Base RPS_Base
# HNHV cases: CPN_HNHV RPN_HNHV CPS_HNHV RPS_HNHV
# dBdt cases: RBY, CBY
case = "RPS_HNHV"

# FOR HNHV - DOUBLE CHECK ONLY ONE IS TRUE!!!!
transient = False  # 280-300 s
post_transient = True  # 330-350 s
new_state = False  # 680-700 s

if "Base" in case:
    input_folder = f"/Volumes/data_backup/mercury/extreme/{case}/plane_product/object/"
    sim_steps = list(range(110000, 115000 + 1, 1000))
elif "HNHV" in case:
    input_folder = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/object/"
    if transient and not post_transient and not new_state:
        sim_steps = range(145000, 150000 + 1, 1000)
    elif post_transient and not transient and not new_state:
        sim_steps = range(170000, 175000 + 1, 1000)
    elif new_state and not post_transient and not transient:
        sim_steps = range(345000, 350000 + 1, 1000)
    else:
        raise ValueError("Too many flags! Set only one of transient, post_transient, or new_state to True")
elif "Y" in case:
    input_folder = f"/Volumes/data_backup/mercury/dBdT/{case}/plane_product/object/"
    sim_steps = range(250000, 270000 + 1, 10000)
else:
    raise ValueError("Unrecognized case! Are you using a case from Base, HNHV, or dBdt?")

plot_id = "Pmag"  # Using current density for standoff distance calculation

PLOT_BG = {
    "Bmag": {
        "key": "Bmag",
        "label": r"|B|\ (\mathrm{nT})",
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 150.0,
        "bs_col": "red",
        "mp_col": "magenta",
    },
    "Jmag": {
        "key": "Jmag",
        "label": r"|J|\ (\mathrm{nA\,m^{-2}})",
        "cmap": "plasma",
        "vmin": 0.0,
        "vmax": 150.0,
        "bs_col": "cyan",
        "mp_col": "limegreen",
    },
    "Pmag": {
        "key": "gradP",
        "label": r"N\ (\mathrm{cm^{-3}})",
        "cmap": "cividis",
        "vmin": 0,
        "vmax": 100.,
        "bs_col": "red",
        "mp_col": "magenta",
    },
}

out_dir = f"/Users/danywaller/Projects/mercury/extreme/boundary_3D_timeseries/{case}_standoff/"
os.makedirs(out_dir, exist_ok=True)

RM_M = 2440.0e3
Z_MAG_KM = 484.0
Z_MAG_RM = Z_MAG_KM / 2440.0

# ----------------------------
# ACCUMULATORS for standoff distances and 3D positions
# ----------------------------
standoff_geo = []  # Geographic equator standoff distances
standoff_mag = []  # Magnetic equator standoff distances
timestamps = []

# 3D position accumulators
geo_x_all = []
geo_y_all = []
geo_z_all = []

mag_x_all = []
mag_y_all = []
mag_z_all = []

# ----------------------------
# LOOP: Calculate standoff distances at each timestep
# ----------------------------
for sim_step in sim_steps:
    filename = f"{sim_step:06d}"
    f_3d = os.path.join(input_folder, f"Amitis_{case}_{filename}_xz_comp.nc")

    if not os.path.exists(f_3d):
        print(f"[WARN] missing 3D file: {f_3d}")
        continue

    time_s = sim_step * 0.002
    print(f"Processing timestep: {time_s:.1f} s")

    # Calculate geographic equator standoff distance
    _, _, _, _, _, geo_standoff = boundary_utils.compute_masks_3d(
        f_3d, plot_id, equator='geographic', debug=debug
    )

    # Calculate magnetic equator standoff distance
    _, _, _, _, _, mag_standoff = boundary_utils.compute_masks_3d(
        f_3d, plot_id, equator='magnetic', debug=debug
    )

    # Accumulate timeseries data
    timestamps.append(time_s)
    standoff_geo.append(geo_standoff)
    standoff_mag.append(mag_standoff)

    # Accumulate 3D positions
    if not np.isnan(geo_standoff):
        geo_x_all.append(geo_standoff)
        geo_y_all.append(0.0)
        geo_z_all.append(0.0)

    if not np.isnan(mag_standoff):
        mag_x_all.append(mag_standoff)
        mag_y_all.append(0.0)
        mag_z_all.append(Z_MAG_RM)

    print(f"  Geographic: {geo_standoff:.4f} R_M ({geo_standoff * 2440:.2f} km)")
    print(f"  Magnetic:   {mag_standoff:.4f} R_M ({mag_standoff * 2440:.2f} km)")

print("\nStandoff distance calculation complete.")

# ----------------------------
# Convert to numpy arrays
# ----------------------------
timestamps = np.array(timestamps)
standoff_geo = np.array(standoff_geo)
standoff_mag = np.array(standoff_mag)

geo_x_all = np.array(geo_x_all)
geo_y_all = np.array(geo_y_all)
geo_z_all = np.array(geo_z_all)

mag_x_all = np.array(mag_x_all)
mag_y_all = np.array(mag_y_all)
mag_z_all = np.array(mag_z_all)

# Remove NaN values for statistics
geo_valid = standoff_geo[~np.isnan(standoff_geo)]
mag_valid = standoff_mag[~np.isnan(standoff_mag)]

# ----------------------------
# STATISTICS
# ----------------------------
print("\n" + "=" * 70)
print(f"{case} - Magnetopause Standoff Distance Statistics")
print("=" * 70)

if len(geo_valid) > 0:
    geo_mean = np.mean(geo_valid)
    geo_median = np.median(geo_valid)
    geo_std = np.std(geo_valid)
    geo_min = np.min(geo_valid)
    geo_max = np.max(geo_valid)

    print("\nGeographic Equator (Z = 0 R_M):")
    print(f"  Mean:   {geo_mean:.4f} R_M ({geo_mean * 2440:.2f} km)")
    print(f"  Median: {geo_median:.4f} R_M ({geo_median * 2440:.2f} km)")
    print(f"  Std:    {geo_std:.4f} R_M ({geo_std * 2440:.2f} km)")
    print(f"  Min:    {geo_min:.4f} R_M ({geo_min * 2440:.2f} km)")
    print(f"  Max:    {geo_max:.4f} R_M ({geo_max * 2440:.2f} km)")
    print(f"  N:      {len(geo_valid)} timesteps")
else:
    geo_mean = geo_median = geo_std = geo_min = geo_max = np.nan
    print("\nGeographic Equator: No valid standoff distances")

if len(mag_valid) > 0:
    mag_mean = np.mean(mag_valid)
    mag_median = np.median(mag_valid)
    mag_std = np.std(mag_valid)
    mag_min = np.min(mag_valid)
    mag_max = np.max(mag_valid)

    print(f"\nMagnetic Equator (Z = {Z_MAG_KM:.0f} km = {Z_MAG_RM:.4f} R_M):")
    print(f"  Mean:   {mag_mean:.4f} R_M ({mag_mean * 2440:.2f} km)")
    print(f"  Median: {mag_median:.4f} R_M ({mag_median * 2440:.2f} km)")
    print(f"  Std:    {mag_std:.4f} R_M ({mag_std * 2440:.2f} km)")
    print(f"  Min:    {mag_min:.4f} R_M ({mag_min * 2440:.2f} km)")
    print(f"  Max:    {mag_max:.4f} R_M ({mag_max * 2440:.2f} km)")
    print(f"  N:      {len(mag_valid)} timesteps")
else:
    mag_mean = mag_median = mag_std = mag_min = mag_max = np.nan
    print("\nMagnetic Equator: No valid standoff distances")

print("=" * 70)

# ----------------------------
# SAVE TIMESERIES DATA
# ----------------------------
ts_df = pd.DataFrame({
    'time_s': timestamps,
    'geo_standoff_rm': standoff_geo,
    'geo_standoff_km': standoff_geo * 2440,
    'mag_standoff_rm': standoff_mag,
    'mag_standoff_km': standoff_mag * 2440
})

if "Base" in case:
    ts_csv = os.path.join(out_dir, f"{case}_standoff_timeseries.csv")
elif "HNHV" in case:
    if transient:
        ts_csv = os.path.join(out_dir, f"{case}_standoff_timeseries_transient.csv")
    elif post_transient:
        ts_csv = os.path.join(out_dir, f"{case}_standoff_timeseries_post-transient.csv")
    elif new_state:
        ts_csv = os.path.join(out_dir, f"{case}_standoff_timeseries_newstate.csv")
elif "Y" in case:
    ts_csv = os.path.join(out_dir, f"{case}_standoff_timeseries.csv")

ts_df.to_csv(ts_csv, index=False)
print(f"\nSaved timeseries: {ts_csv}")

# ----------------------------
# SAVE STATISTICS
# ----------------------------
stats_df = pd.DataFrame([
    {
        'equator': 'Geographic (Z=0)',
        'mean_r_rm': geo_mean,
        'median_r_rm': geo_median,
        'std_r_rm': geo_std,
        'min_r_rm': geo_min,
        'max_r_rm': geo_max,
        'mean_r_km': geo_mean * 2440 if not np.isnan(geo_mean) else np.nan,
        'median_r_km': geo_median * 2440 if not np.isnan(geo_median) else np.nan,
        'std_r_km': geo_std * 2440 if not np.isnan(geo_std) else np.nan,
        'min_r_km': geo_min * 2440 if not np.isnan(geo_min) else np.nan,
        'max_r_km': geo_max * 2440 if not np.isnan(geo_max) else np.nan,
        'n_timesteps': len(geo_valid)
    },
    {
        'equator': f'Magnetic (Z={Z_MAG_KM:.0f}km)',
        'mean_r_rm': mag_mean,
        'median_r_rm': mag_median,
        'std_r_rm': mag_std,
        'min_r_rm': mag_min,
        'max_r_rm': mag_max,
        'mean_r_km': mag_mean * 2440 if not np.isnan(mag_mean) else np.nan,
        'median_r_km': mag_median * 2440 if not np.isnan(mag_median) else np.nan,
        'std_r_km': mag_std * 2440 if not np.isnan(mag_std) else np.nan,
        'min_r_km': mag_min * 2440 if not np.isnan(mag_min) else np.nan,
        'max_r_km': mag_max * 2440 if not np.isnan(mag_max) else np.nan,
        'n_timesteps': len(mag_valid)
    }
])

if "Base" in case:
    stats_csv = os.path.join(out_dir, f"{case}_standoff_statistics.csv")
elif "HNHV" in case:
    if transient:
        stats_csv = os.path.join(out_dir, f"{case}_standoff_statistics_transient.csv")
    elif post_transient:
        stats_csv = os.path.join(out_dir, f"{case}_standoff_statistics_post-transient.csv")
    elif new_state:
        stats_csv = os.path.join(out_dir, f"{case}_standoff_statistics_newstate.csv")
elif "Y" in case:
    stats_csv = os.path.join(out_dir, f"{case}_standoff_statistics.csv")

stats_df.to_csv(stats_csv, index=False)
print(f"Saved statistics: {stats_csv}")

# ----------------------------
# PLOT TIMESERIES
# ----------------------------
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(timestamps, standoff_geo, 'o-', color='cyan', label='Geographic Eq. (Z=0)', linewidth=2, markersize=6)
ax.plot(timestamps, standoff_mag, 's-', color='orange', label=f'Magnetic Eq. (Z={Z_MAG_KM:.0f}km)', linewidth=2,
        markersize=6)

# Add mean lines
if not np.isnan(geo_mean):
    ax.axhline(geo_mean, color='cyan', linestyle='--', alpha=0.5, label=f'Geo. Mean: {geo_mean:.3f} R$_M$')
if not np.isnan(mag_mean):
    ax.axhline(mag_mean, color='orange', linestyle='--', alpha=0.5, label=f'Mag. Mean: {mag_mean:.3f} R$_M$')

ax.set_xlabel('Time (s)', fontsize=14)
ax.set_ylabel('Standoff Distance (R$_M$)', fontsize=14)
ax.set_title(f'{case} - Magnetopause Standoff Distance Evolution', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()

if "Base" in case:
    plot_path = os.path.join(out_dir, f"{case}_standoff_timeseries.png")
elif "HNHV" in case:
    if transient:
        plot_path = os.path.join(out_dir, f"{case}_standoff_timeseries_transient.png")
    elif post_transient:
        plot_path = os.path.join(out_dir, f"{case}_standoff_timeseries_post-transient.png")
    elif new_state:
        plot_path = os.path.join(out_dir, f"{case}_standoff_timeseries_newstate.png")
elif "Y" in case:
    plot_path = os.path.join(out_dir, f"{case}_standoff_timeseries.png")

plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved plot: {plot_path}")
plt.close()

# ----------------------------
# 3D PLOTLY VISUALIZATION - ALL ACCUMULATED POINTS
# ----------------------------
print("\nCreating 3D visualization...")

# Create Mercury sphere with day/night hemispheres
plot_depth = 1.0  # Mercury radius in R_M
theta = np.linspace(0, np.pi, 100)  # colatitude
phi = np.linspace(0, 2 * np.pi, 200)  # longitude
theta, phi = np.meshgrid(theta, phi)

xs = plot_depth * np.sin(theta) * np.cos(phi)
ys = plot_depth * np.sin(theta) * np.sin(phi)
zs = plot_depth * np.cos(theta)

eps = 0
mask_pos = xs >= -eps
mask_neg = xs <= eps

cfg = PLOT_BG[plot_id]

# Create single 3D scene
fig = go.Figure()

# Mercury sphere (light grey dayside)
fig.add_trace(
    go.Surface(
        x=np.where(mask_pos, xs, np.nan),
        y=np.where(mask_pos, ys, np.nan),
        z=np.where(mask_pos, zs, np.nan),
        surfacecolor=np.ones_like(xs),
        colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
        cmin=0,
        cmax=1,
        showscale=False,
        lighting=dict(ambient=1, diffuse=0, specular=0),
        hoverinfo='skip',
        showlegend=False
    )
)

# Mercury sphere (black nightside)
fig.add_trace(
    go.Surface(
        x=np.where(mask_neg, xs, np.nan),
        y=np.where(mask_neg, ys, np.nan),
        z=np.where(mask_neg, zs, np.nan),
        surfacecolor=np.zeros_like(xs),
        colorscale=[[0, 'black'], [1, 'black']],
        cmin=0,
        cmax=1,
        showscale=False,
        lighting=dict(ambient=1, diffuse=0, specular=0),
        hoverinfo='skip',
        showlegend=False
    )
)

# Geographic equator - all accumulated points
if len(geo_x_all) > 0:
    fig.add_trace(
        go.Scatter3d(
            x=geo_x_all,
            y=geo_y_all,
            z=geo_z_all,
            mode='markers',
            marker=dict(size=8, color='cyan', opacity=0.8, symbol='circle'),
            name=f'Geo. Eq.: {geo_mean:.2f} ± {geo_std:.2f} R<sub>M</sub>',
            showlegend=True,
            hovertemplate='Geographic Equator<br>X: %{x:.3f} R_M<br>Y: %{y:.3f} R_M<br>Z: %{z:.3f} R_M<extra></extra>'
        )
    )

# Magnetic equator - all accumulated points
if len(mag_x_all) > 0:
    fig.add_trace(
        go.Scatter3d(
            x=mag_x_all,
            y=mag_y_all,
            z=mag_z_all,
            mode='markers',
            marker=dict(size=8, color='orange', opacity=0.8, symbol='square'),
            name=f'Mag. Eq.: {mag_mean:.2f} ± {mag_std:.2f} R<sub>M</sub>',
            showlegend=True,
            hovertemplate=f'Magnetic Equator<br>X: %{{x:.3f}} R_M<br>Y: %{{y:.3f}} R_M<br>Z: {Z_MAG_RM:.3f} R_M<extra></extra>'
        )
    )

# Update layout
camera = dict(
    eye=dict(x=1.8, y=1.2, z=0.8),
    center=dict(x=0, y=0, z=0)
)

fig.update_layout(
    scene=dict(
        xaxis=dict(title='X (R<sub>M</sub>)', range=[-2, 2]),
        yaxis=dict(title='Y (R<sub>M</sub>)', range=[-2, 2]),
        zaxis=dict(title='Z (R<sub>M</sub>)', range=[-2, 2]),
        aspectmode='cube',
        camera=camera
    ),
    height=800,
    width=1000,
    showlegend=True,
    legend=dict(x=0.7, y=0.9, bgcolor='rgba(255,255,255,0.9)'),
    template='plotly_white'
)

# Set title
stitle = f"{case.replace('_', ' ')} - Accumulated Standoff Distances"
if "Base" in case:
    stitle += ": Pre-Transient"
    html_path = os.path.join(out_dir, f"{case}_3D_standoff.html")
elif "HNHV" in case:
    if transient:
        stitle += ": Transient"
        html_path = os.path.join(out_dir, f"{case}_3D_standoff_transient.html")
    elif post_transient:
        stitle += ": Post-Transient"
        html_path = os.path.join(out_dir, f"{case}_3D_standoff_post-transient.html")
    elif new_state:
        stitle += ": New State"
        html_path = os.path.join(out_dir, f"{case}_3D_standoff_newstate.html")
elif "Y" in case:
    html_path = os.path.join(out_dir, f"{case}_3D_standoff.html")

fig.update_layout(title_text=stitle, title_x=0.5, title_font_size=18)

# Save as interactive HTML
fig.write_html(html_path)
print(f"Saved interactive 3D plot: {html_path}")

print(f"\n3D points: {len(geo_x_all)} geographic, {len(mag_x_all)} magnetic")
print("Processing complete!")
