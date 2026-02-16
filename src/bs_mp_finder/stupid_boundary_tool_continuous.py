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

# extreme cases: CPN RPN CPS RPS
# dBdt cases: RBY, CBY
case = "RPN"

if "RP" in case or "CP" in case:
    # sim_steps = list(range(105000, 350000 + 1, 1000))
    sim_steps = list(range(0, 350000 + 1, 1000))
    file_list = []
    for sim_step in sim_steps:
        if sim_step < 115000:
            input_folder = f"/Volumes/data_backup/mercury/extreme/{case}_Base/plane_product/object/"
            filename = f"{sim_step:06d}"
            f_3d = os.path.join(input_folder, f"Amitis_{case}_Base_{filename}_xz_comp.nc")
        else:
            input_folder = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}_HNHV/plane_product/object/"
            filename = f"{sim_step:06d}"
            f_3d = os.path.join(input_folder, f"Amitis_{case}_HNHV_{filename}_xz_comp.nc")
        file_list.append(f_3d)
elif "Y" in case:
    input_folder = f"/Volumes/data_backup/mercury/dBdT/{case}/plane_product/object/"
    sim_steps = range(250000, 270000 + 1, 10000)
    file_list = []
    for sim_step in sim_steps:
        filename = f"{sim_step:06d}"
        f_3d = os.path.join(input_folder, f"Amitis_{case}_{filename}_xz_comp.nc")
        file_list.append(f_3d)
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
for file in file_list:

    if not os.path.exists(file):
        print(f"[WARN] missing 3D file: {file}")
        continue

    time_s = float(file.split("_")[-3]) * 0.002
    print(f"Processing timestep: {time_s:.1f} s")

    # Calculate geographic equator standoff distance
    _, _, _, _, _, geo_standoff = boundary_utils.compute_masks_3d(
        file, plot_id, equator='geographic', debug=debug
    )

    # Calculate magnetic equator standoff distance
    _, _, _, _, _, mag_standoff = boundary_utils.compute_masks_3d(
        file, plot_id, equator='magnetic', debug=debug
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

# ----------------------------
# WINDOWED STATISTICS
# ----------------------------
# Define time windows
windows = {
    'window1': (210, 230),
    'window2': (280, 300),
    'window3': (330, 350),
    'window4': (680, 700)
}


def calculate_window_stats(timestamps, standoff_data, window_start, window_end):
    """Calculate statistics for data within a time window"""
    mask = (timestamps >= window_start) & (timestamps <= window_end)
    windowed_data = standoff_data[mask]
    valid_data = windowed_data[~np.isnan(windowed_data)]

    if len(valid_data) > 0:
        return {
            'mean': np.mean(valid_data),
            'median': np.median(valid_data),
            'std': np.std(valid_data),
            'min': np.min(valid_data),
            'max': np.max(valid_data),
            'n_points': len(valid_data)
        }
    else:
        return {
            'mean': np.nan, 'median': np.nan, 'std': np.nan,
            'min': np.nan, 'max': np.nan, 'n_points': 0
        }


print("\n" + "=" * 70)
print(f"{case} - Magnetopause Standoff Distance Statistics (Windowed)")
print("=" * 70)

# Calculate and print statistics for each window
for window_name, (t_start, t_end) in windows.items():
    print(f"\n{window_name.upper()}: {t_start}-{t_end} s")
    print("-" * 70)

    # Geographic equator
    geo_stats = calculate_window_stats(timestamps, standoff_geo, t_start, t_end)
    print(f"\nGeographic Equator (Z = 0 R_M):")
    if geo_stats['n_points'] > 0:
        print(f"  Mean:   {geo_stats['mean']:.4f} R_M ({geo_stats['mean'] * 2440:.2f} km)")
        print(f"  Median: {geo_stats['median']:.4f} R_M ({geo_stats['median'] * 2440:.2f} km)")
        print(f"  Std:    {geo_stats['std']:.4f} R_M ({geo_stats['std'] * 2440:.2f} km)")
        print(f"  Min:    {geo_stats['min']:.4f} R_M ({geo_stats['min'] * 2440:.2f} km)")
        print(f"  Max:    {geo_stats['max']:.4f} R_M ({geo_stats['max'] * 2440:.2f} km)")
        print(f"  N:      {geo_stats['n_points']} timesteps")
    else:
        print("  No valid data in this window")

    # Magnetic equator
    mag_stats = calculate_window_stats(timestamps, standoff_mag, t_start, t_end)
    print(f"\nMagnetic Equator (Z = {Z_MAG_KM:.0f} km = {Z_MAG_RM:.4f} R_M):")
    if mag_stats['n_points'] > 0:
        print(f"  Mean:   {mag_stats['mean']:.4f} R_M ({mag_stats['mean'] * 2440:.2f} km)")
        print(f"  Median: {mag_stats['median']:.4f} R_M ({mag_stats['median'] * 2440:.2f} km)")
        print(f"  Std:    {mag_stats['std']:.4f} R_M ({mag_stats['std'] * 2440:.2f} km)")
        print(f"  Min:    {mag_stats['min']:.4f} R_M ({mag_stats['min'] * 2440:.2f} km)")
        print(f"  Max:    {mag_stats['max']:.4f} R_M ({mag_stats['max'] * 2440:.2f} km)")
        print(f"  N:      {mag_stats['n_points']} timesteps")
    else:
        print("  No valid data in this window")

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

ts_csv = os.path.join(out_dir, f"{case}_standoff_timeseries.csv")

ts_df.to_csv(ts_csv, index=False)
print(f"\nSaved timeseries: {ts_csv}")

# ----------------------------
# PLOT TIMESERIES
# ----------------------------
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(timestamps, standoff_geo, 'o-', color='cyan', label='Geographic Eq. (Z=0)', linewidth=2, markersize=6)
ax.plot(timestamps, standoff_mag, 's-', color='orange', label=f'Magnetic Eq. (Z={Z_MAG_KM:.0f}km)', linewidth=2,
        markersize=6)

if 0:
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

plot_path = os.path.join(out_dir, f"{case}_standoff_timeseries.png")

plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved plot: {plot_path}")
plt.close()

if 0:
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
    html_path = os.path.join(out_dir, f"{case}_3D_standoff.html")

    fig.update_layout(title_text=stitle, title_x=0.5, title_font_size=18)

    # Save as interactive HTML
    fig.write_html(html_path)
    print(f"Saved interactive 3D plot: {html_path}")

    print(f"\n3D points: {len(geo_x_all)} geographic, {len(mag_x_all)} magnetic")

print("Processing complete!")
