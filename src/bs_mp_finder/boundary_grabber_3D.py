#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import bs_mp_finder.boundary_utils as boundary_utils
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# SETTINGS
# ----------------------------
debug = False

# base cases: CPN_Base RPN_Base CPS_Base RPS_Base
# HNHV cases: CPN_HNHV RPN_HNHV CPS_HNHV RPS_HNHV
# LNHV cases: CPS_HNHV RPS_HNHV
case = "CPS"
mode = "LNHV"

# FOR HNHV - DOUBLE CHECK ONLY ONE IS TRUE!!!!
transient = True  # 280-300 s
post_transient = False  # 330-350 s
new_state = False  # 680-700 s

if "Base" in mode:
    input_folder = f"/Volumes/data_backup/mercury/extreme/{case}_Base/plane_product/object/"
    sim_steps = list(range(105000, 115000 + 1, 1000))
elif "HNHV" in mode or "LNHV" in mode:
    input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/plane_product/object/"
    if transient and not post_transient and not new_state:
        sim_steps = range(140000, 150000 + 1, 1000)
    elif post_transient and not transient and not new_state:
        sim_steps = range(165000, 175000 + 1, 1000)
    elif new_state and not post_transient and not transient:
        sim_steps = range(340000, 350000 + 1, 1000)
    else:
        raise ValueError("Too many flags! Set only one of transient, post_transient, or new_state to True")
else:
    raise ValueError("Unrecognized mode! Are you using one of (Base, HNHV, LNHV)?")

use_slices = ["xy", "xz"]
n_slices = len(use_slices)
slice_tag = "_".join(use_slices)

plot_id = "Pmag"

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

out_dir = f"/Users/danywaller/Projects/mercury/extreme/boundary_3D_timeseries/{case}_{mode}/"
os.makedirs(out_dir, exist_ok=True)
out_folder_ts = os.path.join(out_dir, f"timeseries_{slice_tag}/")
os.makedirs(out_folder_ts, exist_ok=True)

RM_M = 2440.0e3

# ----------------------------
# ACCUMULATORS for 3D positions
# ----------------------------
acc_3d = {
    'bs': {'x': [], 'y': [], 'z': [], 'r': []},
    'mp': {'x': [], 'y': [], 'z': [], 'r': []}
}

# ----------------------------
# LOOP: per-timestep processing + accumulation
# ----------------------------
for sim_step in sim_steps:
    filename = f"{sim_step:06d}"
    f_3d = os.path.join(input_folder, f"Amitis_{case}_{mode}_{filename}_xz_comp.nc")

    if not os.path.exists(f_3d):
        print(f"[WARN] missing 3D file: {f_3d}")
        continue

    print(f"Processing timestep: {sim_step*0.002} s")

    # Compute 3D boundary masks and coords
    x_coords, y_coords, z_coords, plot_bg_3d, bs_mask_3d, mp_mask_3d = boundary_utils.compute_masks_3d(
        f_3d, plot_id, debug=debug
    )

    # max MP radius finder
    mp_max_info = boundary_utils.max_radius_index_xr(mp_mask_3d, x_name="Nx", y_name="Ny", z_name="Nz")

    if mp_max_info is None:
        print(f"[WARN] no MP mask for timestep {sim_step}, skipping.")
        continue

    iy_slice = mp_max_info['iy_max']
    iz_slice = mp_max_info['iz_max']

    if debug:
        print(f"  MP max at r={mp_max_info['r_max']:.2f} R_M: "
              f"x={mp_max_info['x_max']:.2f}, y={mp_max_info['y_max']:.2f}, z={mp_max_info['z_max']:.2f}")

    # XY slice (at z = z_mp) - NUMPY INDEXING
    bg_xy = plot_bg_3d[:, :, iz_slice]
    bs_xy = bs_mask_3d.values[:, :, iz_slice].astype(np.uint8)
    mp_xy = mp_mask_3d.values[:, :, iz_slice].astype(np.uint8)
    x_plot_xy, y_plot_xy = x_coords, y_coords

    # XZ slice (at y = y_mp) - NUMPY INDEXING
    bg_xz = plot_bg_3d[:, iy_slice, :]
    bs_xz = bs_mask_3d.values[:, iy_slice, :].astype(np.uint8)
    mp_xz = mp_mask_3d.values[:, iy_slice, :].astype(np.uint8)
    x_plot_xz, y_plot_xz = x_coords, z_coords

    # create subplots
    fig, axes = plt.subplots(1, n_slices, figsize=(6 * n_slices, 6), constrained_layout=True)
    if n_slices == 1:
        axes = np.array([axes])

    last_im = None
    slice_data = [
        ("xy", x_plot_xy, y_plot_xy, bg_xy, bs_xy, mp_xy, f"Z={z_coords[iz_slice]:.2f} R$_M$"),
        ("xz", x_plot_xz, y_plot_xz, bg_xz, bs_xz, mp_xz, f"Y={y_coords[iy_slice]:.2f} R$_M$")
    ]

    cfg = PLOT_BG[plot_id]

    for ax, (use_slice, x_plot, y_plot, plot_bg, bs_mask, mp_mask, slice_pos) in zip(axes, slice_data):

        last_im = ax.pcolormesh(
            x_plot, y_plot, plot_bg.T, shading="auto",
            cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"]
        )

        if bs_mask.sum() > 0 and mp_mask.sum() > 0:
            ax.contour(x_plot, y_plot, bs_mask.astype(float).T, levels=[0.5],
                       colors=cfg['bs_col'], linewidths=2)
            ax.contour(x_plot, y_plot, mp_mask.astype(float).T, levels=[0.5],
                       colors=cfg['mp_col'], linewidths=2)

        ax.add_patch(plt.Circle((0, 0), 1, edgecolor="white", facecolor="none", linewidth=1))

        xlabel, ylabel = boundary_utils.labels_for_slice(use_slice)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_aspect("equal")
        ax.set_title(f"{use_slice.upper()}\n{slice_pos}")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, location="right", shrink=0.9)
        cbar.set_label(rf"${cfg['label']}$")

    tsec = sim_step * 0.002
    fig.suptitle(
        f"{case} {mode} - BS ({cfg['bs_col']}) and MP ({cfg['mp_col']}) at t = {tsec:.3f} s\n"
        f"Slice through MP max: x={mp_max_info['x_max']:.2f}, y={mp_max_info['y_max']:.2f}, "
        f"z={mp_max_info['z_max']:.2f}, r={mp_max_info['r_max']:.2f} R$_M$",
        fontsize=14, y=0.98
    )

    outpath = os.path.join(out_folder_ts, f"{case}_{mode}_{plot_id.lower()}_boundaries_{slice_tag}_{sim_step:06d}.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    # After computing masks, accumulate 3D positions
    bs_indices = np.argwhere(bs_mask_3d.values > 0)
    mp_indices = np.argwhere(mp_mask_3d.values > 0)

    # Convert indices to R_M coordinates
    bs_x = x_coords[bs_indices[:, 0]]
    bs_y = y_coords[bs_indices[:, 1]]
    bs_z = z_coords[bs_indices[:, 2]]
    bs_r = np.sqrt(bs_x ** 2 + bs_y ** 2 + bs_z ** 2)

    mp_x = x_coords[mp_indices[:, 0]]
    mp_y = y_coords[mp_indices[:, 1]]
    mp_z = z_coords[mp_indices[:, 2]]
    mp_r = np.sqrt(mp_x ** 2 + mp_y ** 2 + mp_z ** 2)

    # Accumulate
    acc_3d['bs']['x'].extend(bs_x)
    acc_3d['bs']['y'].extend(bs_y)
    acc_3d['bs']['z'].extend(bs_z)
    acc_3d['bs']['r'].extend(bs_r)

    acc_3d['mp']['x'].extend(mp_x)
    acc_3d['mp']['y'].extend(mp_y)
    acc_3d['mp']['z'].extend(mp_z)
    acc_3d['mp']['r'].extend(mp_r)

print("Per-timestep 3D accumulation complete.")

# ----------------------------
# POST-LOOP: 3D scatter + statistics
# ----------------------------
from mpl_toolkits.mplot3d import Axes3D

# Convert to numpy arrays
bs_x_all = np.array(acc_3d['bs']['x'])
bs_y_all = np.array(acc_3d['bs']['y'])
bs_z_all = np.array(acc_3d['bs']['z'])
bs_r_all = np.array(acc_3d['bs']['r'])

mp_x_all = np.array(acc_3d['mp']['x'])
mp_y_all = np.array(acc_3d['mp']['y'])
mp_z_all = np.array(acc_3d['mp']['z'])
mp_r_all = np.array(acc_3d['mp']['r'])

print(f"Total BS points: {len(bs_x_all)}, MP points: {len(mp_x_all)}")

# ----------------------------
# EQUATORIAL ANALYSIS
# ----------------------------
print("\n" + "=" * 60)
print("EQUATORIAL MP ANALYSIS")
print("=" * 60)

# Geographic equator: Z = 0 R_M
z_geo = 0.0
z_tolerance = 0.05  # R_M, adjust based on grid resolution

# Magnetic equator: Z = 484 km = 484/2440 R_M
z_mag_km = 484.0
z_mag = z_mag_km / 2440.0
print(f"Magnetic equator height: {z_mag:.4f} R_M ({z_mag_km:.0f} km)")

# Spatial constraints for equatorial analysis
x_min = 1.0  # R_M - dayside only
y_min = -0.25  # R_M
y_max = 0.25  # R_M
print(f"Spatial constraints: X > {x_min} R_M, {y_min} < Y < {y_max} R_M")

# Filter MP points near geographic equator with spatial constraints
geo_mask = (np.abs(mp_z_all - z_geo) < z_tolerance) & \
           (mp_x_all > x_min) & \
           (mp_y_all > y_min) & \
           (mp_y_all < y_max)

mp_geo_x = mp_x_all[geo_mask]
mp_geo_y = mp_y_all[geo_mask]
mp_geo_z = mp_z_all[geo_mask]
mp_geo_r = mp_r_all[geo_mask]

if len(mp_geo_r) > 0:
    # Calculate mean
    mean_geo_r = np.mean(mp_geo_r)

    # Calculate standard deviation
    std_geo_r = np.std(mp_geo_r)

    # Take mean x, y, z coordinates
    mean_geo_x = np.mean(mp_geo_x)
    mean_geo_y = np.mean(mp_geo_y)
    mean_geo_z = np.mean(mp_geo_z)

    # Also calculate statistics
    geo_median = np.median(mp_geo_r)
    geo_q1 = np.percentile(mp_geo_r, 25)
    geo_q3 = np.percentile(mp_geo_r, 75)
    geo_iqr = geo_q3 - geo_q1
    geo_min = np.min(mp_geo_r)
    geo_max = np.max(mp_geo_r)

    print(f"\nGeographic Equator (Z ≈ {z_geo} R_M, tolerance ±{z_tolerance} R_M):")
    print(f"  Number of MP points (after spatial filter): {len(mp_geo_r)}")
    print(f"  Mean MP distance: {mean_geo_r:.4f} ± {std_geo_r:.4f} R_M ({mean_geo_r * 2440.0:.2f} ± {std_geo_r * 2440.0:.2f} km)")
    print(f"  Median: {geo_median:.4f} R_M")
    print(f"  IQR: [{geo_q1:.4f}, {geo_q3:.4f}] R_M, width: {geo_iqr:.4f} R_M")
    print(f"  Range: [{geo_min:.4f}, {geo_max:.4f}] R_M")
    print(f"  Mean position: X={mean_geo_x:.4f}, Y={mean_geo_y:.4f}, Z={mean_geo_z:.4f} R_M")
    print(f"  Cylindrical radius: {np.sqrt(mean_geo_x ** 2 + mean_geo_y ** 2):.4f} R_M")
else:
    print(f"\nGeographic Equator: No MP points found within constraints")
    mean_geo_r = np.nan
    std_geo_r = np.nan
    mean_geo_x = np.nan
    mean_geo_y = np.nan
    mean_geo_z = np.nan
    geo_median = np.nan
    geo_min = np.nan
    geo_max = np.nan

# Filter MP points near magnetic equator with spatial constraints
mag_mask = (np.abs(mp_z_all - z_mag) < z_tolerance) & \
           (mp_x_all > x_min) & \
           (mp_y_all > y_min) & \
           (mp_y_all < y_max)

mp_mag_x = mp_x_all[mag_mask]
mp_mag_y = mp_y_all[mag_mask]
mp_mag_z = mp_z_all[mag_mask]
mp_mag_r = mp_r_all[mag_mask]

# Similar for magnetic equator
if len(mp_mag_r) > 0:
    # Calculate mean
    mean_mag_r = np.mean(mp_mag_r)

    # Calculate standard deviation
    std_mag_r = np.std(mp_mag_r)

    # Take mean x, y, z coordinates
    mean_mag_x = np.mean(mp_mag_x)
    mean_mag_y = np.mean(mp_mag_y)
    mean_mag_z = np.mean(mp_mag_z)

    # Also calculate statistics
    mag_median = np.median(mp_mag_r)
    mag_q1 = np.percentile(mp_mag_r, 25)
    mag_q3 = np.percentile(mp_mag_r, 75)
    mag_iqr = mag_q3 - mag_q1
    mag_min = np.min(mp_mag_r)
    mag_max = np.max(mp_mag_r)

    print(f"\nMagnetic Equator (Z ≈ {z_mag:.4f} R_M, tolerance ±{z_tolerance} R_M):")
    print(f"  Number of MP points (after spatial filter): {len(mp_mag_r)}")
    print(f"  Mean MP distance: {mean_mag_r:.4f} ± {std_mag_r:.4f} R_M ({mean_mag_r * 2440.0:.2f} ± {std_mag_r * 2440.0:.2f} km)")
    print(f"  Median: {mag_median:.4f} R_M")
    print(f"  IQR: [{mag_q1:.4f}, {mag_q3:.4f}] R_M, width: {mag_iqr:.4f} R_M")
    print(f"  Range: [{mag_min:.4f}, {mag_max:.4f}] R_M")
    print(f"  Mean position: X={mean_mag_x:.4f}, Y={mean_mag_y:.4f}, Z={mean_mag_z:.4f} R_M")
    print(f"  Cylindrical radius: {np.sqrt(mean_mag_x ** 2 + mean_mag_y ** 2):.4f} R_M")
else:
    print(f"\nMagnetic Equator: No MP points found within constraints")
    mean_mag_r = np.nan
    std_mag_r = np.nan
    mean_mag_x = np.nan
    mean_mag_y = np.nan
    mean_mag_z = np.nan
    mag_median = np.nan
    mag_min = np.nan
    mag_max = np.nan

print("=" * 60 + "\n")

# Calculate overall statistics
bs_r_median = np.median(bs_r_all) if len(bs_r_all) > 0 else np.nan
bs_r_q1 = np.percentile(bs_r_all, 25) if len(bs_r_all) > 0 else np.nan
bs_r_q3 = np.percentile(bs_r_all, 75) if len(bs_r_all) > 0 else np.nan
bs_r_iqr = bs_r_q3 - bs_r_q1

mp_r_median = np.median(mp_r_all) if len(mp_x_all) > 0 else np.nan
mp_r_q1 = np.percentile(mp_r_all, 25) if len(mp_x_all) > 0 else np.nan
mp_r_q3 = np.percentile(mp_r_all, 75) if len(mp_x_all) > 0 else np.nan
mp_r_iqr = mp_r_q3 - mp_r_q1

print(f"BS shell: median={bs_r_median:.3f} R_M, IQR=[{bs_r_q1:.3f}, {bs_r_q3:.3f}], width={bs_r_iqr:.3f} R_M")
print(f"MP shell: median={mp_r_median:.3f} R_M, IQR=[{mp_r_q1:.3f}, {mp_r_q3:.3f}], width={mp_r_iqr:.3f} R_M")

# Downsample for plotting
max_plot_points = 50000
if len(bs_x_all) > max_plot_points:
    bs_sample = np.random.choice(len(bs_x_all), max_plot_points, replace=False)
    bs_x_plot = bs_x_all[bs_sample]
    bs_y_plot = bs_y_all[bs_sample]
    bs_z_plot = bs_z_all[bs_sample]
else:
    bs_x_plot, bs_y_plot, bs_z_plot = bs_x_all, bs_y_all, bs_z_all

if len(mp_x_all) > max_plot_points:
    mp_sample = np.random.choice(len(mp_x_all), max_plot_points, replace=False)
    mp_x_plot = mp_x_all[mp_sample]
    mp_y_plot = mp_y_all[mp_sample]
    mp_z_plot = mp_z_all[mp_sample]
else:
    mp_x_plot, mp_y_plot, mp_z_plot = mp_x_all, mp_y_all, mp_z_all

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

# Create subplots with 3D scenes
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=('Cumulative Boundaries',
                    'Equatorial MP Analysis')
)

# ========================================
# LEFT PLOT: All scatter points
# ========================================
# Bowshock - independent toggle
fig.add_trace(
    go.Scatter3d(
        x=bs_x_plot, y=bs_y_plot, z=bs_z_plot,
        mode='markers',
        marker=dict(size=3, color=cfg['bs_col'], opacity=0.8),
        name='Bowshock',
        showlegend=True
    ),
    row=1, col=1
)

# Magnetopause - independent toggle
fig.add_trace(
    go.Scatter3d(
        x=mp_x_plot, y=mp_y_plot, z=mp_z_plot,
        mode='markers',
        marker=dict(size=3, color=cfg['mp_col'], opacity=0.8),
        name='Magnetopause',
        showlegend=True
    ),
    row=1, col=1
)

# Mercury sphere - left (light grey dayside)
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
    ),
    row=1, col=1
)

# Mercury sphere - left (black nightside)
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
    ),
    row=1, col=1
)

# ========================================
# RIGHT PLOT: Equatorial points only
# ========================================
# Mercury sphere - right (light grey dayside)
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
    ),
    row=1, col=2
)

# Mercury sphere - right (black nightside)
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
    ),
    row=1, col=2
)

# Geographic equator MP points
if len(mp_geo_x) > 0:
    # Downsample if needed
    max_eq_points = 5000
    if len(mp_geo_x) > max_eq_points:
        geo_sample = np.random.choice(len(mp_geo_x), max_eq_points, replace=False)
        geo_x = mp_geo_x[geo_sample]
        geo_y = mp_geo_y[geo_sample]
        geo_z = mp_geo_z[geo_sample]
    else:
        geo_x, geo_y, geo_z = mp_geo_x, mp_geo_y, mp_geo_z

    fig.add_trace(
        go.Scatter3d(
            x=geo_x, y=geo_y, z=geo_z,
            mode='markers',
            marker=dict(size=2, color='cyan', opacity=0.5),
            name=f'Geo. Eq. MP (Z≈0)',
            legendgroup='geo_eq',
            showlegend=True
        ),
        row=1, col=2
    )

# Magnetic equator MP points
if len(mp_mag_x) > 0:
    # Downsample if needed
    if len(mp_mag_x) > max_eq_points:
        mag_sample = np.random.choice(len(mp_mag_x), max_eq_points, replace=False)
        mag_x = mp_mag_x[mag_sample]
        mag_y = mp_mag_y[mag_sample]
        mag_z = mp_mag_z[mag_sample]
    else:
        mag_x, mag_y, mag_z = mp_mag_x, mp_mag_y, mp_mag_z

    fig.add_trace(
        go.Scatter3d(
            x=mag_x, y=mag_y, z=mag_z,
            mode='markers',
            marker=dict(size=2, color='yellow', opacity=0.5),
            name=f'Mag. Eq. MP (Z≈{z_mag:.2f})',
            legendgroup='mag_eq',
            showlegend=True
        ),
        row=1, col=2
    )

# Geographic equator mean
if not np.isnan(mean_geo_r):
    fig.add_trace(
        go.Scatter3d(
            x=[mean_geo_x], y=[mean_geo_y], z=[mean_geo_z],
            mode='markers',
            marker=dict(size=10, color='cyan', symbol='diamond',
                        line=dict(color='black', width=2)),
            name=f'Geo. Eq.: {mean_geo_r:.2f} ± {std_geo_r:.2f} R<sub>M</sub>',
            legendgroup='geo_eq',
            showlegend=True,
            hovertemplate='Mean Point<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>' +
                         f'R: {mean_geo_r:.2f} ± {std_geo_r:.2f} R_M'
        ),
        row=1, col=2
    )

# Magnetic equator mean
if not np.isnan(mean_mag_r):
    fig.add_trace(
        go.Scatter3d(
            x=[mean_mag_x], y=[mean_mag_y], z=[mean_mag_z],
            mode='markers',
            marker=dict(size=10, color='yellow', symbol='diamond',
                        line=dict(color='black', width=2)),
            name=f'Mag. Eq.: {mean_mag_r:.2f} ± {std_mag_r:.2f} R<sub>M</sub>',
            legendgroup='mag_eq',
            showlegend=True,
            hovertemplate='Mean Point<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>' +
                         f'R: {mean_mag_r:.2f} ± {std_mag_r:.2f} R_M'
        ),
        row=1, col=2
    )

# Update layout and camera positions
camera_oblique = dict(
    eye=dict(x=1.5, y=1.5, z=1.0),
    center=dict(x=0, y=0, z=0)
)

camera_equator = dict(
    eye=dict(x=2.0, y=0.5, z=0.5),  # Side view to see equatorial planes
    center=dict(x=0, y=0, z=0)
)

fig.update_layout(
    scene=dict(
        xaxis=dict(title='X (R<sub>M</sub>)', range=[-2, 2]),
        yaxis=dict(title='Y (R<sub>M</sub>)', range=[-2, 2]),
        zaxis=dict(title='Z (R<sub>M</sub>)', range=[-2, 2]),
        aspectmode='cube',
        camera=camera_oblique
    ),
    scene2=dict(
        xaxis=dict(title='X (R<sub>M</sub>)', range=[-2, 2]),
        yaxis=dict(title='Y (R<sub>M</sub>)', range=[-2, 2]),
        zaxis=dict(title='Z (R<sub>M</sub>)', range=[-2, 2]),
        aspectmode='cube',
        camera=camera_equator
    ),
    height=600,
    width=1400,
    showlegend=True,
    legend=dict(x=1.02, y=0.95, bgcolor='rgba(255,255,255,0.9)'),
    template='plotly_white'
)

# Set title
stitle = f"{case} {mode} - Boundary Shells"
if "Base" in mode:
    stitle += ": Pre-Transient"
    html_path = os.path.join(out_dir, f"{case}_{mode}_{plot_id.lower()}_3D_cumulative_scatter.html")
elif "HNHV" in mode or "LNHV" in mode:
    if transient:
        stitle += ": Transient"
        html_path = os.path.join(out_dir, f"{case}_{mode}_{plot_id.lower()}_3D_cumulative_scatter_transient.html")
    elif post_transient:
        stitle += ": Post-Transient"
        html_path = os.path.join(out_dir, f"{case}_{mode}_{plot_id.lower()}_3D_cumulative_scatter_post-transient.html")
    elif new_state:
        stitle += ": New State"
        html_path = os.path.join(out_dir, f"{case}_{mode}_{plot_id.lower()}_3D_cumulative_scatter_newstate.html")

fig.update_layout(title_text=stitle, title_x=0.5, title_font_size=18)

# Save as interactive HTML
fig.write_html(html_path)
print(f"Saved interactive 3D plot: {html_path}")

# ----------------------------
# CSV OUTPUT: 3D shell statistics + equatorial analysis
# ----------------------------
records = [
    {
        "boundary": "BowShock",
        "median_r_re": bs_r_median,
        "q1_r_re": bs_r_q1,
        "q3_r_re": bs_r_q3,
        "iqr_r_re": bs_r_iqr,
        "median_r_km": bs_r_median * 2440.0,
        "iqr_r_km": bs_r_iqr * 2440.0,
        "n_points": len(bs_x_all),
        "geo_eq_mean_r_re": np.nan,
        "geo_eq_median_r_re": np.nan,
        "geo_eq_min_r_re": np.nan,
        "geo_eq_max_r_re": np.nan,
        "geo_eq_std_r_re": np.nan,
        "geo_eq_mean_r_km": np.nan,
        "geo_eq_median_r_km": np.nan,
        "geo_eq_min_r_km": np.nan,
        "geo_eq_max_r_km": np.nan,
        "geo_eq_std_r_km": np.nan,
        "geo_eq_n_points": 0,
        "mag_eq_mean_r_re": np.nan,
        "mag_eq_median_r_re": np.nan,
        "mag_eq_min_r_re": np.nan,
        "mag_eq_max_r_re": np.nan,
        "mag_eq_std_r_re": np.nan,
        "mag_eq_mean_r_km": np.nan,
        "mag_eq_median_r_km": np.nan,
        "mag_eq_min_r_km": np.nan,
        "mag_eq_max_r_km": np.nan,
        "mag_eq_std_r_km": np.nan,
        "mag_eq_n_points": 0
    },
    {
        "boundary": "Magnetopause",
        "median_r_re": mp_r_median,
        "q1_r_re": mp_r_q1,
        "q3_r_re": mp_r_q3,
        "iqr_r_re": mp_r_iqr,
        "median_r_km": mp_r_median * 2440.0,
        "iqr_r_km": mp_r_iqr * 2440.0,
        "n_points": len(mp_x_all),
        "geo_eq_mean_r_re": mean_geo_r if not np.isnan(mean_geo_r) else np.nan,
        "geo_eq_median_r_re": geo_median if not np.isnan(mean_geo_r) else np.nan,
        "geo_eq_min_r_re": geo_min if not np.isnan(mean_geo_r) else np.nan,
        "geo_eq_max_r_re": geo_max if not np.isnan(mean_geo_r) else np.nan,
        "geo_eq_std_r_re": std_geo_r if not np.isnan(std_geo_r) else np.nan,
        "geo_eq_mean_r_km": mean_geo_r * 2440.0 if not np.isnan(mean_geo_r) else np.nan,
        "geo_eq_median_r_km": geo_median * 2440.0 if not np.isnan(mean_geo_r) else np.nan,
        "geo_eq_min_r_km": geo_min * 2440.0 if not np.isnan(mean_geo_r) else np.nan,
        "geo_eq_max_r_km": geo_max * 2440.0 if not np.isnan(mean_geo_r) else np.nan,
        "geo_eq_std_r_km": std_geo_r * 2440.0 if not np.isnan(std_geo_r) else np.nan,
        "geo_eq_n_points": len(mp_geo_r) if len(mp_geo_r) > 0 else 0,
        "mag_eq_mean_r_re": mean_mag_r if not np.isnan(mean_mag_r) else np.nan,
        "mag_eq_median_r_re": mag_median if not np.isnan(mean_mag_r) else np.nan,
        "mag_eq_min_r_re": mag_min if not np.isnan(mean_mag_r) else np.nan,
        "mag_eq_max_r_re": mag_max if not np.isnan(mean_mag_r) else np.nan,
        "mag_eq_std_r_re": std_mag_r if not np.isnan(std_mag_r) else np.nan,
        "mag_eq_mean_r_km": mean_mag_r * 2440.0 if not np.isnan(mean_mag_r) else np.nan,
        "mag_eq_median_r_km": mag_median * 2440.0 if not np.isnan(mean_mag_r) else np.nan,
        "mag_eq_min_r_km": mag_min * 2440.0 if not np.isnan(mean_mag_r) else np.nan,
        "mag_eq_max_r_km": mag_max * 2440.0 if not np.isnan(mean_mag_r) else np.nan,
        "mag_eq_std_r_km": std_mag_r * 2440.0 if not np.isnan(std_mag_r) else np.nan,
        "mag_eq_n_points": len(mp_mag_r) if len(mp_mag_r) > 0 else 0
    }
]

df = pd.DataFrame(records)
if "Base" in mode:
    csv_path = os.path.join(out_dir, f"{case}_{mode}_{plot_id.lower()}_3D_shell_statistics.csv")
elif "HNHV" in mode or "LNHV" in mode:
    if transient:
        csv_path = os.path.join(out_dir, f"{case}_{mode}_{plot_id.lower()}_3D_shell_statistics_transient.csv")
    elif post_transient:
        csv_path = os.path.join(out_dir, f"{case}_{mode}_{plot_id.lower()}_3D_shell_statistics_post-transient.csv")
    elif new_state:
        csv_path = os.path.join(out_dir, f"{case}_{mode}_{plot_id.lower()}_3D_shell_statistics_newstate.csv")

df.to_csv(csv_path, index=False)
print(f"Saved 3D shell statistics: {csv_path}")

# Print only MP equatorial statistics
print(f"\n{case} - Magnetopause Equatorial Statistics:")
print("="*70)

mp_row = df[df['boundary'] == 'Magnetopause'].iloc[0]

print("\nGeographic Equator (Z ≈ 0 R_M):")
print(f"  Mean:   {mp_row['geo_eq_mean_r_re']:.2f} R_M ({mp_row['geo_eq_mean_r_km']:.2f} km)")
print(f"  Median: {mp_row['geo_eq_median_r_re']:.2f} R_M ({mp_row['geo_eq_median_r_km']:.2f} km)")
print(f"  Std:    {mp_row['geo_eq_std_r_re']:.2f} R_M ({mp_row['geo_eq_std_r_km']:.2f} km)")
print(f"  Min:    {mp_row['geo_eq_min_r_re']:.2f} R_M ({mp_row['geo_eq_min_r_km']:.2f} km)")
print(f"  Max:    {mp_row['geo_eq_max_r_re']:.2f} R_M ({mp_row['geo_eq_max_r_km']:.2f} km)")
print(f"  N:      {mp_row['geo_eq_n_points']:.0f} points")

print(f"\nMagnetic Equator (Z ≈ {z_mag:.2f} R_M):")
print(f"  Mean:   {mp_row['mag_eq_mean_r_re']:.2f} R_M ({mp_row['mag_eq_mean_r_km']:.2f} km)")
print(f"  Median: {mp_row['mag_eq_median_r_re']:.2f} R_M ({mp_row['mag_eq_median_r_km']:.2f} km)")
print(f"  Std:    {mp_row['mag_eq_std_r_re']:.2f} R_M ({mp_row['mag_eq_std_r_km']:.2f} km)")
print(f"  Min:    {mp_row['mag_eq_min_r_re']:.2f} R_M ({mp_row['mag_eq_min_r_km']:.2f} km)")
print(f"  Max:    {mp_row['mag_eq_max_r_re']:.2f} R_M ({mp_row['mag_eq_max_r_km']:.2f} km)")
print(f"  N:      {mp_row['mag_eq_n_points']:.0f} points")

print("="*70)
