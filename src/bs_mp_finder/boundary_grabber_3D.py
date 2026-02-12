#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import bs_mp_finder.boundary_utils as boundary_utils

# ----------------------------
# SETTINGS
# ----------------------------
debug = False

# base cases: CPN_Base RPN_Base CPS_Base RPS_Base
# HNHV cases: CPN_HNHV RPN_HNHV CPS_HNHV RPS_HNHV
case = "CPS_Base"

# FOR HNHV - DOUBLE CHECK ONLY ONE IS TRUE!!!!
transient = False  # 280-300 s
post_transient = False  # 330-350 s
new_state = False  # 680-700 s

if "Base" in case:
    input_folder = f"/Volumes/data_backup/mercury/extreme/{case}/plane_product/object/"
    sim_steps = list(range(105000, 115000 + 1, 1000))
elif "HNHV" in case:
    input_folder = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/object/"
    if transient and not post_transient and not new_state:
        sim_steps = range(140000, 150000 + 1, 1000)
    elif post_transient and not transient and not new_state:
        sim_steps = range(165000, 175000 + 1, 1000)
    elif new_state and not post_transient and not transient:
        sim_steps = range(340000, 350000 + 1, 1000)
    else:
        raise ValueError("Too many flags! Set only one of transient, post_transient, or new_state to True")
else:
    raise ValueError("Unrecognized case! Are you using one of Base or HNHV?")

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

out_dir = f"/Users/danywaller/Projects/mercury/extreme/boundary_3D_timeseries/{case}/"
os.makedirs(out_dir, exist_ok=True)
out_folder_ts = os.path.join(out_dir, f"timeseries_{slice_tag}/")
os.makedirs(out_folder_ts, exist_ok=True)

RM_M = 2440.0e3

# ----------------------------
# ACCUMULATORS for 3D positions (MODIFIED)
# ----------------------------
acc_3d = {
    'bs': {'x': [], 'y': [], 'z': [], 'r': []},
    'mp': {'x': [], 'y': [], 'z': [], 'r': []}
}

# ----------------------------
# LOOP: per-timestep 3D processing + 2D slicing + accumulation
# ----------------------------
for sim_step in sim_steps:
    filename = f"{sim_step:06d}"
    f_3d = os.path.join(input_folder, f"Amitis_{case}_{filename}_xz_comp.nc")

    if not os.path.exists(f_3d):
        print(f"[WARN] missing 3D file: {f_3d}")
        continue

    print(f"Processing timestep {sim_step}: {filename}")

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
        f"{case.replace('_', ' ')} - BS ({cfg['bs_col']}) and MP ({cfg['mp_col']}) at t = {tsec:.3f} s\n"
        f"Slice through MP max: x={mp_max_info['x_max']:.2f}, y={mp_max_info['y_max']:.2f}, "
        f"z={mp_max_info['z_max']:.2f}, r={mp_max_info['r_max']:.2f} R$_M$",
        fontsize=14, y=0.98
    )

    outpath = os.path.join(out_folder_ts, f"{case}_{plot_id.lower()}_boundaries_{slice_tag}_{sim_step:06d}.png")
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

# Calculate statistics
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
max_plot_points = 10000
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

# Create 3D scatter plot
cfg = PLOT_BG[plot_id]
fig = plt.figure(figsize=(16, 7))

# Left: oblique view
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(bs_x_plot, bs_y_plot, bs_z_plot, c=cfg['bs_col'], s=0.5, alpha=0.2, label='Bowshock')
ax1.scatter(mp_x_plot, mp_y_plot, mp_z_plot, c=cfg['mp_col'], s=0.5, alpha=0.2, label='Magnetopause')

# Mercury sphere
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sph = np.outer(np.cos(u), np.sin(v))
y_sph = np.outer(np.sin(u), np.sin(v))
z_sph = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_surface(x_sph, y_sph, z_sph, color='gray', alpha=0.6, edgecolor='none')

ax1.set_xlabel(r'$X\ (R_M)$', fontsize=12)
ax1.set_ylabel(r'$Y\ (R_M)$', fontsize=12)
ax1.set_zlabel(r'$Z\ (R_M)$', fontsize=12)
ax1.set_xlim([-4, 4])
ax1.set_ylim([-4, 4])
ax1.set_zlim([-4, 4])
ax1.view_init(elev=20, azim=45)
ax1.legend(loc='upper left', markerscale=10)
ax1.set_box_aspect([1, 1, 1])
ax1.set_title('Cumulative 3D Boundaries\n(All timesteps)', fontsize=12)

# Right: top-down view (XZ plane)
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(bs_x_plot, bs_y_plot, bs_z_plot, c=cfg['bs_col'], s=0.5, alpha=0.2, label='Bowshock')
ax2.scatter(mp_x_plot, mp_y_plot, mp_z_plot, c=cfg['mp_col'], s=0.5, alpha=0.2, label='Magnetopause')
ax2.plot_surface(x_sph, y_sph, z_sph, color='gray', alpha=0.6, edgecolor='none')

ax2.set_xlabel(r'$X\ (R_M)$', fontsize=12)
ax2.set_ylabel(r'$Y\ (R_M)$', fontsize=12)
ax2.set_zlabel(r'$Z\ (R_M)$', fontsize=12)
ax2.set_xlim([-4, 4])
ax2.set_ylim([-4, 4])
ax2.set_zlim([-4, 4])
ax2.view_init(elev=90, azim=-90)  # Top-down
ax2.legend(loc='upper left', markerscale=10)
ax2.set_box_aspect([1, 1, 1])
ax2.set_title(f'BS: r={bs_r_median:.2f}±{bs_r_iqr / 2:.2f} R$_M$\n'
              f'MP: r={mp_r_median:.2f}±{mp_r_iqr / 2:.2f} R$_M$', fontsize=12)

stitle = f"{case.replace('_', ' ')} - 3D Boundary Shells"
if "Base" in case:
    stitle += ": Pre-Transient"
    scatter_path = os.path.join(out_dir, f"{case}_{plot_id.lower()}_3D_cumulative_scatter.png")
elif "HNHV" in case:
    if transient:
        stitle += ": Transient"
        scatter_path = os.path.join(out_dir, f"{case}_{plot_id.lower()}_3D_cumulative_scatter_transient.png")
    elif post_transient:
        stitle += ": Post-Transient"
        scatter_path = os.path.join(out_dir, f"{case}_{plot_id.lower()}_3D_cumulative_scatter_post-transient.png")
    elif new_state:
        stitle += ": New State"
        scatter_path = os.path.join(out_dir, f"{case}_{plot_id.lower()}_3D_cumulative_scatter_newstate.png")

fig.suptitle(stitle, fontsize=16, y=0.98)
fig.savefig(scatter_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved 3D scatter plot: {scatter_path}")

# ----------------------------
# CSV OUTPUT: 3D shell statistics
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
        "n_points": len(bs_x_all)
    },
    {
        "boundary": "Magnetopause",
        "median_r_re": mp_r_median,
        "q1_r_re": mp_r_q1,
        "q3_r_re": mp_r_q3,
        "iqr_r_re": mp_r_iqr,
        "median_r_km": mp_r_median * 2440.0,
        "iqr_r_km": mp_r_iqr * 2440.0,
        "n_points": len(mp_x_all)
    }
]

df = pd.DataFrame(records)
if "Base" in case:
    csv_path = os.path.join(out_dir, f"{case}_{plot_id.lower()}_3D_shell_statistics.csv")
elif "HNHV" in case:
    if transient:
        csv_path = os.path.join(out_dir, f"{case}_{plot_id.lower()}_3D_shell_statistics_transient.csv")
    elif post_transient:
        csv_path = os.path.join(out_dir, f"{case}_{plot_id.lower()}_3D_shell_statistics_post-transient.csv")
    elif new_state:
        csv_path = os.path.join(out_dir, f"{case}_{plot_id.lower()}_3D_shell_statistics_newstate.csv")

df.to_csv(csv_path, index=False)
print(f"Saved 3D shell statistics: {csv_path}")
print(f"\n{case} Statistics:")
print(df.to_string(index=False))