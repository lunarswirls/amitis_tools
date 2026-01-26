#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports:
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.bs_mp_finder.boundary_utils import *

# ----------------------------
# SETTINGS
# ----------------------------
debug = False

# base cases: CPN_Base RPN_Base CPS_Base RPS_Base
# HNHV cases: CPN_HNHV RPN_HNHV CPS_HNHV RPS_HNHV
case = "CPS_Base"

if "Base" in case:
    base_dir = f"/Volumes/data_backup/mercury/extreme/{case}/plane_product/"
elif "HNHV" in case:
    base_dir = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/"

# use_slices = ["xy", "xz", "yz"]  # plot all 3
use_slices = ["xy", "xz"]  # plot only 2
n_slices = len(use_slices)       # number of requested slices
slice_tag = "_".join(use_slices)

# Plot background selection
plot_id = "Pmag"   # options: "Bmag", "Jmag", "Pmag"

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

# take last 15-ish seconds
sim_steps = list(range(98000, 115000 + 1, 1000))

out_dir = f"/Users/danywaller/Projects/mercury/extreme/slice_bowshock/{case}/"
os.makedirs(out_dir, exist_ok=True)

out_folder_ts = os.path.join(out_dir, f"timeseries_{slice_tag}/")
os.makedirs(out_folder_ts, exist_ok=True)

RM_M = 2440.0e3

# ----------------------------
# ACCUMULATORS for median plot
# ----------------------------
acc = {
    s: dict(plot_bg=[], bs=[], mp=[], x_plot=None, y_plot=None)
    for s in use_slices
}

# ----------------------------
# LOOP: per-timestep plotting + accumulation
# ----------------------------
for sim_step in sim_steps:
    filename = f"{sim_step:06d}"

    # create only as many subplots as requested
    fig, axes = plt.subplots(1, n_slices, figsize=(6 * n_slices, 6), constrained_layout=True)
    if n_slices == 1:
        axes = np.array([axes])

    last_im = None

    for ax, use_slice in zip(axes, use_slices):
        input_folder = os.path.join(base_dir, f"fig_{use_slice}")
        f = os.path.join(input_folder, f"Amitis_{case}_{filename}_{use_slice}_comp.nc")

        if not os.path.exists(f):
            ax.axis("off")
            ax.set_title(f"{use_slice.upper()} missing")
            print(f"[WARN] missing: {f}")
            continue

        ds = xr.open_dataset(f)
        x_plot, y_plot, plot_bg, bs_mask, mp_mask = compute_masks_one_timestep(ds, use_slice, plot_id)
        ds.close()

        if bs_mask.size != 0 and mp_mask.size != 0:
            # accumulate for post-loop median plot
            acc[use_slice]["plot_bg"].append(plot_bg)
            acc[use_slice]["bs"].append(bs_mask.values.astype(np.uint8))  # store as 0/1
            acc[use_slice]["mp"].append(mp_mask.values.astype(np.uint8))
            acc[use_slice]["x_plot"] = x_plot
            acc[use_slice]["y_plot"] = y_plot

        # per-timestep plot
        cfg = PLOT_BG[plot_id]
        last_im = ax.pcolormesh(x_plot, y_plot, plot_bg, shading="auto", cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"])
        if bs_mask.size != 0 and mp_mask.size != 0:
            ax.contour(x_plot, y_plot, bs_mask.values.astype(float), levels=[0.5], colors=cfg['bs_col'], linewidths=2)
            ax.contour(x_plot, y_plot, mp_mask.values.astype(float), levels=[0.5], colors=cfg['mp_col'], linewidths=2)
        ax.add_patch(plt.Circle((0, 0), 1, edgecolor="white", facecolor="none", linewidth=1))

        xlabel, ylabel = labels_for_slice(use_slice)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_aspect("equal")
        ax.set_title(use_slice.upper())

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, location="right", shrink=0.9)
        cbar.set_label(rf"${cfg['label']}$")

    tsec = sim_step * 0.002
    fig.suptitle(f"{case.replace("_", " ")} - BS ({cfg['bs_col']}) and MP ({cfg['mp_col']}) position at t = {tsec:.3f} s", fontsize=18, y=0.99)

    outpath = os.path.join(out_folder_ts, f"{case}_{plot_id.lower()}_boundaries_{slice_tag}_{sim_step:06d}.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

print("Per-timestep plots complete.")

# ----------------------------
# POST-LOOP: median + IQR figure (1x3)
# ----------------------------
fig, axes = plt.subplots(1, n_slices, figsize=(6 * n_slices, 6), constrained_layout=True)
if n_slices == 1:
    axes = np.array([axes])
last_im = None

for ax, use_slice in zip(axes, use_slices):
    if len(acc[use_slice]["plot_bg"]) == 0:
        ax.axis("off")
        ax.set_title(f"{use_slice.upper()} no data")
        continue

    x_plot = acc[use_slice]["x_plot"]
    y_plot = acc[use_slice]["y_plot"]

    plot_bg_med = np.median(np.stack(acc[use_slice]["plot_bg"], axis=0), axis=0)

    bs_stack = np.stack(acc[use_slice]["bs"], axis=0).astype(float)
    mp_stack = np.stack(acc[use_slice]["mp"], axis=0).astype(float)
    bs_stack[bs_stack == 0] = np.nan
    mp_stack[mp_stack == 0] = np.nan

    # IQR envelopes
    bs_q1 = np.nanpercentile(bs_stack, 25, axis=0)
    bs_q3 = np.nanpercentile(bs_stack, 75, axis=0)
    mp_q1 = np.nanpercentile(mp_stack, 25, axis=0)
    mp_q3 = np.nanpercentile(mp_stack, 75, axis=0)

    # occupancy-threshold median masks (consistent with your earlier approach)
    bs_occ, _, bs_med, _ = occupancy_and_bands(np.stack(acc[use_slice]["bs"], axis=0).astype(bool))
    mp_occ, _, mp_med, _ = occupancy_and_bands(np.stack(acc[use_slice]["mp"], axis=0).astype(bool))

    cfg = PLOT_BG[plot_id]

    last_im = ax.pcolormesh(x_plot, y_plot, plot_bg_med, shading="auto", cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"])

    # IQR filled regions
    ax.contourf(x_plot, y_plot, (bs_q1 > 0) & (bs_q3 > 0), levels=[0.5, 1], colors=cfg['bs_col'], alpha=0.4)
    ax.contourf(x_plot, y_plot, (mp_q1 > 0) & (mp_q3 > 0), levels=[0.5, 1], colors=cfg['mp_col'], alpha=0.4)

    # median contours
    ax.contour(x_plot, y_plot, bs_med.astype(float), levels=[0.5], colors=cfg['bs_col'], linewidths=2)
    ax.contour(x_plot, y_plot, mp_med.astype(float), levels=[0.5], colors=cfg['mp_col'], linewidths=2)

    # --------------------------------------------------
    # max median distance along axis (XY & XZ)
    # --------------------------------------------------
    bs_max = max_axis_distance(bs_med, x_plot, y_plot)
    mp_max = max_axis_distance(mp_med, x_plot, y_plot)

    legend_handles = []

    if bs_max is not None:
        bs_r = bs_max[2].item() if isinstance(bs_max[2], np.ndarray) else float(bs_max[2])

        ax.plot(
            bs_max[0], bs_max[1],
            marker="o",
            markersize=8,
            color=cfg['bs_col'],
            markeredgecolor="white",
            zorder=10
        )

        legend_handles.append(
            Line2D(
                [], [], marker="o", linestyle="None",
                color=cfg['bs_col'],
                markeredgecolor="white",
                markersize=8,
                label=f"BS max: {bs_r:.2f} R$_M$"
            )
        )

    if mp_max is not None:
        mp_r = mp_max[2].item() if isinstance(mp_max[2], np.ndarray) else float(mp_max[2])
        ax.plot(
            mp_max[0], mp_max[1],
            marker="s",
            markersize=8,
            color=cfg['mp_col'],
            markeredgecolor="white",
            zorder=10
        )

        legend_handles.append(
            Line2D(
                [], [], marker="s", linestyle="None",
                color=cfg['mp_col'],
                markeredgecolor="white",
                markersize=8,
                label=f"MP max: {mp_r:.2f} R$_M$"
            )
        )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            fontsize=10
        )

    ax.add_patch(plt.Circle((0, 0), 1, edgecolor="white", facecolor="none", linewidth=1))

    xlabel, ylabel = labels_for_slice(use_slice)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_aspect("equal")
    ax.set_title(use_slice.upper())

if last_im is not None:
    cbar = fig.colorbar(last_im, ax=axes, location="right", shrink=0.75)
    cbar.set_label(rf"$\mathrm{{Median}}\ {cfg['label']}$")

if n_slices > 2:
    fig.suptitle(f"{case.replace("_", " ")} BS ({cfg['bs_col']}) and MP ({cfg['mp_col']}) IQR envelopes with median occupancy contour", fontsize=18, y=0.99)
else:
    fig.suptitle(f"{case.replace("_", " ")} BS ({cfg['bs_col']}) and MP ({cfg['mp_col']}) IQR envelopes with median occupancy contour", fontsize=14, y=0.99)

median_path = os.path.join(out_dir, f"{case}_{plot_id.lower()}_boundaries_{slice_tag}_median.png")
fig.savefig(median_path, dpi=300)
plt.close(fig)

print(f"Saved median plot: {median_path}")

records = []

for boundary, key in [("BowShock", "bs"), ("Magnetopause", "mp")]:

    for use_slice in ["xy", "xz"]:
        if use_slice not in acc:
            continue
        if len(acc[use_slice][key]) == 0:
            continue

        x = acc[use_slice]["x_plot"]
        y = acc[use_slice]["y_plot"]

        # median occupancy contour
        occ, _, med, _ = occupancy_and_bands(
            np.stack(acc[use_slice][key], axis=0).astype(bool)
        )

        # max extraction
        max_pt = max_axis_distance(med, x, y)

        if max_pt is None:
            continue

        xv, av, r = max_pt  # guaranteed floats

        records.append({
            "boundary": boundary,
            "slice": use_slice.upper(),
            "x_re": xv,
            "axis_re": av,                 # Y (XY) or Z (XZ)
            "distance_re": r,
            "distance_km": r * (RM_M * 1e-3)
        })

# ----------------------------
# Write CSV
# ----------------------------
df = pd.DataFrame(records)

csv_path = os.path.join(
    out_dir,
    f"{case}_{plot_id.lower()}_median_boundary_distances_{slice_tag}.csv"
)

df.to_csv(csv_path, index=False)
print(f"Saved median boundary distances: {csv_path}")
