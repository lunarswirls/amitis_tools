#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports:
import pandas as pd
from matplotlib.lines import Line2D
from bs_mp_finder.zzz_archive.boundary_utils import *

# ----------------------------
# SETTINGS
# ----------------------------
debug = False

# base cases: CPN_Base RPN_Base CPS_Base RPS_Base
# HNHV cases: CPN_HNHV RPN_HNHV CPS_HNHV RPS_HNHV
case = "RPS_Base"

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
        "vmax": 100.0,
        "bs_col": "red",
        "mp_col": "magenta",
    },
}

# take discrete timestep
sim_steps = [115000]

out_dir = f"/Users/danywaller/Projects/mercury/extreme/boundary_id_1timestep/{case}/"
os.makedirs(out_dir, exist_ok=True)

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
        x_plot, y_plot, plot_bg, bs_mask, mp_mask = compute_masks_one_timestep(ds, use_slice, plot_id, case, sim_step)
        ds.close()

        bs_max = None
        mp_max = None

        if bs_mask.size != 0:
            bs_max = max_axis_distance(bs_mask.values.astype(bool), x_plot, y_plot)

        if mp_mask.size != 0:
            mp_max = max_axis_distance(mp_mask.values.astype(bool), x_plot, y_plot)

        # per-timestep plot
        cfg = PLOT_BG[plot_id]
        last_im = ax.pcolormesh(x_plot, y_plot, plot_bg, shading="auto", cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"])
        if bs_mask.size != 0 and mp_mask.size != 0:
            ax.contour(x_plot, y_plot, bs_mask.values.astype(float), levels=[0.5], colors=cfg['bs_col'], linewidths=2)
            ax.contour(x_plot, y_plot, mp_mask.values.astype(float), levels=[0.5], colors=cfg['mp_col'], linewidths=2)

        legend_handles = []

        if bs_max is not None:
            xv, av, r = bs_max
            ax.plot(xv, av, marker="o", color=cfg['bs_col'],
                    markeredgecolor="white", markersize=8, zorder=10)

            legend_handles.append(
                Line2D([], [], marker="o", linestyle="None",
                       color=cfg['bs_col'], markeredgecolor="white",
                       label=f"BS: {r:.2f} R$_M$")
            )

        if mp_max is not None:
            xv, av, r = mp_max
            ax.plot(xv, av, marker="s", color=cfg['mp_col'],
                    markeredgecolor="white", markersize=8, zorder=10)

            legend_handles.append(
                Line2D([], [], marker="s", linestyle="None",
                       color=cfg['mp_col'], markeredgecolor="white",
                       label=f"MP: {r:.2f} R$_M$")
            )

        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right")

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

    outpath = os.path.join(out_dir, f"{case}_{plot_id.lower()}_boundaries_{slice_tag}_{sim_step:06d}.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

print("Timestep plot complete.")

records = []

for boundary, max_pt in [("BowShock", bs_max), ("Magnetopause", mp_max)]:
    if max_pt is None:
        continue

    xv, av, r = max_pt

    records.append({
        "timestep": sim_step,
        "boundary": boundary,
        "slice": use_slice.upper(),
        "x_re": xv,
        "axis_re": av,
        "distance_re": r,
        "distance_km": r * (RM_M * 1e-3)
    })

df = pd.DataFrame(records)
csv_path = os.path.join(
    out_dir,
    f"{case}_{plot_id.lower()}_boundary_distances_{slice_tag}_{sim_step:06d}.csv"
)
df.to_csv(csv_path, index=False)
