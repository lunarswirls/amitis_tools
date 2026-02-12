#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from bs_mp_finder.zzz_archive.boundary_utils import labels_for_slice, coords_for_slice

# base cases: CPN_Base RPN_Base CPS_Base RPS_Base
# HNHV cases: CPN_HNHV RPN_HNHV CPS_HNHV RPS_HNHV
case = "CPN_Base"

use_slices = ["xy", "xz", "yz"]  # plot all 3
n_slices = len(use_slices)       # number of requested slices
slice_tag = "_".join(use_slices)

if "Base" in case:
    base_in_dir = f"/Volumes/data_backup/mercury/extreme/{case}/plane_product/"
    c_max = 500
elif "HNHV" in case:
    base_in_dir = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/"
    c_max = 400

out_folder = f"/Users/danywaller/Projects/mercury/extreme/timeseries_vmag_{slice_tag}/{case}/"
os.makedirs(out_folder, exist_ok=True)

# radius of Mercury [meters]
R_M = 2440.0e3

# first stable timestamp approx. 25000 for dt=0.002, numsteps=115000
if "Base" in case:
    # take last 15-ish seconds
    sim_steps = range(98000, 115000 + 1, 1000)
elif "HNHV" in case:
    sim_steps = range(115000, 350000 + 1, 1000)


def extract_slice_fields(ds: xr.Dataset, use_slice: str):
    """
    Pull 2D arrays for a slice:
      xy: Nz ~= 0
      xz: Ny ~= 0
      yz: Nx ~= 0  (dayside view looking along -X)
    """
    s = use_slice.lower().strip()
    if s == "xy":
        sel_kw = dict(Nz=0)
    elif s == "xz":
        sel_kw = dict(Ny=0)
    elif s == "yz":
        sel_kw = dict(Nx=0)
    else:
        raise ValueError(use_slice)

    Vx = ds["vx01"].sel(**sel_kw, method="nearest").squeeze() + ds["vx03"].sel(**sel_kw, method="nearest").squeeze()
    Vy = ds["vy01"].sel(**sel_kw, method="nearest").squeeze() + ds["vy03"].sel(**sel_kw, method="nearest").squeeze()
    Vz = ds["vz01"].sel(**sel_kw, method="nearest").squeeze() + ds["vz03"].sel(**sel_kw, method="nearest").squeeze()

    v_mag = np.sqrt(Vx ** 2 + Vy ** 2 + Vz ** 2)

    return v_mag

for sim_step in sim_steps:
    filetime = f"{sim_step:06d}"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    last_im = None

    near_planes = []

    for ax, use_slice in zip(axes, use_slices):
        input_folder = os.path.join(base_in_dir, f"fig_{use_slice}")
        f = os.path.join(input_folder, f"Amitis_{case}_{filetime}_{use_slice}_comp.nc")
        # input_folder = os.path.join(base_in_dir, f"object")
        # f = os.path.join(input_folder, f"Amitis_{case}_{filetime}_xz_comp.nc")

        if not os.path.exists(f):
            ax.axis("off")
            ax.set_title(f"{use_slice.upper()} missing")
            print(f"[WARN] missing: {f}")
            continue

        ds = xr.open_dataset(f)

        x_plot, y_plot = coords_for_slice(ds, use_slice)

        vmag = extract_slice_fields(ds, use_slice)

        ds.close()

        circle = plt.Circle((0, 0), 1, edgecolor='black', facecolor='cornflowerblue', alpha=0.3, linewidth=1, )
        last_im = ax.pcolormesh(x_plot, y_plot, vmag, shading='auto', cmap='Greens', vmin=0, vmax=c_max)
        ax.add_patch(circle)
        xlabel, ylabel = labels_for_slice(use_slice)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_aspect("equal")
        ax.set_title(f"{use_slice.upper()}")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, location="right", shrink=0.9)
        cbar.set_label(r"|V| [km/s]")

    fig.suptitle(rf"{case.replace("_"," ")} |V| at t = {sim_step * 0.002:.3f} s", fontsize=18, y=0.99)
    # plt.tight_layout()
    fig_path = os.path.join(out_folder, f"{case}_vmag_{sim_step}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"Done plotting |V| to {out_folder}")
