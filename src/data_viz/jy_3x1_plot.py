#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.bs_mp_finder.boundary_utils import labels_for_slice, coords_for_slice

# base cases: CPN_Base RPN_Base CPS_Base RPS_Base
# HNHV cases: CPN_HNHV RPN_HNHV CPS_HNHV RPS_HNHV
case = "CPN_Base"

use_slices = ["xy", "xz", "yz"]  # plot all 3
n_slices = len(use_slices)       # number of requested slices
slice_tag = "_".join(use_slices)

if "Base" in case:
    base_in_dir = f"/Volumes/data_backup/mercury/extreme/{case}/plane_product/"
    c_max = 60
elif "HNHV" in case:
    base_in_dir = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/"
    c_max = 400

out_folder = f"/Users/danywaller/Projects/mercury/extreme/timeseries_jcomp_{slice_tag}/{case}/"
os.makedirs(out_folder, exist_ok=True)

# proton mass [kg]
M_P = 1.6726e-27

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

    Jx = ds["Jx"].sel(**sel_kw, method="nearest").squeeze()
    Jy = ds["Jy"].sel(**sel_kw, method="nearest").squeeze()
    Jz = ds["Jz"].sel(**sel_kw, method="nearest").squeeze()

    return Jy

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

        j_comp = extract_slice_fields(ds, use_slice)

        ds.close()

        circle = plt.Circle((0, 0), 1, edgecolor='black', facecolor='cornflowerblue', alpha=0.3, linewidth=1, )
        last_im = ax.pcolormesh(x_plot, y_plot, j_comp, shading='auto', cmap='bwr', vmin=-c_max, vmax=c_max)
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
        cbar.set_label(r"$J_y$ [nA/m$^2$]")

    fig.suptitle(rf"{case.replace("_"," ")} $J_y$ at t = {sim_step * 0.002:.3f} s", fontsize=18, y=0.99)
    # plt.tight_layout()
    fig_path = os.path.join(out_folder, f"{case}_jy_{sim_step}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"Done plotting J_y to {out_folder}")
