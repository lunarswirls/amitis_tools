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
    c_max = 50
elif "HNHV" in case:
    base_in_dir = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/"
    c_max = 400

out_folder = f"/Users/danywaller/Projects/mercury/extreme/timeseries_brot_{slice_tag}/{case}/"
os.makedirs(out_folder, exist_ok=True)

# radius of Mercury [meters]
R_M = 2440.0e3

# first stable timestamp approx. 25000 for dt=0.002, numsteps=115000
if "Base" in case:
    # take last 15-ish seconds
    sim_steps = range(98000, 115000 + 1, 1000)
elif "HNHV" in case:
    sim_steps = range(115000, 350000 + 1, 1000)


def compute_B_rotation(ds: xr.Dataset, use_slice: str, RM_M=2440.0e3):
    """
    Compute B-field magnitude, in-plane curl, and local rotation angles
    for a given XY/XZ/YZ slice.

    Args:
        ds        : xarray.Dataset containing Bx, By, Bz
        use_slice : "xy", "xz", or "yz"
        RM_M      : planetary radius (for scaling, default in meters)

    Returns:
        dict with keys:
          B_mag       : 2D array
          curl        : 2D array (in-plane component only)
          theta_x_deg : 2D array (rotation along X/grid1 axis)
          theta_y_deg : 2D array (rotation along Y/grid2 axis)
          theta_mag   : 2D array (combined rotation)
          X, Y        : 1D coordinate arrays trimmed to theta_mag
    """
    s = use_slice.lower().strip()
    # --- slice selection ---
    if s == "xy":
        sel_kw = dict(Nz=0)
        axis1, axis2 = "Y", "X"
    elif s == "xz":
        sel_kw = dict(Ny=0)
        axis1, axis2 = "Z", "X"
    elif s == "yz":
        sel_kw = dict(Nx=0)
        axis1, axis2 = "Z", "Y"
    else:
        raise ValueError(f"Unknown slice: {use_slice}")

    # --- extract B components ---
    Bx = ds["Bx"].sel(**sel_kw, method="nearest").squeeze().values
    By = ds["By"].sel(**sel_kw, method="nearest").squeeze().values
    Bz = ds["Bz"].sel(**sel_kw, method="nearest").squeeze().values

    # --- magnitude ---
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

    # --- grid coordinates ---
    X_full, Y_full = coords_for_slice(ds, use_slice, RM_M=RM_M)

    # Determine grid spacings
    dx = np.gradient(X_full)
    dy = np.gradient(Y_full)

    # --- 2D in-plane derivatives ---
    if s == "xy":
        dBx_dy, dBx_dx = np.gradient(Bx, dy, dx, edge_order=2)
        dBy_dy, dBy_dx = np.gradient(By, dy, dx, edge_order=2)
        dBz_dy, dBz_dx = np.gradient(Bz, dy, dx, edge_order=2)
        # only z-component exists in-plane
        curl = dBy_dx - dBx_dy

    elif s == "xz":
        # Rows = Z, Columns = X
        dz = dy  # NOTE: correct dz from dataset, not dy
        dz = float(ds.full_dz) / RM_M
        dx = float(ds.full_dx) / RM_M
        dBx_dz, dBx_dx = np.gradient(Bx, dz, dx, edge_order=2)
        dBz_dz, dBz_dx = np.gradient(Bz, dz, dx, edge_order=2)
        # only y-component exists in-plane
        curl = dBx_dz - dBz_dx

    elif s == "yz":
        # Rows = Z, Columns = Y
        dz = float(ds.full_dz) / RM_M
        dy = float(ds.full_dy) / RM_M
        dBy_dz, dBy_dy = np.gradient(By, dz, dy, edge_order=2)
        dBz_dz, dBz_dy = np.gradient(Bz, dz, dy, edge_order=2)
        # only x-component exists in-plane
        curl = dBz_dy - dBy_dz

    # --- Local rotation angles ---
    # along axis1 (rows)
    B1_x = np.stack([Bx[:-1, :], By[:-1, :], Bz[:-1, :]], axis=-1)
    B2_x = np.stack([Bx[1:, :], By[1:, :], Bz[1:, :]], axis=-1)
    dot_x = np.sum(B1_x * B2_x, axis=-1)
    norms_x = np.linalg.norm(B1_x, axis=-1) * np.linalg.norm(B2_x, axis=-1)
    theta_x_deg = np.degrees(np.arccos(np.clip(dot_x / norms_x, -1.0, 1.0)))

    # along axis2 (columns)
    B1_y = np.stack([Bx[:, :-1], By[:, :-1], Bz[:, :-1]], axis=-1)
    B2_y = np.stack([Bx[:, 1:], By[:, 1:], Bz[:, 1:]], axis=-1)
    dot_y = np.sum(B1_y * B2_y, axis=-1)
    norms_y = np.linalg.norm(B1_y, axis=-1) * np.linalg.norm(B2_y, axis=-1)
    theta_y_deg = np.degrees(np.arccos(np.clip(dot_y / norms_y, -1.0, 1.0)))

    # --- Combine theta_x and theta_y into single magnitude ---
    min_rows = min(theta_x_deg.shape[0], theta_y_deg.shape[0])
    min_cols = min(theta_x_deg.shape[1], theta_y_deg.shape[1])

    theta_x_trim = theta_x_deg[:min_rows, :min_cols]
    theta_y_trim = theta_y_deg[:min_rows, :min_cols]
    theta_mag = np.sqrt(theta_x_trim**2 + theta_y_trim**2)

    # --- Trim coordinates to match theta_mag ---
    X_trim = X_full[:min_cols]
    Y_trim = Y_full[:min_rows]

    return {
        "B_mag": B_mag,
        "curl": curl,
        "theta_x_deg": theta_x_deg,
        "theta_y_deg": theta_y_deg,
        "theta_mag": theta_mag,
        "X": X_trim,
        "Y": Y_trim
    }

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

        slice_data = compute_B_rotation(ds, use_slice)

        ds.close()

        circle = plt.Circle((0, 0), 1, edgecolor='black', facecolor='cornflowerblue', alpha=0.3, linewidth=1, )
        theta_mag = slice_data["theta_mag"]
        X = slice_data["X"][:theta_mag.shape[1]]  # X-axis length = number of columns
        Y = slice_data["Y"][:theta_mag.shape[0]]  # Y-axis length = number of rows

        last_im = ax.pcolormesh(X, Y, theta_mag, shading='auto', cmap="YlOrBr", vmin=0, vmax=c_max)

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
        cbar.set_label(r"Rotation magnitude [$^{\circ}$]")

    fig.suptitle(rf"{case.replace("_"," ")} B rotation at t = {sim_step * 0.002:.3f} s", fontsize=18, y=0.99)
    # plt.tight_layout()
    fig_path = os.path.join(out_folder, f"{case}_brot_{sim_step}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"Done plotting B rotation to {out_folder}")
