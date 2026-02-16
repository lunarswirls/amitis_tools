#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# base cases: CPN_Base RPN_Base CPS_Base RPS_Base
# HNHV cases: CPN_HNHV RPN_HNHV CPS_HNHV RPS_HNHV
case = "RPN_HNHV"

use_slices = ["xy", "xz", "yz"]  # plot all 3
n_slices = len(use_slices)       # number of requested slices
slice_tag = "_".join(use_slices)

if "Base" in case:
    base_in_dir = f"/Volumes/data_backup/mercury/extreme/{case}/plane_product/"
    c_max = 120
elif "HNHV" in case:
    base_in_dir = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/"
    c_max = 400

out_folder = f"/Users/danywaller/Projects/mercury/extreme/timeseries_pdyn_{slice_tag}/{case}/"
os.makedirs(out_folder, exist_ok=True)

# proton mass [kg]
M_P = 1.6726e-27

# radius of Mercury [meters]
R_M = 2440.0e3

# first stable timestamp approx. 25000 for dt=0.002, numsteps=115000
if "Base" in case:
    # take last 10-ish seconds
    sim_steps = range(98000, 115000 + 1, 1000)
elif "HNHV" in case:
    sim_steps = range(115000, 350000 + 1, 1000)


# ----------------------------
# Helpers
# ----------------------------
def labels_for_slice(s: str):
    s = s.lower().strip()
    if s == "xy":
        return r"$X\ (\mathrm{R_M})$", r"$Y\ (\mathrm{R_M})$"
    if s == "xz":
        return r"$X\ (\mathrm{R_M})$", r"$Z\ (\mathrm{R_M})$"
    if s == "yz":
        return r"$Y\ (\mathrm{R_M})$", r"$Z\ (\mathrm{R_M})$"
    raise ValueError(s)


def coords_for_slice(ds: xr.Dataset, use_slice: str):
    xmin = float(ds.full_xmin); xmax = float(ds.full_xmax)
    ymin = float(ds.full_ymin); ymax = float(ds.full_ymax)
    zmin = float(ds.full_zmin); zmax = float(ds.full_zmax)
    dx = float(ds.full_dx); dy = float(ds.full_dy); dz = float(ds.full_dz)

    x = np.arange(xmin, xmax, dx) / R_M
    y = np.arange(ymin, ymax, dy) / R_M
    z = np.arange(zmin, zmax, dz) / R_M

    s = use_slice.lower().strip()
    if s == "xy":
        return x, y
    if s == "xz":
        return x, z
    if s == "yz":
        return y, z
    raise ValueError(use_slice)


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

    # bulk velocity in km/s
    Vx = ds["vx01"].sel(**sel_kw, method="nearest").squeeze() + ds["vx02"].sel(**sel_kw, method="nearest").squeeze() + ds["vx03"].sel(**sel_kw, method="nearest").squeeze() + ds["vx04"].sel(**sel_kw, method="nearest").squeeze()
    Vy = ds["vy01"].sel(**sel_kw, method="nearest").squeeze() + ds["vy02"].sel(**sel_kw, method="nearest").squeeze() + ds["vy03"].sel(**sel_kw, method="nearest").squeeze() + ds["vy04"].sel(**sel_kw, method="nearest").squeeze()
    Vz = ds["vz01"].sel(**sel_kw, method="nearest").squeeze() + ds["vz02"].sel(**sel_kw, method="nearest").squeeze() + ds["vz03"].sel(**sel_kw, method="nearest").squeeze() + ds["vz04"].sel(**sel_kw, method="nearest").squeeze()

    vmag = np.sqrt(Vx ** 2 + Vy ** 2 + Vz ** 2) * 1e3  # km/s -> m/s

    # densities in cm^-3
    den01 = ds["den01"].sel(**sel_kw, method="nearest").squeeze()
    den02 = ds["den02"].sel(**sel_kw, method="nearest").squeeze()
    den03 = ds["den03"].sel(**sel_kw, method="nearest").squeeze()
    den04 = ds["den04"].sel(**sel_kw, method="nearest").squeeze()

    # sum all densities to get total density
    tot_den = (den01 + den02 + den03 + den04) * 1e6  # cm^-3 -> m^-3

    return vmag, tot_den

for sim_step in sim_steps:
    filetime = f"{sim_step:06d}"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    last_im = None

    for ax, use_slice in zip(axes, use_slices):
        input_folder = os.path.join(base_in_dir, f"fig_{use_slice}")
        f = os.path.join(input_folder, f"Amitis_{case}_{filetime}_{use_slice}_comp.nc")

        if not os.path.exists(f):
            ax.axis("off")
            ax.set_title(f"{use_slice.upper()} missing")
            print(f"[WARN] missing: {f}")
            continue

        ds = xr.open_dataset(f)

        x_plot, y_plot = coords_for_slice(ds, use_slice)

        v_mag, den_tot = extract_slice_fields(ds, use_slice)

        ds.close()

        # dynamic pressure (Pa)
        P_pa = den_tot * M_P * v_mag ** 2

        # convert to nPa
        P_npa = P_pa * 1e9

        circle = plt.Circle((0, 0), 1, edgecolor='black', facecolor='cornflowerblue', alpha=0.3, linewidth=1, )
        last_im = ax.pcolormesh(x_plot, y_plot, P_npa, shading='auto', cmap='gist_heat_r', vmin=0, vmax=c_max)
        ax.add_patch(circle)
        xlabel, ylabel = labels_for_slice(use_slice)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_aspect("equal")
        ax.set_title(use_slice.upper())

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, location="right", shrink=0.9)
        cbar.set_label(r"$P_{dyn}$ [nPa]")

    fig.suptitle(rf"{case.replace("_"," ")} $P_{{dyn}}$ at t = {sim_step * 0.002:.3f} s", fontsize=18, y=0.99)
    # plt.tight_layout()
    fig_path = os.path.join(out_folder, f"{case}_pdyn_{sim_step}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"Done plotting P_dyn to {out_folder}")
