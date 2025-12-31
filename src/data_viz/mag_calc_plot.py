#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports:
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ----------------------------
# SETTINGS
# ----------------------------
debug = False

base = "RPN"

use_slices = ["xy", "xz", "yz"]  # plot all 3
n_slices = len(use_slices)       # number of requested slices
slice_tag = "_".join(use_slices)

# Plot background selection
plot_id = "Jmag"   # options: "Bmag", "Jmag", "Pmag"

PLOT_BG = {
    "Bmag": {
        "key": "Bmag",
        "label": r"|B|\ (\mathrm{nT})",
        "title_word": "|B|",
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 150.0,
        "bs_col": "red",
        "mp_col": "magenta",
    },
    "Jmag": {
        "key": "Jmag",
        "label": r"|J|\ (\mathrm{nA\,m^{-2}})",
        "title_word": "|J|",
        "cmap": "plasma",
        "vmin": 0.0,
        "vmax": 150.0,
        "bs_col": "cyan",
        "mp_col": "limegreen",
    },
    "Pmag": {
        "key": "gradP",
        "label": r"N\ (\mathrm{cm^{-3}})",
        "title_word": "Total Density",
        "cmap": "cividis",
        "vmin": 0,
        "vmax": 100.,
        "bs_col": "red",
        "mp_col": "magenta",
    },
}

# first stable timestamp approx. 25000 for dt=0.002, numsteps=115000
# sim_steps = list(range(27000, 115000 + 1, 1000))
sim_steps = list(range(1000, 115000 + 1, 1000))

base_dir = f"/Users/danywaller/Projects/mercury/{base}_Base/"
out_folder = os.path.join(base_dir, f"{plot_id.lower()}/")
os.makedirs(out_folder, exist_ok=True)

out_folder_ts = os.path.join(out_folder, f"timeseries_{slice_tag}/")
os.makedirs(out_folder_ts, exist_ok=True)

RM_M = 2440.0e3

# variables
VAR_X = "Bx"; VAR_Y = "By"; VAR_Z = "Bz"
VAR_V1X = "vx01"; VAR_V1Y = "vy01"; VAR_V1Z = "vz01"
VAR_V3X = "vx03"; VAR_V3Y = "vy03"; VAR_V3Z = "vz03"
VAR_JX = "Jx"; VAR_JY = "Jy"; VAR_JZ = "Jz"
VAR_DEN1 = "den01"; VAR_DEN3 = "den03"


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

    x = np.arange(xmin, xmax, dx) / RM_M
    y = np.arange(ymin, ymax, dy) / RM_M
    z = np.arange(zmin, zmax, dz) / RM_M

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

    BX = ds[VAR_X].sel(**sel_kw, method="nearest").squeeze()  # [units: nT]
    BY = ds[VAR_Y].sel(**sel_kw, method="nearest").squeeze()  # [units: nT]
    BZ = ds[VAR_Z].sel(**sel_kw, method="nearest").squeeze()  # [units: nT]

    vx01 = ds[VAR_V1X].sel(**sel_kw, method="nearest").squeeze() * 1e3  # convert to m/s
    vy01 = ds[VAR_V1Y].sel(**sel_kw, method="nearest").squeeze() * 1e3  # convert to m/s
    vz01 = ds[VAR_V1Z].sel(**sel_kw, method="nearest").squeeze() * 1e3  # convert to m/s

    vx03 = ds[VAR_V3X].sel(**sel_kw, method="nearest").squeeze() * 1e3  # convert to m/s
    vy03 = ds[VAR_V3Y].sel(**sel_kw, method="nearest").squeeze() * 1e3  # convert to m/s
    vz03 = ds[VAR_V3Z].sel(**sel_kw, method="nearest").squeeze() * 1e3  # convert to m/s

    den01 = ds[VAR_DEN1].sel(**sel_kw, method="nearest").squeeze() * 1e6  # convert to m^-3
    den03 = ds[VAR_DEN3].sel(**sel_kw, method="nearest").squeeze() * 1e6  # convert to m^-3

    JX = ds[VAR_JX].sel(**sel_kw, method="nearest").squeeze()  # [units: nA/m^2]
    JY = ds[VAR_JY].sel(**sel_kw, method="nearest").squeeze()  # [units: nA/m^2]
    JZ = ds[VAR_JZ].sel(**sel_kw, method="nearest").squeeze()  # [units: nA/m^2]

    return BX,BY,BZ,vx01,vy01,vz01,vx03,vy03,vz03,den01,den03,JX,JY,JZ


def compute_one_timestep(ds: xr.Dataset, use_slice: str, plot_id: str):
    """
    Compute all fields for one timestep.
    Returns: x_plot, y_plot, plot_bg(np.ndarray)
    """
    x_plot, y_plot = coords_for_slice(ds, use_slice)

    BX,BY,BZ,vx01,vy01,vz01,vx03,vy03,vz03,den01,den03,JX,JY,JZ = extract_slice_fields(ds, use_slice)

    tot_den = den01 + den03
    Pmag = tot_den
    Bmag = np.sqrt(BX**2 + BY**2 + BZ**2)
    Vmag01 = np.sqrt(vx01**2 + vy01**2 + vz01**2)
    Vmag03 = np.sqrt(vx03**2 + vy03**2 + vz03**2)
    Vmag = Vmag01 + Vmag03
    Jmag = np.sqrt(JX**2 + JY**2 + JZ**2)

    B = xr.concat([BX, BY, BZ], dim="comp").assign_coords(comp=["x", "y", "z"])
    Bhat = B / Bmag

    s = use_slice.lower().strip()

    # in-plane derivatives
    if s == "xy":
        dB_du    = Bmag.differentiate("Nx")
        dV_du    = Vmag.differentiate("Nx")
        dJ_du    = Jmag.differentiate("Nx")
        dP_du    = tot_den.differentiate("Nx")
        dBhat_du = Bhat.differentiate("Nx")

        dB_dv    = Bmag.differentiate("Ny")
        dV_dv    = Vmag.differentiate("Ny")
        dJ_dv    = Jmag.differentiate("Ny")
        dP_dv    = tot_den.differentiate("Ny")
        dBhat_dv = Bhat.differentiate("Ny")

    elif s == "xz":
        dB_du    = Bmag.differentiate("Nx")
        dV_du    = Vmag.differentiate("Nx")
        dJ_du    = Jmag.differentiate("Nx")
        dP_du    = tot_den.differentiate("Nx")
        dBhat_du = Bhat.differentiate("Nx")

        dB_dv    = Bmag.differentiate("Nz")
        dV_dv    = Vmag.differentiate("Nz")
        dJ_dv    = Jmag.differentiate("Nz")
        dP_dv    = tot_den.differentiate("Nz")
        dBhat_dv = Bhat.differentiate("Nz")

    elif s == "yz":
        dB_du    = Bmag.differentiate("Ny")
        dV_du    = Vmag.differentiate("Ny")
        dJ_du    = Jmag.differentiate("Ny")
        dP_du    = tot_den.differentiate("Ny")
        dBhat_du = Bhat.differentiate("Ny")

        dB_dv    = Bmag.differentiate("Nz")
        dV_dv    = Vmag.differentiate("Nz")
        dJ_dv    = Jmag.differentiate("Nz")
        dP_dv    = tot_den.differentiate("Nz")
        dBhat_dv = Bhat.differentiate("Nz")
    else:
        raise ValueError(use_slice)

    gradB = np.sqrt(dB_du**2 + dB_dv**2)
    gradV = np.sqrt(dV_du**2 + dV_dv**2)
    gradP = np.sqrt(dP_du**2 + dP_dv**2)
    gradJ = np.sqrt(dJ_du**2 + dJ_dv**2)
    rotation_strength = (dBhat_du**2 + dBhat_dv**2).sum("comp") ** 0.5

    s = use_slice.lower().strip()

    bg_map = {
        "Bmag": Bmag,
        "Jmag": Jmag,
        "Pmag": Pmag,
    }

    if plot_id not in bg_map:
        raise ValueError(f"Invalid plot_id='{plot_id}'. Options: {list(bg_map)}")

    plot_bg = bg_map[plot_id].values

    if plot_id == "Pmag":
        plot_bg = plot_bg * 1e-6  # convert back to cm^-3

    return x_plot, y_plot, plot_bg


def slice_axes_dims(use_slice: str):
    """
    For stacking masks consistently (2D):
      xy -> (Ny, Nx)
      xz -> (Nz, Nx)
      yz -> (Nz, Ny)
    """
    s = use_slice.lower().strip()
    if s == "xy":
        return ("Ny", "Nx")
    if s == "xz":
        return ("Nz", "Nx")
    if s == "yz":
        return ("Nz", "Ny")
    raise ValueError(use_slice)


def occupancy_and_bands(stack_bool: np.ndarray, thresholds=(0.25, 0.125, 0.0625)):
    """
    stack_bool: (T, H, W) boolean
    returns occupancy p (H,W) in [0,1] and masks at thresholds.
    """
    p = stack_bool.mean(axis=0)
    q1_thr, med_thr, q3_thr = thresholds
    q1mask  = (p >= q1_thr)
    medmask = (p >= med_thr)
    q3mask  = (p >= q3_thr)
    return p, q1mask, medmask, q3mask


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
    filename = "Base_" + f"{sim_step:06d}"

    # create only as many subplots as requested
    fig, axes = plt.subplots(1, n_slices, figsize=(6 * n_slices, 6), constrained_layout=True)
    if n_slices == 1:
        axes = np.array([axes])

    last_im = None

    for ax, use_slice in zip(axes, use_slices):
        input_folder = os.path.join(base_dir, f"fig_{use_slice}")
        f = os.path.join(input_folder, f"Amitis_{base}_{filename}_{use_slice}_comp.nc")

        if not os.path.exists(f):
            ax.axis("off")
            ax.set_title(f"{use_slice.upper()} missing")
            print(f"[WARN] missing: {f}")
            continue

        ds = xr.open_dataset(f)
        x_plot, y_plot, plot_bg = compute_one_timestep(ds, use_slice, plot_id)
        ds.close()

        # accumulate for post-loop median plot
        acc[use_slice]["plot_bg"].append(plot_bg)
        acc[use_slice]["x_plot"] = x_plot
        acc[use_slice]["y_plot"] = y_plot

        # per-timestep plot
        cfg = PLOT_BG[plot_id]
        last_im = ax.pcolormesh(x_plot, y_plot, plot_bg, shading="auto", cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"])
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
    if n_slices > 2:
        fig.suptitle(f"{base} Base {cfg['title_word']} at t = {tsec:.3f} s", fontsize=18, y=0.99)
    else:
        fig.suptitle(f"{base} Base {cfg['title_word']} at t = {tsec:.3f} s", fontsize=14, y=0.99)

    outpath = os.path.join(out_folder_ts, f"{base}_{plot_id.lower()}_{slice_tag}_{sim_step:06d}.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

print("Per-timestep plots complete.")

# ----------------------------
# median plot
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

    cfg = PLOT_BG[plot_id]

    last_im = ax.pcolormesh(x_plot, y_plot, plot_bg_med, shading="auto", cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"])
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
    fig.suptitle(f"{base} Base Median {cfg['title_word']}", fontsize=18, y=0.99)
else:
    fig.suptitle(f"{base} Base Median {cfg['title_word']}", fontsize=14, y=0.99)

median_path = os.path.join(out_folder, f"{base}_{plot_id.lower()}_{slice_tag}_median.png")
fig.savefig(median_path, dpi=300)
plt.close(fig)

print(f"Saved median plot: {median_path}")
