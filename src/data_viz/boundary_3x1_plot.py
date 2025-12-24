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

use_slices = ["xy", "xz", "yz"]  # plot all 3

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
        "label": r"N\ (\mathrm{m^{-3}})",
        "cmap": "cividis",
        "vmin": 0,
        "vmax": 0.1e-3,
        "bs_col": "red",
        "mp_col": "magenta",
    },
}

# first stable timestamp approx. 25000 for dt=0.002, numsteps=115000
sim_steps = list(range(27000, 115000 + 1, 1000))

base_dir = "/Users/danywaller/Projects/mercury/RPS_Base/"
out_folder = os.path.join(base_dir, "slice_bowshock/")
os.makedirs(out_folder, exist_ok=True)

out_folder_ts = os.path.join(out_folder, "timeseries_xyz/")
os.makedirs(out_folder_ts, exist_ok=True)

RM_M = 2440.0e3

# threshold percentiles
Bgradmax = 0.3
Vgradmax = 0.10
Pgradmax = 0.10
Jgradmax = 0.25
rotmax   = 0.10

Vgradnmax_mp = 0.25
Pgradmax_mp  = 0.75
Jgradmax_mp  = 0.60
rotmax_mp    = 0.10

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

    den01 = ds[VAR_DEN1].sel(**sel_kw, method="nearest").squeeze() * 1e-6  # convert to m^-3
    den03 = ds[VAR_DEN3].sel(**sel_kw, method="nearest").squeeze() * 1e-6  # convert to m^-3

    JX = ds[VAR_JX].sel(**sel_kw, method="nearest").squeeze()  # [units: nA/m^2]
    JY = ds[VAR_JY].sel(**sel_kw, method="nearest").squeeze()  # [units: nA/m^2]
    JZ = ds[VAR_JZ].sel(**sel_kw, method="nearest").squeeze()  # [units: nA/m^2]

    return BX,BY,BZ,vx01,vy01,vz01,vx03,vy03,vz03,den01,den03,JX,JY,JZ


def compute_masks_one_timestep(ds: xr.Dataset, use_slice: str, plot_id: str):
    """
    Compute Bmag and BS/MP masks for one timestep.
    Returns: x_plot, y_plot, Bmag(np.ndarray), bs_mask(bool ndarray), mp_mask(bool ndarray)
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

    bmag_threshold = Bgradmax * np.nanmax(gradB)
    vmag_threshold = Vgradmax * np.nanmax(gradV)
    jmag_threshold = Jgradmax * np.nanmax(gradJ)
    den_threshold  = Pgradmax * np.nanmax(gradP)

    rot_threshold_mp = rotmax_mp * np.nanmax(rotation_strength)

    s = use_slice.lower().strip()

    if s in ("xy", "xz"):
        # keep existing XY/XZ logic
        magnetopause_mask = (
                (gradJ > jmag_threshold) &
                (gradP > den_threshold) & (dP_du > 0) &
                (gradV < vmag_threshold) & (dV_du < 0) &
                (rotation_strength > rot_threshold_mp)
        )
        bowshock_mask = (
                (gradJ > jmag_threshold) &
                (gradP > den_threshold) & (dP_du < 0) &
                (rotation_strength < rot_threshold_mp)
        )
        bowshock_mask = bowshock_mask & (~magnetopause_mask)

    elif s == "yz":
        # ------------------------------------------------------------
        # YZ dayside view (looking along −X):
        #   horizontal axis = Y
        #   vertical axis   = Z
        #
        # The planet is centered at Y=0, so "inward" direction flips sign:
        #   - On Y>0 side, inward is toward decreasing Y
        #   - On Y<0 side, inward is toward increasing Y
        #
        # Therefore use a symmetric inward-density-gradient proxy:
        #   dP_inward = -sign(Y) * dP/dY
        # ------------------------------------------------------------

        # Horizontal derivative is d/du = d/dY (Ny)
        dP_h = dP_du  # already computed as differentiate("Ny")

        # Build Y–Z coordinate grid (in R_M)
        y_axis = x_plot  # coords_for_slice returns (Y, Z) for yz
        z_axis = y_plot
        Yg, Zg = np.meshgrid(y_axis, z_axis, indexing="xy")
        r = np.sqrt(Yg ** 2 + Zg ** 2)

        # Exclude inside the planetary body
        outside_body = r >= 1.0

        # Rotation field as numpy for fast masking
        gradB = gradB.values
        gradJ = gradJ.values
        gradP = gradP.values
        rot = rotation_strength.values

        # ------------------------------------------------------------
        # Unified candidate gate on strong |J| and density gradients
        # ------------------------------------------------------------
        candidate = (
                (gradJ > jmag_threshold) &
                (gradP > den_threshold) &
                outside_body &
                np.isfinite(rot)
        )

        # cand = candidate.values.astype(bool)
        cand = candidate.astype(bool)

        if debug:
            south = (Zg < -1.5) & (np.abs(Yg) < 2.0) & outside_body

            def cnt(mask):
                return int(np.count_nonzero(mask))

            print(
                f"[YZ step={sim_step}] "
                f"max gradJ={np.nanmax(gradJ):.3g} thr={jmag_threshold:.3g} "
                f"max gradP={np.nanmax(gradP):.3g} thr={den_threshold:.3g} "
                f"max gradB={np.nanmax(gradB):.3g} thr={bmag_threshold:.3g} "
                f"max rot={np.nanmax(rotation_strength):.3g} thr={rot_threshold_mp:.3g} "
                f"cand={cnt(cand)} cand_south={cnt(cand & south)}\n")

        if np.count_nonzero(cand) == 0:
            bowshock_mask = candidate * False
            magnetopause_mask = candidate * False

        else:
            # --------------------------------------------------------
            # Symmetric inward density-gradient proxy
            # --------------------------------------------------------
            dPdy = dP_h.values
            # sign(Y): +1 (Y>0), -1 (Y<0); ignore a thin band near Y=0
            sY = np.sign(Yg)
            eps_rm = 0.001
            y0_band = np.abs(Yg) <= eps_rm
            dP_inward = -sY * dPdy

            # Initial split (symmetric)
            bs_pre = cand & (~y0_band) & (dP_inward > 0.0) & (gradB > bmag_threshold)
            mp_pre = cand & (~y0_band) & (dP_inward < 0.0)

            # --------------------------------------------------------
            # MP: strong rotation; BS: weak rotation
            # --------------------------------------------------------
            mp_gate = rot > float(rot_threshold_mp)
            bs_gate = rot < float(rot_threshold_mp)

            # Secondary split (rotation gates)
            bs_post = bs_pre & bs_gate  # inward increase, weak rotation
            mp_post = mp_pre & mp_gate  # inward decrease, strong rotation

            # Convert back to xarray and enforce exclusivity
            magnetopause_mask = xr.DataArray(mp_pre, coords=gradV.coords, dims=gradV.dims)
            bowshock_mask = xr.DataArray(bs_pre, coords=gradV.coords, dims=gradV.dims)
            bowshock_mask = bowshock_mask & (~magnetopause_mask)
    else:
        raise ValueError(use_slice)

    # Exclusion region (only meaningful when X is in-plane)
    if s in ("xy", "xz"):
        x_bad = x_plot < 0.75
        y_bad = (y_plot > -1.2) & (y_plot < 1.2)
        exclude_region = y_bad[:, None] & x_bad[None, :]
        bowshock_mask = bowshock_mask & (~exclude_region)
        magnetopause_mask = magnetopause_mask & (~exclude_region)

    bg_map = {
        "Bmag": Bmag,
        "Jmag": Jmag,
        "Pmag": Pmag,
    }

    if plot_id not in bg_map:
        raise ValueError(f"Invalid plot_id='{plot_id}'. Options: {list(bg_map)}")

    plot_bg = bg_map[plot_id].values

    return x_plot, y_plot, plot_bg, bowshock_mask, magnetopause_mask


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

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    last_im = None

    for ax, use_slice in zip(axes, use_slices):
        input_folder = os.path.join(base_dir, f"fig_{use_slice}")
        f = os.path.join(input_folder, f"Amitis_RPS_{filename}_{use_slice}_comp.nc")

        if not os.path.exists(f):
            ax.axis("off")
            ax.set_title(f"{use_slice.upper()} missing")
            print(f"[WARN] missing: {f}")
            continue

        ds = xr.open_dataset(f)
        x_plot, y_plot, plot_bg, bs_mask, mp_mask = compute_masks_one_timestep(ds, use_slice, plot_id)
        ds.close()

        # accumulate for post-loop median plot
        acc[use_slice]["plot_bg"].append(plot_bg)
        acc[use_slice]["bs"].append(bs_mask.values.astype(np.uint8))  # store as 0/1
        acc[use_slice]["mp"].append(mp_mask.values.astype(np.uint8))
        acc[use_slice]["x_plot"] = x_plot
        acc[use_slice]["y_plot"] = y_plot

        # per-timestep plot
        cfg = PLOT_BG[plot_id]
        last_im = ax.pcolormesh(x_plot, y_plot, plot_bg, shading="auto", cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"])
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
    fig.suptitle(f"BS ({cfg['bs_col']}) + MP ({cfg['mp_col']}) position at t = {tsec:.3f} s", fontsize=18, y=0.99)

    outpath = os.path.join(out_folder_ts, f"rps_{plot_id.lower()}_boundaries_xyz_{sim_step:06d}.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

print("Per-timestep plots complete.")

# ----------------------------
# POST-LOOP: median + IQR figure (1x3)
# ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
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
    cbar.set_label(rf"$\mathrm{{Median}}\ {cfg['label']}$")

fig.suptitle(f"BS ({cfg['bs_col']}) + MP ({cfg['mp_col']}) IQR envelopes with median occupancy contour", fontsize=18, y=0.99)

median_path = os.path.join(out_folder, f"rps_{plot_id.lower()}_boundaries_xyz_median.png")
fig.savefig(median_path, dpi=300)
plt.close(fig)

print(f"Saved median plot: {median_path}")
