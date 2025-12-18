#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pick slice for input files
use_slice = "xz"

# first stable timestamp approx. 25000 for dt=0.002, numsteps=115000
sim_steps = list(range(27000, 115000 + 1, 1000))

# directories
input_folder = "/Users/danywaller/Projects/mercury/RPS_Base/fig_" + use_slice + "/"
out_folder = "/Users/danywaller/Projects/mercury/RPS_Base/slice_bowshock/"
os.makedirs(out_folder, exist_ok=True)

# plot labels
if use_slice == "xy":
    ylab = r"$Y\ (\mathrm{R_M})$"
elif use_slice == "xz":
    ylab = r"$Z\ (\mathrm{R_M})$"
else:
    print("\nInvalid use_slice argument: " + use_slice)
    print("Must be either 'xy' or 'xz'")
    sys.exit(1)

# percentile thresholds
Bgradmax = 0.35   # 0.25–0.45
Vgradmax = 0.10   # 0.05–0.15
Pgradmax = 0.1   # 0.20–0.35
Jgradmax = 0.25   # 0.20–0.35
rotmax    = 0.10  # 0.05–0.15  (weak rotation)

Vgradnmax_mp = 0.25   # 0.20–0.35
Pgradmax_mp  = 0.75   # 0.85–0.98
Jgradmax_mp  = 0.60   # 0.50–0.75
rotmax_mp    = 0.1   # 0.60–0.85  (strong rotation)

# name of variables inside the NetCDF file
VAR_X = "Bx"   # x-component of B  [units: nT]
VAR_Y = "By"   # y-component of B  [units: nT]
VAR_Z = "Bz"   # z-component of B  [units: nT]

VAR_V1X = "vx01"   # x-component of species 1 velocity  [units: km/s]
VAR_V1Y = "vy01"   # y-component of species 1 velocity  [units: km/s]
VAR_V1Z = "vz01"   # z-component of species 1 velocity  [units: km/s]
VAR_V3X = "vx03"   # x-component of species 3 velocity  [units: km/s]
VAR_V3Y = "vy03"   # y-component of species 3 velocity  [units: km/s]
VAR_V3Z = "vz03"   # z-component of species 3 velocity  [units: km/s]

VAR_JX = "Jx"   # x-component of J  [units: nA/m^2]
VAR_JY = "Jy"   # y-component of J  [units: nA/m^2]
VAR_JZ = "Jz"   # z-component of J  [units: nA/m^2]

VAR_DEN1 = "den01"   # density of particle species #1  [units: cm^-3]
VAR_DEN3 = "den03"   # density of particle species #3  [units: cm^-3]

# preallocate list for bowshock and magnetopause masks to calculate median positions later
bs_positions = []
mp_positions = []

# preallocate list for Bmag for plot background later
Bmag_list = []

for sim_step in sim_steps:
    filename = 'Base_' + "%06d" % sim_step

    f = input_folder + "Amitis_RPS_" + filename + "_" + use_slice + "_comp.nc"

    ds = xr.open_dataset(f)

    # Extract physical domain extents
    xmin = float(ds.full_xmin)
    xmax = float(ds.full_xmax)
    ymin = float(ds.full_ymin)
    ymax = float(ds.full_ymax)
    zmin = float(ds.full_zmin)
    zmax = float(ds.full_zmax)

    dx = float(ds.full_dx)
    dy = float(ds.full_dy)
    dz = float(ds.full_dz)

    # Build coordinate arrays
    x = np.arange(xmin, xmax, dx)  # [units: m]

    if use_slice == "xy":
        y = np.arange(ymin, ymax, dy)  # [units: m]
    elif use_slice == "xz":
        y = np.arange(zmin, zmax, dz)  # [units: m]
    else:
        print("\nInvalid use_slice argument: " + use_slice)
        print("Must be either 'xy' or 'xz'")
        sys.exit(1)

    # convert to R_m for plotting
    x_plot = x / 2440.e3
    y_plot = y / 2440.e3

    if use_slice == "xy":
        # Extract arrays
        BX = ds[VAR_X].sel(Nz=0, method="nearest").squeeze()  # [units: nT]
        BY = ds[VAR_Y].sel(Nz=0, method="nearest").squeeze()  # [units: nT]
        BZ = ds[VAR_Z].sel(Nz=0, method="nearest").squeeze()  # [units: nT]

        vx01 = ds[VAR_V1X].sel(Nz=0, method="nearest").squeeze()*1.e3   # convert to m/s
        vy01 = ds[VAR_V1Y].sel(Nz=0, method="nearest").squeeze()*1.e3   # convert to m/s
        vz01 = ds[VAR_V1Z].sel(Nz=0, method="nearest").squeeze()*1.e3   # convert to m/s

        vx03 = ds[VAR_V3X].sel(Nz=0, method="nearest").squeeze()*1.e3   # convert to m/s
        vy03 = ds[VAR_V3Y].sel(Nz=0, method="nearest").squeeze()*1.e3   # convert to m/s
        vz03 = ds[VAR_V3Z].sel(Nz=0, method="nearest").squeeze()*1.e3   # convert to m/s

        den01 = ds[VAR_DEN1].sel(Nz=0, method="nearest").squeeze()*1.e-6  # convert to m^-3
        den03 = ds[VAR_DEN3].sel(Nz=0, method="nearest").squeeze()*1.e-6  # convert to m^-3

        JX = ds[VAR_JX].sel(Nz=0, method="nearest").squeeze()  # [units: nA/m^2]
        JY = ds[VAR_JY].sel(Nz=0, method="nearest").squeeze()  # [units: nA/m^2]
        JZ = ds[VAR_JZ].sel(Nz=0, method="nearest").squeeze()  # [units: nA/m^2]
    elif use_slice == "xz":
        # Extract arrays
        BX = ds[VAR_X].sel(Ny=0, method="nearest").squeeze()  # [units: nT]
        BY = ds[VAR_Y].sel(Ny=0, method="nearest").squeeze()  # [units: nT]
        BZ = ds[VAR_Z].sel(Ny=0, method="nearest").squeeze()  # [units: nT]

        vx01 = ds[VAR_V1X].sel(Ny=0, method="nearest").squeeze() * 1.e3  # convert to m/s
        vy01 = ds[VAR_V1Y].sel(Ny=0, method="nearest").squeeze() * 1.e3  # convert to m/s
        vz01 = ds[VAR_V1Z].sel(Ny=0, method="nearest").squeeze() * 1.e3  # convert to m/s

        vx03 = ds[VAR_V3X].sel(Ny=0, method="nearest").squeeze() * 1.e3  # convert to m/s
        vy03 = ds[VAR_V3Y].sel(Ny=0, method="nearest").squeeze() * 1.e3  # convert to m/s
        vz03 = ds[VAR_V3Z].sel(Ny=0, method="nearest").squeeze() * 1.e3  # convert to m/s

        den01 = ds[VAR_DEN1].sel(Ny=0, method="nearest").squeeze() * 1.e-6  # convert to m^-3
        den03 = ds[VAR_DEN3].sel(Ny=0, method="nearest").squeeze() * 1.e-6  # convert to m^-3

        JX = ds[VAR_JX].sel(Ny=0, method="nearest").squeeze()  # [units: nA/m^2]
        JY = ds[VAR_JY].sel(Ny=0, method="nearest").squeeze()  # [units: nA/m^2]
        JZ = ds[VAR_JZ].sel(Ny=0, method="nearest").squeeze()  # [units: nA/m^2]
    else:
        print("\nInvalid use_slice argument: " + use_slice)
        print("Must be either 'xy' or 'xz'")
        sys.exit(1)

    # calculate total density
    tot_den = den01 + den03

    # calculate field magnitudes
    Bmag = np.sqrt(BX ** 2 + BY ** 2 + BZ ** 2)
    Vmag01 = np.sqrt(vx01 ** 2 + vy01 ** 2 + vz01 ** 2)
    Vmag03 = np.sqrt(vx03 ** 2 + vy03 ** 2 + vz03 ** 2)
    Vmag = Vmag01 + Vmag03
    Jmag = np.sqrt(JX ** 2 + JY ** 2 + JZ ** 2)

    # check magnetic field rotation
    B = xr.concat([BX, BY, BZ], dim="comp")
    B = B.assign_coords(comp=["x", "y", "z"])
    Bhat = B / Bmag

    # compute gradients along x and y
    dB_dx = Bmag.differentiate("Nx")
    dV_dx = Vmag.differentiate("Nx")
    dJ_dx = Jmag.differentiate("Nx")
    dP_dx = tot_den.differentiate("Nx")
    dBhat_dx = Bhat.differentiate("Nx")

    if use_slice == "xy":
        dB_dz = Bmag.differentiate("Ny")
        dV_dz = Vmag.differentiate("Ny")
        dJ_dz = Jmag.differentiate("Ny")
        dP_dz = tot_den.differentiate("Ny")
        dBhat_dz = Bhat.differentiate("Ny")
    elif use_slice == "xz":
        dB_dz = Bmag.differentiate("Nz")
        dV_dz = Vmag.differentiate("Nz")
        dJ_dz = Jmag.differentiate("Nz")
        dP_dz = tot_den.differentiate("Nz")
        dBhat_dz = Bhat.differentiate("Nz")
    else:
        print("\nInvalid use_slice argument: " + use_slice)
        print("Must be either 'xy' or 'xz'")
        sys.exit(1)

    # magnitude of gradient
    gradB = np.sqrt(dB_dx ** 2 + dB_dz ** 2)
    gradV = np.sqrt(dV_dx ** 2 + dV_dz ** 2)
    gradP = np.sqrt(dP_dx ** 2 + dP_dz ** 2)
    gradJ = np.sqrt(dJ_dx ** 2 + dJ_dz ** 2)
    rotation_strength = (dBhat_dx ** 2 + dBhat_dz ** 2).sum("comp") ** 0.5

    bmag_threshold = Bgradmax * np.nanmax(gradB)
    vmag_threshold = Vgradmax * np.nanmax(gradV)
    jmag_threshold = Jgradmax * np.nanmax(gradJ)
    den_threshold = Pgradmax * np.nanmax(gradP)
    den_threshold_mp = Pgradmax_mp * np.nanmax(gradP)

    vmag_threshold_mp = Vgradnmax_mp * np.nanmax(gradV)
    jmag_threshold_mp = Jgradmax_mp * np.nanmax(gradJ)

    rot_threshold = rotmax * np.nanmax(rotation_strength)
    rot_threshold_mp = rotmax_mp * np.nanmax(rotation_strength)

    if 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(x_plot, y_plot, gradV, vmin=0, vmax=750.0, shading='auto', cmap='cividis')
        circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(circle)
        plt.colorbar(im, label="∇|V| (m/s)")
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.xlabel(r"$X\ (\mathrm{R_M})$")
        plt.ylabel(ylab)
        plt.title(f"Velocity magnitude gradient,  t = {sim_step * 0.002} seconds")
        fig_path = os.path.join(out_folder, f"rps_{use_slice}_vmag_gradient_{sim_step}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    if 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(x_plot, y_plot, gradP, vmin=0, vmax=.5e-6, shading='auto', cmap='cividis')
        circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(circle)
        plt.colorbar(im, label=r"$\nabla N_{sw}\text{ (m^-3)}$")
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.xlabel(r"$X\ (\mathrm{R_M})$")
        plt.ylabel(ylab)
        plt.title(f"Total density gradient,  t = {sim_step * 0.002} seconds")
        fig_path = os.path.join(out_folder, f"rps_{use_slice}_pmag_gradient_{sim_step}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    if 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(x_plot, y_plot, rotation_strength, vmin=0, vmax=0.005, shading='auto', cmap='cividis')
        circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(circle)
        plt.colorbar(im, label="|rotation|")
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.xlabel(r"$X\ (\mathrm{R_M})$")
        plt.ylabel(ylab)
        plt.title(f"Rotation strength,  t = {sim_step * 0.002} seconds")
        fig_path = os.path.join(out_folder, f"rps_{use_slice}_rotation_strength_{sim_step}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    magnetopause_mask = (
            (gradJ > jmag_threshold) &
            (gradP > den_threshold) & (dP_dx > 0) &
            (gradV < vmag_threshold) & (dV_dx < 0) &
            (rotation_strength > rot_threshold_mp)
    )

    bowshock_mask  = ((gradJ > jmag_threshold) &
                      (gradP > den_threshold) & (dP_dx < 0) &
                      (rotation_strength < rot_threshold_mp))


    # check no bowshock points have been classified as magnetopause points
    bowshock_mask &= ~magnetopause_mask

    # exclude plasma sheet
    x_bad = x_plot < 0.75
    y_bad = (y_plot > -1.2) & (y_plot < 1.2)

    exclude_region = y_bad[:, None] & x_bad[None, :]

    bowshock_mask &= ~exclude_region

    magnetopause_mask &= ~exclude_region

    # append for later use
    bs_positions.append(bowshock_mask)
    mp_positions.append(magnetopause_mask)
    Bmag_list.append(Bmag)

    if 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(x_plot, y_plot, Bmag, vmin=0, vmax=150, shading='auto', cmap='viridis')
        ax.contour(x_plot, y_plot, bowshock_mask, levels=[0.5], colors='red', linewidths=2)
        ax.contour(x_plot, y_plot, magnetopause_mask, levels=[0.5], colors='magenta', linewidths=2)
        circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(circle)

        ax.set_xlabel(r"$X\ (\mathrm{R_M})$")
        ax.set_ylabel(ylab)
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        ax.set_aspect("equal")
        plt.colorbar(im, label="|B| (nT)")
        plt.title(f"Bow Shock (red) + MP (pink),  t = {sim_step * 0.002} seconds")
        fig_path = os.path.join(out_folder, f"rps_{use_slice}_bmag_bowshock_{sim_step}_den01-only.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    if 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(x_plot, y_plot, Jmag, vmin=0, vmax=150, shading='auto', cmap='plasma')
        ax.contour(x_plot, y_plot, bowshock_mask, levels=[0.5], colors='aqua', linewidths=2)
        ax.contour(x_plot, y_plot, magnetopause_mask, levels=[0.5], colors='magenta', linewidths=2)
        circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(circle)

        ax.set_xlabel(r"$X\ (\mathrm{R_M})$")
        ax.set_ylabel(ylab)
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        ax.set_aspect("equal")
        plt.colorbar(im, label="|J| (nA/m²")
        plt.title(f"Bow Shock (blue) + MP (pink),  t = {sim_step * 0.002} seconds")
        fig_path = os.path.join(out_folder, f"rps_{use_slice}_jmag_bowshock_{sim_step}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    ds.close()


# stack all Bmag arrays and take median for plotting
Bmag_med = np.median(np.stack(Bmag_list, axis=0), axis=0)

# stack and sum all bowshock and magnetopause positions for plotting
bs_all = np.sum(np.stack(bs_positions, axis=0), axis=0)
mp_all = np.sum(np.stack(mp_positions, axis=0), axis=0)

# stack all bowshock and magnetopause positions for envelope fitting
bs_stack = np.stack(bs_positions, axis=0).astype(float)
bs_stack[bs_stack == 0] = np.nan
mp_stack = np.stack(mp_positions, axis=0).astype(float)
mp_stack[mp_stack == 0] = np.nan

# bowshock IQR from 0.25 to 0.75
bs_q1 = np.nanpercentile(bs_stack, 25, axis=0)
bs_q3 = np.nanpercentile(bs_stack, 75, axis=0)
bs_iqr = bs_q3 - bs_q1

# magnetopause IQR from 0.25 to 0.75
mp_q1 = np.nanpercentile(mp_stack, 25, axis=0)
mp_q3 = np.nanpercentile(mp_stack, 75, axis=0)
mp_iqr = mp_q3 - mp_q1


def _slice_axes(def_slice: str) -> tuple[str, str]:
    """
    Map slice name to (Y-dim, X-dim) used in the 2D masks.
      use_slice="xy" -> (Ny, Nx)
      use_slice="xz" -> (Nz, Nx)
    """
    def_slice = def_slice.lower().strip()
    if def_slice == "xy":
        return "Ny", "Nx"
    if def_slice == "xz":
        return "Nz", "Nx"
    raise ValueError("use_slice must be 'xy' or 'xz'")


def occupancy_and_bands(
    positions: list[xr.DataArray] | xr.DataArray,
    *,
    thresholds: tuple[float, float, float] = (0.25, 0.125, 0.0625),
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Statistically consistent summary for binary curve masks.

    Computes per-pixel occupancy probability p in [0,1], then threshold bands:
      q1mask  = p >= 0.25
      medmask = p >= 0.125
      q3mask  = p >= 0.0625

    positions:
      - list of N xarray.DataArray (2D bool masks) each with a scalar 'time' coord, OR
      - a single xarray.DataArray with dims including 'time' and the 2D slice dims.

    use_slice:
      - "xy" -> masks are (Ny, Nx)
      - "xz" -> masks are (Nz, Nx)
    """
    Ydim, Xdim = _slice_axes(use_slice)

    # Normalize input -> a single stacked DataArray "stack" with dim time
    if isinstance(positions, xr.DataArray):
        da = positions
        if "time" in da.dims:
            stack = da.astype(np.uint8)
            ref = da.isel(time=0)
        else:
            # Single 2D mask
            stack = da.expand_dims(time=[np.datetime64("NaT")]).astype(np.uint8)
            ref = da
    else:
        masks = list(positions)
        if len(masks) == 0:
            raise ValueError("positions is empty")
        for m in masks:
            if not isinstance(m, xr.DataArray):
                raise TypeError("Each element of positions must be an xarray.DataArray")
            if set((Ydim, Xdim)) - set(m.dims):
                raise ValueError(f"Each mask must include dims ({Ydim},{Xdim}); got {m.dims}")

        stack = xr.concat([m.astype(np.uint8) for m in masks], dim="time")
        # Preserve per-mask time if present (common in your data)
        if "time" in masks[0].coords:
            stack = stack.assign_coords(time=[m.coords["time"].values for m in masks])
        ref = masks[0]

    # Enforce consistent 2D ordering and coords
    stack = stack.transpose("time", Ydim, Xdim)
    p = stack.mean("time").rename("occupancy")
    p = p.assign_coords({Ydim: ref[Ydim], Xdim: ref[Xdim]})

    q1_thr, med_thr, q3_thr = thresholds
    q1mask  = (p >= q1_thr).rename("q1mask")
    medmask = (p >= med_thr).rename("medmask")
    q3mask  = (p >= q3_thr).rename("q3mask")
    return p, q1mask, medmask, q3mask


# Consistent occupancy + bands
bs_p, _, bs_med, _ = occupancy_and_bands(bs_positions)
mp_p, _, mp_med, _ = occupancy_and_bands(mp_positions)


def farthest_standoff_at_y0_from_mask(
    band_mask: xr.DataArray,
    *,
    y0: float = 0.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    y_band: int = 50,
    metric: str = "xmax",   # "xmax" (sunward-style) or "euclidean"
) -> dict:
    """
    From a boolean band_mask (e.g., q1mask/medmask/q3mask), find the farthest standoff point
    from an origin, restricted to the row(s) closest to Y=y0.

    use_slice:
        - "xy" uses (Ny, Nx) and interprets y0/origin_y in Ny units
        - "xz" uses (Nz, Nx) and interprets y0/origin_y in Nz units

    metric:
        - "xmax": maximize X if "standoff" means most sunward at Y≈0
        - "euclidean": maximize sqrt((X-origin_x)^2 + (Y-origin_y)^2)
    """
    Ydim, Xdim = _slice_axes(use_slice)

    if not isinstance(band_mask, xr.DataArray):
        raise TypeError("band_mask must be an xarray.DataArray")
    if set((Ydim, Xdim)) - set(band_mask.dims):
        raise ValueError(f"band_mask must include dims ({Ydim},{Xdim}); got {band_mask.dims}")

    band_mask = band_mask.transpose(Ydim, Xdim)
    arr = band_mask.values.astype(bool)
    H, W = arr.shape

    Y_vals = band_mask[Ydim].values
    X_vals = band_mask[Xdim].values

    # closest Y row to y0 (in physical units)
    y0_row = int(np.argmin(np.abs(Y_vals - y0)))
    lo = max(0, y0_row - y_band)
    hi = min(H - 1, y0_row + y_band)

    sub = arr[lo:hi + 1, :]
    iy_rel, ix = np.nonzero(sub)

    if iy_rel.size == 0:
        return dict(
            r_far=np.nan, X_far=np.nan, Y_far=np.nan,
            x_idx=np.nan, y_idx=np.nan,
            n_candidates=0, y_used=float(Y_vals[y0_row]),
            band=(lo, hi),
            use_slice=use_slice,
            metric=metric,
        )

    iy = iy_rel + lo
    Y = Y_vals[iy]
    X = X_vals[ix]

    metric = metric.lower().strip()
    if metric == "euclidean":
        score = (X - origin_x) ** 2 + (Y - origin_y) ** 2
        k = int(np.argmax(score))
        r_far = float(np.sqrt(score[k]))
    elif metric == "xmax":
        k = int(np.argmax(X))
        r_far = float(np.sqrt((X[k] - origin_x) ** 2 + (Y[k] - origin_y) ** 2))
    else:
        raise ValueError("metric must be 'euclidean' or 'xmax'")

    return dict(
        r_far=r_far,
        X_far=float(X[k]),
        Y_far=float(Y[k]),
        x_idx=int(ix[k]),
        y_idx=int(iy[k]),
        n_candidates=int(ix.size),
        y_used=float(Y_vals[y0_row]),
        band=(lo, hi),
        metric=metric,
    )


# Farthest standoff near Y=0 on the median band
bs_far = farthest_standoff_at_y0_from_mask(bs_med, y0=0.0, y_band=50, metric="xmax")
mp_far = farthest_standoff_at_y0_from_mask(mp_med, y0=0.0, y_band=50, metric="xmax")

# bs_far and mp_far are dicts from farthest_standoff_at_y0_from_mask
standoff_df = pd.DataFrame([
    dict(region="bow_shock", **bs_far),
    dict(region="magnetopause", **mp_far),
])

standoff_df.to_csv(os.path.join(out_folder, f"rps_{use_slice}_standoff_summary.csv"), index=False)

if 1:
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im2 = ax2.pcolormesh(x_plot, y_plot, Bmag_med, vmin=0, vmax=150, shading='auto', cmap='viridis')
    # Bow shock IQR envelope
    ax2.contourf(x_plot, y_plot, (bs_q1 > 0) & (bs_q3 > 0), levels=[0.5, 1], colors='red', alpha=0.7)

    # Magnetopause IQR envelope
    ax2.contourf(x_plot, y_plot, (mp_q1 > 0) & (mp_q3 > 0), levels=[0.5, 1], colors='magenta', alpha=0.7)
    ax2.contour(x_plot, y_plot, bs_med, levels=[0.5], colors='red', linewidths=2)
    ax2.contour(x_plot, y_plot, mp_med, levels=[0.5], colors='magenta', linewidths=2)

    circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=1)
    ax2.add_patch(circle)

    if np.isfinite(bs_far["X_far"]) and np.isfinite(bs_far["Y_far"]):
        ax2.scatter(
            bs_far["X_far"]/2440.0,
            bs_far["Y_far"]/2440.0,
            marker="D",
            s=60,
            facecolor="red",
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
            label=f"Mdn. BS: {round(bs_far["r_far"]/2440.0, 3)} Rₘ",
        )

    if np.isfinite(mp_far["X_far"]) and np.isfinite(mp_far["Y_far"]):
        ax2.scatter(
            mp_far["X_far"]/2440.0,
            mp_far["Y_far"]/2440.0,
            marker="D",
            s=60,
            facecolor="magenta",
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
            label=f"Mdn. MP: {round(mp_far["r_far"]/2440.0, 3)} Rₘ",
        )
    ax2.legend(loc="upper right")

    ax2.set_xlabel(r"$X\ (\mathrm{R_M})$")
    ax2.set_ylabel(ylab)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    ax2.set_aspect("equal")
    plt.colorbar(im2, label="|B| (nT)")
    plt.title("Bow Shock (red) + MP (pink) Location (IQR + Median)")
    fig_path = os.path.join(out_folder, f"rps_{use_slice}_bmag_boundary_labels.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

if 0:
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im2 = ax2.pcolormesh(x, z, Jmag_med, vmin=0, vmax=150, shading='auto', cmap='plasma')
    ax2.contour(x, z, bowshock_med_mask, levels=[0.5], colors='aqua', linewidths=2)
    circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=1)
    ax2.add_patch(circle)

    ax2.set_xlabel(r"$X\ (\mathrm{R_M})$")
    ax2.set_ylabel(r"$Z\ (\mathrm{R_M})$")
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    ax2.set_aspect("equal")
    plt.colorbar(im2, label="|J| (nA/m²)")
    plt.title("Median Bow Shock Identification (blue contour)")
    fig_path = os.path.join(out_folder, f"rps_jmag_bowshock_median.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()