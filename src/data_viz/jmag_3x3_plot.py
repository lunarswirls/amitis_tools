#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recreate a 3x3 slice figure (like the provided paper figure):
  Row 1: equatorial plane (z=0)   -> X_MSE vs Y_MSE
  Row 2: day-night plane  (y=0)   -> X_MSE vs Z_MSE
  Row 3: terminator plane (x=0)   -> Y_MSE vs Z_MSE
Columns are colored by Jx, Jy, Jz. Quivers show projected current direction.
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

# ----------------------------
# SETTINGS
# ----------------------------
case = "RPS"
input_folder = f"/Users/danywaller/Projects/mercury/{case}_Base/object/"
out_folder   = f"/Users/danywaller/Projects/mercury/{case}_Base/slice_current/"
os.makedirs(out_folder, exist_ok=True)

# name of variables inside the NetCDF file
VAR_U = "Jx"
VAR_V = "Jy"
VAR_W = "Jz"

SUB = 4  # quiver subsampling

# Color limits, set to None to autoscale each panel.
CLIM = (-50.0, 50.0)

# If your J is not already nA/m^2, set a scale factor here.
# Example: if J is in A/m^2, use J_SCALE = 1e9 to convert to nA/m^2.
J_SCALE = 1.0

# Mercury radius
RM_M = 2440e3

# first stable timestamp approx. 25000 for dt=0.002, numsteps=115000
sim_steps = list(range(27000, 115000 + 1, 1000))

# ----------------------------
# Helpers
# ----------------------------
def _coords_from_ds(ds: xr.Dataset):
    """Build x,y,z coordinate arrays from ds attrs."""
    xmin, xmax = float(ds.full_xmin), float(ds.full_xmax)
    ymin, ymax = float(ds.full_ymin), float(ds.full_ymax)
    zmin, zmax = float(ds.full_zmin), float(ds.full_zmax)

    dx = float(ds.full_dx)
    dy = float(ds.full_dy) if "full_dy" in ds else float(ds.full_dx)
    dz = float(ds.full_dz)

    x = np.arange(xmin, xmax, dx) / RM_M
    y = np.arange(ymin, ymax, dy) / RM_M
    z = np.arange(zmin, zmax, dz) / RM_M
    return x, y, z


def _planet_disk(ax, plane: str):
    """
    Draw a unit-radius disk and a simple day/night half-shading.
    plane:
      'xy' -> split by x (night at x<0)
      'xz' -> split by x (night at x<0)
      'yz' -> split by y (night at y<0)
    """
    # Base (dayside) disk
    ax.add_patch(Circle((0, 0), 1.0, facecolor="white", edgecolor="black", lw=1.0, zorder=5))

    # Nightside half (black)
    if plane in ("xy", "xz"):
        # left half (x<0): wedge from 90 to 270 degrees
        ax.add_patch(Wedge((0, 0), 1.0, 90, 270, facecolor="black", edgecolor="none", zorder=6))
    elif plane == "yz":
        # View is along -X from dayside; no meaningful in-plane split.
        pass


def _mask_inside_body(X, Y, Z):
    """Mask values inside r<1."""
    r = np.sqrt(X**2 + Y**2)
    Zm = np.array(Z, dtype=float)
    Zm[r < 1.0] = np.nan
    return Zm


def _unit_quiver(u, v, eps=1e-12):
    """Return unit vectors for quiver plotting; avoid divide-by-zero."""
    mag = np.sqrt(u*u + v*v)
    mag = np.where(mag < eps, np.nan, mag)
    return u/mag, v/mag


def plot_panel(ax, X, Y, comp, uproj, vproj, *,
               plane: str,
               title: str,
               xlabel: str,
               ylabel: str,
               clim=None,
               cmap="RdBu_r"):
    """Single panel: pcolormesh of component + unit quivers + planet disk."""
    comp_m = _mask_inside_body(X, Y, comp)

    vmin, vmax = (None, None) if clim is None else clim
    pm = ax.pcolormesh(X, Y, comp_m, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)

    # Quiver: project + normalize; subsample for readability
    uu, vv = _unit_quiver(uproj, vproj)
    uu = _mask_inside_body(X, Y, uu)
    vv = _mask_inside_body(X, Y, vv)

    ax.quiver(
        X[::SUB, ::SUB], Y[::SUB, ::SUB],
        uu[::SUB, ::SUB], vv[::SUB, ::SUB],
        angles="xy", scale_units="xy", scale=12.0,
        width=0.002, headwidth=3, headlength=4, headaxislength=3,
        color="k", alpha=0.85, zorder=3
    )

    _planet_disk(ax, plane=plane)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    return pm


# ----------------------------
# Main loop
# ----------------------------
for sim_step in sim_steps:
    filename = "Base_" + f"{sim_step:06d}"
    f = os.path.join(input_folder, f"Amitis_{case}_{filename}_xz_comp.nc")

    print(f"Processing {os.path.basename(f)} ...")
    ds = xr.open_dataset(f)

    # Coordinates (in R_M)
    x, y, z = _coords_from_ds(ds)

    # Pull 4D -> 3D (time, z, y, x) -> (z, y, x)
    Jx3 = (ds[VAR_U].values * J_SCALE).squeeze(axis=0)
    Jy3 = (ds[VAR_V].values * J_SCALE).squeeze(axis=0)
    Jz3 = (ds[VAR_W].values * J_SCALE).squeeze(axis=0)

    # Build meshgrids for each slice plane (indexing='xy' conventions)
    # Assumes dimensions are ordered (Nz, Ny, Nx) OR have coords named Nz,Ny,Nx.
    # We'll select by coordinate labels if present; otherwise fall back to nearest indices.

    # --- Slice indices nearest to 0 ---
    def _nearest_idx(arr):
        return int(np.nanargmin(np.abs(arr)))

    ix0 = _nearest_idx(x)
    iy0 = _nearest_idx(y)
    iz0 = _nearest_idx(z)

    # Equatorial (z = 0): X vs Y
    # Data on (Ny, Nx)
    Jx_xy = Jx3[iz0, :, :]
    Jy_xy = Jy3[iz0, :, :]
    Jz_xy = Jz3[iz0, :, :]
    Xxy, Yxy = np.meshgrid(x, y, indexing="xy")

    # Polar (y = 0): X vs Z
    # Data on (Nz, Nx)
    Jx_xz = Jx3[:, iy0, :]
    Jy_xz = Jy3[:, iy0, :]
    Jz_xz = Jz3[:, iy0, :]
    Xxz, Zxz = np.meshgrid(x, z, indexing="xy")

    # Dayside (x = 0): Y vs Z
    # Data on (Nz, Ny)
    Jx_yz = Jx3[:, :, ix0]
    Jy_yz = Jy3[:, :, ix0]
    Jz_yz = Jz3[:, :, ix0]
    Yyz, Zyz = np.meshgrid(y, z, indexing="xy")

    # ----------------------------
    # Build 3x3 figure
    # ----------------------------
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)

    # Column headers / colorbar labels
    col_titles = [r"Current density, $J_x$", r"Current density, $J_y$", r"Current density, $J_z$"]

    # Row 1: equatorial (xy) quiver uses (Jx,Jy)
    pms = []
    pms.append(plot_panel(
        axes[0, 0], Xxy, Yxy, Jx_xy, Jx_xy, Jy_xy,
        plane="xy",
        title=col_titles[0],
        xlabel=r"$X\ (R_M)$",
        ylabel=r"$Y\ (R_M)$",
        clim=CLIM
    ))
    pms.append(plot_panel(
        axes[0, 1], Xxy, Yxy, Jy_xy, Jx_xy, Jy_xy,
        plane="xy",
        title=col_titles[1],
        xlabel=r"$X\ (R_M)$",
        ylabel=r"$Y\ (R_M)$",
        clim=CLIM
    ))
    pms.append(plot_panel(
        axes[0, 2], Xxy, Yxy, Jz_xy, Jx_xy, Jy_xy,
        plane="xy",
        title=col_titles[2],
        xlabel=r"$X\ (R_M)$",
        ylabel=r"$Y\ (R_M)$",
        clim=CLIM
    ))

    # Row 2: polar (xz) quiver uses (Jx,Jz)
    pms.append(plot_panel(
        axes[1, 0], Xxz, Zxz, Jx_xz, Jx_xz, Jz_xz,
        plane="xz",
        title="",
        xlabel=r"$X\ (R_M)$",
        ylabel=r"$Z\ (R_M)$",
        clim=CLIM
    ))
    pms.append(plot_panel(
        axes[1, 1], Xxz, Zxz, Jy_xz, Jx_xz, Jz_xz,
        plane="xz",
        title="",
        xlabel=r"$X\ (R_M)$",
        ylabel=r"$Z\ (R_M)$",
        clim=CLIM
    ))
    pms.append(plot_panel(
        axes[1, 2], Xxz, Zxz, Jz_xz, Jx_xz, Jz_xz,
        plane="xz",
        title="",
        xlabel=r"$X\ (R_M)$",
        ylabel=r"$Z\ (R_M)$",
        clim=CLIM
    ))

    # Row 3: dayside (yz) quiver uses (Jy,Jz)
    pms.append(plot_panel(
        axes[2, 0], Yyz, Zyz, Jx_yz, Jy_yz, Jz_yz,
        plane="yz",
        title="",
        xlabel=r"$Y\ (R_M)$",
        ylabel=r"$Z\ (R_M)$",
        clim=CLIM
    ))
    pms.append(plot_panel(
        axes[2, 1], Yyz, Zyz, Jy_yz, Jy_yz, Jz_yz,
        plane="yz",
        title="",
        xlabel=r"$Y\ (R_M)$",
        ylabel=r"$Z\ (R_M)$",
        clim=CLIM
    ))
    pms.append(plot_panel(
        axes[2, 2], Yyz, Zyz, Jz_yz, Jy_yz, Jz_yz,
        plane="yz",
        title="",
        xlabel=r"$Y\ (R_M)$",
        ylabel=r"$Z\ (R_M)$",
        clim=CLIM
    ))

    # Panel letters a–i (upper-left of each axes)
    letters = list("abcdefghi")
    k = 0
    for r in range(3):
        for c in range(3):
            axes[r, c].text(0.02, 0.98, letters[k], transform=axes[r, c].transAxes,
                            ha="left", va="top", fontsize=11, fontweight="bold")
            k += 1

    # One colorbar above each column
    for c in range(3):
        cb = fig.colorbar(pms[c], ax=axes[:, c], location="top", pad=0.02, fraction=0.04, shrink=0.9)
        cb.ax.set_title(r"nA m$^{-2}$", fontsize=9)

    tsec = sim_step * 0.002
    fig.suptitle(f"{case} Current density slices at t = {tsec:.3f} s ({filename})", fontsize=12)

    # Save
    outpath = os.path.join(out_folder, f"{case}_current_slices_{sim_step:06d}.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    ds.close()

print("Done.")
