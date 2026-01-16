#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Identify bow shock and magnetopause in AMITIS RPS slice outputs (XY/XZ),
plot per-timestep boundaries, plot per-timestep signed rotation, then plot median/IQR summary.

Additions vs your original:
  - IMF estimation from furthest-upstream slab near ds.full_xmax
  - Analytic MP rotation-sign enforcement for XZ @ Y=0 with dipole ~ +X:
        expected_sign = sign(IMF_Bz)
    and it gates MP pixels by:
        sign( (Bhat × ∂x Bhat) · ŷ ) == expected_sign
  - Per-timestep plot of signed rotation (rot_signed) with sign and MP overlay.
"""

# Imports:
import os
import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# SETTINGS
# -------------------------
use_slice = "xz"  # "xy" or "xz" (this script supports both)

# first stable timestamp approx. 25000 for dt=0.002, numsteps=115000
sim_steps = list(range(27000, 115000 + 1, 1000))

# directories
input_folder = f"/Users/danywaller/Projects/mercury/RPS_Base/fig_{use_slice}/"
out_folder = f"/Users/danywaller/Projects/mercury/RPS_Base/slice_bowshock/{use_slice}_imf_rotation/"
os.makedirs(out_folder, exist_ok=True)

# plot labels
if use_slice == "xy":
    ylab = r"$Y\ (\mathrm{R_M})$"
elif use_slice == "xz":
    ylab = r"$Z\ (\mathrm{R_M})$"
else:
    print("\nInvalid use_slice argument:", use_slice)
    print("Must be either 'xy' or 'xz'")
    sys.exit(1)

# -------------------------
# Threshold settings (your originals)
# -------------------------
Bgradmax = 0.35   # 0.25–0.45
Vgradmax = 0.10   # 0.05–0.15
Pgradmax = 0.10   # 0.20–0.35 (note: you set 0.1)
Jgradmax = 0.25   # 0.20–0.35
rotmax   = 0.10   # 0.05–0.15  (weak rotation)

Vgradnmax_mp = 0.25   # 0.20–0.35
Pgradmax_mp  = 0.75   # 0.85–0.98
Jgradmax_mp  = 0.60   # 0.50–0.75
rotmax_mp    = 0.10   # 0.60–0.85  (strong rotation)  (note: you set 0.1)

# -------------------------
# Variable names in the NetCDF
# -------------------------
VAR_X = "Bx"   # nT
VAR_Y = "By"   # nT
VAR_Z = "Bz"   # nT

VAR_V1X = "vx01"  # km/s
VAR_V1Y = "vy01"
VAR_V1Z = "vz01"
VAR_V3X = "vx03"
VAR_V3Y = "vy03"
VAR_V3Z = "vz03"

VAR_JX = "Jx"   # nA/m^2
VAR_JY = "Jy"
VAR_JZ = "Jz"

VAR_DEN1 = "den01"  # cm^-3
VAR_DEN3 = "den03"  # cm^-3

RM_M = 2440e3  # meters per R_M


# -------------------------
# Helpers
# -------------------------
def cross_3vec(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """
    Cross product for xarray DataArrays with dim 'comp' labeled ['x','y','z'].
    """
    ax, ay, az = a.sel(comp="x"), a.sel(comp="y"), a.sel(comp="z")
    bx, by, bz = b.sel(comp="x"), b.sel(comp="y"), b.sel(comp="z")

    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx

    out = xr.concat([cx, cy, cz], dim="comp")
    return out.assign_coords(comp=["x", "y", "z"])


def estimate_msph_direction(
    BX: xr.DataArray,
    BY: xr.DataArray,
    BZ: xr.DataArray,
    x_plot: np.ndarray,
    z_plot: np.ndarray,
    *,
    rmin_rm: float = 0.6,
    rmax_rm: float = 0.7,
    z_abs_max_rm: float = 0.1,
    robust: str = "median",
) -> dict:
    """
    Estimate the *magnetospheric (inside)* magnetic-field direction from a
    near-planet, dayside region in an XZ slice (Y = 0).

    This function is explicitly designed for Mercury-like geometries, where:
      - the magnetospheric field direction cannot be assumed to be +X everywhere,
      - the dipole may be offset or tilted,
      - and the magnetopause rotation sense depends on the *relative* orientation
        between IMF and magnetospheric fields.

    The goal is to recover a representative unit vector
        b̂_msph = (bhx, bhy, bhz)
    describing the *internal* field direction that the IMF rotates into across
    the magnetopause.

    Parameters
    ----------
    BX, BY, BZ : xr.DataArray
        Magnetic field components on the selected slice (units: nT).
        Expected shape is (Nz, Nx) for an XZ slice.

    x_plot, z_plot : np.ndarray
        1D arrays of X and Z coordinates in units of R_M (Mercury radii),
        corresponding to the grid of BX/BY/BZ.

    rmin_rm, rmax_rm : float
        Radial bounds (in R_M) defining a shell inside the magnetosphere.
        This excludes the planetary interior and avoids the MP itself.

    z_abs_max_rm : float
        Maximum |Z| (in R_M) to exclude the plasma sheet and high-latitude regions,
        ensuring a clean sampling of the dayside magnetospheric field.

    robust : {"median", "mean"}
        Statistical estimator for the representative field direction.
        Median is preferred for robustness against boundary contamination.

    Returns
    -------
    dict with keys:
        npts : int
            Number of grid points used in the estimate.

        bhx, bhy, bhz : float
            Components of the unit magnetospheric field direction.

        Bx, By, Bz : float
            Raw (non-normalized) representative field components.

        Bmag : float
            Magnitude of the representative field vector.
    """

    # ------------------------------------------------------------------
    # Construct 2D coordinate grids matching the magnetic field arrays.
    # BX has shape (Nz, Nx). We broadcast x_plot and z_plot accordingly.
    # ------------------------------------------------------------------
    X2 = np.broadcast_to(x_plot[None, :], BX.shape)
    Z2 = np.broadcast_to(z_plot[:, None], BX.shape)

    # Radial distance in the XZ plane (units: R_M)
    r = np.sqrt(X2**2 + Z2**2)

    # ------------------------------------------------------------------
    # Define a *magnetospheric interior* mask:
    #   - dayside only (X > 0),
    #   - within a radial shell inside the MP,
    #   - near the equatorial plane to avoid plasma sheet effects.
    # ------------------------------------------------------------------
    mask = (
        (X2 > 0.0) &
        (r >= rmin_rm) & (r <= rmax_rm) &
        (np.abs(Z2) <= z_abs_max_rm)
    )

    # Extract masked magnetic-field components
    bx = BX.where(mask).values
    by = BY.where(mask).values
    bz = BZ.where(mask).values

    # Remove NaNs introduced by masking or missing data
    bx = bx[np.isfinite(bx)]
    by = by[np.isfinite(by)]
    bz = bz[np.isfinite(bz)]

    # Number of valid points used in the estimate
    n = bx.size

    # If too few points are available, return NaNs to signal failure
    if n < 10:
        return dict(
            npts=int(n),
            bhx=np.nan, bhy=np.nan, bhz=np.nan,
            Bx=np.nan, By=np.nan, Bz=np.nan,
            Bmag=np.nan
        )

    # ------------------------------------------------------------------
    # Compute a representative magnetospheric field vector.
    # Median is preferred to suppress boundary/sheath contamination.
    # ------------------------------------------------------------------
    if robust == "median":
        Bx0, By0, Bz0 = np.nanmedian(bx), np.nanmedian(by), np.nanmedian(bz)
    else:
        Bx0, By0, Bz0 = np.nanmean(bx), np.nanmean(by), np.nanmean(bz)

    # Magnitude of the representative field
    Bmag0 = np.sqrt(Bx0**2 + By0**2 + Bz0**2)

    # Guard against pathological cases
    if Bmag0 == 0 or not np.isfinite(Bmag0):
        return dict(
            npts=int(n),
            bhx=np.nan, bhy=np.nan, bhz=np.nan,
            Bx=Bx0, By=By0, Bz=Bz0,
            Bmag=Bmag0
        )

    # ------------------------------------------------------------------
    # Return normalized direction and diagnostics
    # ------------------------------------------------------------------
    return dict(
        npts=int(n),
        bhx=Bx0 / Bmag0,
        bhy=By0 / Bmag0,
        bhz=Bz0 / Bmag0,
        Bx=Bx0, By=By0, Bz=Bz0,
        Bmag=Bmag0
    )


def estimate_imf_from_furthest_upstream_slice(
    ds: xr.Dataset,
    BX: xr.DataArray,
    BY: xr.DataArray,
    BZ: xr.DataArray,
    x_plot_1d: np.ndarray,
    y_plot_1d: np.ndarray,
    *,
    slab_width_rm: float = 0.2,
    y_abs_max_rm: float = 2.0,
    r_min_rm: float = 1.05,
    robust: str = "median",
) -> dict:
    """
    Estimate the upstream interplanetary magnetic field (IMF) direction
    from the *furthest +X slab* of the simulation domain.

    This function assumes:
      - +X is sunward (MSO-like coordinates),
      - the upstream region is approximately uniform,
      - and the furthest +X edge of the domain is uncontaminated by bow shock
        or magnetosheath structure.

    The output provides a unit vector b̂_IMF and standard IMF diagnostics
    (cone and clock angles).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset providing domain metadata (full_xmax).

    BX, BY, BZ : xr.DataArray
        Magnetic field components on the selected slice.

    x_plot_1d, y_plot_1d : np.ndarray
        1D coordinate arrays (R_M) corresponding to BX/BY/BZ grid.

    slab_width_rm : float
        Thickness (in R_M) of the upstream sampling slab.

    y_abs_max_rm : float
        Maximum |Y| allowed, excluding flanks and downstream regions.

    r_min_rm : float
        Minimum radial distance to exclude the planetary interior.

    robust : {"median", "mean"}
        Statistical estimator for the upstream IMF.

    Returns
    -------
    dict with keys:
        npts, bhx, bhy, bhz,
        Bx, By, Bz, Bmag,
        x_slice_rm,
        cone_deg, clock_deg
    """

    # Convert domain max X from meters to R_M
    x_max_rm = float(ds.full_xmax) / RM_M
    x_lo_rm = x_max_rm - slab_width_rm

    # Build 2D coordinate grids
    X2 = np.broadcast_to(x_plot_1d[None, :], BX.shape)
    Y2 = np.broadcast_to(y_plot_1d[:, None], BX.shape)

    # Radial distance in the slice plane
    r = np.sqrt(X2**2 + Y2**2)

    # Upstream IMF mask
    m = (
        (X2 >= x_lo_rm) & (X2 <= x_max_rm) &
        (np.abs(Y2) <= y_abs_max_rm) &
        (r >= r_min_rm)
    )

    # Extract and clean IMF samples
    bx = BX.where(m).values
    by = BY.where(m).values
    bz = BZ.where(m).values

    bx = bx[np.isfinite(bx)]
    by = by[np.isfinite(by)]
    bz = bz[np.isfinite(bz)]
    n = bx.size

    # Fail gracefully if insufficient data
    if n < 10:
        return dict(
            npts=int(n), bhx=np.nan, bhy=np.nan, bhz=np.nan,
            Bx=np.nan, By=np.nan, Bz=np.nan, Bmag=np.nan,
            x_slice_rm=(x_lo_rm, x_max_rm),
            cone_deg=np.nan, clock_deg=np.nan
        )

    # Robust IMF estimate
    if robust.lower() == "median":
        Bx0, By0, Bz0 = float(np.nanmedian(bx)), float(np.nanmedian(by)), float(np.nanmedian(bz))
    else:
        Bx0, By0, Bz0 = float(np.nanmean(bx)), float(np.nanmean(by)), float(np.nanmean(bz))

    Bmag0 = float(np.sqrt(Bx0**2 + By0**2 + Bz0**2))

    if not np.isfinite(Bmag0) or Bmag0 == 0.0:
        return dict(
            npts=int(n), bhx=np.nan, bhy=np.nan, bhz=np.nan,
            Bx=Bx0, By=By0, Bz=Bz0, Bmag=Bmag0,
            x_slice_rm=(x_lo_rm, x_max_rm),
            cone_deg=np.nan, clock_deg=np.nan
        )

    # Normalize IMF direction
    bhx, bhy, bhz = Bx0 / Bmag0, By0 / Bmag0, Bz0 / Bmag0

    return dict(
        npts=int(n),
        bhx=bhx, bhy=bhy, bhz=bhz,
        Bx=Bx0, By=By0, Bz=Bz0, Bmag=Bmag0,
        x_slice_rm=(x_lo_rm, x_max_rm),
        cone_deg=float(np.degrees(np.arccos(np.clip(bhx, -1.0, 1.0)))),
        clock_deg=float(np.degrees(np.arctan2(bhz, bhy))),
    )


def expected_mp_rotation_sign(
    imf: dict,
    msph: dict,
    *,
    slice_plane: str = "xz",
):
    """
    Compute the *expected* magnetopause rotation sign based on the
    relative orientation of the upstream IMF and the internal
    magnetospheric field.

    Physically, this enforces:
        sign[(b̂_IMF × b̂_msph) · n̂_slice]

    where n̂_slice is the normal to the 2D slice plane.

    This is the correct topological criterion for the sense of field
    rotation across a tangential discontinuity / magnetopause.

    Parameters
    ----------
    imf : dict
        Output of estimate_imf_from_furthest_upstream_slice().
        Must contain bhx, bhy, bhz.

    msph : dict
        Output of estimate_msph_direction().
        Must contain bhx, bhy, bhz.

    slice_plane : {"xz", "xy"}
        Defines the plane of the slice:
          - "xz" → rotation normal is +Y
          - "xy" → rotation normal is +Z

    Returns
    -------
    int or None
        +1 or −1 for the expected rotation sign.
        Returns None if the sign cannot be robustly determined.
    """

    # Require valid unit vectors from both regions
    if not np.isfinite(imf["bhx"]) or not np.isfinite(msph["bhx"]):
        return None

    # Assemble unit vectors
    b_imf = np.array([imf["bhx"], imf["bhy"], imf["bhz"]], float)
    b_ms  = np.array([msph["bhx"], msph["bhy"], msph["bhz"]], float)

    # Cross product gives rotation sense from IMF → magnetosphere
    c = np.cross(b_imf, b_ms)

    # Select component normal to the plotting plane
    if slice_plane == "xz":
        s = c[1]   # y-component
    elif slice_plane == "xy":
        s = c[2]   # z-component
    else:
        raise ValueError("slice_plane must be 'xz' or 'xy'")

    # Reject ambiguous or undefined cases
    if not np.isfinite(s) or s == 0:
        return None

    return int(np.sign(s))


# -------------------------
# Storage for median plots
# -------------------------
bs_positions: list[xr.DataArray] = []
mp_positions: list[xr.DataArray] = []
Bmag_list: list[xr.DataArray] = []

# -------------------------
# Main loop
# -------------------------
for sim_step in sim_steps:
    filename = "Base_" + f"{sim_step:06d}"
    f = input_folder + f"Amitis_RPS_{filename}_{use_slice}_comp.nc"

    print(f"Processing {os.path.basename(f)} ...")
    ds = xr.open_dataset(f)

    # Extract physical domain extents (meters)
    xmin = float(ds.full_xmin)
    xmax = float(ds.full_xmax)
    ymin = float(ds.full_ymin)
    ymax = float(ds.full_ymax)
    zmin = float(ds.full_zmin)
    zmax = float(ds.full_zmax)

    dx = float(ds.full_dx)
    dy = float(ds.full_dy)
    dz = float(ds.full_dz)

    # Build coordinate arrays (meters)
    x = np.arange(xmin, xmax, dx)

    if use_slice == "xy":
        y = np.arange(ymin, ymax, dy)
    elif use_slice == "xz":
        y = np.arange(zmin, zmax, dz)
    else:
        print("\nInvalid use_slice argument:", use_slice)
        sys.exit(1)

    # Convert to R_M for plotting / region masks
    x_plot = x / RM_M
    y_plot = y / RM_M

    # Extract arrays on the slice
    if use_slice == "xy":
        BX = ds[VAR_X].sel(Nz=0, method="nearest").squeeze()
        BY = ds[VAR_Y].sel(Nz=0, method="nearest").squeeze()
        BZ = ds[VAR_Z].sel(Nz=0, method="nearest").squeeze()

        vx01 = ds[VAR_V1X].sel(Nz=0, method="nearest").squeeze() * 1e3
        vy01 = ds[VAR_V1Y].sel(Nz=0, method="nearest").squeeze() * 1e3
        vz01 = ds[VAR_V1Z].sel(Nz=0, method="nearest").squeeze() * 1e3
        vx03 = ds[VAR_V3X].sel(Nz=0, method="nearest").squeeze() * 1e3
        vy03 = ds[VAR_V3Y].sel(Nz=0, method="nearest").squeeze() * 1e3
        vz03 = ds[VAR_V3Z].sel(Nz=0, method="nearest").squeeze() * 1e3

        den01 = ds[VAR_DEN1].sel(Nz=0, method="nearest").squeeze() * 1e-6  # cm^-3 -> m^-3
        den03 = ds[VAR_DEN3].sel(Nz=0, method="nearest").squeeze() * 1e-6

        JX = ds[VAR_JX].sel(Nz=0, method="nearest").squeeze()
        JY = ds[VAR_JY].sel(Nz=0, method="nearest").squeeze()
        JZ = ds[VAR_JZ].sel(Nz=0, method="nearest").squeeze()

    else:  # xz
        BX = ds[VAR_X].sel(Ny=0, method="nearest").squeeze()
        BY = ds[VAR_Y].sel(Ny=0, method="nearest").squeeze()
        BZ = ds[VAR_Z].sel(Ny=0, method="nearest").squeeze()

        vx01 = ds[VAR_V1X].sel(Ny=0, method="nearest").squeeze() * 1e3
        vy01 = ds[VAR_V1Y].sel(Ny=0, method="nearest").squeeze() * 1e3
        vz01 = ds[VAR_V1Z].sel(Ny=0, method="nearest").squeeze() * 1e3
        vx03 = ds[VAR_V3X].sel(Ny=0, method="nearest").squeeze() * 1e3
        vy03 = ds[VAR_V3Y].sel(Ny=0, method="nearest").squeeze() * 1e3
        vz03 = ds[VAR_V3Z].sel(Ny=0, method="nearest").squeeze() * 1e3

        den01 = ds[VAR_DEN1].sel(Ny=0, method="nearest").squeeze() * 1e-6
        den03 = ds[VAR_DEN3].sel(Ny=0, method="nearest").squeeze() * 1e-6

        JX = ds[VAR_JX].sel(Ny=0, method="nearest").squeeze()
        JY = ds[VAR_JY].sel(Ny=0, method="nearest").squeeze()
        JZ = ds[VAR_JZ].sel(Ny=0, method="nearest").squeeze()

    # Total density
    tot_den = den01 + den03

    # Magnitudes
    Bmag = np.sqrt(BX**2 + BY**2 + BZ**2)
    Vmag01 = np.sqrt(vx01**2 + vy01**2 + vz01**2)
    Vmag03 = np.sqrt(vx03**2 + vy03**2 + vz03**2)
    Vmag = Vmag01 + Vmag03
    Jmag = np.sqrt(JX**2 + JY**2 + JZ**2)

    # Unit vector field
    B = xr.concat([BX, BY, BZ], dim="comp").assign_coords(comp=["x", "y", "z"])
    Bhat = B / Bmag

    # Derivatives (x is always Nx)
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
    else:  # xz
        dB_dz = Bmag.differentiate("Nz")
        dV_dz = Vmag.differentiate("Nz")
        dJ_dz = Jmag.differentiate("Nz")
        dP_dz = tot_den.differentiate("Nz")
        dBhat_dz = Bhat.differentiate("Nz")

    # Gradient magnitudes
    gradB = np.sqrt(dB_dx**2 + dB_dz**2)
    gradV = np.sqrt(dV_dx**2 + dV_dz**2)
    gradP = np.sqrt(dP_dx**2 + dP_dz**2)
    gradJ = np.sqrt(dJ_dx**2 + dJ_dz**2)

    # Rotation strength (rad/m)
    rotation_strength = (dBhat_dx**2 + dBhat_dz**2).sum("comp") ** 0.5

    # Thresholds
    vmag_threshold = Vgradmax * np.nanmax(gradV)
    jmag_threshold = Jgradmax * np.nanmax(gradJ)
    den_threshold = Pgradmax * np.nanmax(gradP)
    rot_threshold_mp = rotmax_mp * np.nanmax(rotation_strength)

    # IMF from furthest upstream slice
    imf = estimate_imf_from_furthest_upstream_slice(
        ds, BX, BY, BZ, x_plot, y_plot,
        slab_width_rm=0.2,
        y_abs_max_rm=2.0,
        r_min_rm=1.05,
    )

    # Magnetospheric field direction
    msph = estimate_msph_direction(
        BX, BY, BZ,
        x_plot=x_plot,
        z_plot=y_plot,  # y_plot is z for XZ slice
    )

    # Expected MP rotation sign
    expected_sign = expected_mp_rotation_sign(
        imf, msph, slice_plane=use_slice
    )

    # Signed rotation across +X in plane normal direction:
    rot_vec = cross_3vec(Bhat, dBhat_dx)
    if use_slice == "xz":
        rot_signed = rot_vec.sel(comp="y")  # signed (rad/m)
        rot_sign_field = xr.apply_ufunc(np.sign, rot_signed)
    elif use_slice == "xy":
        rot_signed = rot_vec.sel(comp="z")
        rot_sign_field = xr.apply_ufunc(np.sign, rot_signed)
    else:
        rot_signed = None
        rot_sign_field = None

    # Boundary masks (original + sign gate on MP for XZ)
    magnetopause_mask = (
        (gradJ > jmag_threshold) &
        (gradP > den_threshold) & (dP_dx > 0) &
        (gradV < vmag_threshold) & (dV_dx < 0) &
        (rotation_strength > rot_threshold_mp)
    )
    if expected_sign is not None:
        magnetopause_mask = magnetopause_mask & (rot_sign_field == expected_sign)

    bowshock_mask = (
        (gradJ > jmag_threshold) &
        (gradP > den_threshold) & (dP_dx < 0) &
        (rotation_strength < rot_threshold_mp)
    )
    bowshock_mask &= ~magnetopause_mask

    # Exclude plasma sheet
    x_bad = x_plot < 0.75
    y_bad = (y_plot > -1.2) & (y_plot < 1.2)
    exclude_region = y_bad[:, None] & x_bad[None, :]
    bowshock_mask &= ~exclude_region
    magnetopause_mask &= ~exclude_region

    # Store for later median plots
    bs_positions.append(bowshock_mask)
    mp_positions.append(magnetopause_mask)
    Bmag_list.append(Bmag)

    # -------------------------
    # Per-timestep plot: boundaries over Bmag
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    imB = ax.pcolormesh(x_plot, y_plot, Bmag, vmin=0, vmax=150, shading="auto", cmap="viridis")
    ax.contour(x_plot, y_plot, bowshock_mask, levels=[0.5], colors="red", linewidths=2)
    ax.contour(x_plot, y_plot, magnetopause_mask, levels=[0.5], colors="magenta", linewidths=2)
    ax.add_patch(plt.Circle((0, 0), 1, edgecolor="white", facecolor="none", linewidth=1))

    ax.set_xlabel(r"$X\ (\mathrm{R_M})$")
    ax.set_ylabel(ylab)
    ax.set_aspect("equal")
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])

    tsec = sim_step * 0.002
    ax.set_title(f"{use_slice.upper()} BS (red) + MP (magenta), t={tsec:.3f}s")
    plt.colorbar(imB, label="|B| (nT)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"rps_{use_slice}_imf_rot_bmag_boundaries_{sim_step:06d}.png"), dpi=300)
    plt.close()

    # -------------------------
    # Per-timestep plot: signed rotation sign
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot sign field, use vmin/vmax to make it stable between timesteps
    imS = ax.pcolormesh(
        x_plot, y_plot,
        rot_sign_field,
        vmin=-1, vmax=1,
        shading="auto",
        cmap="bwr"
    )

    # Overlay MP and BS contours for context
    # ax.contour(x_plot, y_plot, bowshock_mask, levels=[0.5], colors="black", linewidths=1.0)
    # ax.contour(x_plot, y_plot, magnetopause_mask, levels=[0.5], colors="lime", linewidths=1.5)

    ax.add_patch(plt.Circle((0, 0), 1, edgecolor="white", facecolor="none", linewidth=2))

    ax.set_xlabel(r"$X\ (\mathrm{R_M})$")
    ax.set_ylabel(ylab)
    ax.set_aspect("equal")
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])

    if expected_sign is None:
        ax.set_title(f"{use_slice.upper()} Plane: Rotation sign, t={tsec:.3f}s")
    else:
        ax.set_title(
            f"{use_slice.upper()} Plane: Rotation sign, t={tsec:.3f}s | expected MP sign {expected_sign:+d}"
        )

    # Colorbar with ticks
    cb = plt.colorbar(imS, label="rotation sign")
    cb.set_ticks([-1, 0, 1])
    cb.set_ticklabels(["-1", "0", "+1"])

    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"rps_{use_slice}_imf_rot_sign_{sim_step:06d}.png"), dpi=300)
    plt.close()

    ds.close()

# -------------------------
# Median / occupancy summary
# -------------------------
Bmag_med = np.median(np.stack(Bmag_list, axis=0), axis=0)

bs_stack = np.stack(bs_positions, axis=0).astype(float)
bs_stack[bs_stack == 0] = np.nan
mp_stack = np.stack(mp_positions, axis=0).astype(float)
mp_stack[mp_stack == 0] = np.nan

bs_q1 = np.nanpercentile(bs_stack, 25, axis=0)
bs_q3 = np.nanpercentile(bs_stack, 75, axis=0)
mp_q1 = np.nanpercentile(mp_stack, 25, axis=0)
mp_q3 = np.nanpercentile(mp_stack, 75, axis=0)

def _slice_axes(def_slice: str) -> tuple[str, str]:
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
    Ydim, Xdim = _slice_axes(use_slice)

    if isinstance(positions, xr.DataArray):
        da = positions
        if "time" in da.dims:
            stack = da.astype(np.uint8)
            ref = da.isel(time=0)
        else:
            stack = da.expand_dims(time=[np.datetime64("NaT")]).astype(np.uint8)
            ref = da
    else:
        masks = list(positions)
        if len(masks) == 0:
            raise ValueError("positions is empty")
        stack = xr.concat([m.astype(np.uint8) for m in masks], dim="time")
        ref = masks[0]

    stack = stack.transpose("time", Ydim, Xdim)
    p = stack.mean("time").rename("occupancy")
    p = p.assign_coords({Ydim: ref[Ydim], Xdim: ref[Xdim]})

    q1_thr, med_thr, q3_thr = thresholds
    q1mask  = (p >= q1_thr).rename("q1mask")
    medmask = (p >= med_thr).rename("medmask")
    q3mask  = (p >= q3_thr).rename("q3mask")
    return p, q1mask, medmask, q3mask

bs_p, _, bs_med, _ = occupancy_and_bands(bs_positions)
mp_p, _, mp_med, _ = occupancy_and_bands(mp_positions)

# -------------------------
# Final median plot
# -------------------------
fig2, ax2 = plt.subplots(figsize=(8, 6))
im2 = ax2.pcolormesh(x_plot, y_plot, Bmag_med, vmin=0, vmax=150, shading="auto", cmap="viridis")

# IQR envelopes
ax2.contourf(x_plot, y_plot, (bs_q1 > 0) & (bs_q3 > 0), levels=[0.5, 1], colors="red", alpha=0.35)
ax2.contourf(x_plot, y_plot, (mp_q1 > 0) & (mp_q3 > 0), levels=[0.5, 1], colors="magenta", alpha=0.25)

# Median contours
ax2.contour(x_plot, y_plot, bs_med, levels=[0.5], colors="red", linewidths=2)
ax2.contour(x_plot, y_plot, mp_med, levels=[0.5], colors="magenta", linewidths=2)

ax2.add_patch(plt.Circle((0, 0), 1, edgecolor="white", facecolor="none", linewidth=1))

ax2.set_xlabel(r"$X\ (\mathrm{R_M})$")
ax2.set_ylabel(ylab)
ax2.set_aspect("equal")
ax2.set_xlim([-5, 5])
ax2.set_ylim([-5, 5])

plt.colorbar(im2, label="|B| (nT)")
ax2.set_title("Bow Shock (red) + MP (magenta): IQR fill + median contours")
plt.tight_layout()
plt.savefig(os.path.join(out_folder, f"rps_{use_slice}_imf_rot_bmag_boundaries_median.png"), dpi=300)
plt.close()

print("Done.")
