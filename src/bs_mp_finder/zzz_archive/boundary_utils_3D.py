#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports:
import numpy as np
import xarray as xr
from bs_mp_finder.boundary_thresholds import THRESHOLDS as th


# ----------------------------
# helper: max-radius index in 3D mask
# ----------------------------
def max_radius_index_xr(mask_da, x_name="Nx", y_name="Ny", z_name="Nz"):
    """
    Find max radial distance in 3D xarray mask.

    Returns dict with ix_max, iy_max, iz_max, x_max, y_max, z_max, r_max
    or None if mask is empty.
    """

    # Get 1D coordinate arrays
    x_coords = mask_da.coords[x_name].values
    y_coords = mask_da.coords[y_name].values
    z_coords = mask_da.coords[z_name].values

    # Convert mask to numpy for easier processing
    mask_np = mask_da.values.astype(bool)

    if not mask_np.any():
        return None

    # Create 3D radial distance field
    Xg, Yg, Zg = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    Rg = np.sqrt(Xg ** 2 + Yg ** 2 + Zg ** 2)

    # Mask radii where boundary exists
    Rg_masked = np.where(mask_np, Rg, -np.inf)

    # Find flat index of max
    i_flat = np.argmax(Rg_masked)

    # Unravel to 3D indices
    ix, iy, iz = np.unravel_index(i_flat, mask_np.shape)

    return {
        'ix_max': int(ix),
        'iy_max': int(iy),
        'iz_max': int(iz),
        'x_max': float(x_coords[ix]),
        'y_max': float(y_coords[iy]),
        'z_max': float(z_coords[iz]),
        'r_max': float(Rg[ix, iy, iz])
    }


def labels_for_slice(s: str):
    s = s.lower().strip()
    if s == "xy":
        return r"$X\ (\mathrm{R_M})$", r"$Y\ (\mathrm{R_M})$"
    if s == "xz":
        return r"$X\ (\mathrm{R_M})$", r"$Z\ (\mathrm{R_M})$"
    if s == "yz":
        return r"$Y\ (\mathrm{R_M})$", r"$Z\ (\mathrm{R_M})$"
    raise ValueError(s)


def fetch_coords(ds: xr.Dataset, debug: bool, RM_M=2440.0e3):
    xmin = float(ds.full_xmin)
    xmax = float(ds.full_xmax)
    ymin = float(ds.full_ymin)
    ymax = float(ds.full_ymax)
    zmin = float(ds.full_zmin)
    zmax = float(ds.full_zmax)
    dx = float(ds.full_dx)
    dy = float(ds.full_dy)
    dz = float(ds.full_dz)

    dx_km = dx/1e3
    dy_km = dy/1e3
    dz_km = dz/1e3

    x = np.arange(xmin, xmax, dx) / RM_M
    y = np.arange(ymin, ymax, dy) / RM_M
    z = np.arange(zmin, zmax, dz) / RM_M

    if debug:
        print(f"Grid spacing: dx={dx:.1f} km, dy={dy:.1f} km, dz={dz:.1f} km")
        print(f"Domain: X=[{x[0]:.2f}, {x[-1]:.2f}] R_M ({len(x)} cells)")
        print(f"        Y=[{y[0]:.2f}, {y[-1]:.2f}] R_M ({len(y)} cells)")
        print(f"        Z=[{z[0]:.2f}, {z[-1]:.2f}] R_M ({len(z)} cells)")

    return dx_km, dy_km, dz_km, x, y, z


def extract_3d_fields(ds: xr.Dataset):
    """
    Extract all 3D fields needed for boundary detection
    """
    # Magnetic field
    BX = ds["Bx_tot"].isel(time=0).values  # nT
    BY = ds["By_tot"].isel(time=0).values
    BZ = ds["Bz_tot"].isel(time=0).values
    # Transpose Nz, Ny, Nx → Nx, Ny, Nz
    BX = np.transpose(BX, (2, 1, 0))
    BY = np.transpose(BY, (2, 1, 0))
    BZ = np.transpose(BZ, (2, 1, 0))

    # Proton velocities (pre-ICME + ICME)
    vx_proton = ds["vx01"].isel(time=0).values + ds["vx02"].isel(time=0).values  # km/s
    vy_proton = ds["vy01"].isel(time=0).values + ds["vy02"].isel(time=0).values
    vz_proton = ds["vz01"].isel(time=0).values + ds["vz02"].isel(time=0).values
    # Transpose Nz, Ny, Nx → Nx, Ny, Nz
    vx_proton = np.transpose(vx_proton, (2, 1, 0))
    vy_proton = np.transpose(vy_proton, (2, 1, 0))
    vz_proton = np.transpose(vz_proton, (2, 1, 0))

    # Alpha velocities (pre-ICME + ICME)
    vx_alpha = ds["vx03"].isel(time=0).values + ds["vx04"].isel(time=0).values  # km/s
    vy_alpha = ds["vy03"].isel(time=0).values + ds["vy04"].isel(time=0).values
    vz_alpha = ds["vz03"].isel(time=0).values + ds["vz04"].isel(time=0).values
    # Transpose Nz, Ny, Nx → Nx, Ny, Nz
    vx_alpha = np.transpose(vx_alpha, (2, 1, 0))
    vy_alpha = np.transpose(vy_alpha, (2, 1, 0))
    vz_alpha = np.transpose(vz_alpha, (2, 1, 0))

    # Densities (pre-ICME + ICME)
    den_proton = ds["den01"].isel(time=0).values + ds["den02"].isel(time=0).values  # cm^-3
    den_alpha = ds["den03"].isel(time=0).values + ds["den04"].isel(time=0).values
    tot_den = den_proton + den_alpha
    # Transpose Nz, Ny, Nx → Nx, Ny, Nz
    tot_den = np.transpose(tot_den, (2, 1, 0))

    # Currents
    JX = ds["Jx"].isel(time=0).values  # nA/m^2
    JY = ds["Jy"].isel(time=0).values
    JZ = ds["Jz"].isel(time=0).values
    # Transpose Nz, Ny, Nx → Nx, Ny, Nz
    JX = np.transpose(JX, (2, 1, 0))
    JY = np.transpose(JY, (2, 1, 0))
    JZ = np.transpose(JZ, (2, 1, 0))

    return BX, BY, BZ, vx_proton, vy_proton, vz_proton, vx_alpha, vy_alpha, vz_alpha, den_proton, den_alpha, JX, JY, JZ, tot_den


def compute_masks_3d(f_3d: str, plot_id: str, debug: bool = False):
    """
    Compute 3D bowshock and magnetopause masks from 3D NetCDF file.
    Returns: x_coords, y_coords, z_coords, plot_bg, bs_mask_3d, mp_mask_3d
    """
    ds = xr.open_dataset(f_3d)

    if debug:
        print("DEBUG: compute_masks_3d")

    # Extract 3D fields (already numpy arrays)
    BX, BY, BZ, vx01, vy01, vz01, vx03, vy03, vz03, den01, den03, JX, JY, JZ, tot_den = extract_3d_fields(ds)

    # Derived scalar fields (all numpy)
    Bmag = np.sqrt(BX ** 2 + BY ** 2 + BZ ** 2)  # nT
    Vmag01 = np.sqrt(vx01 ** 2 + vy01 ** 2 + vz01 ** 2)  # km/s
    Vmag03 = np.sqrt(vx03 ** 2 + vy03 ** 2 + vz03 ** 2)  # km/s
    Vmag = Vmag01 + Vmag03  # km/s
    Jmag = np.sqrt(JX ** 2 + JY ** 2 + JZ ** 2)  # nA/m^2
    Pmag = tot_den  # cm^-3

    if debug:
        print(f"Bmag min/med/max={np.nanmin(Bmag):.3f}, {np.nanmedian(Bmag):.3f}, {np.nanmax(Bmag):.3f}")
        print(f"Vmag min/med/max={np.nanmin(Vmag):.3f}, {np.nanmedian(Vmag):.3f}, {np.nanmax(Vmag):.3f}")
        print(f"Pmag min/med/max={np.nanmin(Pmag):.3f}, {np.nanmedian(Pmag):.3f}, {np.nanmax(Pmag):.3f}")

    dx_km, dy_km, dz_km, x_coords, y_coords, z_coords = fetch_coords(ds, debug=debug)

    # Compute gradients WITH PROPER SPACING (in physical units: field/km)
    dB_du = np.gradient(Bmag, dx_km, axis=0)  # nT/km
    dV_du = np.gradient(Vmag, dx_km, axis=0)  # (km/s)/km = 1/s
    dJ_du = np.gradient(Jmag, dx_km, axis=0)  # (nA/m^2)/km
    dJy_du = np.gradient(JY, dx_km, axis=0)  # (nA/m^2)/km
    dP_du = np.gradient(Pmag, dx_km, axis=0)  # (cm^-3)/km

    dB_dv = np.gradient(Bmag, dy_km, axis=1)  # nT/km
    dV_dv = np.gradient(Vmag, dy_km, axis=1)  # 1/s
    dJ_dv = np.gradient(Jmag, dy_km, axis=1)
    dJy_dv = np.gradient(JY, dy_km, axis=1)
    dP_dv = np.gradient(Pmag, dy_km, axis=1)

    dB_dw = np.gradient(Bmag, dz_km, axis=2)  # nT/km
    dV_dw = np.gradient(Vmag, dz_km, axis=2)  # 1/s
    dJ_dw = np.gradient(Jmag, dz_km, axis=2)
    dJy_dw = np.gradient(JY, dz_km, axis=2)
    dP_dw = np.gradient(Pmag, dz_km, axis=2)

    # 3D gradient magnitudes
    gradB_mag = np.sqrt(dB_du ** 2 + dB_dv ** 2 + dB_dw ** 2)  # nT/km
    gradV_mag = np.sqrt(dV_du ** 2 + dV_dv ** 2 + dV_dw ** 2)  # 1/s
    gradP_mag = np.sqrt(dP_du ** 2 + dP_dv ** 2 + dP_dw ** 2)  # cm^-3/km
    gradJ_mag = np.sqrt(dJ_du ** 2 + dJ_dv ** 2 + dJ_dw ** 2)  # (nA/m^2)/km
    gradJy_mag = np.sqrt(dJy_du ** 2 + dJy_dv ** 2 + dJy_dw ** 2)

    # Planetary exclusion (1 < r < 4 R_M)
    Xg, Yg, Zg = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    Rg = np.sqrt(Xg ** 2 + Yg ** 2 + Zg ** 2)
    outside_body = (Rg >= 1.0) & (Rg < 4.0)

    if debug:
        print(f"Valid volume: {np.sum(outside_body)} / {outside_body.size} cells ({100 * np.mean(outside_body):.1f}%)")

    # Apply exclusion
    gradB_mag = np.where(outside_body, gradB_mag, np.nan)
    gradV_mag = np.where(outside_body, gradV_mag, np.nan)
    gradP_mag = np.where(outside_body, gradP_mag, np.nan)
    gradJ_mag = np.where(outside_body, gradJ_mag, np.nan)
    gradJy_mag = np.where(outside_body, gradJy_mag, np.nan)
    dV_du = np.where(outside_body, dV_du, np.nan)
    dJ_du = np.where(outside_body, dJ_du, np.nan)
    dP_du = np.where(outside_body, dP_du, np.nan)
    dJy_du = np.where(outside_body, dJy_du, np.nan)
    dB_du = np.where(outside_body, dB_du, np.nan)

    if debug:
        print(
            f"gradBmag min/med/max={np.nanmin(gradB_mag):.6f}, {np.nanmedian(gradB_mag):.6f}, {np.nanmax(gradB_mag):.6f} nT/km")
        print(
            f"gradVmag min/med/max={np.nanmin(gradV_mag):.6f}, {np.nanmedian(gradV_mag):.6f}, {np.nanmax(gradV_mag):.6f} s^-1")
        print(
            f"gradPmag min/med/max={np.nanmin(gradP_mag):.6f}, {np.nanmedian(gradP_mag):.6f}, {np.nanmax(gradP_mag):.6f} cm^-3/km")

    # Thresholds based on maxima
    bmag_threshold_bs = th["Bgradmax_bs"] * np.nanmax(np.abs(gradB_mag))
    vmag_threshold_bs = th["Vgradmax_bs"] * np.nanmax(np.abs(gradV_mag))
    jmag_threshold_bs = th["Jgradmax_bs"] * np.nanmax(np.abs(gradJ_mag))
    den_threshold_bs = th["Pgradmax_bs"] * np.nanmax(np.abs(gradP_mag))

    bmag_threshold_mp = th["Bgradmax_mp"] * np.nanmax(np.abs(gradB_mag))
    vmag_threshold_mp = th["Vgradmax_mp"] * np.nanmax(np.abs(gradV_mag))
    jmag_threshold_mp = th["Jgradmax_mp"] * np.nanmax(np.abs(gradJy_mag))
    den_threshold_mp = th["Pgradmax_mp"] * np.nanmax(np.abs(gradP_mag))

    if debug:
        print(
            f"Thresholds - BS: B={bmag_threshold_bs:.6f}, V={vmag_threshold_bs:.6f}, J={jmag_threshold_bs:.6f}, P={den_threshold_bs:.6f}")
        print(
            f"Thresholds - MP: B={bmag_threshold_mp:.6f}, V={vmag_threshold_mp:.6f}, Jy={jmag_threshold_mp:.6f}, P={den_threshold_mp:.6f}")

    # Bowshock: outward velocity gradient, strong current, density decrease along X
    bowshock_mask = (
            (gradV_mag > vmag_threshold_bs) & (dV_du > 0) &
            (gradJ_mag > jmag_threshold_bs) & (dJ_du < 0) &
            (gradP_mag > den_threshold_bs) & (dP_du < 0) &
            outside_body
    )

    # Magnetopause: strong Jy gradient, density increase along X, B gradient
    magnetopause_mask = (
            (gradJy_mag > jmag_threshold_mp) & (dJy_du > 0) &
            (gradP_mag > den_threshold_mp) & (dP_du > 0) &
            (gradB_mag > bmag_threshold_mp) & (dB_du < 0) &
            outside_body
    )

    # Mutual exclusivity
    bowshock_mask = bowshock_mask & ~magnetopause_mask

    bg_map = {"Bmag": Bmag, "Jmag": Jmag, "Pmag": Pmag}
    if plot_id not in bg_map:
        raise ValueError(f"Invalid plot_id='{plot_id}'. Options: {list(bg_map)}")
    plot_bg = bg_map[plot_id]

    if debug:
        print(f"3D BS candidates: {np.sum(bowshock_mask)}")
        print(f"3D MP candidates: {np.sum(magnetopause_mask)}")
        print(f"BS volume fraction: {np.mean(bowshock_mask):.6f}")
        print(f"MP volume fraction: {np.mean(magnetopause_mask):.6f}\n")

    ds.close()

    # Convert masks to xarray DataArrays
    bs_mask_da = xr.DataArray(
        bowshock_mask.astype(np.uint8),
        coords={'Nx': x_coords, 'Ny': y_coords, 'Nz': z_coords},
        dims=['Nx', 'Ny', 'Nz']
    )
    mp_mask_da = xr.DataArray(
        magnetopause_mask.astype(np.uint8),
        coords={'Nx': x_coords, 'Ny': y_coords, 'Nz': z_coords},
        dims=['Nx', 'Ny', 'Nz']
    )

    return x_coords, y_coords, z_coords, plot_bg, bs_mask_da, mp_mask_da


def slice_axes_dims(use_slice: str):
    s = use_slice.lower().strip()
    if s == "xy":
        return "Ny", "Nx"
    if s == "xz":
        return "Nz", "Nx"
    if s == "yz":
        return "Nz", "Ny"
    raise ValueError(use_slice)


def occupancy_and_bands(stack_bool: np.ndarray, thresholds=(0.25, 0.125, 0.0625)):
    p = stack_bool.mean(axis=0)
    q1_thr, med_thr, q3_thr = thresholds
    return p, (p >= q1_thr), (p >= med_thr), (p >= q3_thr)


def max_axis_distance(mask, x, y, width_km=500):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None

    width_re = width_km / 2440.0
    x_vals, a_vals = [], []

    for iy, ix in pts:
        av = float(y[iy])
        if abs(av) > width_re: continue
        xv = float(x[ix])
        x_vals.append(xv)
        a_vals.append(av)

    if not x_vals:
        return None

    return float(np.median(x_vals)), float(np.median(a_vals)), float(np.hypot(*np.median([x_vals, a_vals], axis=0)))
