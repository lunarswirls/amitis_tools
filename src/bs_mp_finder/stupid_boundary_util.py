#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports:
import numpy as np
import xarray as xr

# Constants
RM_M = 2440.0e3  # Mercury radius in meters
Z_MAG_KM = 484.0  # Magnetic equator height in km
Z_MAG_RM = Z_MAG_KM / 2440.0  # Magnetic equator in R_M


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

    dx_km = dx / 1e3
    dy_km = dy / 1e3
    dz_km = dz / 1e3

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
    Extract 3D fields needed for boundary detection
    """
    # Currents
    JX = ds["Jx"].isel(time=0).values  # nA/m^2
    JY = ds["Jy"].isel(time=0).values
    JZ = ds["Jz"].isel(time=0).values
    # Transpose Nz, Ny, Nx → Nx, Ny, Nz
    JX = np.transpose(JX, (2, 1, 0))
    JY = np.transpose(JY, (2, 1, 0))
    JZ = np.transpose(JZ, (2, 1, 0))

    return JX, JY, JZ


def find_mp_standoff_distance(x_coords, y_coords, z_coords, Jmag, equator='geographic', debug=False):
    """
    Find magnetopause standoff distance by locating maximum current density
    along the positive x-axis at y=0 and specified equator.

    Parameters:
    -----------
    equator : str
        'geographic' for Z=0 R_M, 'magnetic' for Z=484 km (0.1984 R_M)

    Returns: standoff_distance (R_M), standoff_index_x, standoff_index_y, standoff_index_z
    """
    # Validate equator choice
    equator = equator.lower()
    if equator not in ['geographic', 'magnetic']:
        raise ValueError(f"equator must be 'geographic' or 'magnetic', got '{equator}'")

    # Determine z coordinate based on equator choice
    if equator == 'geographic':
        z_target = 0.0
        z_label = "Z=0 R_M (Geographic Equator)"
    else:  # magnetic
        z_target = Z_MAG_RM
        z_label = f"Z={Z_MAG_KM:.0f} km = {Z_MAG_RM:.4f} R_M (Magnetic Equator)"

    # Find indices closest to y=0 and target z
    iy_zero = np.argmin(np.abs(y_coords))
    iz_target = np.argmin(np.abs(z_coords - z_target))

    if debug:
        print(f"\nSearching along: {z_label}")
        print(f"  y=0 closest index: {iy_zero}, y={y_coords[iy_zero]:.4f} R_M")
        print(f"  z={z_target:.4f} closest index: {iz_target}, z={z_coords[iz_target]:.4f} R_M")

    # Extract current density along x-axis at y=0, z=target
    J_along_x = Jmag[:, iy_zero, iz_target]

    # Only consider positive x (dayside) AND outside planet (x > (1.0 R_M - 1 grid cell))
    valid_x_mask = (x_coords > (1.0 - 75/2440))
    x_valid = x_coords[valid_x_mask]
    J_valid = J_along_x[valid_x_mask]

    if len(J_valid) == 0:
        if debug:
            print("WARNING: No valid x-axis points found (x > 1.0 R_M)")
        return np.nan, None, None, None

    # Find maximum current density location
    ix_max_relative = np.argmax(J_valid)
    ix_max_absolute = np.where(valid_x_mask)[0][ix_max_relative]

    standoff_distance = x_valid[ix_max_relative]
    max_current = J_valid[ix_max_relative]

    print(f"\nMagnetopause Standoff Distance ({equator.capitalize()} Equator):")
    print(f"  X = {standoff_distance:.4f} R_M ({standoff_distance * 2440:.2f} km)")
    print(f"  Y = {y_coords[iy_zero]:.4f} R_M")
    print(f"  Z = {z_coords[iz_target]:.4f} R_M")
    print(f"  J_max = {max_current:.4f} nA/m²")
    print(f"  Index: ix={ix_max_absolute}, iy={iy_zero}, iz={iz_target}")
    print(f"  Confirmed outside planet: r = {standoff_distance:.4f} R_M > 1.0 R_M")

    return standoff_distance, ix_max_absolute, iy_zero, iz_target


def compute_masks_3d(f_3d: str, plot_id: str, equator: str = 'geographic', debug: bool = False):
    """
    Compute 3D magnetopause mask from 3D NetCDF file.
    Finds MP standoff distance along x-axis at y=0 and specified equator using max current density.

    Parameters:
    -----------
    f_3d : str
        Path to 3D NetCDF file
    plot_id : str
        Background field for plotting (currently only 'Jmag' supported)
    equator : str
        'geographic' for Z=0 R_M, 'magnetic' for Z=484 km
    debug : bool
        Enable debug output

    Returns: x_coords, y_coords, z_coords, plot_bg, mp_mask_3d, standoff_distance
    """
    ds = xr.open_dataset(f_3d)

    if debug:
        print("DEBUG: compute_masks_3d (MP only)")
        print(f"Equator selection: {equator}")

    # Extract 3D current fields
    JX, JY, JZ = extract_3d_fields(ds)

    # Total current magnitude
    Jmag = np.sqrt(JX ** 2 + JY ** 2 + JZ ** 2)  # nA/m^2

    if debug:
        print(f"Jmag min/med/max={np.nanmin(Jmag):.3f}, {np.nanmedian(Jmag):.3f}, {np.nanmax(Jmag):.3f} nA/m²")

    dx_km, dy_km, dz_km, x_coords, y_coords, z_coords = fetch_coords(ds, debug=debug)

    # Find MP standoff distance along specified equator
    standoff_distance, ix_standoff, iy_standoff, iz_standoff = find_mp_standoff_distance(
        x_coords, y_coords, z_coords, Jmag, equator=equator, debug=debug
    )

    # Create empty mask
    magnetopause_mask = np.zeros((len(x_coords), len(y_coords), len(z_coords)), dtype=bool)

    # Only mark the standoff distance location as True
    if not np.isnan(standoff_distance) and ix_standoff is not None:
        magnetopause_mask[ix_standoff, iy_standoff, iz_standoff] = True

        if debug:
            print(f"\nMP mask: Single point marked at standoff distance")
            print(f"  Location: ix={ix_standoff}, iy={iy_standoff}, iz={iz_standoff}")
            print(
                f"  Coordinates: x={x_coords[ix_standoff]:.4f}, y={y_coords[iy_standoff]:.4f}, z={z_coords[iz_standoff]:.4f} R_M")
    else:
        if debug:
            print("\nWARNING: No valid standoff distance found, mask is all False")

    if debug:
        print(f"3D MP mask: {np.sum(magnetopause_mask)} points")
        print(f"MP volume fraction: {np.mean(magnetopause_mask):.6e}\n")

    # Background field for plotting
    bg_map = {"Jmag": Jmag}
    if plot_id not in bg_map:
        plot_id = "Jmag"  # Default to current magnitude
    plot_bg = bg_map[plot_id]

    ds.close()

    # Convert mask to xarray DataArray
    mp_mask_da = xr.DataArray(
        magnetopause_mask.astype(np.uint8),
        coords={'Nx': x_coords, 'Ny': y_coords, 'Nz': z_coords},
        dims=['Nx', 'Ny', 'Nz']
    )

    return x_coords, y_coords, z_coords, plot_bg, mp_mask_da, standoff_distance

