#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from numba import njit

# --------------------------
# Numba functions
# --------------------------
@njit
def trilinear_interp(x_grid, y_grid, z_grid, B, xi, yi, zi):
    i = np.searchsorted(x_grid, xi) - 1
    j = np.searchsorted(y_grid, yi) - 1
    k = np.searchsorted(z_grid, zi) - 1
    i = max(0, min(i, len(x_grid)-2))
    j = max(0, min(j, len(y_grid)-2))
    k = max(0, min(k, len(z_grid)-2))
    xd = (xi - x_grid[i]) / (x_grid[i+1]-x_grid[i])
    yd = (yi - y_grid[j]) / (y_grid[j+1]-y_grid[j])
    zd = (zi - z_grid[k]) / (z_grid[k+1]-z_grid[k])
    c000 = B[i,j,k]
    c100 = B[i+1,j,k]
    c010 = B[i,j+1,k]
    c001 = B[i,j,k+1]
    c101 = B[i+1,j,k+1]
    c011 = B[i,j+1,k+1]
    c110 = B[i+1,j+1,k]
    c111 = B[i+1,j+1,k+1]
    c00 = c000*(1-xd)+c100*xd
    c01 = c001*(1-xd)+c101*xd
    c10 = c010*(1-xd)+c110*xd
    c11 = c011*(1-xd)+c111*xd
    c0 = c00*(1-yd)+c10*yd
    c1 = c01*(1-yd)+c11*yd
    return c0*(1-zd)+c1*zd

@njit
def get_B(r, Bx, By, Bz, x_grid, y_grid, z_grid):
    bx = trilinear_interp(x_grid, y_grid, z_grid, Bx, r[0], r[1], r[2])
    by = trilinear_interp(x_grid, y_grid, z_grid, By, r[0], r[1], r[2])
    bz = trilinear_interp(x_grid, y_grid, z_grid, Bz, r[0], r[1], r[2])
    B = np.array([bx, by, bz])
    norm = np.linalg.norm(B)
    if norm == 0.0:
        return np.zeros(3)
    return B / norm

@njit
def cartesian_to_latlon(r):
    rmag = np.linalg.norm(r)
    lat = np.degrees(np.arcsin(r[2]/rmag))
    lon = np.degrees(np.arctan2(r[1], r[0]))
    return lat, lon

@njit
def rk45_step(f, r, h, Bx, By, Bz, x_grid, y_grid, z_grid):
    k1 = f(r, Bx, By, Bz, x_grid, y_grid, z_grid)
    k2 = f(r + h*k1*0.25, Bx, By, Bz, x_grid, y_grid, z_grid)
    k3 = f(r + h*(3*k1+9*k2)/32, Bx, By, Bz, x_grid, y_grid, z_grid)
    k4 = f(r + h*(1932*k1 - 7200*k2 + 7296*k3)/2197, Bx, By, Bz, x_grid, y_grid, z_grid)
    k5 = f(r + h*(439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104), Bx, By, Bz, x_grid, y_grid, z_grid)
    k6 = f(r + h*(-8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40), Bx, By, Bz, x_grid, y_grid, z_grid)
    r_next = r + h*(16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55)
    return r_next

@njit
def trace_field_line_rk(seed, Bx, By, Bz, x_grid, y_grid, z_grid, RM, max_steps=5000, h=50.0, surface_tol=-1.0):
    traj = np.empty((max_steps, 3), dtype=np.float64)
    traj[0] = seed
    r = seed.copy()
    exit_y_boundary = False
    for i in range(1, max_steps):
        B = get_B(r, Bx, By, Bz, x_grid, y_grid, z_grid)
        if np.all(B == 0.0):
            return traj[:i], exit_y_boundary
        r_next = rk45_step(get_B, r, h, Bx, By, Bz, x_grid, y_grid, z_grid)
        traj[i] = r_next
        r = r_next
        if np.linalg.norm(r) <= RM + surface_tol:
            return traj[:i+1], exit_y_boundary
        if (r[0]<x_grid[0] or r[0]>x_grid[-1] or
            r[2]<z_grid[0] or r[2]>z_grid[-1]):
            return traj[:i+1], exit_y_boundary
        if r[1]<y_grid[0] or r[1]>y_grid[-1]:
            exit_y_boundary = True
            return traj[:i+1], exit_y_boundary
    return traj, exit_y_boundary

@njit
def classify(traj_fwd, traj_bwd, RM, exit_fwd_y=False, exit_bwd_y=False):
    # check if last point in trajectory is equal to or less than Mercury radius
    hit_fwd = np.linalg.norm(traj_fwd[-1]) <= RM
    hit_bwd = np.linalg.norm(traj_bwd[-1]) <= RM
    if exit_fwd_y or exit_bwd_y:
        # line ran into domain boundary - terminate as unknown
        return "TBD"
    if hit_fwd and hit_bwd:
        # line originated from and returned to planet surface - closed
        return "closed"
    elif hit_fwd or hit_bwd:
        # line connected to planet surface at only ONE end - open
        return "open"
    else:
        return "TBD"


def compute_open_fraction(
        df,
        lon0,
        dlon=5.0,
        lat_bins=np.linspace(0, 90, 91),
        hemisphere="north"
):
    """
    Compute the fraction of open magnetic field lines as a function of latitude
    for a given longitude slice.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the columns:
        'latitude_deg', 'longitude_deg', 'median_classification'.
        'median_classification' should be one of 'open', 'closed', or 'solar_wind'.
    lon0 : float
        Target longitude (degrees) for the slice.
    dlon : float, optional
        Width of longitude window around lon0 to select points (default 5.0 deg).
    lat_bins : array_like, optional
        Array of latitude bin edges in degrees.
    hemisphere : str, optional
        'north' or 'south' hemisphere to consider (default 'north').

    Returns
    -------
    lat_centers : ndarray
        Centers of latitude bins (degrees).
    frac_open : ndarray
        Fraction of open field lines in each latitude bin. NaN if insufficient points.

    Notes
    -----
    - For the southern hemisphere, latitudes are mirrored to positive values
      for consistent computation from pole toward equator.
    - Bins with fewer than 3 points are assigned NaN.
    """

    # Select points near target longitude, accounting for periodicity
    sub = df[np.abs(((df["longitude_deg"] - lon0 + 180) % 360) - 180) < dlon]

    # Select hemisphere and mirror south latitudes to positive
    if hemisphere == "north":
        sub = sub[sub["latitude_deg"] > 0]
        lat_vals = sub["latitude_deg"].values
    else:
        sub = sub[sub["latitude_deg"] < 0]
        lat_vals = -sub["latitude_deg"].values  # mirror south for consistent handling

    # Skip if too few points to compute a reliable fraction
    if len(sub) < 10:
        return None, None

    # Boolean mask: True where the field line is classified as 'open'
    open_mask = (sub["classification"] == "open").values

    frac_open = []
    # Compute bin centers
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])

    # Loop over latitude bins
    for lo, hi in zip(lat_bins[:-1], lat_bins[1:]):
        mask = (lat_vals >= lo) & (lat_vals < hi)
        if np.sum(mask) < 3:
            # Insufficient points → NaN
            frac_open.append(np.nan)
        else:
            # Fraction of points in bin classified as 'open'
            frac_open.append(np.mean(open_mask[mask]))

    return lat_centers, np.array(frac_open)


def find_transition_lat(lat, frac_open, threshold):
    """
    Find the latitude where topology transitions from open to closed
    when scanning equatorward.
    """
    valid = np.isfinite(frac_open)
    lat = lat[valid]
    frac_open = frac_open[valid]

    if len(lat) < 5:
        return None

    # sort from pole → equator
    order = np.argsort(lat)[::-1]
    lat = lat[order]
    frac_open = frac_open[order]

    for i in range(1, len(lat)):
        if frac_open[i-1] >= threshold and frac_open[i] < threshold:
            # linear interpolation
            w = ((threshold - frac_open[i-1]) /
                 (frac_open[i] - frac_open[i-1]))
            return lat[i-1] + w * (lat[i] - lat[i-1])

    return None


def compute_ocb_transition(
    df,
    lon_bins,
    hemisphere="north",
    threshold=0.5,
    max_jump_deg=10.0
):
    """
    Compute the open-closed boundary (OCB) transition latitude as a function of longitude.

    Scans equatorward from the pole to find where the fraction of open field lines
    drops below the threshold. If a computed boundary point jumps more than `max_jump_deg`
    toward the pole relative to the previous point, the previous latitude is reused.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'latitude_deg', 'longitude_deg', 'median_classification'.
    lon_bins : array-like
        Longitudes (deg) at which to compute the boundary.
    hemisphere : {'north', 'south'}
        Hemisphere to compute.
    threshold : float
        Fraction of open field lines defining the transition.
    max_jump_deg : float
        Maximum allowed poleward jump in latitude between neighboring points along the boundary.

    Returns
    -------
    lons : ndarray
        Longitudes of boundary points.
    lats : ndarray
        Latitudes of boundary points.
    """

    lons_all = []
    lats_all = []

    prev_lat = None

    for lon0 in lon_bins:
        lat_c, frac_open = compute_open_fraction(df, lon0, hemisphere=hemisphere)
        if lat_c is None:
            continue

        lat_b = find_transition_lat(lat_c, frac_open, threshold)
        if lat_b is None:
            continue

        if hemisphere == "south":
            lat_b = -lat_b

        # If previous latitude exists, check poleward jump
        if prev_lat is not None:
            jump = lat_b - prev_lat if hemisphere == "north" else prev_lat - lat_b
            if jump > max_jump_deg:
                # Reject jump: use previous value
                lat_b = prev_lat

        lons_all.append(lon0)
        lats_all.append(lat_b)
        prev_lat = lat_b

    return np.array(lons_all), np.array(lats_all)