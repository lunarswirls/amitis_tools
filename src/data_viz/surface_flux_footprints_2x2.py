#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# -------------------------------
# Configuration
# -------------------------------
cases = ["RPS", "CPS", "RPN", "CPN"]
output_folder = f"/Users/danywaller/Projects/mercury/extreme/surface_flux/"
os.makedirs(output_folder, exist_ok=True)

debug = False

R_M = 2440.0        # Mercury radius [km]
LAT_BINS = 180      # Surface latitude bins
LON_BINS = 360      # Surface longitude bins


def compute_radial_flux(ds, x, y, z):
    """
    Compute radial particle flux F_r = n (v · r_hat)

    """

    # densities converted from cm^-3 to km^-3
    den01 = ds["den01"].isel(time=0).values * 1e15
    den02 = ds["den02"].isel(time=0).values * 1e15
    den03 = ds["den03"].isel(time=0).values * 1e15
    den04 = ds["den04"].isel(time=0).values * 1e15

    # sum all densities to get total density
    den_tot = (den01 + den02 + den03 + den04)

    # density-weighted bulk velocity (km/s)
    vx_bulk = np.zeros_like(den_tot)
    vy_bulk = np.zeros_like(den_tot)
    vz_bulk = np.zeros_like(den_tot)

    # mask where total density is > 0
    mask = den_tot > 0

    vx_bulk[mask] = (den01[mask] * ds["vx01"].isel(time=0).values[mask] +
                     den02[mask] * ds["vx02"].isel(time=0).values[mask] +
                     den03[mask] * ds["vx03"].isel(time=0).values[mask] +
                     den04[mask] * ds["vx04"].isel(time=0).values[mask]) / den_tot[mask]

    vy_bulk[mask] = (den01[mask] * ds["vy01"].isel(time=0).values[mask] +
                     den02[mask] * ds["vy02"].isel(time=0).values[mask] +
                     den03[mask] * ds["vy03"].isel(time=0).values[mask] +
                     den04[mask] * ds["vy04"].isel(time=0).values[mask]) / den_tot[mask]

    vz_bulk[mask] = (den01[mask] * ds["vz01"].isel(time=0).values[mask] +
                     den02[mask] * ds["vz02"].isel(time=0).values[mask] +
                     den03[mask] * ds["vz03"].isel(time=0).values[mask] +
                     den04[mask] * ds["vz04"].isel(time=0).values[mask]) / den_tot[mask]

    # build position grids (Nz, Ny, Nx)
    Zg, Yg, Xg = np.meshgrid(z, y, x, indexing="ij")

    r_mag = np.sqrt(Xg ** 2 + Yg ** 2 + Zg ** 2)
    mask_r = r_mag > 0

    # Inward radial unit vector
    nx = np.zeros_like(r_mag)
    ny = np.zeros_like(r_mag)
    nz = np.zeros_like(r_mag)

    nx[mask_r] = -Xg[mask_r] / r_mag[mask_r]
    ny[mask_r] = -Yg[mask_r] / r_mag[mask_r]
    nz[mask_r] = -Zg[mask_r] / r_mag[mask_r]

    # radial velocity [km/s]
    v_dot_r = vx_bulk * nx + vy_bulk * ny + vz_bulk * nz

    # weighted flux [km^-2 s^-1]
    flux = den_tot * v_dot_r

    # convert from km^-2 s^-1 to cm^-2 s^-1
    flux *= 1e-10

    return flux


def lon_diff(a, b):
    """
    Minimal angular difference in degrees.
    """
    return np.abs(((a - b + 180) % 360) - 180)


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


# -------------------------------
# Prepare figure
# -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 8), subplot_kw={"projection": "hammer"})

for case in cases:

    input_folder1  = f"/Users/danywaller/Projects/mercury/extreme/{case}_Base/object/"

    input_folder2 = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology/{case}_Base/"
    # csv_file = os.path.join(input_folder2, f"{case}_last_10_footprints_median_class.csv")  # CSV with footprints
    csv_file = os.path.join(input_folder2, f"{case}_115000_footprints_class.csv")  # CSV with footprints

    # -------------------------------
    # Load footprint CSV
    # -------------------------------
    if os.path.exists(csv_file):
        df_footprints = pd.read_csv(csv_file)
        # print(f"Loaded {len(df_footprints)} footprints for {case}")
    else:
        print(f"No footprint CSV found for {case}, skipping footprints")
        df_footprints = pd.DataFrame(columns=["latitude_deg", "longitude_deg", "classification"])

    # -------------------------------
    # Load grid (assume first file is representative)
    # -------------------------------
    first_file = sorted([f for f in os.listdir(input_folder1) if f.endswith("_xz_comp.nc")])[0]
    ds0 = xr.open_dataset(os.path.join(input_folder1, first_file))

    x = ds0["Nx"].values
    y = ds0["Ny"].values
    z = ds0["Nz"].values

    # -------------------------------
    # Time-average total radial flux
    # -------------------------------
    flux_sum = None
    count = 0

    # Consider last N steps (adjust as needed)
    sim_steps = range(115000, 115000 + 1, 1000)

    for step in sim_steps:
        nc_file = os.path.join(input_folder1, f"Amitis_{case}_Base_{step:06d}_xz_comp.nc")
        ds = xr.open_dataset(nc_file)

        flux = compute_radial_flux(ds, x, y, z)

        if flux_sum is None:
            flux_sum = np.zeros_like(flux, dtype=np.float64)
        flux_sum += flux
        count += 1

    flux_avg = flux_sum / count
    print(f"Computed time-averaged radial flux for {case}")

    # -------------------------------
    # Interpolate radial flux onto Mercury surface
    # -------------------------------
    lat = np.linspace(-90, 90, LAT_BINS)
    lon = np.linspace(-180, 180, LON_BINS)

    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    Xs = R_M * np.cos(lat_r[:, None]) * np.cos(lon_r[None, :])
    Ys = R_M * np.cos(lat_r[:, None]) * np.sin(lon_r[None, :])
    Zs = R_M * np.sin(lat_r[:, None]) * np.ones_like(lon_r[None, :])

    points_surface = np.stack((Zs, Ys, Xs), axis=-1).reshape(-1, 3)
    interp = RegularGridInterpolator((z, y, x), flux_avg, bounds_error=False, fill_value=np.nan)
    flux_surface = interp(points_surface).reshape(LAT_BINS, LON_BINS)
    flux_surface = flux_surface[::-1, :]  # flip latitude for plotting

    # Mask non-positive values
    flux_surface_masked = np.where(flux_surface > 0, flux_surface, np.nan)

    # Log10
    log_flux_surface = np.log10(flux_surface_masked)

    # -------------------------------
    # Plot
    # -------------------------------
    if case == "RPN": row, col = 0, 0
    elif case == "CPN": row, col = 1, 0
    elif case == "RPS": row, col = 0, 1
    elif case == "CPS": row, col = 1, 1

    ax = axs[row, col]

    quick_cmin = 6
    quick_cmax = 8

    # Plot flux
    lon_grid, lat_grid = np.meshgrid(lon_r, lat_r)  # radians
    # shift lon to [-pi, pi]
    lon_grid = np.where(lon_grid > np.pi, lon_grid - 2*np.pi, lon_grid)

    # Surface flux
    sc = ax.pcolormesh(lon_grid, lat_grid, log_flux_surface, cmap="viridis", shading="auto", vmin=quick_cmin, vmax=quick_cmax)
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.5)
    cbar.set_label(r"$\log_{10}$(F [cm$^{-2}$ s$^{-1}$])")

    # Overlay footprints
    if 0:
        # for topo, color in [("closed", "blue"), ("open", "white")]:
        for topo, color in [("open", "white")]:
            subset = df_footprints[df_footprints['median_classification'] == topo]
            if not subset.empty:
                ax.scatter(subset['longitude_deg'], subset['latitude_deg'], s=1, color=color, facecolor=None, alpha=0.2)

    # Open–Closed Boundary (OCB)
    lon_bins = np.linspace(-180, 180, 180)
    lon_n, lat_n = compute_ocb_transition(df_footprints, lon_bins, "north")
    lon_s, lat_s = compute_ocb_transition(df_footprints, lon_bins, "south")

    # Convert to radians for Mollweide
    lon_n_rad = np.deg2rad(lon_n)
    lat_n_rad = np.deg2rad(lat_n)
    lon_s_rad = np.deg2rad(lon_s)
    lat_s_rad = np.deg2rad(lat_s)

    # Mollweide longitude in matplotlib goes from -pi to pi (radians)
    # Latitude stays as is
    ax.plot(lon_n_rad, lat_n_rad, color="magenta", lw=2, label="OCB North")
    ax.plot(lon_s_rad, lat_s_rad, color="magenta", lw=2, ls="--", label="OCB South")

    # Longitude ticks (-170 to 170 every n °)
    lon_ticks_deg = np.arange(-120, 121, 60)
    lon_ticks_rad = np.deg2rad(lon_ticks_deg)

    # Latitude ticks (-90 to 90 every n °)
    lat_ticks_deg = np.arange(-60, 61, 30)
    lat_ticks_rad = np.deg2rad(lat_ticks_deg)

    # Apply to the current axis
    ax.set_xticks(lon_ticks_rad)
    ax.set_yticks(lat_ticks_rad)

    # Label ticks in degrees
    ax.set_xticklabels([f"{int(l)}°" for l in lon_ticks_deg])
    ax.set_yticklabels([f"{int(l)}°" for l in lat_ticks_deg])

    ax.set_title(case)
    ax.grid(True, alpha=0.3, color="grey")

# Save figure
plt.tight_layout()
outfile_png = os.path.join(output_folder, "all_cases_surface_flux_with_footprints_115000.png")
plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
print("Saved figure:", outfile_png)