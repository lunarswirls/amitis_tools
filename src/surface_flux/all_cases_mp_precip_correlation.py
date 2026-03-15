#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation between dayside magnetopause stand-off Δr and precipitation flux.

Inputs per case
---------------
1) Precipitation snapshots: precipitation_timeseries/*.npz
2) Magnetopause mask timeseries: <case>_mp_mask_timeseries.nc

Outputs
-------
- 2D histogram per timestep:
    x-axis: Δr [R_M]
    y-axis: log10(total inward number flux) [#/m^2/s]
- Median flux vs Δr trend overlaid
"""

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import src.surface_flux.flux_utils as flux_utils
import matplotlib.colors as mcolors

# =============================================================================
# USER SETTINGS
# =============================================================================
cases = ["RPS_HNHV", "CPS_HNHV", "RPN_HNHV", "CPN_HNHV"]
selected_times = [230, 270, 330, 700]  # s

base_precip_dir = "/Volumes/data_backup/mercury/extreme/High_HNHV"
base_mp_dir = "/Users/danywaller/Projects/mercury/extreme/magnetopause_3D_timeseries"

out_dir = "/Users/danywaller/Projects/mercury/extreme/gridded_dr_flux"
os.makedirs(out_dir, exist_ok=True)

sim_ppc = np.array([24, 24, 11, 11], dtype=float)
sim_den = np.array([38.0e6, 76.0e6, 1.0e6, 2.0e6], dtype=float)
species_mass = np.array([1.0, 1.0, 4.0, 4.0], dtype=float)
AMU = 1.66053906660e-27
m_kg_by_sid = species_mass * AMU

sim_dx = sim_dy = sim_dz = 75e3
R_M = 2440e3
DELTA_R_M = 0.5 * sim_dx

dlat, dlon = 2.0, 2.0
lat_bin_edges = np.arange(-90, 90 + dlat, dlat)
lon_bin_edges = np.arange(-180, 180 + dlon, dlon)
lat_centers = 0.5 * (lat_bin_edges[:-1] + lat_bin_edges[1:])
lon_centers = 0.5 * (lon_bin_edges[:-1] + lon_bin_edges[1:])

DAY_LON_MIN, DAY_LON_MAX = -75, 75  # dayside longitude for correlation
DAY_LAT_MIN, DAY_LAT_MAX = -65, 65

V_cell = sim_dx * sim_dy * sim_dz
W_by_sid = flux_utils.macro_weights(sim_den, sim_ppc, V_cell)

# Ray-trace settings
N_THETA = 90
N_PHI = 360
RAY_RMIN_RM = 1.0
RAY_RMAX_RM = 2.0
RAY_DR_OVERRIDE_RM = None
EPS_ANG = 1e-6

FILL_NAN_WITH_NEAREST = True
SMOOTH_SIGMA_THETA = 1.0
SMOOTH_SIGMA_PHI = 1.0

DR_BINS = 60

# =============================================================================
# Helpers
# =============================================================================
def regrid_delta_r_raytrace(delta_r, theta, phi):
    """
    Regrid the ray-traced Δr (theta/phi) onto the precipitation lat/lon grid.
    """
    # Convert theta/phi to lat/lon
    lat = 90.0 - np.degrees(theta)
    lon = np.degrees(phi)

    # Interpolator expects ascending axes
    interp = RegularGridInterpolator((lat, lon), delta_r, bounds_error=False, fill_value=np.nan)

    # Build grid points in precipitation coordinates
    LATp, LONp = np.meshgrid(lat_centers, lon_centers, indexing="ij")
    pts = np.column_stack([LATp.ravel(), LONp.ravel()])
    return interp(pts).reshape(LATp.shape)

def _read_npz_time(npz_path):
    """Extract simulation time from a precipitation npz snapshot."""
    with np.load(npz_path) as p:
        return float(np.asarray(p["time"]).item())

def _nearest_index(values, target):
    return int(np.argmin(np.abs(values - target)))

def _build_theta_phi_grid():
    theta = np.linspace(EPS_ANG, np.pi - EPS_ANG, N_THETA)
    phi = np.linspace(-np.pi/2 + EPS_ANG, np.pi/2 - EPS_ANG, N_PHI)
    lat_deg = 90.0 - np.degrees(theta)
    lon_deg = np.degrees(phi)
    return theta, phi, lat_deg, lon_deg

def _build_ray_cache(x, y, z, theta, phi):
    """Precompute nearest-neighbor voxel indices for all sampled rays."""
    if RAY_DR_OVERRIDE_RM is None:
        dx = float(np.min(np.diff(x)))
        dy = float(np.min(np.diff(y)))
        dz = float(np.min(np.diff(z)))
        dr = min(dx, dy, dz)
    else:
        dr = float(RAY_DR_OVERRIDE_RM)

    r_samp = np.arange(RAY_RMIN_RM, RAY_RMAX_RM + 0.5 * dr, dr)

    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    sx = np.sin(TH) * np.cos(PH)
    sy = np.sin(TH) * np.sin(PH)
    sz = np.cos(TH)

    X = r_samp[:, None, None] * sx[None, :, :]
    Y = r_samp[:, None, None] * sy[None, :, :]
    Z = r_samp[:, None, None] * sz[None, :, :]

    ix = np.searchsorted(x, X)
    iy = np.searchsorted(y, Y)
    iz = np.searchsorted(z, Z)

    def nearest_idx(grid, idx, coord):
        idx0 = np.clip(idx, 1, len(grid) - 1)
        left = grid[idx0 - 1]
        right = grid[idx0]
        choose_left = np.abs(coord - left) <= np.abs(coord - right)
        return np.where(choose_left, idx0 - 1, idx0)

    ixn = nearest_idx(x, ix, X).astype(np.int32)
    iyn = nearest_idx(y, iy, Y).astype(np.int32)
    izn = nearest_idx(z, iz, Z).astype(np.int32)

    valid = (
        (ixn >= 0) & (ixn < len(x)) &
        (iyn >= 0) & (iyn < len(y)) &
        (izn >= 0) & (izn < len(z))
    )

    return {"r_samp": r_samp, "ix": ixn, "iy": iyn, "iz": izn, "valid": valid, "dr": dr}

def _fill_nans_nearest(arr2d):
    out = np.array(arr2d, dtype=float, copy=True)
    nan_mask = ~np.isfinite(out)
    if not np.any(nan_mask) or np.all(nan_mask):
        return out
    nearest_idx_arr = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
    nearest_vals = out[tuple(nearest_idx_arr[d] for d in range(out.ndim))]
    out[nan_mask] = nearest_vals[nan_mask]
    return out

def _nan_gaussian_smooth(arr2d, sigma):
    out = np.array(arr2d, dtype=float, copy=True)
    valid = np.isfinite(out)
    if not np.any(valid):
        return out
    vals = np.where(valid, out, 0.0)
    w = valid.astype(float)
    vals_blur = gaussian_filter(vals, sigma=sigma, mode="nearest")
    w_blur = gaussian_filter(w, sigma=sigma, mode="nearest")
    smoothed = np.full_like(out, np.nan)
    good = w_blur > 1e-8
    smoothed[good] = vals_blur[good] / w_blur[good]
    return smoothed

def _compute_dayside_delta_r_map(mask_xyz, ray_cache):
    """
    Compute dayside stand-off map Δr = r_mp - 1 [R_M] along rays.
    **Use the innermost magnetopause voxel** along each ray as the MP position.
    """
    r_samp = ray_cache["r_samp"]
    ix, iy, iz = ray_cache["ix"], ray_cache["iy"], ray_cache["iz"]
    valid = ray_cache["valid"]
    out_shape = valid.shape[1:]

    if not np.any(mask_xyz):
        return np.full(out_shape, np.nan)

    # Build a 3D "inside MP" array along rays
    inside = np.zeros_like(valid, dtype=bool)
    v = valid
    inside[v] = mask_xyz[ix[v], iy[v], iz[v]]

    # Find rays where any voxel is inside MP
    any_true = np.any(inside, axis=0)
    if not np.any(any_true):
        return np.full(out_shape, np.nan)

    # Find the first (innermost) voxel along each ray
    j_first = np.argmax(inside, axis=0)  # index of first True along r
    r_map = np.full(out_shape, np.nan)
    r_map[any_true] = r_samp[j_first[any_true]]

    delta_r = r_map  # - 1.0

    # Fill NaNs and smooth
    if FILL_NAN_WITH_NEAREST:
        delta_r = _fill_nans_nearest(delta_r)
    if SMOOTH_SIGMA_THETA > 0.0 or SMOOTH_SIGMA_PHI > 0.0:
        delta_r = _nan_gaussian_smooth(delta_r, sigma=(SMOOTH_SIGMA_THETA, SMOOTH_SIGMA_PHI))

    return delta_r

def _regrid_to_precip_grid(delta_r, lat_mp, lon_mp):
    """Interpolate Δr onto precipitation grid for dayside longitudes."""
    from scipy.interpolate import RegularGridInterpolator
    LATp, LONp = np.meshgrid(lat_centers, lon_centers, indexing="ij")
    interp = RegularGridInterpolator((lat_mp, lon_mp), delta_r, bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([LATp.ravel(), LONp.ravel()])
    return interp(pts).reshape(LATp.shape)[:, (lon_centers >= DAY_LON_MIN) & (lon_centers <= DAY_LON_MAX)]

# =============================================================================
# PRECOMPUTE
# =============================================================================
theta, phi, lat_mp_deg, lon_mp_deg = _build_theta_phi_grid()

all_delta_r = {}
all_flux = {}

for case in cases:
    print(f"\nProcessing case: {case}")

    precip_files = sorted(glob.glob(f"{base_precip_dir}/{case}/precipitation_timeseries/*.npz"))
    if len(precip_files) == 0:
        continue

    precip_times = np.array([_read_npz_time(f) for f in precip_files])

    mp_nc = os.path.join(base_mp_dir, case, f"{case}_mp_mask_timeseries.nc")
    with xr.open_dataset(mp_nc) as ds:
        mp_mask_4d = ds["mp_mask"].values
        mp_times = np.asarray(ds["time"].values)
        x, y, z = np.asarray(ds["x"].values), np.asarray(ds["y"].values), np.asarray(ds["z"].values)

    ray_cache = _build_ray_cache(x, y, z, theta, phi)

    all_delta_r[case] = {}
    all_flux[case] = {}

    for t_sel in selected_times:
        print(f"  t={t_sel:.1f}s")

        idx_precip = _nearest_index(precip_times, t_sel)
        npz_path = precip_files[idx_precip]
        t_precip = precip_times[idx_precip]

        # match closest MP mask
        mp_idx = _nearest_index(mp_times, t_precip)

        delta_r_ray = _compute_dayside_delta_r_map(mp_mask_4d[mp_idx].astype(bool), ray_cache)
        delta_r_interp = regrid_delta_r_raytrace(delta_r_ray, theta, phi)

        # precipitation flux
        _, flux_all, *_ = flux_utils.flux_maps_snapshot_shell(
            npz_path, R=R_M, delta_r_m=DELTA_R_M,
            lat_bin_edges=lat_bin_edges, lon_bin_edges=lon_bin_edges,
            W_by_sid=W_by_sid, m_kg_by_sid=m_kg_by_sid
        )

        dayside_mask = (lon_centers >= DAY_LON_MIN) & (lon_centers <= DAY_LON_MAX)
        all_delta_r[case][t_sel] = delta_r_interp[:, dayside_mask]
        all_flux[case][t_sel] = flux_all[:, dayside_mask]

# =============================================================================
# PLOT CORRELATION HEATMAP (4x4 style: times × cases)
# =============================================================================
if 0:
    n_rows, n_cols = len(selected_times), len(cases)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False, sharex=True, sharey=True, constrained_layout=True
    )

    for col, case in enumerate(cases):
        for row, t_sel in enumerate(selected_times):
            ax = axes[row, col]
            dr = all_delta_r[case][t_sel].flatten()
            flux_log = np.log10(all_flux[case][t_sel].flatten())

            valid = np.isfinite(dr) & np.isfinite(flux_log) & (flux_log > 0)
            dr, flux_log = dr[valid], flux_log[valid]

            # 2D histogram
            h = ax.hist2d(
                dr, flux_log,
                bins=[DR_BINS, 80],
                cmap="cividis",
                range=[[1.0, 1.9], [8, 14]],
                vmin=0, vmax=5   # <-- fixed colormap limits
            )

            # median trend
            bins = np.linspace(1.0, 1.9, DR_BINS)
            digitized = np.digitize(dr, bins)
            med = [np.nanmedian(flux_log[digitized == i]) for i in range(1, len(bins))]
            centers = 0.5 * (bins[:-1] + bins[1:])
            # ax.plot(centers, med, color="red", lw=2, label="median")

            ax.set_title(f"{case} t={t_sel:.1f} s", fontsize=10)
            ax.set_xlim(1.0, 1.9)
            ax.set_ylim(8, 14)
            if row == n_rows - 1: ax.set_xlabel("Δr [R_M]")
            if col == 0: ax.set_ylabel("log10 flux")
            ax.grid(alpha=0.3)

    fig.colorbar(
        h[3], ax=axes, orientation="vertical",
        fraction=0.02, pad=0.01, label="counts",
        ticks=np.linspace(0, 5, 6)  # fixed tick labels
    )
    plt.show()

# =============================================================================
# PLOT DAYSIDE Δr AND PRECIPITATION ON HAMMER PROJECTION (per case)
# =============================================================================
if 0:
    # raw hammer plots
    for case in cases:
        fig, axes = plt.subplots(
            nrows=4, ncols=2, figsize=(12, 12), constrained_layout=True,
            subplot_kw={'projection': 'hammer'}
        )

        for row, t_sel in enumerate(selected_times):
            ax_dr = axes[row, 0]
            ax_flux = axes[row, 1]

            dr = all_delta_r[case][t_sel]
            flux = all_flux[case][t_sel]

            # lat/lon in radians for plotting
            lat_rad = np.radians(lat_centers)
            lon_rad = np.radians(lon_centers[(lon_centers >= DAY_LON_MIN) & (lon_centers <= DAY_LON_MAX)])

            # build edge arrays for pcolormesh
            lat_edges = np.radians(np.linspace(-90, 90, dr.shape[0] + 1))
            lon_edges = np.radians(np.linspace(DAY_LON_MIN, DAY_LON_MAX, dr.shape[1] + 1))

            # Plot Δr
            pcm = ax_dr.pcolormesh(
                lon_edges, lat_edges, dr,
                shading='auto', cmap='viridis', vmin=1.0, vmax=1.75
            )
            ax_dr.set_title(f"{case.split("_")[0]} Δr t={t_sel:.1f}s")
            ax_dr.set_xticks(np.radians(np.linspace(-120, 120, 7)))  # 7 ticks from -120° to +120°
            ax_dr.set_yticks(np.radians(np.linspace(-60, 60, 7)))  # 7 ticks from -60° to +60°
            ax_dr.grid(alpha=0.3)

            # Plot precipitation flux
            pcm_f = ax_flux.pcolormesh(
                lon_edges, lat_edges, flux,
                shading='auto', cmap='plasma', norm=mcolors.LogNorm(vmin=1e8, vmax=1e14)
            )
            ax_flux.set_title(f"{case.split("_")[0]} flux t={t_sel:.1f}s")
            ax_flux.set_xticks(np.radians(np.linspace(-120, 120, 7)))  # 7 ticks from -120° to +120°
            ax_flux.set_yticks(np.radians(np.linspace(-60, 60, 7)))  # 7 ticks from -60° to +60°
            ax_flux.grid(alpha=0.3)

        fig.colorbar(pcm, ax=axes[:, 0], orientation='horizontal', fraction=0.05, pad=0.02, label='Δr [R_M]')
        fig.colorbar(pcm_f, ax=axes[:, 1], orientation='horizontal', fraction=0.05, pad=0.02, label='Flux [#/m²/s]')
        out_png = os.path.join(out_dir, f"{case}_delta_r_flux_gridded_raw.png")
        fig.savefig(out_png, dpi=250)
        plt.show()


    # log hammer plots
    for case in cases:
        fig, axes = plt.subplots(
            nrows=4, ncols=2, figsize=(12, 12), constrained_layout=True,
            subplot_kw={'projection': 'hammer'}
        )

        for row, t_sel in enumerate(selected_times):
            ax_dr = axes[row, 0]
            ax_flux = axes[row, 1]

            dr = np.log10(all_delta_r[case][t_sel])
            flux = np.log10(all_flux[case][t_sel])

            # lat/lon in radians for plotting
            lat_rad = np.radians(lat_centers)
            lon_rad = np.radians(lon_centers[(lon_centers >= DAY_LON_MIN) & (lon_centers <= DAY_LON_MAX)])

            # build edge arrays for pcolormesh
            lat_edges = np.radians(np.linspace(-90, 90, dr.shape[0] + 1))
            lon_edges = np.radians(np.linspace(DAY_LON_MIN, DAY_LON_MAX, dr.shape[1] + 1))

            # Plot Δr
            pcm = ax_dr.pcolormesh(
                lon_edges, lat_edges, dr,
                shading='auto', cmap='viridis', vmin=np.log10(1.0), vmax=np.log10(1.75)
            )
            ax_dr.set_title(f"{case.split("_")[0]} log10(Δr) t={t_sel:.1f}s")
            ax_dr.set_xticks(np.radians(np.linspace(-120, 120, 7)))
            ax_dr.set_yticks(np.radians(np.linspace(-60, 60, 7)))
            ax_dr.grid(alpha=0.3)

            # Plot precipitation flux
            pcm_f = ax_flux.pcolormesh(
                lon_edges, lat_edges, flux,
                shading='auto', cmap='plasma', vmin=np.log10(1e8), vmax=np.log10(1e14)
            )
            ax_flux.set_title(f"{case.split("_")[0]} log10(flux) t={t_sel:.1f}s")
            ax_flux.set_xticks(np.radians(np.linspace(-120, 120, 7)))
            ax_flux.set_yticks(np.radians(np.linspace(-60, 60, 7)))
            ax_flux.grid(alpha=0.3)

        fig.colorbar(pcm, ax=axes[:, 0], orientation='horizontal', fraction=0.05, pad=0.02, label='log10(Δr [R_M])')
        fig.colorbar(pcm_f, ax=axes[:, 1], orientation='horizontal', fraction=0.05, pad=0.02, label='log10(Flux [#/m²/s])')
        out_png = os.path.join(out_dir, f"{case}_delta_r_flux_gridded_log.png")
        fig.savefig(out_png, dpi=250)
        plt.show()

# =============================================================================
# HEATMAP CORRELATION: Δr vs flux with latitude limited to ±75°
# =============================================================================

lat_mask = (lat_centers >= DAY_LAT_MIN) & (lat_centers <= DAY_LAT_MAX)
# lat_mask = (lat_centers >= -90) & (lat_centers <= 90)

if 0:
    for case in cases:
        fig, axes = plt.subplots(
            nrows=len(selected_times), ncols=1, figsize=(6, 3*len(selected_times)), constrained_layout=True
        )

        for row, t_sel in enumerate(selected_times):
            ax = axes[row] if len(selected_times) > 1 else axes

            # select only the dayside longitudes and ±75° latitude
            dr = all_delta_r[case][t_sel][lat_mask, :]
            flux = all_flux[case][t_sel][lat_mask, :]

            # log-transform for correlation
            dr_log = np.log10(dr)
            flux_log = np.log10(flux)

            # flatten arrays and remove NaNs or -inf from log10
            # valid = np.isfinite(dr_log) & np.isfinite(flux_log)
            valid = np.isfinite(flux_log)
            # dr_flat = dr_log[valid]
            dr_flat = dr[valid]
            flux_flat = flux_log[valid]

            if dr_flat.size == 0:
                print(f"Skipping {case} t={t_sel}s: no valid data in ±75° latitude")
                continue

            # 2D histogram
            h = ax.hist2d(
                dr_flat, flux_flat,
                bins=[DR_BINS, 80],
                cmap="cividis",
                range=[[1.0, 1.75], [8, 14]]
            )

            # median trend
            bins = np.linspace(1.0, 1.75, DR_BINS)
            digitized = np.digitize(dr_flat, bins)
            med = [np.nanmedian(flux_flat[digitized == i]) if np.any(digitized == i) else np.nan for i in range(1, len(bins))]
            centers = 0.5 * (bins[:-1] + bins[1:])
            ax.plot(centers, med, color='red', lw=2, label='median')

            ax.set_title(f"{case} t={t_sel:.1f}s")
            ax.set_xlabel("Δr [R_M]")
            ax.set_ylabel("log10(Flux [#/m²/s])")
            ax.grid(alpha=0.3)
            ax.legend()

        fig.colorbar(h[3], ax=axes, orientation='vertical', fraction=0.02, pad=0.01, label='counts')
        plt.show()


fig, axes = plt.subplots(
    nrows=len(selected_times), ncols=len(cases),
    figsize=(12, 10), constrained_layout=True, sharex=True, sharey=True
)

for i, t_sel in enumerate(selected_times):
    for j, case in enumerate(cases):

        ax = axes[i, j]

        # select only the dayside longitudes and ±75° latitude
        dr = all_delta_r[case][t_sel][lat_mask, :]
        flux = all_flux[case][t_sel][lat_mask, :]

        # log-transform flux
        dr_log = np.log10(dr)
        flux_log = np.log10(flux)

        # flatten arrays and remove invalid values
        valid = np.isfinite(flux_log)
        dr_flat = dr[valid]
        flux_flat = flux_log[valid]

        if dr_flat.size == 0:
            print(f"Skipping {case} t={t_sel}s: no valid data in ±75° latitude")
            continue

        # 2D histogram
        h = ax.hist2d(
            dr_flat, flux_flat,
            bins=[DR_BINS, 80],
            cmap="cividis",
            range=[[1.0, 1.9], [8.5, 14.5]],
            vmin=0, vmax=5
        )

        # median trend
        bins = np.linspace(1.0, 1.9, DR_BINS)
        digitized = np.digitize(dr_flat, bins)
        med = [
            np.nanmedian(flux_flat[digitized == k]) if np.any(digitized == k) else np.nan
            for k in range(1, len(bins))
        ]
        centers = 0.5 * (bins[:-1] + bins[1:])
        # ax.plot(centers, med, color='red', lw=2)

        # titles
        # panel label (a1, a2, ..., b1, b2, ...)
        row_letter = chr(ord('a') + i)
        panel_label = f"({row_letter}{j + 1})"

        ax.set_title(f"{panel_label} {case.split('_')[0]}  t = {t_sel:.0f} s", fontsize=14, fontweight='bold')

        # labels only on outer panels
        if i == len(selected_times) - 1:
            ax.set_xlabel("Δr [R$_M$]", fontsize=11)
        if j == 0:
            ax.set_ylabel("log10(Flux [#/m²/s])", fontsize=11)

        ax.grid(alpha=0.3)

# shared colorbar
fig.colorbar(
    h[3],
    ax=axes,
    orientation='vertical',
    fraction=0.02,
    pad=0.01,
    label='counts'
)

out_png = os.path.join(out_dir, f"all_cases_delta_r_fluxlog_correlation_lat{DAY_LAT_MIN}—{DAY_LAT_MAX}.png")
fig.savefig(out_png, dpi=250)
plt.show()

def draw_latlon_grid(ax):
    # latitude lines
    lats = np.arange(-60, 90, 30)
    lons = np.linspace(-90, 90, 361)

    for lat in lats:
        lat_r = np.radians(lat)
        lon_r = np.radians(lons)

        x = np.cos(lat_r) * np.sin(lon_r)
        y = np.sin(lat_r) * np.ones_like(lon_r)

        mask = np.cos(lat_r) * np.cos(lon_r) > 0
        ax.plot(x[mask], y[mask], color="gray", lw=0.5, alpha=0.5)

    # longitude lines
    lons = np.arange(-90, 91, 30)
    lats = np.linspace(-90, 90, 361)

    for lon in lons:
        lat_r = np.radians(lats)
        lon_r = np.radians(lon)

        x = np.cos(lat_r) * np.sin(lon_r)
        y = np.sin(lat_r)

        mask = np.cos(lat_r) * np.cos(lon_r) > 0
        ax.plot(x[mask], y[mask], color="silver", lw=0.5, alpha=0.5)

    # longitude labels
    for lon in np.arange(-60, 61, 30):
        lon_r = np.radians(lon)
        x = np.sin(lon_r)
        ax.text(x, -1.05, f"{lon}°", ha="center", va="top", fontsize=8)

    # latitude labels
    for lat in [-60, -30, 0, 30, 60]:
        y = np.sin(np.radians(lat))
        ax.text(-1.05, y, f"{lat}°", ha="right", va="center", fontsize=8)


# =============================================================================
# CIRCULAR DAYSIDE PROJECTION (SUBSOLAR DISK) — ALL CASES / ALL TIMES
# =============================================================================

fig, axes = plt.subplots(
    nrows=len(selected_times),
    ncols=len(cases),
    figsize=(12, 10),
    constrained_layout=True
)

for i, t_sel in enumerate(selected_times):
    for j, case in enumerate(cases):

        ax = axes[i, j]

        dr_log = np.log10(all_delta_r[case][t_sel][lat_mask, :])

        lon_day = lon_centers[(lon_centers >= DAY_LON_MIN) & (lon_centers <= DAY_LON_MAX)]
        lat_day = lat_centers[(lat_centers >= DAY_LAT_MIN) & (lat_centers <= DAY_LAT_MAX)]
        LAT, LON = np.meshgrid(lat_day, lon_day, indexing="ij")

        lat_r = np.radians(LAT)
        lon_r = np.radians(LON)

        # Orthographic projection centered on subsolar point
        X = np.cos(lat_r) * np.sin(lon_r)
        Y = np.sin(lat_r)

        # Mask nightside
        mask = np.cos(lat_r) * np.cos(lon_r) > 0
        dr_log_plot = np.where(mask, dr_log, np.nan)

        pcm = ax.pcolormesh(
            X, Y, dr_log_plot,
            shading="auto",
            cmap="viridis",
            vmin=np.log10(1.0),
            vmax=np.log10(1.6)
        )

        ax.set_aspect("equal")
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.axis("off")

        # draw dayside boundary
        circle = plt.Circle((0,0),1, color="black", fill=False, lw=1.2)
        ax.add_patch(circle)

        draw_latlon_grid(ax)

        # panel label
        row_letter = chr(ord('a') + i)
        panel_label = f"({row_letter}{j+1})"

        ax.set_title(
            f"{panel_label} {case.split('_')[0]}  t = {t_sel:.0f} s",
            fontsize=14, fontweight='bold'
        )

# shared colorbar
fig.colorbar(
    pcm,
    ax=axes,
    orientation="vertical",
    fraction=0.02,
    pad=0.01,
    label="log10(Δr [R$_M$])"
)

out_png = os.path.join(out_dir, f"all_cases_dayside_disk_projection_log_lat{DAY_LAT_MIN}—{DAY_LAT_MAX}.png")
fig.savefig(out_png, dpi=250)

plt.show()

if 0:
    # =============================================================================
    # CIRCULAR DAYSIDE PROJECTION (SUBSOLAR DISK) — LOG VERSION
    # =============================================================================

    for case in cases:
        fig, axes = plt.subplots(
            nrows=4, ncols=2, figsize=(10,12),
            constrained_layout=True
        )

        for row, t_sel in enumerate(selected_times):

            ax_dr = axes[row,0]
            ax_flux = axes[row,1]

            dr = np.log10(all_delta_r[case][t_sel])
            flux = np.log10(all_flux[case][t_sel])

            lon_day = lon_centers[(lon_centers >= DAY_LON_MIN) & (lon_centers <= DAY_LON_MAX)]
            LAT, LON = np.meshgrid(lat_centers, lon_day, indexing="ij")

            lat_r = np.radians(LAT)
            lon_r = np.radians(LON)

            # Orthographic projection centered on subsolar point
            X = np.cos(lat_r) * np.sin(lon_r)
            Y = np.sin(lat_r)

            # Mask nightside
            mask = np.cos(lat_r) * np.cos(lon_r) > 0

            dr_plot = np.where(mask, dr, np.nan)
            flux_plot = np.where(mask, flux, np.nan)

            pcm = ax_dr.pcolormesh(
                X, Y, dr_plot,
                shading="auto",
                cmap="viridis",
                vmin=np.log10(1.0),
                vmax=np.log10(1.75)
            )

            pcm_f = ax_flux.pcolormesh(
                X, Y, flux_plot,
                shading="auto",
                cmap="plasma",
                vmin=np.log10(1e8),
                vmax=np.log10(1e14)
            )

            for ax in (ax_dr, ax_flux):
                ax.set_aspect("equal")
                ax.set_xlim(-1,1)
                ax.set_ylim(-1,1)
                ax.axis("off")

                # draw dayside boundary
                circle = plt.Circle((0,0),1,color="black",fill=False,lw=1)
                ax.add_patch(circle)

                draw_latlon_grid(ax)

            ax_dr.set_title(f"{case.split('_')[0]} log10(Δr) t={t_sel:.1f}s")
            ax_flux.set_title(f"{case.split('_')[0]} log10(flux) t={t_sel:.1f}s")

        fig.colorbar(
            pcm, ax=axes[:,0],
            orientation="horizontal",
            fraction=0.05,
            pad=0.05,
            label="log10(Δr [R_M])"
        )

        fig.colorbar(
            pcm_f, ax=axes[:,1],
            orientation="horizontal",
            fraction=0.05,
            pad=0.05,
            label="log10(Flux [#/m²/s])"
        )

        out_png = os.path.join(out_dir, f"{case}_dayside_disk_projection_log.png")
        fig.savefig(out_png, dpi=250)
        plt.show()
