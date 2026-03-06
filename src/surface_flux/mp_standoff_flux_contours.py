#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay dayside magnetopause stand-off contours on total precipitation number-flux maps.

Inputs per case
---------------
1) Precipitation snapshots: precipitation_timeseries/*.npz
2) Magnetopause mask timeseries: <case>_mp_mask_timeseries.nc

Output
------
Hammer-projection PNG per matched timestep with:
- background: log10(total inward number flux) [#/m^2/s]
- contours:   MP stand-off from surface, Delta r = r_mp - 1 [R_M]
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import patheffects
from scipy.ndimage import distance_transform_edt, gaussian_filter

import src.surface_flux.flux_utils as flux_utils


# =============================================================================
# USER SETTINGS
# =============================================================================
cases = ["RPS_HNHV", "CPS_HNHV", "RPN_HNHV", "CPN_HNHV"]

base_precip_dir = "/Volumes/T9/mercury/extreme/High_HNHV"
base_mp_dir = "/Volumes/T9/mercury/magnetopause_3D_timeseries"
base_out_dir = "/Users/danywaller/Projects/mercury/extreme/surface_flux_timeseries_mp_overlay"

# Precipitation setup (same species ordering used in case_comparison_timeseries.py)
sim_ppc = np.array([24, 24, 11, 11], dtype=float)
sim_den = np.array([38.0e6, 76.0e6, 1.0e6, 2.0e6], dtype=float)   # [m^-3]
species_mass = np.array([1.0, 1.0, 4.0, 4.0], dtype=float)         # [amu]

AMU = 1.66053906660e-27
m_kg_by_sid = species_mass * AMU

sim_dx = 75.0e3
sim_dy = 75.0e3
sim_dz = 75.0e3
R_M = 2440.0e3
DELTA_R_M = 0.5 * sim_dx

dlat = 2.0
dlon = 2.0
lat_bin_edges = np.arange(-90.0, 90.0 + dlat, dlat)
lon_bin_edges = np.arange(-180.0, 180.0 + dlon, dlon)

# Number-flux plotting (same style/range as case_comparison_timeseries.py)
eps = 1e-30
CMAP = "cividis"
NF_VMIN, NF_VMAX = 9.0, 14.0

# MP stand-off mapping settings
N_THETA = 90
N_PHI = 360
DAYSIDE_LON_LIMIT_DEG = 90.0
EPS_ANG = 1e-6

# Ray-trace stand-off solver settings
RAY_RMIN_RM = 1.0
RAY_RMAX_RM = 3.0
RAY_DR_OVERRIDE_RM = None  # set float to force radial sampling step [R_M]

# Surface smoothing settings for Delta r
FILL_NAN_WITH_NEAREST = True
SMOOTH_SIGMA_THETA = 2.0
SMOOTH_SIGMA_PHI = 2.0

# Contour levels for Delta r = r - 1 [R_M]
DELTA_R_LEVELS = np.arange(0.05, 0.9, 0.1)
CONTOUR_CMAP = "cool"

# Time matching tolerance (seconds). If None, auto = 0.51 * median(mp_dt).
TIME_MATCH_TOL_S = None


# =============================================================================
# Helpers: Hammer formatting (kept consistent with case_comparison_timeseries.py)
# =============================================================================
def _deg_edges_to_rad(lon_edges_deg, lat_edges_deg):
    return np.deg2rad(lon_edges_deg), np.deg2rad(lat_edges_deg)


def _set_hammer_degree_grid(ax, lon_step_deg=60, lat_step_deg=30):
    xt = np.deg2rad(np.arange(-150, 151, lon_step_deg))
    yt = np.deg2rad(np.arange(-60, 60, lat_step_deg))
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([f"{d:d}" for d in np.arange(-150, 151, lon_step_deg)])
    ax.set_yticklabels([f"{d:d}" for d in np.arange(-60, 60, lat_step_deg)])


def _read_npz_time(npz_path):
    with np.load(npz_path) as p:
        return float(np.asarray(p["time"]).item())


def _build_theta_phi_grid():
    theta = np.linspace(EPS_ANG, np.pi - EPS_ANG, N_THETA)
    phi_lim = np.deg2rad(DAYSIDE_LON_LIMIT_DEG)
    phi = np.linspace(-phi_lim + EPS_ANG, phi_lim - EPS_ANG, N_PHI)
    lat_deg = 90.0 - np.degrees(theta)
    lon_deg = np.degrees(phi)
    return theta, phi, lat_deg, lon_deg


def _nearest_index(values, target):
    i = int(np.argmin(np.abs(values - target)))
    return i, float(np.abs(values[i] - target))


def _build_ray_cache(x, y, z, theta, phi):
    """
    Precompute nearest-neighbor voxel indices for all sampled rays.
    """
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

    return {
        "r_samp": r_samp,
        "ix": ixn,
        "iy": iyn,
        "iz": izn,
        "valid": valid,
        "dr": dr,
    }


def _fill_nans_nearest(arr2d):
    out = np.array(arr2d, dtype=float, copy=True)
    nan_mask = ~np.isfinite(out)
    if not np.any(nan_mask) or np.all(nan_mask):
        return out

    nearest_idx = distance_transform_edt(
        nan_mask, return_distances=False, return_indices=True
    )
    nearest_vals = out[tuple(nearest_idx[d] for d in range(out.ndim))]
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

    smoothed = np.full_like(out, np.nan, dtype=float)
    good = w_blur > 1e-8
    smoothed[good] = vals_blur[good] / w_blur[good]
    return smoothed


def _compute_dayside_delta_r_map(mask_xyz, ray_cache):
    """
    Compute dayside stand-off map Delta r = r_mp - 1 [R_M] on a theta/phi grid.
    """
    r_samp = ray_cache["r_samp"]
    ix = ray_cache["ix"]
    iy = ray_cache["iy"]
    iz = ray_cache["iz"]
    valid = ray_cache["valid"]

    out_shape = valid.shape[1:]
    if not np.any(mask_xyz):
        return np.full(out_shape, np.nan, dtype=float)

    inside = np.zeros_like(valid, dtype=bool)
    v = valid
    inside[v] = mask_xyz[ix[v], iy[v], iz[v]]

    any_true = np.any(inside, axis=0)
    if not np.any(any_true):
        return np.full(out_shape, np.nan, dtype=float)

    inside_rev = inside[::-1, :, :]
    j_from_end = np.argmax(inside_rev, axis=0)
    j_last = (len(r_samp) - 1) - j_from_end

    r_map = np.full(out_shape, np.nan, dtype=float)
    r_map[any_true] = r_samp[j_last[any_true]]
    delta_r = r_map - 1.0

    if FILL_NAN_WITH_NEAREST:
        delta_r = _fill_nans_nearest(delta_r)

    if SMOOTH_SIGMA_THETA > 0.0 or SMOOTH_SIGMA_PHI > 0.0:
        delta_r = _nan_gaussian_smooth(delta_r, sigma=(SMOOTH_SIGMA_THETA, SMOOTH_SIGMA_PHI))

    return delta_r


def _save_overlay_map(
    outpath,
    flux2d,
    delta_r2d,
    lon_mp_deg,
    lat_mp_deg,
    *,
    title,
):
    lon_e_rad, lat_e_rad = _deg_edges_to_rad(lon_bin_edges, lat_bin_edges)

    plot_flux = np.full_like(flux2d, np.nan, dtype=float)
    mf = np.isfinite(flux2d)
    plot_flux[mf] = np.log10(np.maximum(flux2d[mf], eps))

    fig, ax = plt.subplots(
        figsize=(10, 4.8),
        constrained_layout=True,
        subplot_kw={"projection": "hammer"},
    )

    pm = ax.pcolormesh(
        lon_e_rad,
        lat_e_rad,
        plot_flux,
        shading="flat",
        cmap=CMAP,
        vmin=NF_VMIN,
        vmax=NF_VMAX,
    )

    # Overlay MP Delta r contours on the dayside patch.
    if np.isfinite(delta_r2d).any():
        lon_mp_rad = np.deg2rad(lon_mp_deg)
        lat_mp_rad = np.deg2rad(lat_mp_deg)
        lon2d, lat2d = np.meshgrid(lon_mp_rad, lat_mp_rad)

        cs = ax.contour(
            lon2d,
            lat2d,
            delta_r2d,
            levels=DELTA_R_LEVELS,
            cmap=CONTOUR_CMAP,
            linewidths=1.2,
        )

        labels = ax.clabel(cs, inline=True, fmt="%.2f", fontsize=6)
        for txt in labels:
            txt.set_path_effects([patheffects.withStroke(linewidth=0.9, foreground="black")])

    _set_hammer_degree_grid(ax)
    ax.grid(True, alpha=0.35)
    ax.set_title(title)

    cb = fig.colorbar(pm, ax=ax, pad=0.03, shrink=0.9)
    cb.set_label("log10(#/m^2/s)")

    # Legend-like note for contour quantity.
    ax.text(
        -0.01,
        0.03,
        "Contours: $\\Delta r = r-1\ [R_M]$",
        transform=ax.transAxes,
        fontsize=10,
        color="black",
        ha="left",
        va="bottom",
    )

    fig.savefig(outpath, dpi=250)
    plt.close(fig)


V_cell = sim_dx * sim_dy * sim_dz
W_by_sid = flux_utils.macro_weights(sim_den, sim_ppc, V_cell)

theta, phi, lat_mp_deg, lon_mp_deg = _build_theta_phi_grid()

for case in cases:
    print(f"\n{'=' * 72}")
    print(f"Processing case: {case}")
    print(f"{'=' * 72}")

    precip_glob = f"{base_precip_dir}/{case}/precipitation_timeseries/*.npz"
    mp_nc = os.path.join(base_mp_dir, case, f"{case}_mp_mask_timeseries.nc")
    out_dir = os.path.join(base_out_dir, case, "total_with_mp_standoff_contours")
    os.makedirs(out_dir, exist_ok=True)

    precip_files = sorted(glob.glob(precip_glob))
    if len(precip_files) == 0:
        print(f"  WARNING: no precipitation files found ({precip_glob})")
        continue
    if not os.path.exists(mp_nc):
        print(f"  WARNING: mp mask file not found ({mp_nc})")
        continue

    precip_times = np.array([_read_npz_time(f) for f in precip_files], dtype=float)

    with xr.open_dataset(mp_nc) as ds:
        mp_mask_4d = ds["mp_mask"].values
        mp_times = np.asarray(ds["time"].values, dtype=float)
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
        z = np.asarray(ds["z"].values, dtype=float)

    if len(mp_times) == 0:
        print("  WARNING: no mp times present; skipping.")
        continue

    if TIME_MATCH_TOL_S is None:
        if len(mp_times) > 1:
            match_tol = 0.51 * float(np.nanmedian(np.diff(mp_times)))
        else:
            match_tol = 0.5
    else:
        match_tol = float(TIME_MATCH_TOL_S)

    print(f"  Time matching tolerance: {match_tol:.4f} s")
    ray_cache = _build_ray_cache(x, y, z, theta, phi)
    print(
        f"  Ray sampler: r=[{ray_cache['r_samp'][0]:.2f}, {ray_cache['r_samp'][-1]:.2f}] R_M, "
        f"dr~{ray_cache['dr']:.4f} R_M, n_r={len(ray_cache['r_samp'])}"
    )

    standoff_cache = {}
    n_saved = 0
    n_skipped = 0

    for i, npz_path in enumerate(precip_files):
        t_precip = precip_times[i]
        mp_idx, dt = _nearest_index(mp_times, t_precip)
        if dt > match_tol:
            print(f"  [{i+1:03d}/{len(precip_files):03d}] skip (no time match): dt={dt:.3f}s")
            n_skipped += 1
            continue

        if mp_idx not in standoff_cache:
            standoff_cache[mp_idx] = _compute_dayside_delta_r_map(
                mp_mask_4d[mp_idx].astype(bool),
                ray_cache,
            )

        (
            _flux_by_sid, flux_all,
            _mass_flux_by_sid, _mass_flux_all,
            _energy_flux_by_sid, _energy_flux_all,
            _vrabs_map, t_flux,
            _total_rate, _total_rate_by_sid,
            _total_mass_rate, _total_mass_rate_by_sid,
            _total_power, _total_power_by_sid
        ) = flux_utils.flux_maps_snapshot_shell(
            npz_path,
            R=R_M,
            delta_r_m=DELTA_R_M,
            lat_bin_edges=lat_bin_edges,
            lon_bin_edges=lon_bin_edges,
            W_by_sid=W_by_sid,
            m_kg_by_sid=m_kg_by_sid,
        )

        base = os.path.basename(npz_path).replace(".npz", "")
        out_png = os.path.join(out_dir, f"{base}_flux_total_with_mp_delta_r_contours.png")

        _save_overlay_map(
            out_png,
            flux_all,
            standoff_cache[mp_idx],
            lon_mp_deg,
            lat_mp_deg,
            title=(
                f"{case.replace('_', ' ')} total inward surface number flux\n"
                f"MP stand-off from surface contours ($\\Delta r$), t={t_flux:.3f} s"
            ),
        )

        n_saved += 1
        print(
            f"  [{i+1:03d}/{len(precip_files):03d}] "
            f"saved (t_flux={t_flux:.3f}s, t_mp={mp_times[mp_idx]:.3f}s, dt={dt:.3f}s)"
        )

    print(f"  Complete: saved={n_saved}, skipped={n_skipped}, out_dir={out_dir}")
