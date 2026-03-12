#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
import src.surface_flux.flux_utils as flux_utils

# =============================================================================
# USER SETTINGS
# =============================================================================
cases = ["RPN_HNHV", "CPN_HNHV", "RPS_HNHV", "CPS_HNHV"]
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

V_cell = sim_dx * sim_dy * sim_dz
W_by_sid = flux_utils.macro_weights(sim_den, sim_ppc, V_cell)

# =============================================================================
# Helpers
# =============================================================================

def _read_npz_time(npz_path):
    """Extract simulation time from a precipitation npz snapshot."""
    with np.load(npz_path) as p:
        return float(np.asarray(p["time"]).item())

def _nearest_index(values, target):
    return int(np.argmin(np.abs(values - target)))

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

# =============================================================================
# PRECOMPUTE
# =============================================================================

all_flux = {}

for case in cases:
    print(f"\nProcessing case: {case}")

    precip_files = sorted(glob.glob(f"{base_precip_dir}/{case}/precipitation_timeseries/*.npz"))
    if len(precip_files) == 0:
        continue

    precip_times = np.array([_read_npz_time(f) for f in precip_files])

    all_flux[case] = {}

    for t_sel in selected_times:
        print(f"  t={t_sel:.1f}s")

        idx_precip = _nearest_index(precip_times, t_sel)
        npz_path = precip_files[idx_precip]
        t_precip = precip_times[idx_precip]

        # precipitation flux
        _, flux_all, *_ = flux_utils.flux_maps_snapshot_shell(
            npz_path, R=R_M, delta_r_m=DELTA_R_M,
            lat_bin_edges=lat_bin_edges, lon_bin_edges=lon_bin_edges,
            W_by_sid=W_by_sid, m_kg_by_sid=m_kg_by_sid
        )

        all_flux[case][t_sel] = flux_all

# =============================================================================
# 4x4 GRID: GLOBAL PRECIPITATION (HAMMER PROJECTION)
# =============================================================================

n_rows = len(selected_times)
n_cols = len(cases)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(3*n_cols, 2*n_rows),
    subplot_kw={"projection": "hammer"},
    constrained_layout=True
)

# ------------------------------------------------------------
# Determine global log10 limits for consistent colorbar
# ------------------------------------------------------------
all_logs = []

for case in cases:
    for t_sel in selected_times:
        flux = all_flux[case][t_sel]
        valid = flux > 0
        if np.any(valid):
            all_logs.append(np.log10(flux[valid]))

all_logs = np.concatenate(all_logs)

# vmin = np.nanmin(all_logs)
# vmax = np.nanmax(all_logs)
vmin = 7.5
vmax = 14.5

print(f"log10 flux limits: {vmin:.2f} → {vmax:.2f}")

# ------------------------------------------------------------
# Build mesh edges for Hammer projection
# ------------------------------------------------------------
lat_edges = np.radians(np.linspace(-90, 90, len(lat_centers) + 1))
lon_edges = np.radians(np.linspace(-180, 180, len(lon_centers) + 1))

# ------------------------------------------------------------
# Plot panels
# ------------------------------------------------------------
for col, case in enumerate(cases):
    for row, t_sel in enumerate(selected_times):

        ax = axes[row, col]

        flux = all_flux[case][t_sel]
        flux_log = np.log10(flux)

        pcm = ax.pcolormesh(
            lon_edges,
            lat_edges,
            flux_log,
            cmap="plasma",
            shading="auto",
            vmin=vmin,
            vmax=vmax
        )

        # ax.set_title(f"{case.split('_')[0]}\nt = {t_sel:.0f} s", fontsize=12, fontweight="bold")
        col_letter = chr(ord('a') + col)
        panel_label = f"({col_letter}{row + 1})"

        ax.set_title(
            f"{panel_label} {case.split('_')[0]} t = {t_sel:.0f} s",
            fontsize=12,
            fontweight="bold"
        )

        ax.grid(alpha=0.3, color="k")

        # nicer ticks for hammer
        ticks = [-90, 0, 90]

        ax.set_xticks(np.radians(ticks))
        ax.set_xticklabels([f"{t}°" for t in ticks])
        # ax.set_xticks(np.radians(np.linspace(-150, 150, 7)))
        ax.set_yticks(np.radians(np.linspace(-60, 60, 5)))

# ------------------------------------------------------------
# Shared colorbar
# ------------------------------------------------------------
cbar = fig.colorbar(
    pcm,
    ax=axes,
    orientation="vertical",
    fraction=0.025,
    pad=0.02
)

cbar.set_label("log10(Flux [#/m²/s])")

# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------
out_png = os.path.join(out_dir, "precip_flux_global_4x4.png")
fig.savefig(out_png, dpi=300)

plt.show()