#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correlation between dayside magnetopause stand-off distance and
surface precipitation number flux.

Outputs two plots per timestep:
1) Δr vs flux (scatter + density)
2) Δr vs flux vs latitude (latitude binned median trend)
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

import src.surface_flux.flux_utils as flux_utils

# =============================================================================
# USER SETTINGS
# =============================================================================

cases = ["RPS_HNHV", "CPS_HNHV", "RPN_HNHV", "CPN_HNHV"]

base_precip_dir = "/Volumes/T9/mercury/extreme/High_HNHV"
base_mp_dir = "/Volumes/T9/mercury/magnetopause_3D_timeseries"
base_out_dir = "/Users/danywaller/Projects/mercury/extreme/mp_precip_correlation"

# simulation parameters
sim_ppc = np.array([24, 24, 11, 11], dtype=float)
sim_den = np.array([38.0e6, 76.0e6, 1.0e6, 2.0e6], dtype=float)
species_mass = np.array([1.0, 1.0, 4.0, 4.0], dtype=float)

AMU = 1.66053906660e-27
m_kg_by_sid = species_mass * AMU

sim_dx = 75e3
sim_dy = 75e3
sim_dz = 75e3

R_M = 2440e3
DELTA_R_M = 0.5 * sim_dx

# precipitation grid
dlat = 2.0
dlon = 2.0
lat_bin_edges = np.arange(-90, 90 + dlat, dlat)
lon_bin_edges = np.arange(-180, 180 + dlon, dlon)

# correlation settings
DR_BINS = 40
LAT_BINS = 20
eps = 1e-30

# dayside longitude range
DAY_LON_MIN = -90
DAY_LON_MAX = 90

# =============================================================================
# HELPERS
# =============================================================================

def read_npz_time(path):
    with np.load(path) as p:
        return float(np.asarray(p["time"]).item())


def nearest_index(values, target):
    return int(np.argmin(np.abs(values - target)))


def compute_delta_r(mask_xyz, x, y, z):
    """Compute dayside Δr map on theta/phi grid"""
    theta = np.linspace(1e-6, np.pi - 1e-6, 90)
    phi = np.linspace(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6, 360)
    r_samples = np.linspace(1.0, 3.0, 200)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    sx = np.sin(TH) * np.cos(PH)
    sy = np.sin(TH) * np.sin(PH)
    sz = np.cos(TH)
    delta_r = np.full_like(TH, np.nan)

    for i in range(len(theta)):
        for j in range(len(phi)):
            for r in r_samples[::-1]:
                px = r * sx[i, j]
                py = r * sy[i, j]
                pz = r * sz[i, j]
                ix = np.argmin(np.abs(x - px))
                iy = np.argmin(np.abs(y - py))
                iz = np.argmin(np.abs(z - pz))
                if mask_xyz[ix, iy, iz]:
                    delta_r[i, j] = r - 1.0
                    break
    return delta_r


def regrid_delta_r(delta_r):
    """Interpolate MP Δr onto precipitation grid"""
    mp_lat = np.linspace(-90, 90, delta_r.shape[0])
    mp_lon = np.linspace(-90, 90, delta_r.shape[1])
    precip_lat = 0.5 * (lat_bin_edges[:-1] + lat_bin_edges[1:])
    precip_lon = 0.5 * (lon_bin_edges[:-1] + lon_bin_edges[1:])
    LATp, LONp = np.meshgrid(precip_lat, precip_lon, indexing="ij")
    interp = RegularGridInterpolator(
        (mp_lat, mp_lon),
        delta_r,
        bounds_error=False,
        fill_value=np.nan
    )
    pts = np.column_stack([LATp.ravel(), LONp.ravel()])
    delta_r_interp = interp(pts).reshape(LATp.shape)
    return delta_r_interp


def plot_correlation(outpath, delta_r, flux, lat_centers, t_flux=None):
    """Scatter + density plot: Δr vs flux"""
    dr = delta_r.flatten()
    flux = flux.flatten()
    valid = np.isfinite(dr) & np.isfinite(flux) & (flux > 0)
    dr = dr[valid]
    flux = np.log10(flux[valid])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(dr, flux, s=2, alpha=0.25)
    h = ax.hist2d(dr, flux, bins=[DR_BINS, 80], cmap="cividis")
    plt.colorbar(h[3], ax=ax, label="counts")

    # median trend
    bins = np.linspace(np.min(dr), np.max(dr), DR_BINS)
    digitized = np.digitize(dr, bins)
    med = [np.nanmedian(flux[digitized == i]) for i in range(1, len(bins))]
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax.plot(centers, med, color="red", lw=2, label="median")

    # add time to title if provided
    title = "Δr vs flux"
    if t_flux is not None:
        title += f", t={t_flux:.1f} s"
    ax.set_title(title)

    ax.set_xlabel("Magnetopause stand-off Δr  [R_M]")
    ax.set_ylabel("log10 precipitation flux  [#/m²/s]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(outpath, dpi=250)
    plt.close()


def plot_vs_latitude(outpath, delta_r, flux, lat_centers):
    """Latitude-binned median of flux vs Δr"""
    dr = delta_r.flatten()
    flux = flux.flatten()
    lat = np.repeat(lat_centers, delta_r.shape[1])

    valid = np.isfinite(dr) & np.isfinite(flux) & (flux > 0)
    dr = dr[valid]
    flux = np.log10(flux[valid])
    lat = lat[valid]

    bins = np.linspace(-90, 90, LAT_BINS + 1)
    medians = []
    dr_centers = []

    for i in range(LAT_BINS):
        mask = (lat >= bins[i]) & (lat < bins[i + 1])
        if np.any(mask):
            medians.append(np.median(flux[mask]))
            dr_centers.append(np.median(dr[mask]))
        else:
            medians.append(np.nan)
            dr_centers.append(np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(dr_centers, medians, 'o-', color='blue', lw=2)
    ax.set_xlabel("Magnetopause stand-off Δr  [R_M]")
    ax.set_ylabel("log10 precipitation flux  [#/m²/s]")
    ax.set_title("Latitude-binned median flux vs Δr")
    ax.grid(alpha=0.3)
    fig.savefig(outpath, dpi=250)
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

V_cell = sim_dx * sim_dy * sim_dz
W_by_sid = flux_utils.macro_weights(sim_den, sim_ppc, V_cell)
lat_centers = 0.5 * (lat_bin_edges[:-1] + lat_bin_edges[1:])
lon_centers = 0.5 * (lon_bin_edges[:-1] + lon_bin_edges[1:])

for case in cases:
    print(f"\nProcessing case: {case}")
    precip_glob = f"{base_precip_dir}/{case}/precipitation_timeseries/*.npz"
    mp_nc = os.path.join(base_mp_dir, case, f"{case}_mp_mask_timeseries.nc")
    out_dir = os.path.join(base_out_dir, case)
    os.makedirs(out_dir, exist_ok=True)

    precip_files = sorted(glob.glob(precip_glob))
    precip_times = np.array([read_npz_time(f) for f in precip_files])

    with xr.open_dataset(mp_nc) as ds:
        mp_mask_4d = ds["mp_mask"].values
        mp_times = np.asarray(ds["time"].values)
        x = np.asarray(ds["x"].values)
        y = np.asarray(ds["y"].values)
        z = np.asarray(ds["z"].values)

    for i, npz_path in enumerate(precip_files):
        t_precip = precip_times[i]
        print(f"\tProcessing timestep: {t_precip}")
        mp_idx = nearest_index(mp_times, t_precip)

        # --- compute magnetopause stand-off ---
        delta_r = compute_delta_r(mp_mask_4d[mp_idx].astype(bool), x, y, z)
        delta_r_interp = regrid_delta_r(delta_r)

        # --- compute precipitation flux maps ---
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

        # --- clip to dayside ---
        dayside_mask = (lon_centers >= DAY_LON_MIN) & (lon_centers <= DAY_LON_MAX)
        flux_dayside = flux_all[:, dayside_mask]
        delta_r_dayside = delta_r_interp[:, dayside_mask]

        # --- use physical time from flux computation in filenames and titles ---
        out_png1 = os.path.join(out_dir, f"{case}_mp_vs_precip_t{t_flux:.1f}s.png")
        out_png2 = os.path.join(out_dir, f"{case}_mp_vs_precip_vs_lat_t{t_flux:.1f}s.png")

        title_corr = f"{case} Δr vs flux, t={t_flux:.1f} s"

        plot_correlation(out_png1, delta_r_dayside, flux_dayside, lat_centers, t_flux=t_flux)
        plot_vs_latitude(out_png2, delta_r_dayside, flux_dayside, lat_centers)

        print(f"Saved plots for t_flux={t_flux:.1f} s")