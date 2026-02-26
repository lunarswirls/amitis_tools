#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports:
import numpy as np
import pandas as pd


# -----------------------
# Helpers
# -----------------------
def bin_areas_sphere(R, lat_bin_edges_deg, lon_bin_edges_deg):
    lat = np.deg2rad(lat_bin_edges_deg)
    lon = np.deg2rad(lon_bin_edges_deg)
    dlon = np.diff(lon)[None, :]
    sin_dlat = (np.sin(lat[1:]) - np.sin(lat[:-1]))[:, None]
    return (R**2) * sin_dlat * dlon


def macro_weights(sim_den_arr, sim_ppc_arr, V_cell):
    return sim_den_arr * V_cell / sim_ppc_arr


def nan_safe_sum2(a, b):
    out = np.nan_to_num(a, nan=0.0) + np.nan_to_num(b, nan=0.0)
    nodata = ~np.isfinite(a) & ~np.isfinite(b)
    out[nodata] = np.nan
    return out


def mask_zeros_to_nan(x):
    y = np.array(x, dtype=float, copy=True)
    y[(y == 0.0) & np.isfinite(y)] = np.nan
    return y


def flux_maps_snapshot_shell(npz_path, R, delta_r_m, lat_bin_edges, lon_bin_edges, W_by_sid,
                             m_kg_by_sid):
    """
    Same as before, but ENERGY FLUX is computed using each particle's actual radial speed.

    Energy flux definition used here:
      For each particle, energy per particle uses inward normal speed vn_in:
        E_part = 0.5 * m * vn_in^2  [J]
      and energy flux through the surface uses:
        energy_flux = sum( W * (vn_in/delta_r) * E_part ) / A  [W/m^2]
                   = sum( W * 0.5*m * vn_in^3 / delta_r ) / A
    This is consistent with computing the kinetic power carried by precipitating flow
    normal to the surface.
    """
    p = np.load(npz_path)

    rx, ry, rz = p["rx"], p["ry"], p["rz"]
    vx, vy, vz = p["vx"], p["vy"], p["vz"]
    sid = p["sid"].astype(int)

    t = np.asarray(p["time"]).item()

    Ns = len(W_by_sid)
    Nlat = len(lat_bin_edges) - 1
    Nlon = len(lon_bin_edges) - 1

    r = np.sqrt(rx*rx + ry*ry + rz*rz)
    invr = np.where(r > 0, 1.0 / r, 0.0)
    nx, ny, nz = rx * invr, ry * invr, rz * invr

    vr = vx * nx + vy * ny + vz * nz
    vn_in = np.maximum(0.0, -vr)

    shell = (r > R) & (r < (R + delta_r_m)) & (vn_in > 0.0)

    A = bin_areas_sphere(R, lat_bin_edges, lon_bin_edges)

    if not np.any(shell):
        nan_sid = np.full((Ns, Nlat, Nlon), np.nan, dtype=float)
        nan_2d = np.full((Nlat, Nlon), np.nan, dtype=float)

        flux_by_sid = nan_sid.copy()
        flux_all = nan_2d.copy()
        mass_flux_by_sid = nan_sid.copy()
        mass_flux_all = nan_2d.copy()
        energy_flux_by_sid = nan_sid.copy()
        energy_flux_all = nan_2d.copy()
        vrabs_map = nan_2d.copy()

        total_rate_by_sid = np.zeros(Ns, dtype=float)
        total_mass_rate_by_sid = np.zeros(Ns, dtype=float)
        total_power_by_sid = np.zeros(Ns, dtype=float)

        return (flux_by_sid, flux_all,
                mass_flux_by_sid, mass_flux_all,
                energy_flux_by_sid, energy_flux_all,
                vrabs_map, t,
                0.0, total_rate_by_sid,
                0.0, total_mass_rate_by_sid,
                0.0, total_power_by_sid)

    lat = np.rad2deg(np.arcsin(rz[shell] / r[shell]))
    lon = np.rad2deg(np.arctan2(ry[shell], rx[shell]))
    lon = (lon + 180.0) % 360.0 - 180.0

    ilat = np.searchsorted(lat_bin_edges, lat, side="right") - 1
    ilon = np.searchsorted(lon_bin_edges, lon, side="right") - 1

    ok = (ilat >= 0) & (ilat < Nlat) & (ilon >= 0) & (ilon < Nlon)
    ilat = ilat[ok]
    ilon = ilon[ok]
    sid_s = sid[shell][ok]
    vn_in_s = vn_in[shell][ok]
    vrabs_s = np.abs(vr[shell][ok])

    valid_sid = (sid_s >= 0) & (sid_s < Ns)
    ilat = ilat[valid_sid]
    ilon = ilon[valid_sid]
    sid_s = sid_s[valid_sid]
    vn_in_s = vn_in_s[valid_sid]
    vrabs_s = vrabs_s[valid_sid]

    # Accumulators
    rate_by_sid = np.zeros((Ns, Nlat, Nlon), float)        # number-rate surface-density * area => (#/m^2/s)*m^2
    e_rate_by_sid = np.zeros((Ns, Nlat, Nlon), float)      # power surface-density * area => (W/m^2)*m^2
    count_by_sid = np.zeros((Ns, Nlat, Nlon), dtype=np.int32)

    vrabs_sum_all = np.zeros((Nlat, Nlon), dtype=float)
    count_all = np.zeros((Nlat, Nlon), dtype=np.int32)

    for s in range(Ns):
        m = (sid_s == s)
        if not np.any(m):
            continue

        # NUMBER FLUX contribution
        np.add.at(rate_by_sid[s], (ilat[m], ilon[m]),
                  W_by_sid[s] * (vn_in_s[m] / delta_r_m))

        # ENERGY FLUX contribution using each particle's vn_in:
        # power density term: W * (vn_in/delta_r) * (0.5*m*vn_in^2) = W * 0.5*m*vn_in^3 / delta_r
        np.add.at(e_rate_by_sid[s], (ilat[m], ilon[m]),
                  W_by_sid[s] * (0.5 * m_kg_by_sid[s]) * (vn_in_s[m]**3) / delta_r_m)

        np.add.at(count_by_sid[s], (ilat[m], ilon[m]), 1)

        np.add.at(vrabs_sum_all, (ilat[m], ilon[m]), vrabs_s[m])
        np.add.at(count_all, (ilat[m], ilon[m]), 1)

    # Convert accumulators to surface fluxes by dividing by bin area
    flux_by_sid = rate_by_sid / A[None, :, :]       # (#/m^2/s)
    energy_flux_by_sid = e_rate_by_sid / A[None, :, :]  # (W/m^2)

    flux_all_unmasked = np.sum(flux_by_sid, axis=0)
    energy_flux_all_unmasked = np.sum(energy_flux_by_sid, axis=0)

    # Integrated number rates (#/s)
    total_rate_by_sid = np.sum(flux_by_sid * A[None, :, :], axis=(1, 2))
    total_rate = float(np.sum(flux_all_unmasked * A))

    # Integrated power (W)
    total_power_by_sid = np.nansum(energy_flux_by_sid * A[None, :, :], axis=(1, 2))
    total_power = float(np.nansum(energy_flux_all_unmasked * A))

    # Mask no-data bins as NaN for plotting
    count_all_from_sid = np.sum(count_by_sid, axis=0)

    flux_all = flux_all_unmasked.astype(float, copy=True)
    flux_all[count_all_from_sid == 0] = np.nan

    flux_by_sid = flux_by_sid.astype(float, copy=True)
    flux_by_sid[count_by_sid == 0] = np.nan

    energy_flux_all = energy_flux_all_unmasked.astype(float, copy=True)
    energy_flux_all[count_all_from_sid == 0] = np.nan

    energy_flux_by_sid = energy_flux_by_sid.astype(float, copy=True)
    energy_flux_by_sid[count_by_sid == 0] = np.nan

    # MASS FLUX derived from number flux and species mass
    mass_flux_by_sid = flux_by_sid * m_kg_by_sid[:, None, None]
    mass_flux_all = np.nansum(mass_flux_by_sid, axis=0)

    nodata_all = np.all(~np.isfinite(flux_by_sid), axis=0)
    mass_flux_all[nodata_all] = np.nan

    # Integrated mass rates (kg/s)
    total_mass_rate_by_sid = np.nansum(mass_flux_by_sid * A[None, :, :], axis=(1, 2))
    total_mass_rate = float(np.nansum(mass_flux_all * A))

    vrabs_map = np.full((Nlat, Nlon), np.nan, dtype=float)
    mvr = (count_all > 0)
    vrabs_map[mvr] = vrabs_sum_all[mvr] / count_all[mvr]

    return (flux_by_sid, flux_all,
            mass_flux_by_sid, mass_flux_all,
            energy_flux_by_sid, energy_flux_all,
            vrabs_map, t,
            total_rate, total_rate_by_sid,
            total_mass_rate, total_mass_rate_by_sid,
            total_power, total_power_by_sid)


# -----------------------
# Map stats function
# -----------------------
def compute_flux_statistics(flux_map, lat_centers, lon_centers, R_M,
                            flux_threshold=None, case_name="Case"):
    """
    Compute summary statistics for a georeferenced flux map.
    """
    def signed_ratio(A, B):
        # Signed ratio in [-1, +1]: (A-B)/(A+B)
        denom = A + B
        return (A - B) / denom if denom > 0 else np.nan

    stats = {
        'case_name': case_name,
        'total_integrated_flux': np.nan,

        # Peak pixel
        'peak_flux_value': np.nan,
        'peak_flux_lat': np.nan,
        'peak_flux_lon': np.nan,

        # Area above threshold
        'spatial_extent_area': np.nan,
        'spatial_extent_percentage': np.nan,

        # Hemispheres (integrated)
        'northern_hemisphere_flux': np.nan,
        'southern_hemisphere_flux': np.nan,
        'hemispheric_asymmetry_ratio': np.nan,
        'dayside_flux': np.nan,
        'nightside_flux': np.nan,
        'dayside_nightside_ratio': np.nan,
        'dawnside_flux': np.nan,
        'duskside_flux': np.nan,
        'dawn_dusk_ratio': np.nan,
        'dawn_dusk_asym_index': np.nan,

        # signed ratios
        'signed_ratio_north_south': np.nan,
        'signed_ratio_day_night': np.nan,
        'signed_ratio_dawn_dusk': np.nan,

        # Moments
        'mean_flux': np.nan,
        'median_flux': np.nan,

        # Threshold tracking
        'threshold_used': 0.0,
    }

    if np.all(flux_map == 0) or np.all(np.isnan(flux_map)):
        return stats

    flux_map_clean = np.nan_to_num(flux_map, nan=0.0)

    n_lat = len(lat_centers)
    n_lon = len(lon_centers)

    dlat = lat_centers[1] - lat_centers[0] if n_lat > 1 else 1.0
    dlon = lon_centers[1] - lon_centers[0] if n_lon > 1 else 1.0

    lat_edges = np.linspace(lat_centers[0] - dlat / 2, lat_centers[-1] + dlat / 2, n_lat + 1)
    lon_edges = np.linspace(lon_centers[0] - dlon / 2, lon_centers[-1] + dlon / 2, n_lon + 1)

    lat_edges_rad = np.radians(lat_edges)
    lon_edges_rad = np.radians(lon_edges)
    dlon_rad = lon_edges_rad[1] - lon_edges_rad[0]

    area_per_bin = np.zeros((n_lat, n_lon))
    for i in range(n_lat):
        area_per_bin[i, :] = R_M ** 2 * dlon_rad * (np.sin(lat_edges_rad[i + 1]) - np.sin(lat_edges_rad[i]))

    # Work in SI for integrated totals (m^-2 s^-1)
    flux_map_si = flux_map_clean * 1e4  # cm^-2 -> m^-2
    total_flux = np.sum(flux_map_si * area_per_bin)
    stats['total_integrated_flux'] = total_flux

    # Peak flux pixel (cm^-2 s^-1)
    peak_idx = np.unravel_index(np.nanargmax(flux_map), flux_map.shape)
    stats['peak_flux_value'] = float(flux_map[peak_idx])
    stats['peak_flux_lat'] = float(lat_centers[peak_idx[0]])
    stats['peak_flux_lon'] = float(lon_centers[peak_idx[1]])

    # Threshold: 5% of peak flux
    if flux_threshold is None:
        flux_threshold = 0.05 * stats['peak_flux_value']
    stats['threshold_used'] = float(flux_threshold)

    above_threshold = flux_map_clean >= flux_threshold
    area_above_threshold = np.sum(area_per_bin[above_threshold])
    total_surface_area = 4 * np.pi * R_M ** 2
    stats['spatial_extent_area'] = float(area_above_threshold)
    stats['spatial_extent_percentage'] = float((area_above_threshold / total_surface_area) * 100.0)

    lat_grid, lon_grid = np.meshgrid(lat_centers, lon_centers, indexing='ij')
    lat_grid_rad = np.radians(lat_grid)
    lon_grid_rad = np.radians(lon_grid)

    # North vs South
    northern_mask = lat_grid >= 0
    southern_mask = lat_grid < 0
    northern_flux = float(np.sum(flux_map_si[northern_mask] * area_per_bin[northern_mask]))
    southern_flux = float(np.sum(flux_map_si[southern_mask] * area_per_bin[southern_mask]))
    stats['northern_hemisphere_flux'] = northern_flux
    stats['southern_hemisphere_flux'] = southern_flux
    stats['hemispheric_asymmetry_ratio'] = (northern_flux / southern_flux) if southern_flux > 0 else (np.inf if northern_flux > 0 else np.nan)
    stats['signed_ratio_north_south'] = signed_ratio(northern_flux, southern_flux)  # (N-S)/(N+S)

    # Day vs Night: use x = cos(lat)*cos(lon), day if x>0 (subsolar hemisphere)
    x_grid = np.cos(lat_grid_rad) * np.cos(lon_grid_rad)
    dayside_mask = x_grid > 0
    nightside_mask = x_grid <= 0
    dayside_flux = float(np.sum(flux_map_si[dayside_mask] * area_per_bin[dayside_mask]))
    nightside_flux = float(np.sum(flux_map_si[nightside_mask] * area_per_bin[nightside_mask]))
    stats['dayside_flux'] = dayside_flux
    stats['nightside_flux'] = nightside_flux
    stats['dayside_nightside_ratio'] = (dayside_flux / nightside_flux) if nightside_flux > 0 else (np.inf if dayside_flux > 0 else np.nan)
    stats['signed_ratio_day_night'] = signed_ratio(dayside_flux, nightside_flux)  # (Day-Night)/(Day+Night)

    # Dawn vs Dusk: use y = cos(lat)*sin(lon)
    y_grid = np.cos(lat_grid_rad) * np.sin(lon_grid_rad)
    duskside_mask = y_grid > 0
    dawnside_mask = y_grid < 0
    duskside_flux = float(np.sum(flux_map_si[duskside_mask] * area_per_bin[duskside_mask]))
    dawnside_flux = float(np.sum(flux_map_si[dawnside_mask] * area_per_bin[dawnside_mask]))
    stats['duskside_flux'] = duskside_flux
    stats['dawnside_flux'] = dawnside_flux
    stats['dawn_dusk_ratio'] = (dawnside_flux / duskside_flux) if duskside_flux > 0 else (np.inf if dawnside_flux > 0 else np.nan)
    denom = dawnside_flux + duskside_flux
    stats['dawn_dusk_asym_index'] = (dawnside_flux - duskside_flux) / denom if denom > 0 else np.nan
    stats['signed_ratio_dawn_dusk'] = signed_ratio(dawnside_flux, duskside_flux)  # (Dawn-Dusk)/(Dawn+Dusk)

    nonzero_flux = flux_map_clean[flux_map_clean > 0]
    if len(nonzero_flux) > 0:
        stats['mean_flux'] = float(np.mean(nonzero_flux))
        stats['median_flux'] = float(np.median(nonzero_flux))

    return stats
