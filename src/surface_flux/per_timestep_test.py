#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports:
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Upstream conditions
# -----------------------
case = "RPN_HNHV"
species = np.array(['H+', 'H+', 'He++', 'He++'])
sim_ppc = np.array([24, 24, 11, 11], dtype=float)  # macroparticles per cell
sim_den = np.array([38.0e6, 76.0e6, 1.0e6, 2.0e6], dtype=float)  # particles / m^3
sim_vel = np.array([400.e3, 700.0e3, 400.e3, 700.0e3], dtype=float)  # m/s
species_mass = np.array([1.0, 1.0, 4.0, 4.0], dtype=float)
species_charge = np.array([1.0, 1.0, 2.0, 2.0], dtype=float)

AMU = 1.66053906660e-27  # kg
QE  = 1.602176634e-19    # J/eV

# Particle masses per SID
m_kg_by_sid = species_mass * AMU                 # (Ns,)

# Simulation grid and obstacle radius
sim_dx = 75.e3
sim_dy = 75.e3
sim_dz = 75.e3
R_M = 2440.e3
DELTA_R_M = 0.5 * sim_dx

# -----------------------
# SETTINGS
# -----------------------
npz_glob = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/precipitation_timeseries/*.npz"

out_dir = f"/Users/danywaller/Projects/mercury/extreme/surface_flux_maps_test/{case}/"
os.makedirs(out_dir, exist_ok=True)

# Lat/lon bins (edges)
dlat = 1.0
dlon = 1.0
lat_bin_edges = np.arange(-90.0, 90.0 + dlat, dlat)
lon_bin_edges = np.arange(-180.0, 180.0 + dlon, dlon)

lat_centers = 0.5 * (lat_bin_edges[:-1] + lat_bin_edges[1:])
lon_centers = 0.5 * (lon_bin_edges[:-1] + lon_bin_edges[1:])

# Plot controls
plot_log10 = True
eps = 1e-30
CMAP = "jet"
save_per_species = True

# Fixed log10 color scales (global)
NF_VMIN, NF_VMAX = 10.0, 14.0    # log10(#/m^2/s)
MF_VMIN, MF_VMAX = -17.5, -12.5  # log10(kg/m^2/s)
EF_VMIN, EF_VMAX = -6.5, -1.5    # log10(W/m^2)

# Fixed log10 color scales (protons vs alphas combos)
NF_PROTON_VMIN, NF_PROTON_VMAX = 10.0, 14.0
NF_ALPHA_VMIN,  NF_ALPHA_VMAX  = 9.0, 13.0

MF_PROTON_VMIN, MF_PROTON_VMAX = -18.5, -12.5
MF_ALPHA_VMIN,  MF_ALPHA_VMAX  = -16.5, -13.5

EF_PROTON_VMIN, EF_PROTON_VMAX = -5.5,  -2.5
EF_ALPHA_VMIN,  EF_ALPHA_VMAX  = -6.5,  -1.5

# For mass/energy maps
mass_eps = 1e-60
energy_eps = 1e-60

# -----------------------
# Map stats function
# -----------------------
def compute_flux_statistics(flux_map, lat_centers, lon_centers, R_M,
                            flux_threshold=None, case_name="Case", debug=False):
    """
    Compute summary statistics for a georeferenced flux map.

    flux_map : [cm^-2 s^-1]
    Returns integrated fluxes in [#/s] using proper spherical geometry.
    """
    stats = {
        'case_name': case_name,
        'total_integrated_flux': 0.0,
        'peak_flux_value': 0.0,
        'peak_flux_lat': 0.0,
        'peak_flux_lon': 0.0,
        'spatial_extent_area': 0.0,
        'spatial_extent_percentage': 0.0,
        'northern_hemisphere_flux': 0.0,
        'southern_hemisphere_flux': 0.0,
        'hemispheric_asymmetry_ratio': 0.0,
        'dayside_flux': 0.0,
        'nightside_flux': 0.0,
        'dayside_nightside_ratio': 0.0,
        'dawnside_flux': 0.0,
        'duskside_flux': 0.0,
        'dawn_dusk_ratio': 0.0,
        'dawn_dusk_asym_index': 0.0,
        'mean_flux': 0.0,
        'median_flux': 0.0,
        'threshold_used': 0.0
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

    flux_map_si = flux_map_clean * 1e4  # cm^-2 -> m^-2
    total_flux = np.sum(flux_map_si * area_per_bin)
    stats['total_integrated_flux'] = total_flux

    peak_idx = np.unravel_index(np.nanargmax(flux_map), flux_map.shape)
    stats['peak_flux_value'] = flux_map[peak_idx]
    stats['peak_flux_lat'] = lat_centers[peak_idx[0]]
    stats['peak_flux_lon'] = lon_centers[peak_idx[1]]

    if flux_threshold is None:
        flux_threshold = 0.001 * stats['peak_flux_value']
    stats['threshold_used'] = flux_threshold

    above_threshold = flux_map_clean >= flux_threshold
    area_above_threshold = np.sum(area_per_bin[above_threshold])
    total_surface_area = 4 * np.pi * R_M ** 2
    stats['spatial_extent_area'] = area_above_threshold
    stats['spatial_extent_percentage'] = (area_above_threshold / total_surface_area) * 100

    lat_grid, lon_grid = np.meshgrid(lat_centers, lon_centers, indexing='ij')
    lat_grid_rad = np.radians(lat_grid)
    lon_grid_rad = np.radians(lon_grid)

    northern_mask = lat_grid >= 0
    southern_mask = lat_grid < 0
    northern_flux = np.sum(flux_map_si[northern_mask] * area_per_bin[northern_mask])
    southern_flux = np.sum(flux_map_si[southern_mask] * area_per_bin[southern_mask])
    stats['northern_hemisphere_flux'] = northern_flux
    stats['southern_hemisphere_flux'] = southern_flux
    stats['hemispheric_asymmetry_ratio'] = (northern_flux / southern_flux) if southern_flux > 0 else (np.inf if northern_flux > 0 else 0.0)

    x_grid = np.cos(lat_grid_rad) * np.cos(lon_grid_rad)
    dayside_mask = x_grid > 0
    nightside_mask = x_grid <= 0
    dayside_flux = np.sum(flux_map_si[dayside_mask] * area_per_bin[dayside_mask])
    nightside_flux = np.sum(flux_map_si[nightside_mask] * area_per_bin[nightside_mask])
    stats['dayside_flux'] = dayside_flux
    stats['nightside_flux'] = nightside_flux
    stats['dayside_nightside_ratio'] = (dayside_flux / nightside_flux) if nightside_flux > 0 else (np.inf if dayside_flux > 0 else 0.0)

    y_grid = np.cos(lat_grid_rad) * np.sin(lon_grid_rad)
    duskside_mask = y_grid > 0
    dawnside_mask = y_grid < 0
    duskside_flux = np.sum(flux_map_si[duskside_mask] * area_per_bin[duskside_mask])
    dawnside_flux = np.sum(flux_map_si[dawnside_mask] * area_per_bin[dawnside_mask])
    stats['duskside_flux'] = duskside_flux
    stats['dawnside_flux'] = dawnside_flux
    stats['dawn_dusk_ratio'] = (dawnside_flux / duskside_flux) if duskside_flux > 0 else (np.inf if dawnside_flux > 0 else 0.0)
    denom = dawnside_flux + duskside_flux
    stats['dawn_dusk_asym_index'] = (dawnside_flux - duskside_flux) / denom if denom > 0 else 0.0

    nonzero_flux = flux_map_clean[flux_map_clean > 0]
    if len(nonzero_flux) > 0:
        stats['mean_flux'] = np.mean(nonzero_flux)
        stats['median_flux'] = np.median(nonzero_flux)

    return stats


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


def _deg_edges_to_rad(lon_edges_deg, lat_edges_deg):
    lon_e = np.deg2rad(lon_edges_deg)
    lat_e = np.deg2rad(lat_edges_deg)
    return lon_e, lat_e


def _set_hammer_degree_grid(ax, lon_step_deg=60, lat_step_deg=30):
    xt = np.deg2rad(np.arange(-150, 151, lon_step_deg))
    yt = np.deg2rad(np.arange(-60,  60,  lat_step_deg))
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([f"{d:d}°" for d in np.arange(-150, 151, lon_step_deg)])
    ax.set_yticklabels([f"{d:d}°" for d in np.arange(-60,  60,  lat_step_deg)])


def save_flux_map_png(outpath, lon_bin_edges, lat_bin_edges, flux2d, *,
                      title, plot_log10=True, eps=1e-30, cmap="inferno",
                      cbar_label=None, vmin=None, vmax=None):
    if plot_log10:
        plot = np.full_like(flux2d, np.nan, dtype=float)
        m = np.isfinite(flux2d)
        plot[m] = np.log10(np.maximum(flux2d[m], eps))
    else:
        plot = flux2d

    lon_e_rad, lat_e_rad = _deg_edges_to_rad(lon_bin_edges, lat_bin_edges)

    fig, ax = plt.subplots(figsize=(10, 4.8),
                           constrained_layout=True,
                           subplot_kw={"projection": "hammer"})

    pm = ax.pcolormesh(lon_e_rad, lat_e_rad, plot, shading="flat",
                       cmap=cmap, vmin=vmin, vmax=vmax)

    _set_hammer_degree_grid(ax, lon_step_deg=60, lat_step_deg=30)
    ax.grid(True, alpha=0.35)
    ax.set_title(title)

    cb = fig.colorbar(pm, ax=ax, pad=0.03, shrink=0.9)
    if cbar_label is None:
        cbar_label = "log10(#/m²/s)" if plot_log10 else "#/m²/s"
    cb.set_label(cbar_label)

    fig.savefig(outpath, dpi=250)
    plt.close(fig)


def save_scalar_map_png(outpath, lon_bin_edges, lat_bin_edges, field2d, *,
                        title, cmap="viridis", cbar_label=None, vmin=None, vmax=None):
    lon_e_rad, lat_e_rad = _deg_edges_to_rad(lon_bin_edges, lat_bin_edges)

    fig, ax = plt.subplots(figsize=(10, 4.8),
                           constrained_layout=True,
                           subplot_kw={"projection": "hammer"})

    pm = ax.pcolormesh(lon_e_rad, lat_e_rad, field2d, shading="flat",
                       cmap=cmap, vmin=vmin, vmax=vmax)

    _set_hammer_degree_grid(ax, lon_step_deg=60, lat_step_deg=30)
    ax.grid(True, alpha=0.35)
    ax.set_title(title)

    cb = fig.colorbar(pm, ax=ax, pad=0.03, shrink=0.9)
    if cbar_label is not None:
        cb.set_label(cbar_label)

    fig.savefig(outpath, dpi=250)
    plt.close(fig)


def save_triptych_timeseries(outpath, times, series_list, titles, ylabels, *,
                             scatter=False, logy=True, legend_ncol=2, suptitle=None):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)

    for j, ax in enumerate(axes):
        if logy:
            ax.set_yscale("log")

        total = series_list[j]["total"]
        by_sid = series_list[j]["by_sid"]
        species_lbl = series_list[j]["species"]

        if scatter:
            ax.scatter(times, total, s=18, color="k", label="Total")
            for s in range(by_sid.shape[1]):
                ax.scatter(times, by_sid[:, s], s=14, label=f"Species {s}: {species_lbl[s]}")
        else:
            ax.plot(times, total, lw=2.5, color="k", label="Total")
            for s in range(by_sid.shape[1]):
                ax.plot(times, by_sid[:, s], lw=1.8, label=f"Species {s}: {species_lbl[s]}")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabels[j])
        ax.grid(True, alpha=0.3)
        ax.set_title(titles[j])

        if j == 2:
            ax.legend(ncol=legend_ncol, fontsize=9, frameon=False)

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.savefig(outpath, dpi=250)
    plt.show()


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
# Main
# -----------------------
files = sorted(glob.glob(npz_glob))
if len(files) == 0:
    raise FileNotFoundError(f"No files matched: {npz_glob}")

V_cell = sim_dx * sim_dy * sim_dz
W_by_sid = macro_weights(sim_den, sim_ppc, V_cell)
Ns = len(W_by_sid)

times = []

# integrated number flux (#/s)
total_rates = []
total_rates_by_sid = []

# integrated mass flux (kg/s)
total_mass_rates = []
total_mass_rates_by_sid = []

# integrated energy flux (W)
total_powers = []
total_powers_by_sid = []

# Map-statistics time series (number flux only)
stats_keys = [
    "dayside_nightside_ratio",
    "dawn_dusk_ratio",
    "dawn_dusk_asym_index",
    "hemispheric_asymmetry_ratio",
    "spatial_extent_percentage",
    "peak_flux_value",
]
stats_total_ts = {k: [] for k in stats_keys}
stats_by_sid_ts = {k: [] for k in stats_keys}

for i, f in enumerate(files):
    base = os.path.basename(f).replace(".npz", "")
    print(f"[{i+1}/{len(files)}] Processing {base} ...")

    (flux_by_sid, flux_all,
     mass_flux_by_sid, mass_flux_all,
     energy_flux_by_sid, energy_flux_all,
     vrabs_map, t,
     total_rate, total_rate_by_sid,
     total_mass_rate, total_mass_rate_by_sid,
     total_power, total_power_by_sid) = flux_maps_snapshot_shell(
        f, R=R_M, delta_r_m=DELTA_R_M,
        lat_bin_edges=lat_bin_edges,
        lon_bin_edges=lon_bin_edges,
        W_by_sid=W_by_sid,
        m_kg_by_sid=m_kg_by_sid
    )

    times.append(t)

    total_rates.append(total_rate)
    total_rates_by_sid.append(total_rate_by_sid)

    total_mass_rates.append(total_mass_rate)
    total_mass_rates_by_sid.append(total_mass_rate_by_sid)

    total_powers.append(total_power)
    total_powers_by_sid.append(total_power_by_sid)

    # Mask zeros -> NaN BEFORE stats, then convert to cm^-2 s^-1
    flux_all_for_stats_cm = mask_zeros_to_nan(flux_all) * 1e-4
    stats_total = compute_flux_statistics(
        flux_all_for_stats_cm, lat_centers, lon_centers, R_M,
        flux_threshold=None, case_name=f"{case}_{base}_TOTAL", debug=False
    )
    for k in stats_keys:
        stats_total_ts[k].append(stats_total.get(k, np.nan))

    sid_vals = {k: np.full(Ns, np.nan, dtype=float) for k in stats_keys}
    for s in range(Ns):
        fm_cm = mask_zeros_to_nan(flux_by_sid[s]) * 1e-4
        st = compute_flux_statistics(
            fm_cm, lat_centers, lon_centers, R_M,
            flux_threshold=None, case_name=f"{case}_{base}_SID{s:02d}", debug=False
        )
        for k in stats_keys:
            sid_vals[k][s] = st.get(k, np.nan)

    for k in stats_keys:
        stats_by_sid_ts[k].append(sid_vals[k])

    # -----------------------
    # Per-timestep plots: TOTAL maps
    # -----------------------
    total_dir = os.path.join(out_dir, "total")
    os.makedirs(total_dir, exist_ok=True)
    out_total = os.path.join(total_dir, f"{base}_flux_total.png")
    save_flux_map_png(
        out_total, lon_bin_edges, lat_bin_edges, flux_all,
        title=f"{case.replace('_', ' ')} Total (all) inward surface number flux, t={t:.3f} s",
        plot_log10=True, eps=eps, cmap=CMAP,
        cbar_label="log10(#/m²/s)",
        vmin=NF_VMIN, vmax=NF_VMAX
    )

    mass_total_dir = os.path.join(out_dir, "mass_total")
    os.makedirs(mass_total_dir, exist_ok=True)
    out_mass_total = os.path.join(mass_total_dir, f"{base}_mass_flux_total.png")
    save_flux_map_png(
        out_mass_total, lon_bin_edges, lat_bin_edges, mass_flux_all,
        title=f"{case.replace('_', ' ')} Total inward surface mass flux, t={t:.3f} s",
        plot_log10=True, eps=mass_eps, cmap="magma",
        cbar_label="log10(kg/m²/s)",
        vmin=MF_VMIN, vmax=MF_VMAX
    )

    energy_total_dir = os.path.join(out_dir, "energy_total")
    os.makedirs(energy_total_dir, exist_ok=True)
    out_energy_total = os.path.join(energy_total_dir, f"{base}_energy_flux_total.png")
    save_flux_map_png(
        out_energy_total, lon_bin_edges, lat_bin_edges, energy_flux_all,
        title=f"{case.replace('_', ' ')} Total inward surface energy flux (using vn_in), t={t:.3f} s",
        plot_log10=True, eps=energy_eps, cmap="plasma",
        cbar_label="log10(W/m²)",
        vmin=EF_VMIN, vmax=EF_VMAX
    )

    # Per-SID plots
    if save_per_species:
        for s in range(Ns):
            species_dir = os.path.join(out_dir, f"sid{s:02d}_{species[s]}")
            os.makedirs(species_dir, exist_ok=True)
            out_s = os.path.join(species_dir, f"{base}_flux_sid{s:02d}_{species[s]}.png")
            save_flux_map_png(
                out_s, lon_bin_edges, lat_bin_edges, flux_by_sid[s],
                title=f"{case.replace('_', ' ')} {species[s]} ({s}) inward surface number flux, t={t:.3f} s",
                plot_log10=True, eps=eps, cmap=CMAP,
                cbar_label="log10(#/m²/s)",
                vmin=NF_VMIN, vmax=NF_VMAX
            )

            mass_dir = os.path.join(out_dir, f"mass_sid{s:02d}_{species[s]}")
            os.makedirs(mass_dir, exist_ok=True)
            out_ms = os.path.join(mass_dir, f"{base}_mass_flux_sid{s:02d}_{species[s]}.png")
            save_flux_map_png(
                out_ms, lon_bin_edges, lat_bin_edges, mass_flux_by_sid[s],
                title=f"{case.replace('_', ' ')} {species[s]} ({s}) inward surface mass flux, t={t:.3f} s",
                plot_log10=True, eps=mass_eps, cmap="magma",
                cbar_label="log10(kg/m²/s)",
                vmin=MF_VMIN, vmax=MF_VMAX
            )

            energy_dir = os.path.join(out_dir, f"energy_sid{s:02d}_{species[s]}")
            os.makedirs(energy_dir, exist_ok=True)
            out_es = os.path.join(energy_dir, f"{base}_energy_flux_sid{s:02d}_{species[s]}.png")
            save_flux_map_png(
                out_es, lon_bin_edges, lat_bin_edges, energy_flux_by_sid[s],
                title=f"{case.replace('_', ' ')} {species[s]} ({s}) inward surface energy flux (vn_in), t={t:.3f} s",
                plot_log10=True, eps=energy_eps, cmap="plasma",
                cbar_label="log10(W/m²)",
                vmin=EF_VMIN, vmax=EF_VMAX
            )

    # Combo maps (with separate nf/mf/ef dirs)
    combos = [
        ((0, 1), "protons_sid00_01_sum", "H+ (sid00+sid01)",
         (NF_PROTON_VMIN, NF_PROTON_VMAX),
         (MF_PROTON_VMIN, MF_PROTON_VMAX),
         (EF_PROTON_VMIN, EF_PROTON_VMAX)),
        ((2, 3), "alphas_sid02_03_sum", "He++ (sid02+sid03)",
         (NF_ALPHA_VMIN, NF_ALPHA_VMAX),
         (MF_ALPHA_VMIN, MF_ALPHA_VMAX),
         (EF_ALPHA_VMIN, EF_ALPHA_VMAX)),
    ]

    for (s0, s1), combo_dirname, combo_label, (nf_vmin, nf_vmax), (mf_vmin, mf_vmax), (ef_vmin, ef_vmax) in combos:
        combo_root = os.path.join(out_dir, combo_dirname)

        nf_dir = os.path.join(combo_root, "nf")
        mf_dir = os.path.join(combo_root, "mf")
        ef_dir = os.path.join(combo_root, "ef")

        os.makedirs(nf_dir, exist_ok=True)
        os.makedirs(mf_dir, exist_ok=True)
        os.makedirs(ef_dir, exist_ok=True)

        nf_combo = nan_safe_sum2(flux_by_sid[s0], flux_by_sid[s1])
        out_nf = os.path.join(nf_dir, f"{base}_number_flux_{combo_dirname}.png")
        save_flux_map_png(
            out_nf, lon_bin_edges, lat_bin_edges, nf_combo,
            title=f"{case.replace('_', ' ')} {combo_label} inward surface number flux, t={t:.3f} s",
            plot_log10=True, eps=eps, cmap=CMAP,
            cbar_label="log10(#/m²/s)",
            vmin=nf_vmin, vmax=nf_vmax
        )

        mf_combo = nan_safe_sum2(mass_flux_by_sid[s0], mass_flux_by_sid[s1])
        out_mf = os.path.join(mf_dir, f"{base}_mass_flux_{combo_dirname}.png")
        save_flux_map_png(
            out_mf, lon_bin_edges, lat_bin_edges, mf_combo,
            title=f"{case.replace('_', ' ')} {combo_label} inward surface mass flux, t={t:.3f} s",
            plot_log10=True, eps=mass_eps, cmap="magma",
            cbar_label="log10(kg/m²/s)",
            vmin=mf_vmin, vmax=mf_vmax
        )

        ef_combo = nan_safe_sum2(energy_flux_by_sid[s0], energy_flux_by_sid[s1])
        out_ef = os.path.join(ef_dir, f"{base}_energy_flux_{combo_dirname}.png")
        save_flux_map_png(
            out_ef, lon_bin_edges, lat_bin_edges, ef_combo,
            title=f"{case.replace('_', ' ')} {combo_label} inward surface energy flux (vn_in), t={t:.3f} s",
            plot_log10=True, eps=energy_eps, cmap="plasma",
            cbar_label="log10(W/m²)",
            vmin=ef_vmin, vmax=ef_vmax
        )

    # vrabs map
    vrabs_dir = os.path.join(out_dir, "radial_velocity")
    os.makedirs(vrabs_dir, exist_ok=True)
    out_vrabs = os.path.join(vrabs_dir, f"{base}_vrabs_mean.png")
    save_scalar_map_png(
        out_vrabs, lon_bin_edges, lat_bin_edges, vrabs_map * 1e-3,
        title=f"{case.replace('_', ' ')} Mean |v$_r$| in inward shell (bins with density>0), t={t:.3f} s",
        cmap="viridis",
        cbar_label=r"|v$_r$| [(km/s)]", vmin=0, vmax=500
    )

# -----------------------
# time series: sort + plots
# -----------------------
times = np.asarray(times)

total_rates = np.asarray(total_rates)
total_rates_by_sid = np.asarray(total_rates_by_sid)

total_mass_rates = np.asarray(total_mass_rates)
total_mass_rates_by_sid = np.asarray(total_mass_rates_by_sid)

total_powers = np.asarray(total_powers)
total_powers_by_sid = np.asarray(total_powers_by_sid)

for k in stats_keys:
    stats_total_ts[k] = np.asarray(stats_total_ts[k])
    stats_by_sid_ts[k] = np.asarray(stats_by_sid_ts[k])

order = np.argsort(times)
times = times[order]

total_rates = total_rates[order]
total_rates_by_sid = total_rates_by_sid[order]

total_mass_rates = total_mass_rates[order]
total_mass_rates_by_sid = total_mass_rates_by_sid[order]

total_powers = total_powers[order]
total_powers_by_sid = total_powers_by_sid[order]

for k in stats_keys:
    stats_total_ts[k] = stats_total_ts[k][order]
    stats_by_sid_ts[k] = stats_by_sid_ts[k][order]

# 1x3 line
out_line = os.path.join(out_dir, f"{case}_integrated_number_mass_energy_vs_time_LINE.png")
save_triptych_timeseries(
    out_line, times,
    series_list=[
        dict(total=total_rates,      by_sid=total_rates_by_sid,      species=species),
        dict(total=total_mass_rates, by_sid=total_mass_rates_by_sid, species=species),
        dict(total=total_powers,     by_sid=total_powers_by_sid,     species=species),
    ],
    titles=[
        "Integrated number precipitation rate vs time",
        "Integrated mass precipitation rate vs time",
        "Integrated energy precipitation power vs time",
    ],
    ylabels=[
        "Integrated inward rate [#/s] (log)",
        "Integrated inward mass rate [kg/s] (log)",
        "Integrated inward power [W] (log)",
    ],
    scatter=False, logy=True, legend_ncol=2,
    suptitle=f"{case.replace('_',' ')}: integrated number/mass/energy vs time"
)

# 1x3 scatter
out_scatter = os.path.join(out_dir, f"{case}_integrated_number_mass_energy_vs_time_SCATTER.png")
save_triptych_timeseries(
    out_scatter, times,
    series_list=[
        dict(total=total_rates,      by_sid=total_rates_by_sid,      species=species),
        dict(total=total_mass_rates, by_sid=total_mass_rates_by_sid, species=species),
        dict(total=total_powers,     by_sid=total_powers_by_sid,     species=species),
    ],
    titles=[
        "Integrated number precipitation rate vs time (scatter)",
        "Integrated mass precipitation rate vs time (scatter)",
        "Integrated energy precipitation power vs time (scatter)",
    ],
    ylabels=[
        "Integrated inward rate [#/s] (log)",
        "Integrated inward mass rate [kg/s] (log)",
        "Integrated inward power [W] (log)",
    ],
    scatter=True, logy=True, legend_ncol=2,
    suptitle=f"{case.replace('_',' ')}: integrated number/mass/energy vs time"
)

# -----------------------
# hemispheric_asymmetry_ratio, dayside_nightside_ratio, dawn_dusk_asym_index
# Total + per-SID lines (same style as existing)
# -----------------------
stats_triptych = [
    ("hemispheric_asymmetry_ratio", "Hemispheric asymmetry index", "North/South Ratio"),
    ("dayside_nightside_ratio", "Dayside/Nightside asymmetry index", "Day/Night Ratio"),
    ("dawn_dusk_asym_index", "Dawn/Dusk asymmetry index", "Dawn/Dusk Ratio"),
]

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)

for j, (k, title, ylabel) in enumerate(stats_triptych):
    ax = axes[j]

    # ax.plot(times, stats_total_ts[k], lw=2.5, color="k", label="Total")
    for s in range(Ns):
        ax.plot(times, stats_by_sid_ts[k][:, s], lw=1.8, label=f"Species {s}: {species[s]}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    # Put legend only on the last panel (keeps the other two clean)
    if j == 2:
        ax.legend(ncol=2, fontsize=9, frameon=False)
        ax.set_ylim([-1.5, 1.5])

fig.suptitle(f"{case.replace('_',' ')}")
out_stats_triptych = os.path.join(
    out_dir, f"{case}_number_flux_stats_triptych_hemi_daynight_dawndusk_LINE.png"
)
fig.savefig(out_stats_triptych, dpi=250)
plt.show()

print(f"Done. Wrote maps + timeseries + stats to:\n  {out_dir}")
