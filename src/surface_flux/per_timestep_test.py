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
species = np.array(['H+', 'H+', 'He++', 'He++'])
sim_ppc = np.array([24, 24, 11, 11], dtype=float)
sim_den = np.array([38.0e6, 76.0e6, 1.0e6, 2.0e6], dtype=float)
sim_vel = np.array([400.e3, 700.0e3, 400.e3, 700.0e3], dtype=float)
species_mass = np.array([1.0, 1.0, 4.0, 4.0], dtype=float)
species_charge = np.array([1.0, 1.0, 2.0, 2.0], dtype=float)

# Simulation grid and obstacle radius
sim_dx = 75.e3
sim_dy = 75.e3
sim_dz = 75.e3
R_M = 2440.e3
DELTA_R_M = 0.5 * sim_dx

# -----------------------
# SETTINGS
# -----------------------
npz_glob = "/Volumes/data_backup/mercury/extreme/High_HNHV/CPN_HNHV/precipitation_timeseries/*.npz"

out_dir = "/Users/danywaller/Projects/mercury/extreme/surface_flux_maps_test/"
os.makedirs(out_dir, exist_ok=True)

# Lat/lon bins (edges)
dlat = 1.0
dlon = 1.0
lat_bin_edges = np.arange(-90.0, 90.0 + dlat, dlat)
lon_bin_edges = np.arange(-180.0, 180.0 + dlon, dlon)

# Plot controls
plot_log10 = True
eps = 1e-30
CMAP = "inferno"
plot_log10_vmin = 9.0
plot_log10_vmax = 14.0
save_per_species = True

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


def flux_maps_snapshot_shell(npz_path, R, delta_r_m, lat_bin_edges, lon_bin_edges, W_by_sid):
    """
    Returns
    -------
    flux_by_sid : (Ns, Nlat, Nlon) #/m^2/s, NaN where count_by_sid==0
    flux_all    : (Nlat, Nlon)     #/m^2/s, NaN where count_all==0
    vrabs_map   : (Nlat, Nlon)     m/s, mean(|v_r|) where count_all>0 else NaN
    t           : float, seconds
    total_rate  : float, #/s
    total_rate_by_sid : (Ns,) float, #/s per SID
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

    # radial component (outward positive)
    vr = vx * nx + vy * ny + vz * nz
    vn_in = np.maximum(0.0, -vr)

    shell = (r > R) & (r < (R + delta_r_m)) & (vn_in > 0.0)

    A = bin_areas_sphere(R, lat_bin_edges, lon_bin_edges)

    if not np.any(shell):
        flux_by_sid = np.full((Ns, Nlat, Nlon), np.nan, dtype=float)
        flux_all = np.full((Nlat, Nlon), np.nan, dtype=float)
        vrabs_map = np.full((Nlat, Nlon), np.nan, dtype=float)
        total_rate_by_sid = np.zeros(Ns, dtype=float)
        return flux_by_sid, flux_all, vrabs_map, t, 0.0, total_rate_by_sid

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

    rate_by_sid = np.zeros((Ns, Nlat, Nlon), float)
    count_by_sid = np.zeros((Ns, Nlat, Nlon), dtype=np.int32)

    vrabs_sum_all = np.zeros((Nlat, Nlon), dtype=float)
    count_all = np.zeros((Nlat, Nlon), dtype=np.int32)

    for s in range(Ns):
        m = (sid_s == s)
        if not np.any(m):
            continue

        np.add.at(rate_by_sid[s], (ilat[m], ilon[m]),
                  W_by_sid[s] * (vn_in_s[m] / delta_r_m))
        np.add.at(count_by_sid[s], (ilat[m], ilon[m]), 1)

        np.add.at(vrabs_sum_all, (ilat[m], ilon[m]), vrabs_s[m])
        np.add.at(count_all, (ilat[m], ilon[m]), 1)

    flux_by_sid = rate_by_sid / A[None, :, :]
    flux_all_unmasked = np.sum(flux_by_sid, axis=0)

    # Integrated rates!
    total_rate_by_sid = np.sum(flux_by_sid * A[None, :, :], axis=(1, 2))
    total_rate = np.asarray(np.sum(flux_all_unmasked * A)).item()

    # mask zero-density bins as NaN for plotting
    count_all_from_sid = np.sum(count_by_sid, axis=0)

    flux_all = flux_all_unmasked.astype(float, copy=True)
    flux_all[count_all_from_sid == 0] = np.nan

    flux_by_sid = flux_by_sid.astype(float, copy=True)
    flux_by_sid[count_by_sid == 0] = np.nan

    vrabs_map = np.full((Nlat, Nlon), np.nan, dtype=float)
    mvr = (count_all > 0)
    vrabs_map[mvr] = vrabs_sum_all[mvr] / count_all[mvr]

    return flux_by_sid, flux_all, vrabs_map, t, total_rate, total_rate_by_sid


def _deg_edges_to_rad(lon_edges_deg, lat_edges_deg):
    lon_e = np.deg2rad(lon_edges_deg)
    lat_e = np.deg2rad(lat_edges_deg)
    # Hammer proj expects longitude in [-pi, pi]
    return lon_e, lat_e


def _set_hammer_degree_grid(ax, lon_step_deg=60, lat_step_deg=30):
    # ticks in radians on geo projections
    xt = np.deg2rad(np.arange(-150, 151, lon_step_deg))
    yt = np.deg2rad(np.arange(-60,  60,  lat_step_deg))
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([f"{d:d}°" for d in np.arange(-150, 151, lon_step_deg)])
    ax.set_yticklabels([f"{d:d}°" for d in np.arange(-60,  60,  lat_step_deg)])


def save_flux_map_png(outpath, lon_bin_edges, lat_bin_edges, flux2d, *,
                      title, plot_log10=True, eps=1e-30, cmap="inferno",
                      cbar_label=None, vmin=None, vmax=None):
    # NaN-safe log10: only transform finite cells
    if plot_log10:
        plot = np.full_like(flux2d, np.nan, dtype=float)
        m = np.isfinite(flux2d)
        plot[m] = np.log10(np.maximum(flux2d[m], eps))
    else:
        plot = flux2d

    lon_e_rad, lat_e_rad = _deg_edges_to_rad(lon_bin_edges, lat_bin_edges)

    fig, ax = plt.subplots(
        figsize=(10, 4.8),
        constrained_layout=True,
        subplot_kw={"projection": "hammer"}
    )

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

    fig, ax = plt.subplots(
        figsize=(10, 4.8),
        constrained_layout=True,
        subplot_kw={"projection": "hammer"}
    )

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
total_rates = []
total_rates_by_sid = []  # list of (Ns,) arrays

for i, f in enumerate(files):
    base = os.path.basename(f).replace(".npz", "")
    print(f"[{i+1}/{len(files)}] Processing {base} ...")

    flux_by_sid, flux_all, vrabs_map, t, total_rate, total_rate_by_sid = flux_maps_snapshot_shell(
        f, R=R_M, delta_r_m=DELTA_R_M,
        lat_bin_edges=lat_bin_edges,
        lon_bin_edges=lon_bin_edges,
        W_by_sid=W_by_sid
    )

    times.append(t)
    total_rates.append(total_rate)
    total_rates_by_sid.append(total_rate_by_sid)

    # Total flux map
    total_dir = os.path.join(out_dir, "total")
    os.makedirs(total_dir, exist_ok=True)
    out_total = os.path.join(total_dir, f"{base}_flux_total.png")
    save_flux_map_png(
        out_total, lon_bin_edges, lat_bin_edges, flux_all,
        title=f"Total (all) inward surface flux, t={t:.3f} s",
        plot_log10=plot_log10, eps=eps, cmap=CMAP,
        cbar_label="log10(#/m²/s)" if plot_log10 else "#/m²/s",
        vmin=plot_log10_vmin if plot_log10 else None,
        vmax=plot_log10_vmax if plot_log10 else None
    )

    # Binned mean |v_r| map
    vrabs_dir = os.path.join(out_dir, "radial_velocity")
    os.makedirs(vrabs_dir, exist_ok=True)
    out_vrabs = os.path.join(vrabs_dir, f"{base}_vrabs_mean.png")
    save_scalar_map_png(
        out_vrabs, lon_bin_edges, lat_bin_edges, vrabs_map*1e-3,  # convert m/s -> km/s
        title=f"Mean |v$_r$| in inward shell (bins with density>0), t={t:.3f} s",
        cmap="viridis",
        cbar_label=r"|v$_r$| [(km/s)]", vmin=0, vmax=500
    )

    # Per-species flux maps
    if save_per_species:
        for s in range(Ns):
            species_dir = os.path.join(out_dir, f"sid{s:02d}_{species[s]}")
            os.makedirs(species_dir, exist_ok=True)
            out_s = os.path.join(species_dir, f"{base}_flux_sid{s:02d}_{species[s]}.png")
            save_flux_map_png(
                out_s, lon_bin_edges, lat_bin_edges, flux_by_sid[s],
                title=f"{species[s]} ({s}) inward surface flux, t={t:.3f} s",
                plot_log10=plot_log10, eps=eps, cmap=CMAP,
                cbar_label="log10(#/m²/s)" if plot_log10 else "#/m²/s",
                vmin=plot_log10_vmin if plot_log10 else None,
                vmax=plot_log10_vmax if plot_log10 else None
            )

    # --- Combined flux maps ---
    # Sum like species
    # Use np.nansum so bins missing one component but present in the other stay finite.
    combos = [
        ((0, 1), "sid00_01_sum", f"{species[0]} (species00+species01)"),
        ((2, 3), "sid02_03_sum", f"{species[2]} (species02+species03)"),
    ]

    for (s0, s1), combo_dirname, combo_label in combos:
        combo_dir = os.path.join(out_dir, combo_dirname)
        os.makedirs(combo_dir, exist_ok=True)

        a = flux_by_sid[s0]
        b = flux_by_sid[s1]

        # Sum treating NaN as 0, but keep NaN where BOTH are NaN (i.e., truly no data)
        flux_combo = np.nan_to_num(a, nan=0.0) + np.nan_to_num(b, nan=0.0)
        nodata = ~np.isfinite(a) & ~np.isfinite(b)
        flux_combo[nodata] = np.nan

        out_combo = os.path.join(combo_dir, f"{base}_flux_{combo_dirname}.png")
        save_flux_map_png(
            out_combo, lon_bin_edges, lat_bin_edges, flux_combo,
            title=f"{combo_label} inward surface flux, t={t:.3f} s",
            plot_log10=plot_log10, eps=eps, cmap=CMAP,
            cbar_label="log10(#/m²/s)" if plot_log10 else "#/m²/s",
            vmin=plot_log10_vmin if plot_log10 else None,
            vmax=plot_log10_vmax if plot_log10 else None
        )


# --- time series: total + per SID ---
times = np.asarray(times)
total_rates = np.asarray(total_rates)
total_rates_by_sid = np.asarray(total_rates_by_sid)  # (Nt, Ns)

order = np.argsort(times)
times = times[order]
total_rates = total_rates[order]
total_rates_by_sid = total_rates_by_sid[order]

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.plot(times, total_rates, lw=2.5, color="k", label="Total")

for s in range(Ns):
    ax.plot(times, total_rates_by_sid[:, s], lw=1.8, label=f"Species {s}: {species[s]}")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Integrated inward rate [log10(#/s)]")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.set_title("Total + per-species precipitation rate vs time")
ax.legend(ncol=2, fontsize=9, frameon=False)

out_ts = os.path.join(out_dir, "total_and_perSID_surface_precip_rate_vs_time.png")
fig.savefig(out_ts, dpi=250)
plt.show()

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

# Set log scale before scatter for autoscaling
ax.set_yscale("log")

# Total (all species IDs)
ax.scatter(times, total_rates, s=18, color="k", label="Total")

# Per-species ID
for s in range(Ns):
    ax.scatter(times, total_rates_by_sid[:, s], s=14, label=f"Species {s}: {species[s]}")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Integrated inward rate [log10(#/s)]")
ax.grid(True, alpha=0.3)
ax.set_title("Total + per-species precipitation rate vs time")
ax.legend(ncol=2, fontsize=9, frameon=False)

out_ts = os.path.join(out_dir, "total_and_perSID_surface_precip_rate_vs_time_scatter.png")
fig.savefig(out_ts, dpi=250)
plt.show()

print(f"Done. Wrote maps + timeseries to:\n  {out_dir}")
