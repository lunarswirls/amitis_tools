#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports:
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import src.surface_flux.flux_utils as flux_utils

# -----------------------
# Upstream conditions
# -----------------------
case = "CPN_HNHV"
species = np.array(['H+', 'H+', 'He++', 'He++'])
sim_ppc = np.array([24, 24, 11, 11], dtype=float)  # macroparticles per cell
sim_den = np.array([38.0e6, 76.0e6, 1.0e6, 2.0e6], dtype=float)  # particles/m^3
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
# Helpers
# -----------------------
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


# -----------------------
# Main
# -----------------------
files = sorted(glob.glob(npz_glob))
if len(files) == 0:
    raise FileNotFoundError(f"No files matched: {npz_glob}")

V_cell = sim_dx * sim_dy * sim_dz
W_by_sid = flux_utils.macro_weights(sim_den, sim_ppc, V_cell)
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

# Map-statistics time series
stats_keys = [
    # Signed ratios
    "signed_ratio_day_night",
    "signed_ratio_north_south",
    "signed_ratio_dawn_dusk",

    # Peak + location
    "peak_flux_value",
    "peak_flux_lat",
    "peak_flux_lon",

    # Percent of total surface area receiving > 5% peak
    "spatial_extent_percentage",
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
     total_power, total_power_by_sid) = flux_utils.flux_maps_snapshot_shell(
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
    flux_all_for_stats_cm = flux_utils.mask_zeros_to_nan(flux_all) * 1e-4
    stats_total = flux_utils.compute_flux_statistics(
        flux_all_for_stats_cm, lat_centers, lon_centers, R_M,
        flux_threshold=None, case_name=f"{case}_{base}_TOTAL"
    )
    for k in stats_keys:
        stats_total_ts[k].append(stats_total.get(k, np.nan))

    sid_vals = {k: np.full(Ns, np.nan, dtype=float) for k in stats_keys}
    for s in range(Ns):
        fm_cm = flux_utils.mask_zeros_to_nan(flux_by_sid[s]) * 1e-4
        st = flux_utils.compute_flux_statistics(
            fm_cm, lat_centers, lon_centers, R_M,
            flux_threshold=None, case_name=f"{case}_{base}_SID{s:02d}"
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

        nf_combo = flux_utils.nan_safe_sum2(flux_by_sid[s0], flux_by_sid[s1])
        out_nf = os.path.join(nf_dir, f"{base}_number_flux_{combo_dirname}.png")
        save_flux_map_png(
            out_nf, lon_bin_edges, lat_bin_edges, nf_combo,
            title=f"{case.replace('_', ' ')} {combo_label} surface number flux, t={t:.3f} s",
            plot_log10=True, eps=eps, cmap=CMAP,
            cbar_label="log10(#/m²/s)",
            vmin=nf_vmin, vmax=nf_vmax
        )

        mf_combo = flux_utils.nan_safe_sum2(mass_flux_by_sid[s0], mass_flux_by_sid[s1])
        out_mf = os.path.join(mf_dir, f"{base}_mass_flux_{combo_dirname}.png")
        save_flux_map_png(
            out_mf, lon_bin_edges, lat_bin_edges, mf_combo,
            title=f"{case.replace('_', ' ')} {combo_label} surface mass flux, t={t:.3f} s",
            plot_log10=True, eps=mass_eps, cmap="magma",
            cbar_label="log10(kg/m²/s)",
            vmin=mf_vmin, vmax=mf_vmax
        )

        ef_combo = flux_utils.nan_safe_sum2(energy_flux_by_sid[s0], energy_flux_by_sid[s1])
        out_ef = os.path.join(ef_dir, f"{base}_energy_flux_{combo_dirname}.png")
        save_flux_map_png(
            out_ef, lon_bin_edges, lat_bin_edges, ef_combo,
            title=f"{case.replace('_', ' ')} {combo_label} surface energy flux, t={t:.3f} s",
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
        title=f"{case.replace('_', ' ')} Mean |v$_r$| in bins with density>0, t={t:.3f} s",
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
        "Integrated number precipitation rate",
        "Integrated mass precipitation rate",
        "Integrated energy precipitation power",
    ],
    ylabels=[
        "log10([#/s])",
        "log10([kg/s])",
        "log10([W])",
    ],
    scatter=False, logy=True, legend_ncol=2,
    suptitle=f"{case.split('_')[0]}: integrated number/mass/energy"
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
        "Integrated number precipitation rate",
        "Integrated mass precipitation rate",
        "Integrated energy precipitation power",
    ],
    ylabels=[
        "log10([#/s])",
        "log10([kg/s])",
        "log10([W])",
    ],
    scatter=True, logy=True, legend_ncol=2,
    suptitle=f"{case.split('_')[0]}: integrated number/mass/energy"
)

# -----------------------
# hemispheric_asymmetry_ratio, dayside_nightside_ratio, dawn_dusk_asym_index
# Total + per-SID lines (same style as existing)
# -----------------------
stats_triptych = [
    ("signed_ratio_north_south", "North/South precipitation ratio", "(N-S)/(N+S)"),
    ("signed_ratio_day_night",   "Day/Night precipitation ratio", "(Day-Night)/(Day+Night)"),
    ("signed_ratio_dawn_dusk",   "Dawn/Dusk precipitation ratio", "(Dawn-Dusk)/(Dawn+Dusk)"),
]

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)

for j, (k, title, ylabel) in enumerate(stats_triptych):
    ax = axes[j]

    # for s in range(Ns):
        # ax.plot(times, stats_by_sid_ts[k][:, s], lw=1.8, alpha=0.7, label=f"Species {s}: {species[s]}")

    # ax.plot(times, stats_total_ts[k], lw=2.5, color="k", label="Total")

    for s in range(Ns):
        ax.scatter(times, stats_by_sid_ts[k][:, s], s=14, alpha=0.7, label=f"Species {s}: {species[s]}")
    # ax.scatter(times, stats_total_ts[k], s=18, color="k", label="Total")
    ax.plot(times, stats_total_ts[k], lw=2.5, color="k", label="Total")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    # Put legend only on one panel (keeps the other two clean)
    if j == 0:
        ax.legend(ncol=2, fontsize=9, loc="upper left")

for ax in axes:
    ax.set_ylim([-1.05, 1.05])

fig.suptitle(f"{case.split('_')[0]}", fontsize=18)
out_stats_triptych = os.path.join(
    out_dir, f"{case}_number_flux_stats_triptych_hemi_daynight_dawndusk_total.png"
)
fig.savefig(out_stats_triptych, dpi=250)
plt.show()

# -----------------------
# peak flux value, peak lat, peak lon, percent area > 5% max
# -----------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), constrained_layout=True)

panels = [
    ("peak_flux_value",          "Peak Precipitation",          "Peak flux [cm⁻² s⁻¹]", True),
    # ("peak_flux_lat",            "Peak latitude vs time",              "Peak lat [deg]",      False),
    # ("peak_flux_lon",            "Peak longitude vs time",             "Peak lon [deg]",      False),
    ("spatial_extent_percentage","Precipitation area > 5% of peak", "Area [% of surface]",  False),
]

for j, (k, title, ylabel, logy) in enumerate(panels):
    ax = axes[j]
    if logy:
        ax.set_yscale("log")

    # for s in range(Ns):
        # ax.plot(times, stats_by_sid_ts[k][:, s], lw=1.8, label=f"Species {s}: {species[s]}")

    for s in range(Ns):
        ax.scatter(times, stats_by_sid_ts[k][:, s], s=14, alpha=0.7, label=f"Species {s}: {species[s]}")
    ax.plot(times, stats_total_ts[k], lw=2.5, color="k", label="Total")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    if j == 1:
        ax.legend(ncol=2, fontsize=9, loc="upper right")

fig.suptitle(f"{case.split('_')[0]}", fontsize=18)
out_peak_area = os.path.join(out_dir, f"{case}_stats_peak_lat_lon_percentarea_1x4_LINE.png")
fig.savefig(out_peak_area, dpi=250)
plt.show()

print(f"Done. Wrote maps + timeseries + stats to:\n  {out_dir}")
