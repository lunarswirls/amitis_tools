#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.surface_flux.flux_utils as flux_utils

case = "CPN_Base"
main_path = f'/Volumes/data_backup/mercury/extreme/{case}/05/'
output_folder = f"/Users/danywaller/Projects/mercury/extreme/surface_flux/"

plot_meth = "raw"  # raw, log, lognorm
run_species = "all"  # 'all' or 'protons' or 'alphas'

species = np.array(['H+', 'tbd', 'He++', 'tbd2'])  # The order is important and it should be based on Amitis.inp file
sim_ppc = [24, 0, 11, 0]  # Number of particles per species, based on Amitis.inp
sim_den = [38.0e6, 0, 1.0e6, 0]   # [/m^3]
sim_vel = [400.e3, 0, 400.e3, 0]  # [km/s]

sim_dx = 75.e3  # simulation cell size based on Amitis.inp
sim_dy = 75.e3  # simulation cell size based on Amitis.inp
sim_dz = 75.e3  # simulation cell size based on Amitis.inp
sim_robs = 2440.e3  # obstacle radius based on Amitis.inp

nlat = 180
nlon = 360

select_R = 2480.e3  # the radius of a sphere + 1/2 grid cell above the surface for particle selection

if "CPZ" in case:
    input_folder2 = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology/{case}_largerxdomain_smallergridsize/"
    csv_file = os.path.join(input_folder2, f"{case}_largerxdomain_smallergridsize_115000_ocb_curve.csv")  # single timestep CSV with OCB curve

    # Load footprint CSV
    if os.path.exists(csv_file):
        df_footprints = pd.read_csv(csv_file)
        print(f"Loaded {len(df_footprints)} footprints for {case}")

all_particles_directory = main_path + 'precipitation/'
all_particles_filename = all_particles_directory + "all_particles_at_surface.npz"

flux_cm, lat_centers, lon_centers, v_r_map, count_map, n_shell_map = \
    flux_utils.compute_radial_flux(
        all_particles_filename=all_particles_filename,
        sim_dx=sim_dx, sim_dy=sim_dy, sim_dz=sim_dz,
        sim_ppc=sim_ppc, sim_den=sim_den, spec_map=species,
        R_M=sim_robs, select_R=select_R,
        species=run_species,
        n_lat=nlat, n_lon=nlon
    )

n_lat = len(lat_centers)
n_lon = len(lon_centers)

# Rebuild bin edges consistent with centers
lon_edges = np.linspace(-180.0, 180.0, n_lon+1)
lat_edges = np.linspace(-90.0, 90.0, n_lat+1)

# ========== 2D maps with units ==========
cnts = count_map.copy()     # [# particles]
den  = n_shell_map.copy()   # [m^-3] shell volume density
vr   = v_r_map.copy()       # [km/s]
flux = flux_cm.copy()       # [cm^-2 s^-1]

vr_abs = np.abs(vr)         # [km/s]
flux_abs = np.abs(flux)     # [cm^-2 s^-1]

# Set low-count pixels to NaN
mask = count_map <= 1e-20
cnts[mask] = np.nan
den[mask]  = np.nan
vr_abs[mask] = np.nan
flux_abs[mask] = np.nan

# ========== Unit conversions ==========
den_cm3 = den * 1e-6  # [m^-3] → [cm^-3]

def safe_log10(arr, vmin=1e-30):
    """Safe log10 that handles zeros/negatives."""
    out = np.full_like(arr, np.nan, dtype=float)
    mask = arr > vmin
    out[mask] = np.log10(arr[mask])
    return out

# ========== Logarithmic maps ==========
log_cnts = safe_log10(cnts)
log_den  = safe_log10(den_cm3)  # log10(cm^-3)
log_vel  = safe_log10(vr_abs)   # log10(km/s)
log_flx  = safe_log10(flux_abs) # log10(cm^-2 s^-1)

# ========== Normalized maps ==========
# Total upstream density [m^-3]
sim_den_tot = np.sum(sim_den)

# Upstream velocity [km/s]
sim_vel_tot = np.mean(sim_vel) * 1e-3  # [m/s] → [km/s]

# Upstream flux [cm^-2 s^-1]
sim_flux_upstream = sim_den_tot * np.mean(sim_vel) * 1e-4  # [m^-3 * m/s] → [cm^-2 s^-1]

# Normalized quantities
log_den_norm = safe_log10(den_cm3 / (sim_den_tot * 1e-6))  # [cm^-3] / [cm^-3]
log_vel_norm = safe_log10(vr_abs / sim_vel_tot)            # [km/s] / [km/s]
log_flx_norm = safe_log10(flux_abs / sim_flux_upstream)    # [cm^-2 s^-1] / [cm^-2 s^-1]

# Define fields for plotting
fields_raw = [
    (cnts, (1, 200), "viridis", "# particles", "magenta"),
    (den_cm3, (1, 140), "cividis", r"$n$ [cm$^{-3}$]", "magenta"),
    (vr_abs, (1, 250), "plasma", r"$|v_r|$ [km/s]", "magenta"),
    (flux_abs, (0.05e9, 8.5e8), "jet", r"$F_r$ [cm$^{-2}$ s$^{-1}$]", "magenta")
]

fields_log = [
    (cnts, (np.nanmin(cnts), np.nanmax(cnts)), "viridis", "# particles", "magenta"),
    (log_den, (np.nanmin(log_den), np.nanmax(log_den)), "cividis", r"log$_{10}$($n$) [cm$^{-3}$]", "magenta"),
    (log_vel, (np.nanmin(log_vel), np.nanmax(log_vel)), "plasma", r"log$_{10}$($|v_r|$) [km s$^{-1}$]", "magenta"),
    (log_flx, (np.nanmin(log_flx), np.nanmax(log_flx)), "jet", r"log$_{10}$($F_r$) [cm$^{-2}$ s$^{-1}$]", "magenta")
]

fields_log_norm = [
    (cnts, (np.nanmin(cnts), np.nanmax(cnts)), "viridis", "# particles", "magenta"),
    (log_den_norm, (-1, 1), "cividis", r"log$_{10}$($n/n_0$)", "magenta"),
    (log_vel_norm, (-1.0, 0.0), "plasma", r"log$_{10}$($|v_r|/v_0$)", "magenta"),
    (log_flx_norm, (-2, 1), "jet", r"log$_{10}$($F_r/F_0$)", "magenta")
]

if plot_meth == 'raw':
    use_fields = fields_raw
elif plot_meth == 'log':
    use_fields = fields_log
elif plot_meth == 'lognorm':
    use_fields = fields_log_norm

titles = ["Counts", "Density", "Radial velocity", "Flux"]

if "CPZ" in case:
    # Split north and south hemispheres
    df_north = df_footprints[df_footprints["hemisphere"] == "north"]
    df_south = df_footprints[df_footprints["hemisphere"] == "south"]

    # Convert to radians for Mollweide/Hammer projection
    lon_n_rad = np.deg2rad(df_north["longitude_deg"])
    lat_n_rad = np.deg2rad(df_north["ocb_latitude_deg"])

    lon_s_rad = np.deg2rad(df_south["longitude_deg"])
    lat_s_rad = np.deg2rad(df_south["ocb_latitude_deg"])


# ---- 3. Plot in Hammer projection (no normalization) ----
fig, axes = plt.subplots(
    2, 2, figsize=(14, 9),
    subplot_kw={"projection": "hammer"}
)

fig.patch.set_facecolor("white")
axes = axes.flatten()

for ax, (data, clim, cmap, cblabel, ocb_col), title in zip(axes, use_fields, titles):
    ax.set_facecolor("white")
    ax.grid(True, linestyle="dotted", color="gray")

    # IMPORTANT: use edges (length n+1) and data (n_lat, n_lon)
    pcm = ax.pcolormesh(
        np.radians(lon_edges),  # X: shape (n_lon+1,)
        np.radians(lat_edges),  # Y: shape (n_lat+1,)
        data,                   # C: shape (n_lat, n_lon)
        cmap=cmap,
        shading="flat"
    )
    pcm.set_clim(*clim)

    if "CPZ" in case:
        # Plot
        ax.plot(lon_n_rad, lat_n_rad, color=ocb_col, lw=2, label="OCB North")
        ax.plot(lon_s_rad, lat_s_rad, color=ocb_col, lw=2, ls="--", label="OCB South")

    cbar = plt.colorbar(
        pcm,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        shrink=0.85
    )
    cbar.set_label(cblabel, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.set_title(title, fontsize=20)

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

# Generate title based on species selection
if run_species == "all":
    stitle = f"{case.replace('_', ' ')}: All species"
    plot_fname = f"{case}_surface_flux_all_species_{plot_meth}vals"
elif run_species == "protons":
    stitle = f"{case.replace('_', ' ')}: H+"
    plot_fname = f"{case}_surface_flux_H+_{plot_meth}vals"
elif run_species == "alphas":
    stitle = f"{case.replace('_', ' ')}: He++"
    plot_fname = f"{case}_surface_flux_He++_{plot_meth}vals"

if "CPZ" in case:
    stitle = stitle + "\nOCB footprints"
    plot_fname = plot_fname + "_footprints.png"
else:
    stitle = stitle
    plot_fname = plot_fname + ".png"

fig.suptitle(stitle, fontsize=20, y=0.95)
plt.tight_layout()

outfile_png = os.path.join(output_folder, plot_fname)
plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
print("Saved figure:", outfile_png)
# plt.show()
plt.close(fig)
