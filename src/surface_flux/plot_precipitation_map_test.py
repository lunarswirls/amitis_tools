#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pyamitis.amitis_particle import *
import src.surface_flux.flux_utils as flux_utils
import src.helper_utils as helper_utils

case = "CPS_Base"
main_path = f'/Volumes/data_backup/mercury/extreme/{case}/05/'
output_folder = f"/Users/danywaller/Projects/mercury/extreme/surface_flux/"

species = np.array(['H+', 'He++'])  # The order is important and it should be based on Amitis.inp file
sim_ppc = [24, 11]  # Number of macroparticles per species, based on Amitis.inp
sim_den = [38.0e6, 1.0e6]  # [/m^3]
sim_vel = [400.e3, 400.e3]  # [km/s]

sim_dx = 75.e3  # simulation cell size based on Amitis.inp
sim_dy = 75.e3  # simulation cell size based on Amitis.inp
sim_dz = 75.e3  # simulation cell size based on Amitis.inp
R_M = 2440.0e3  # planet radius in meters

select_R = 2480.e3  # the radius of a sphere above the surface for particle selection

all_particles_directory = main_path + 'precipitation/'
os.makedirs(all_particles_directory, exist_ok=True)
all_particles_filename = all_particles_directory + "all_particles_at_surface.npz"
moments_filename = all_particles_directory + "moments"

flux_cm, lat_centers, lon_centers, v_r_map, count_map, n_shell_map = \
    flux_utils.compute_radial_flux(
        all_particles_filename=all_particles_filename,
        sim_dx=sim_dx, sim_dy=sim_dy, sim_dz=sim_dz,
        sim_ppc=sim_ppc, sim_den=sim_den, spec_map=species,
        R_M=R_M, select_R=select_R,
        species="all",
        n_lat=180, n_lon=360
    )

n_lat = len(lat_centers)
n_lon = len(lon_centers)

# Rebuild bin edges consistent with centers
lon_edges = np.linspace(-180.0, 180.0, n_lon+1)
lat_edges = np.linspace(-90.0, 90.0, n_lat+1)

# Your 2D fields: shape (n_lat, n_lon)
cnts = count_map          # # particles (weighted)
den  = n_shell_map        # m^-3
vr   = v_r_map            # km/s
flux = flux_cm            # cm^-2 s^-1

# Unit conversions
den_cm3 = den * 1e-6      # m^-3 -> cm^-3
vr_abs  = np.abs(vr)      # km/s -> |km/s|

log_cnts = helper_utils.safe_log10(cnts)
log_den  = helper_utils.safe_log10(den_cm3)
log_vel  = helper_utils.safe_log10(vr_abs)
log_flx  = helper_utils.safe_log10(vr_abs*den_cm3)

sim_den_tot = np.sum(sim_den)
log_den_norm  = helper_utils.safe_log10(den / sim_den_tot)
sim_vel_tot = np.sum(sim_vel) * 1e-3
log_vel_norm  = helper_utils.safe_log10(vr_abs / sim_vel_tot)
sim_flx_tot = sim_den_tot * sim_vel_tot
log_flx_norm  = helper_utils.safe_log10((vr_abs*den)/sim_flx_tot)

# Define fields for plotting
fields_raw = [
    (cnts, (np.nanmin(cnts),    np.nanmax(cnts)),    "PuBu", "# particles"),
    (den_cm3, (np.nanmin(den_cm3), np.nanmax(den_cm3)), "YlGn", r"$n$ [cm$^{-3}$]"),
    (vr_abs, (np.nanmin(vr_abs), np.nanmax(vr_abs)), "YlOrBr",  r"$|v_r|$ [km s$^{-1}$]"),
    (flux,  (np.nanmin(flux),   np.nanmax(flux)),   "BuPu",     r"$F_r$ [cm$^{-2}$ s$^{-1}$]")
]

fields_log = [
    (log_cnts, (np.nanmin(log_cnts),    np.nanmax(log_cnts)),    "PuBu", "log10(# particles)"),
    (log_den, (np.nanmin(log_den), np.nanmax(log_den)), "YlGn", r"log10($n$) [cm$^{-3}$]"),
    (log_vel, (np.nanmin(log_vel), np.nanmax(log_vel)), "YlOrBr",  r"log10($|v_r|$) [km s$^{-1}$]"),
    (log_flx,  (np.nanmin(log_flx),   np.nanmax(log_flx)),   "BuPu",     r"log10($F_r$) [cm$^{-2}$ s$^{-1}$]")
]

fields_log_norm = [
    (log_cnts, (np.nanmin(log_cnts),    np.nanmax(log_cnts)),    "PuBu", "log10(# particles)"),
    (log_den_norm,   (-2, 1), "YlGn", r"log10($n/n_0$)"),
    (log_vel_norm, (-1, 1), "YlOrBr",  r"log10($|v_r|/v_0$)"),
    (log_flx_norm,  (-2, 1),   "BuPu",     r"log10($F_r /F_0$)")
]

titles = ["Counts", "Density", "Radial velocity", "Flux"]


# ---- 3. Plot in Hammer projection (no normalization) ----
fig, axes = plt.subplots(
    2, 2, figsize=(14, 9),
    subplot_kw={"projection": "hammer"}
)

fig.patch.set_facecolor("white")
axes = axes.flatten()

for ax, (data, clim, cmap, cblabel), title in zip(axes, fields_log, titles):
    ax.set_facecolor("white")
    ax.grid(True, linestyle="dotted", color="gray")

    # IMPORTANT: use edges (length n+1) and data (n_lat, n_lon)
    pcm = ax.pcolormesh(
        np.radians(lon_edges),  # X: shape (n_lon+1,)
        np.radians(lat_edges),  # Y: shape (n_lat+1,)
        data,  # C: shape (n_lat, n_lon)
        cmap=cmap,
        shading="flat"  # no shape error
    )
    pcm.set_clim(*clim)

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

plt.tight_layout()
plt.show()