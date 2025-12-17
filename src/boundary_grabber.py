#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# directories
input_folder = "/Users/danywaller/Projects/mercury/RPS_Base/fig_xz/"
out_folder = "/Users/danywaller/Projects/mercury/RPS_Base/slice_bowshock/"
os.makedirs(out_folder, exist_ok=True)

# percentile thresholds
Bgradmax = 0.35   # 0.25–0.45
Vgradmax = 0.10   # 0.05–0.15 velocity jump is strongest at BS
Pgradmax = 0.1   # 0.20–0.35
Jgradmax = 0.25   # 0.20–0.35
rotmax    = 0.10  # 0.05–0.15  (weak rotation)

Vgradnmax_mp = 0.25   # 0.20–0.35
Pgradmax_mp  = 0.75   # 0.85–0.98   (exclude pressure jump)
Jgradmax_mp  = 0.60   # 0.50–0.75
rotmax_mp    = 0.1   # 0.60–0.85   main discriminator

# name of variables inside the NetCDF file
VAR_X = "Bx"   # x-component of B  [units: nT]
VAR_Y = "By"   # y-component of B  [units: nT]
VAR_Z = "Bz"   # z-component of B  [units: nT]

VAR_V1X = "vx01"   # x-component of species 1 velocity  [units: km/s]
VAR_V1Y = "vy01"   # y-component of species 1 velocity  [units: km/s]
VAR_V1Z = "vz01"   # z-component of species 1 velocity  [units: km/s]
VAR_V3X = "vx03"   # x-component of species 3 velocity  [units: km/s]
VAR_V3Y = "vy03"   # y-component of species 3 velocity  [units: km/s]
VAR_V3Z = "vz03"   # z-component of species 3 velocity  [units: km/s]

VAR_JX = "Jx"   # x-component of J  [units: nA/m^2]
VAR_JY = "Jy"   # y-component of J  [units: nA/m^2]
VAR_JZ = "Jz"   # z-component of J  [units: nA/m^2]

VAR_DEN1 = "den01"   # density of particle species #1  [units: cm^-3]
VAR_DEN3 = "den03"   # density of particle species #3  [units: cm^-3]

# preallocate list for bowshock and magnetopause masks to calculate median positions later
bs_positions = []
mp_positions = []

Bmag_list = []

# first stable timestamp approx. 25000 for dt=0.002, numsteps=115000
sim_steps = list(range(27000, 115000 + 1, 1000))

for sim_step in sim_steps:
    filename = 'Base_' + "%06d" % sim_step

    f = input_folder + "Amitis_RPS_" + filename + "_xz_comp.nc"

    ds = xr.open_dataset(f)

    # Extract physical domain extents
    xmin = float(ds.full_xmin)
    xmax = float(ds.full_xmax)
    ymin = float(ds.full_ymin)
    ymax = float(ds.full_ymax)
    zmin = float(ds.full_zmin)
    zmax = float(ds.full_zmax)

    dx = float(ds.full_dx)
    dy = float(ds.full_dy)
    dz = float(ds.full_dz)

    # Build coordinate arrays
    x = np.arange(xmin, xmax, dx)  # [units: m]
    y = np.arange(zmin, zmax, dz)  # [units: m]
    # z = np.arange(zmin, zmax, dz)

    # convert to R_m for plotting
    x_plot = x / 2440.e3
    y_plot = y / 2440.e3
    # z_plot = z / 2440.e3

    # Extract arrays
    BX = ds[VAR_X].sel(Ny=0, method="nearest").squeeze()  # [units: nT]
    BY = ds[VAR_Y].sel(Ny=0, method="nearest").squeeze()  # [units: nT]
    BZ = ds[VAR_Z].sel(Ny=0, method="nearest").squeeze()  # [units: nT]

    vx01 = ds[VAR_V1X].sel(Ny=0, method="nearest").squeeze()*1.e3   # convert to m/s
    vy01 = ds[VAR_V1Y].sel(Ny=0, method="nearest").squeeze()*1.e3   # convert to m/s
    vz01 = ds[VAR_V1Z].sel(Ny=0, method="nearest").squeeze()*1.e3   # convert to m/s

    vx03 = ds[VAR_V3X].sel(Ny=0, method="nearest").squeeze()*1.e3   # convert to m/s
    vy03 = ds[VAR_V3Y].sel(Ny=0, method="nearest").squeeze()*1.e3   # convert to m/s
    vz03 = ds[VAR_V3Z].sel(Ny=0, method="nearest").squeeze()*1.e3   # convert to m/s

    den01 = ds[VAR_DEN1].sel(Ny=0, method="nearest").squeeze()*1.e-6  # convert to m^-3
    den03 = ds[VAR_DEN3].sel(Ny=0, method="nearest").squeeze()*1.e-6  # convert to m^-3
    tot_den = den01 + den03

    JX = ds[VAR_JX].sel(Ny=0, method="nearest").squeeze()  # [units: nA/m^2]
    JY = ds[VAR_JY].sel(Ny=0, method="nearest").squeeze()  # [units: nA/m^2]
    JZ = ds[VAR_JZ].sel(Ny=0, method="nearest").squeeze()  # [units: nA/m^2]

    Bmag = np.sqrt(BX ** 2 + BY ** 2 + BZ ** 2)
    Vmag01 = np.sqrt(vx01 ** 2 + vy01 ** 2 + vz01 ** 2)
    Vmag03 = np.sqrt(vx03 ** 2 + vy03 ** 2 + vz03 ** 2)
    Vmag = Vmag01 + Vmag03
    Jmag = np.sqrt(JX ** 2 + JY ** 2 + JZ ** 2)

    # compute gradients along x and z
    dB_dx = Bmag.differentiate("Nx")
    dB_dz = Bmag.differentiate("Nz")

    dV_dx = Vmag.differentiate("Nx")
    dV_dz = Vmag.differentiate("Nz")

    dJ_dx = Jmag.differentiate("Nx")
    dJ_dz = Jmag.differentiate("Nz")

    dP_dx = tot_den.differentiate("Nx")
    dP_dz = tot_den.differentiate("Nz")

    # magnitude of gradient
    gradB = np.sqrt(dB_dx ** 2 + dB_dz ** 2)
    gradV = np.sqrt(dV_dx ** 2 + dV_dz ** 2)
    gradP = np.sqrt(dP_dx ** 2 + dP_dz ** 2)
    gradJ = np.sqrt(dJ_dx ** 2 + dJ_dz ** 2)

    bmag_threshold = Bgradmax * np.nanmax(gradB)
    vmag_threshold = Vgradmax * np.nanmax(gradV)
    jmag_threshold = Jgradmax * np.nanmax(gradJ)
    den_threshold = Pgradmax * np.nanmax(gradP)
    den_threshold_mp = Pgradmax_mp * np.nanmax(gradP)

    vmag_threshold_mp = Vgradnmax_mp * np.nanmax(gradV)
    jmag_threshold_mp = Jgradmax_mp * np.nanmax(gradJ)

    # check magnetic field rotation
    B = xr.concat([BX, BY, BZ], dim="comp")
    B = B.assign_coords(comp=["x", "y", "z"])
    Bhat = B / Bmag

    dBdx = Bhat.differentiate("Nx")
    dBdz = Bhat.differentiate("Nz")

    rotation_strength = (dBdx ** 2 + dBdz ** 2).sum("comp") ** 0.5
    rot_threshold = rotmax * np.nanmax(rotation_strength)
    rot_threshold_mp = rotmax_mp * np.nanmax(rotation_strength)

    if 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(x_plot, y_plot, gradV, vmin=0, vmax=750.0, shading='auto', cmap='cividis')
        circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(circle)
        plt.colorbar(im, label="∇|V| (m/s)")
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.xlabel(r"$X\ (\mathrm{R_M})$")
        plt.ylabel(r"$Z\ (\mathrm{R_M})$")
        plt.title(f"Velocity magnitude gradient,  t = {sim_step * 0.002} seconds")
        fig_path = os.path.join(out_folder, f"rps_xz_vmag_gradient_{sim_step}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    if 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(x_plot, y_plot, gradP, vmin=0, vmax=.5e-6, shading='auto', cmap='cividis')
        circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(circle)
        plt.colorbar(im, label=r"$\nabla N_{sw}\text{ (m^-3)}$")
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.xlabel(r"$X\ (\mathrm{R_M})$")
        plt.ylabel(r"$Z\ (\mathrm{R_M})$")
        plt.title(f"Total density gradient,  t = {sim_step * 0.002} seconds")
        fig_path = os.path.join(out_folder, f"rps_xz_pmag_gradient_{sim_step}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    if 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(x_plot, y_plot, rotation_strength, vmin=0, vmax=0.005, shading='auto', cmap='cividis')
        circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(circle)
        plt.colorbar(im, label="|rotation|")
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.xlabel(r"$X\ (\mathrm{R_M})$")
        plt.ylabel(r"$Y\ (\mathrm{R_M})$")
        plt.title(f"Rotation strength,  t = {sim_step * 0.002} seconds")
        fig_path = os.path.join(out_folder, f"rps_xy_rotation_strength_{sim_step}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    magnetopause_mask = (
            (gradJ > jmag_threshold) &
            (gradP > den_threshold) & (dP_dx > 0) &
            (gradV < vmag_threshold) & (dV_dx < 0) &
            (rotation_strength > rot_threshold_mp)
    )

    bowshock_mask  = ((gradJ > jmag_threshold) &
                      (gradP > den_threshold) & (dP_dx < 0) &
                      (rotation_strength < rot_threshold_mp))

    # check no bowshock points have been classified as magnetopause points
    bowshock_mask &= ~magnetopause_mask

    # exclude plasma sheet
    x_bad = x_plot < 0
    y_bad = (y_plot > -0.75) & (y_plot < 0.75)

    exclude_region = y_bad[:, None] & x_bad[None, :]

    bowshock_mask &= ~exclude_region

    magnetopause_mask &= ~exclude_region

    # append for later use
    bs_positions.append(bowshock_mask)
    mp_positions.append(magnetopause_mask)
    Bmag_list.append(Bmag)

    if 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(x_plot, y_plot, Bmag, vmin=0, vmax=150, shading='auto', cmap='viridis')
        ax.contour(x_plot, y_plot, bowshock_mask, levels=[0.5], colors='red', linewidths=2)
        ax.contour(x_plot, y_plot, magnetopause_mask, levels=[0.5], colors='magenta', linewidths=2)
        circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(circle)

        ax.set_xlabel(r"$X\ (\mathrm{R_M})$")
        ax.set_ylabel(r"$Z\ (\mathrm{R_M})$")
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        ax.set_aspect("equal")
        plt.colorbar(im, label="|B| (nT)")
        plt.title(f"Bow Shock (red) + MP (pink),  t = {sim_step * 0.002} seconds")
        fig_path = os.path.join(out_folder, f"rps_xz_bmag_bowshock_{sim_step}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    if 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(x_plot, y_plot, Jmag, vmin=0, vmax=150, shading='auto', cmap='plasma')
        ax.contour(x_plot, y_plot, bowshock_mask, levels=[0.5], colors='aqua', linewidths=2)
        ax.contour(x_plot, y_plot, magnetopause_mask, levels=[0.5], colors='magenta', linewidths=2)
        circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(circle)

        ax.set_xlabel(r"$X\ (\mathrm{R_M})$")
        ax.set_ylabel(r"$Z\ (\mathrm{R_M})$")
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        ax.set_aspect("equal")
        plt.colorbar(im, label="|J| (nA/m²")
        plt.title(f"Bow Shock (blue) + MP (pink),  t = {sim_step * 0.002} seconds")
        fig_path = os.path.join(out_folder, f"rps_xz_jmag_bowshock_{sim_step}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    ds.close()

bs_all = np.sum(np.stack(bs_positions, axis=0), axis=0)
mp_all = np.sum(np.stack(mp_positions, axis=0), axis=0)

bs_stack = np.stack(bs_positions, axis=0).astype(float)
bs_stack[bs_stack == 0] = np.nan

bs_q1 = np.nanpercentile(bs_stack, 25, axis=0)
bs_med = np.nanpercentile(bs_stack, 50, axis=0)
bs_q3 = np.nanpercentile(bs_stack, 75, axis=0)

bs_iqr = bs_q3 - bs_q1

mp_stack = np.stack(mp_positions, axis=0).astype(float)
mp_stack[mp_stack == 0] = np.nan

mp_q1 = np.nanpercentile(mp_stack, 25, axis=0)
mp_med = np.nanpercentile(mp_stack, 50, axis=0)
mp_q3 = np.nanpercentile(mp_stack, 75, axis=0)

mp_iqr = mp_q3 - mp_q1

Bmag_med = np.median(np.stack(Bmag_list, axis=0), axis=0)

if 1:
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im2 = ax2.pcolormesh(x_plot, y_plot, Bmag_med, vmin=0, vmax=150, shading='auto', cmap='viridis')
    # Bow shock IQR envelope
    # ax2.contourf(x_plot, y_plot, (bs_q1 > 0) & (bs_q3 > 0), levels=[0.5, 1], colors='red', alpha=0.7)

    # Magnetopause IQR envelope
    # ax2.contourf(x_plot, y_plot, (mp_q1 > 0) & (mp_q3 > 0), levels=[0.5, 1], colors='magenta', alpha=0.7)
    ax2.contour(x_plot, y_plot, bs_med, levels=[0.5], colors='red', linewidths=2)
    ax2.contour(x_plot, y_plot, mp_med, levels=[0.5], colors='magenta', linewidths=2)

    circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=1)
    ax2.add_patch(circle)

    ax2.set_xlabel(r"$X\ (\mathrm{R_M})$")
    ax2.set_ylabel(r"$Z\ (\mathrm{R_M})$")
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    ax2.set_aspect("equal")
    plt.colorbar(im2, label="|B| (nT)")
    plt.title("Median Bow Shock (red) + MP (pink)")
    fig_path = os.path.join(out_folder, f"rps_xz_bmag_bowshock_median.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

if 0:
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im2 = ax2.pcolormesh(x, z, Jmag_med, vmin=0, vmax=150, shading='auto', cmap='plasma')
    ax2.contour(x, z, bowshock_med_mask, levels=[0.5], colors='aqua', linewidths=2)
    circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=1)
    ax2.add_patch(circle)

    ax2.set_xlabel(r"$X\ (\mathrm{R_M})$")
    ax2.set_ylabel(r"$Z\ (\mathrm{R_M})$")
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    ax2.set_aspect("equal")
    plt.colorbar(im2, label="|J| (nA/m²)")
    plt.title("Median Bow Shock Identification (blue contour)")
    fig_path = os.path.join(out_folder, f"rps_jmag_bowshock_median.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()