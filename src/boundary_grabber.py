#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# SETTINGS
input_folder = "/Users/danywaller/Projects/mercury/RPS_Base/fig_xy/"
out_folder = "/Users/danywaller/Projects/mercury/RPS_Base/slice_bowshock/"
os.makedirs(out_folder, exist_ok=True)

# set thresholds
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
VAR_X = "Bx"   # x-component of B
VAR_Y = "By"   # y-component of B
VAR_Z = "Bz"   # z-component of B

VAR_V1X = "vx01"   # x-component of species 1 velocity
VAR_V1Y = "vy01"   # y-component of species 1 velocity
VAR_V1Z = "vz01"   # z-component of species 1 velocity
VAR_V3X = "vx03"   # x-component of species 3 velocity
VAR_V3Y = "vy03"   # y-component of species 3 velocity
VAR_V3Z = "vz03"   # z-component of species 3 velocity

VAR_JX = "Jx"   # x-component of J
VAR_JY = "Jy"   # y-component of J
VAR_JZ = "Jz"   # z-component of J

Bx_list, By_list, Bz_list, Bmag_list = [], [], [], []

vx01_list, vy01_list, vz01_list, Vmag01_list = [], [], [], []

vx03_list, vy03_list, vz03_list, Vmag03_list = [], [], [], []

den01_list = []

Jx_list, Jy_list, Jz_list, Jmag_list = [], [], [], []

# first stable timestamp approx. 25000 for dt=0.002, numsteps=115000
sim_steps = list(range(27000, 115000 + 1, 1000))

for sim_step in sim_steps:
    filename = 'Base_' + "%06d" % sim_step

    f = input_folder + "Amitis_RPS_" + filename + "_xy_comp.nc"

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
    x = np.arange(xmin, xmax, dx)
    y = np.arange(ymin, ymax, dy)
    z = np.arange(zmin, zmax, dz)

    x = x / 2440.e3  # convert to R_m
    y = y / 2440.e3
    z = z / 2440.e3

    # Extract arrays
    BX = ds[VAR_X].sel(Nz=0, method="nearest").squeeze()
    BY = ds[VAR_Y].sel(Nz=0, method="nearest").squeeze()
    BZ = ds[VAR_Z].sel(Nz=0, method="nearest").squeeze()

    vx01 = ds[VAR_V1X].sel(Nz=0, method="nearest").squeeze()
    vy01 = ds[VAR_V1Y].sel(Nz=0, method="nearest").squeeze()
    vz01 = ds[VAR_V1Z].sel(Nz=0, method="nearest").squeeze()

    vx03 = ds[VAR_V3X].sel(Nz=0, method="nearest").squeeze()
    vy03 = ds[VAR_V3Y].sel(Nz=0, method="nearest").squeeze()
    vz03 = ds[VAR_V3Z].sel(Nz=0, method="nearest").squeeze()

    den01 = ds["den01"].sel(Nz=0, method="nearest").squeeze()

    JX = ds[VAR_JX].sel(Nz=0, method="nearest").squeeze()
    JY = ds[VAR_JY].sel(Nz=0, method="nearest").squeeze()
    JZ = ds[VAR_JZ].sel(Nz=0, method="nearest").squeeze()

    Bmag = np.sqrt(BX ** 2 + BY ** 2 + BZ ** 2)
    Vmag01 = np.sqrt(vx01 ** 2 + vy01 ** 2 + vz01 ** 2)
    Vmag03 = np.sqrt(vx03 ** 2 + vy03 ** 2 + vz03 ** 2)
    Jmag = np.sqrt(JX ** 2 + JY ** 2 + JZ ** 2)

    Bx_list.append(BX)
    By_list.append(BY)
    Bz_list.append(BZ)
    Bmag_list.append(Bmag)

    vx01_list.append(vx01)
    vy01_list.append(vy01)
    vz01_list.append(vz01)
    Vmag01_list.append(Vmag01)

    vx03_list.append(vx03)
    vy03_list.append(vy03)
    vz03_list.append(vz03)
    Vmag03_list.append(Vmag03)

    Jx_list.append(JX)
    Jy_list.append(JY)
    Jz_list.append(JZ)
    Jmag_list.append(Jmag)

    den01_list.append(den01)

    # Compute gradients along x and z
    dB_dx = np.gradient(Bmag, x, axis=1)
    dB_dz = np.gradient(Bmag, y, axis=0)

    dV_dx = np.gradient(Vmag01, x, axis=1)
    dV_dz = np.gradient(Vmag01, y, axis=0)

    dJ_dx = np.gradient(Jmag, x, axis=1)
    dJ_dz = np.gradient(Jmag, y, axis=0)

    dP_dx = np.gradient(den01, x, axis=1)
    dP_dz = np.gradient(den01, y, axis=0)

    # Magnitude of gradient
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
    dBdz = Bhat.differentiate("Ny")

    rotation_strength = (dBdx ** 2 + dBdz ** 2).sum("comp") ** 0.5
    rot_threshold = rotmax * np.nanmax(rotation_strength)
    rot_threshold_mp = rotmax_mp * np.nanmax(rotation_strength)

    # BS
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(x, y, gradV, vmin=0, vmax=0.005, shading='auto', cmap='cividis')
    circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=2)
    ax.add_patch(circle)
    plt.colorbar(im, label="|V|")
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.xlabel(r"$X\ (\mathrm{R_M})$")
    plt.ylabel(r"$Y\ (\mathrm{R_M})$")
    plt.title(f"Velocity magnitude gradient,  t = {sim_step * 0.002} seconds")
    fig_path = os.path.join(out_folder, f"rps_xy_vmag_gradient_{sim_step}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # MP
    if 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(x, y, rotation_strength, vmin=0, vmax=0.005, shading='auto', cmap='cividis')
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

    bowshock_mask  = (gradJ > jmag_threshold) & (gradP > den_threshold) & (dP_dx < 0) & (rotation_strength < rot_threshold_mp)

    bowshock_mask &= ~magnetopause_mask

    if 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(x, y, Bmag, vmin=0, vmax=150, shading='auto', cmap='viridis')
        ax.contour(x, y, bowshock_mask, levels=[0.5], colors='red', linewidths=2)
        ax.contour(x, y, magnetopause_mask, levels=[0.5], colors='magenta', linewidths=2)
        circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(circle)

        ax.set_xlabel(r"$X\ (\mathrm{R_M})$")
        ax.set_ylabel(r"$Y\ (\mathrm{R_M})$")
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        ax.set_aspect("equal")
        plt.colorbar(im, label="|B|")
        plt.title(f"Bow Shock (red) + MP (pink),  t = {sim_step * 0.002} seconds")
        fig_path = os.path.join(out_folder, f"rps_xy_bmag_bowshock_{sim_step}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    if 0:
        bowshock_mask = (gradJ > jmag_threshold)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(x, z, Jmag, vmin=0, vmax=150, shading='auto', cmap='plasma')
        ax.contour(x, z, bowshock_mask, levels=[0.5], colors='aqua', linewidths=2)
        circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(circle)

        ax.set_xlabel(r"$X\ (\mathrm{R_M})$")
        ax.set_ylabel(r"$Z\ (\mathrm{R_M})$")
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        ax.set_aspect("equal")
        plt.colorbar(im, label="|J|")
        plt.title(f"Bow Shock Identification (blue contour),  t = {sim_step * 0.002} seconds")
        fig_path = os.path.join(out_folder, f"rps_jmag_bowshock_{sim_step}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    ds.close()

Bx_med = np.median(np.stack(Bx_list, axis=0), axis=0)
By_med = np.median(np.stack(By_list, axis=0), axis=0)
Bz_med = np.median(np.stack(Bz_list, axis=0), axis=0)

Bmag_med = np.sqrt(Bx_med**2 + By_med**2 + Bz_med**2)

vx01_med = np.median(np.stack(vx01_list, axis=0), axis=0)
vy01_med = np.median(np.stack(vy01_list, axis=0), axis=0)
vz01_med = np.median(np.stack(vz01_list, axis=0), axis=0)

Vmag01_med = np.sqrt(vx01_med**2 + vy01_med**2 + vz01_med**2)

vx03_med = np.median(np.stack(vx03_list, axis=0), axis=0)
vy03_med = np.median(np.stack(vy03_list, axis=0), axis=0)
vz03_med = np.median(np.stack(vz03_list, axis=0), axis=0)

Vmag03_med = np.sqrt(vx03_med**2 + vy03_med**2 + vz03_med**2)

den01_med = np.median(np.stack(den01_list, axis=0), axis=0)

Jx_med = np.median(np.stack(Jx_list, axis=0), axis=0)
Jy_med = np.median(np.stack(Jy_list, axis=0), axis=0)
Jz_med = np.median(np.stack(Jz_list, axis=0), axis=0)

Jmag_med = np.sqrt(Jx_med ** 2 + Jy_med ** 2 + Jz_med ** 2)

# Compute gradients along x and z
dB_dx_med = np.gradient(Bmag_med, x, axis=1)
dB_dz_med = np.gradient(Bmag_med, y, axis=0)

dV_dx_med = np.gradient(Vmag01_med, x, axis=1)
dV_dz_med = np.gradient(Vmag01_med, y, axis=0)

dJ_dx_med = np.gradient(Jmag_med, x, axis=1)
dJ_dz_med = np.gradient(Jmag_med, y, axis=0)

dP_dx_med = np.gradient(den01_med, x, axis=1)
dP_dz_med = np.gradient(den01_med, y, axis=0)

# Magnitude of gradient
gradB_med = np.sqrt(dB_dx_med ** 2 + dB_dz_med ** 2)
gradV_med = np.sqrt(dV_dx_med ** 2 + dV_dz_med ** 2)
gradJ_med = np.sqrt(dJ_dx_med ** 2 + dJ_dz_med ** 2)
gradP_med = np.sqrt(dP_dx_med ** 2 + dP_dz_med ** 2)

bmag_threshold = Bgradmax * np.nanmax(gradB_med)
vmag_threshold = Vgradmax * np.nanmax(gradV_med)
jmag_threshold = Jgradmax * np.nanmax(gradJ_med)
den_threshold = Pgradmax * np.nanmax(gradP_med)

vmag_threshold_mp = Vgradnmax_mp * np.nanmax(gradV_med)

# check magnetic field rotation
Bhat = np.stack((Bx_med, By_med, Bz_med), axis=-1) / Bmag_med[..., None]

dBdx, dBdz = np.gradient(Bhat, x, y, axis=(1,0))
rotation_strength_med = np.sqrt(
    np.sum(dBdx**2, axis=-1) + np.sum(dBdz**2, axis=-1)
)

rot_threshold_mp = rotmax_mp * np.nanmax(rotation_strength_med)

magnetopause_med_mask = (
        (gradJ_med > jmag_threshold) &
        (gradP_med > den_threshold) & (dP_dx_med > 0) &
        (gradV_med < vmag_threshold) & (dV_dx_med < 0) &
        (rotation_strength_med > rot_threshold_mp)
)

bowshock_med_mask = (gradJ_med > jmag_threshold) & (gradP_med > den_threshold) & (dP_dx_med < 0) & (rotation_strength_med < rot_threshold_mp)

bowshock_med_mask &= ~magnetopause_med_mask

fig2, ax2 = plt.subplots(figsize=(8, 6))
im2 = ax2.pcolormesh(x, y, Bmag_med, vmin=0, vmax=150, shading='auto', cmap='viridis')
ax2.contour(x, y, bowshock_med_mask, levels=[0.5], colors='red', linewidths=2)
ax2.contour(x, y, magnetopause_med_mask, levels=[0.5], colors='magenta', linewidths=2)
circle = plt.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=1)
ax2.add_patch(circle)

ax2.set_xlabel(r"$X\ (\mathrm{R_M})$")
ax2.set_ylabel(r"$Y\ (\mathrm{R_M})$")
plt.xlim([-5, 5])
plt.ylim([-5, 5])
ax2.set_aspect("equal")
plt.colorbar(im2, label="|B|")
plt.title("Median Bow Shock (red) + MP (pink)")
fig_path = os.path.join(out_folder, f"rps_xy_bmag_bowshock_median.png")
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
    plt.colorbar(im2, label="|J|")
    plt.title("Median Bow Shock Identification (blue contour)")
    fig_path = os.path.join(out_folder, f"rps_jmag_bowshock_median.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()