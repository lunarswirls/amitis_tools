#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from bs_mp_finder.zzz_archive.boundary_utils import labels_for_slice, coords_for_slice

# ======================
# PHYSICAL CONSTANTS
# ======================
k_B = 1.380649e-23  # Boltzmann constant [J/K]
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space [H/m]
k_B_eV = 8.617333262e-5  # Boltzmann constant [eV/K]

# base cases: CPN_Base RPN_Base CPS_Base RPS_Base
# HNHV cases: CPN_HNHV RPN_HNHV CPS_HNHV RPS_HNHV
case = "RPN_Base"

use_slices = ["xy", "xz", "yz"]  # plot all 3
n_slices = len(use_slices)       # number of requested slices
slice_tag = "_".join(use_slices)

if "Base" in case:
    base_in_dir = f"/Volumes/data_backup/mercury/extreme/{case}/plane_product/"
    c_max = 2.5
elif "HNHV" in case:
    base_in_dir = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/"
    c_max = 400

out_folder = f"/Users/danywaller/Projects/mercury/extreme/timeseries_beta_{slice_tag}/{case}/"
os.makedirs(out_folder, exist_ok=True)

# radius of Mercury [meters]
R_M = 2440.0e3

# first stable timestamp approx. 25000 for dt=0.002, numsteps=115000
if "Base" in case:
    # take last 15-ish seconds
    sim_steps = range(98000, 115000 + 1, 1000)
elif "HNHV" in case:
    sim_steps = range(115000, 350000 + 1, 1000)


def extract_slice_fields(ds: xr.Dataset, use_slice: str):
    """
    Pull 2D arrays for a slice:
      xy: Nz ~= 0
      xz: Ny ~= 0
      yz: Nx ~= 0  (dayside view looking along -X)
    """
    s = use_slice.lower().strip()
    if s == "xy":
        sel_kw = dict(Nz=0)
    elif s == "xz":
        sel_kw = dict(Ny=0)
    elif s == "yz":
        sel_kw = dict(Nx=0)
    else:
        raise ValueError(use_slice)

    first_species = True

    # Species temperatures from simulation input file
    s_dict = {
        '1': 1.4e5,  # H+ solar wind protons [K]
        '2': 1.4e5,
        '3': 5.6e5,  # Hot electrons or He++ alphas [K]
        '4': 5.6e5,
    }

    for s in ['1', '2', '3', '4']:
        T_K = s_dict[s]
        n_cm3 = ds[f'den0{s}'].sel(**sel_kw, method="nearest").squeeze()  # [cm^-3]
        n_transposed = n_cm3 * 1e6  # Convert cm^-3 to m^-3

        if n_transposed.max() < 1e3:  # Less than 0.001 cm^-3
            # print(f"Species 0{s}: INACTIVE")
            continue

        # Compute pressure: P = n * k_B * T [Pa]
        P_species_Pa = n_transposed * k_B * T_K

        if first_species:
            P_total_Pa = P_species_Pa.copy()
            first_species = False
        else:
            P_total_Pa += P_species_Pa

        # print(f"Species 0{s}: T={T_K:.2e} K, P_range=[{P_species_nPa.min():.3f}, {P_species_nPa.max():.3f}] nPa")

    Bx_plane = ds["Bx_tot"].sel(**sel_kw, method="nearest").squeeze()  # [nT]
    By_plane = ds["By_tot"].sel(**sel_kw, method="nearest").squeeze()  # [nT]
    Bz_plane = ds["Bz_tot"].sel(**sel_kw, method="nearest").squeeze()  # [nT]

    B_magnitude_nT = np.sqrt(Bx_plane ** 2 + By_plane ** 2 + Bz_plane ** 2)
    B_magnitude_T = B_magnitude_nT * 1e-9

    magnetic_pressure_Pa = B_magnitude_T ** 2 / (2 * mu_0)

    # Avoid division by zero
    magnetic_pressure_Pa_safe = np.where(magnetic_pressure_Pa < 1e-30, 1e-30, magnetic_pressure_Pa)
    plasma_beta = P_total_Pa / magnetic_pressure_Pa_safe

    return plasma_beta


for sim_step in sim_steps:
    filetime = f"{sim_step:06d}"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    last_im = None

    near_planes = []

    for ax, use_slice in zip(axes, use_slices):
        input_folder = os.path.join(base_in_dir, f"fig_{use_slice}")
        f = os.path.join(input_folder, f"Amitis_{case}_{filetime}_{use_slice}_comp.nc")
        # input_folder = os.path.join(base_in_dir, f"object")
        # f = os.path.join(input_folder, f"Amitis_{case}_{filetime}_xz_comp.nc")

        if not os.path.exists(f):
            ax.axis("off")
            ax.set_title(f"{use_slice.upper()} missing")
            print(f"[WARN] missing: {f}")
            continue

        ds = xr.open_dataset(f)

        x_plot, y_plot = coords_for_slice(ds, use_slice)

        pbeta = extract_slice_fields(ds, use_slice)



        ds.close()

        circle = plt.Circle((0, 0), 1, edgecolor='black', facecolor='cornflowerblue', alpha=0.3, linewidth=1, )
        last_im = ax.pcolormesh(x_plot, y_plot, pbeta, shading='auto', cmap='Reds', vmin=0, vmax=c_max)
        ax.add_patch(circle)
        xlabel, ylabel = labels_for_slice(use_slice)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_aspect("equal")
        ax.set_title(f"{use_slice.upper()}")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, location="right", shrink=0.9)
        cbar.set_label(r"$\beta$")

    fig.suptitle(rf"{case.replace("_"," ")} $\beta$ at t = {sim_step * 0.002:.3f} s", fontsize=18, y=0.99)
    # plt.tight_layout()
    fig_path = os.path.join(out_folder, f"{case}_beta_{sim_step}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"Done plotting β to {out_folder}")
