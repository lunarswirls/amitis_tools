#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports:
import numpy as np
import xarray as xr
from src.bs_mp_finder.boundary_thresholds import THRESHOLDS as th
import matplotlib.pyplot as plt

# ----------------------------
# Helpers
# ----------------------------
def labels_for_slice(s: str):
    s = s.lower().strip()
    if s == "xy":
        return r"$X\ (\mathrm{R_M})$", r"$Y\ (\mathrm{R_M})$"
    if s == "xz":
        return r"$X\ (\mathrm{R_M})$", r"$Z\ (\mathrm{R_M})$"
    if s == "yz":
        return r"$Y\ (\mathrm{R_M})$", r"$Z\ (\mathrm{R_M})$"
    raise ValueError(s)


def coords_for_slice(ds: xr.Dataset, use_slice: str, RM_M=2440.0e3):
    xmin = float(ds.full_xmin); xmax = float(ds.full_xmax)
    ymin = float(ds.full_ymin); ymax = float(ds.full_ymax)
    zmin = float(ds.full_zmin); zmax = float(ds.full_zmax)
    dx = float(ds.full_dx); dy = float(ds.full_dy); dz = float(ds.full_dz)

    x = np.arange(xmin, xmax, dx) / RM_M
    y = np.arange(ymin, ymax, dy) / RM_M
    z = np.arange(zmin, zmax, dz) / RM_M

    s = use_slice.lower().strip()
    if s == "xy":
        return x, y
    if s == "xz":
        return x, z
    if s == "yz":
        return y, z
    raise ValueError(use_slice)


def extract_slice_fields(ds: xr.Dataset, use_slice: str):
    """
    Pull 2D arrays for a slice:
      xy: Nz ~= 0
      xz: Ny ~= 0
      yz: Nx ~= 0  (dayside view looking along -X)
    """
    s = use_slice.lower().strip()
    if s == "xy":
        sel_kw = dict(Nz=1)
    elif s == "xz":
        sel_kw = dict(Ny=1)
    elif s == "yz":
        sel_kw = dict(Nx=1)
    else:
        raise ValueError(use_slice)

    BX = ds["Bx_tot"].sel(**sel_kw, method="nearest").squeeze()  # [units: nT]
    BY = ds["By_tot"].sel(**sel_kw, method="nearest").squeeze()  # [units: nT]
    BZ = ds["Bz_tot"].sel(**sel_kw, method="nearest").squeeze()  # [units: nT]

    vx01 = ds["vx01"].sel(**sel_kw, method="nearest").squeeze()  # [units: km/s]
    vy01 = ds["vy01"].sel(**sel_kw, method="nearest").squeeze()  # [units: km/s]
    vz01 = ds["vz01"].sel(**sel_kw, method="nearest").squeeze()  # [units: km/s]

    vx03 = ds["vx03"].sel(**sel_kw, method="nearest").squeeze()  # [units: km/s]
    vy03 = ds["vy03"].sel(**sel_kw, method="nearest").squeeze()  # [units: km/s]
    vz03 = ds["vz03"].sel(**sel_kw, method="nearest").squeeze()  # [units: km/s]

    # pre-ICME protons
    den01 = ds["den01"].sel(**sel_kw, method="nearest").squeeze()  # [units: cm^-3]

    # pre-ICME alphas
    den03 = ds["den03"].sel(**sel_kw, method="nearest").squeeze()  # [units: cm^-3]

    JX = ds["Jx"].sel(**sel_kw, method="nearest").squeeze()  # [units: nA/m^2]
    JY = ds["Jy"].sel(**sel_kw, method="nearest").squeeze()  # [units: nA/m^2]
    JZ = ds["Jz"].sel(**sel_kw, method="nearest").squeeze()  # [units: nA/m^2]

    return BX,BY,BZ,vx01,vy01,vz01,vx03,vy03,vz03,den01,den03,JX,JY,JZ


def compute_masks_one_timestep(ds: xr.Dataset, use_slice: str, plot_id: str, debug: bool=True):
    """
    Compute Bmag and BS/MP masks for one timestep.
    Units of Nx, Ny, Nz are km
    Units of magnetic field are nT
    Units of velocity are km/s
    Units of density are cm^-3
    Units of current are nA/m^2

    Returns: x_plot, y_plot, Bmag(np.ndarray), bs_mask(bool ndarray), mp_mask(bool ndarray)
    """
    if debug:
        print("DEBUGGING: compute_masks_one_timestep")

    x_plot, y_plot = coords_for_slice(ds, use_slice)

    BX,BY,BZ,vx01,vy01,vz01,vx03,vy03,vz03,den01,den03,JX,JY,JZ = extract_slice_fields(ds, use_slice)

    tot_den = den01 + den03  # units: cm^-3
    Pmag = (tot_den * 1e15).assign_attrs(units="km^-3", converted_from="cm^-3", conversion_factor="1e15")  # units: km^-3
    Bmag = np.sqrt(BX**2 + BY**2 + BZ**2)  # units: nT
    Vmag01 = np.sqrt(vx01**2 + vy01**2 + vz01**2)  # units: km/s
    Vmag03 = np.sqrt(vx03**2 + vy03**2 + vz03**2)  # units: km/s
    Vmag = Vmag01 + Vmag03  # units: km/s
    Jmag = (np.sqrt(JX**2 + JY**2 + JZ**2) * 1e6).assign_attrs(units="nA/km^2", converted_from="nA/m^2", conversion_factor="1e6")   # units: nA/km^2

    B = xr.concat([BX, BY, BZ], dim="comp").assign_coords(comp=["x", "y", "z"])
    Bhat = B / Bmag

    s = use_slice.lower().strip()

    # in-plane derivatives
    if s == "xy":
        dB_du    = Bmag.differentiate("Nx")
        dV_du    = Vmag.differentiate("Nx")
        dJ_du    = Jmag.differentiate("Nx")
        dP_du    = Pmag.differentiate("Nx")
        dBhat_du = Bhat.differentiate("Nx")

        dB_dv    = Bmag.differentiate("Ny")
        dV_dv    = Vmag.differentiate("Ny")
        dJ_dv    = Jmag.differentiate("Ny")
        dP_dv    = Pmag.differentiate("Ny")
        dBhat_dv = Bhat.differentiate("Ny")

    elif s == "xz":
        dB_du    = Bmag.differentiate("Nx")
        dV_du    = Vmag.differentiate("Nx")
        dJ_du    = Jmag.differentiate("Nx")
        dP_du    = Pmag.differentiate("Nx")
        dBhat_du = Bhat.differentiate("Nx")

        dB_dv    = Bmag.differentiate("Nz")
        dV_dv    = Vmag.differentiate("Nz")
        dJ_dv    = Jmag.differentiate("Nz")
        dP_dv    = Pmag.differentiate("Nz")
        dBhat_dv = Bhat.differentiate("Nz")

    elif s == "yz":
        dB_du    = Bmag.differentiate("Ny")
        dV_du    = Vmag.differentiate("Ny")
        dJ_du    = Jmag.differentiate("Ny")
        dP_du    = Pmag.differentiate("Ny")
        dBhat_du = Bhat.differentiate("Ny")

        dB_dv    = Bmag.differentiate("Nz")
        dV_dv    = Vmag.differentiate("Nz")
        dJ_dv    = Jmag.differentiate("Nz")
        dP_dv    = Pmag.differentiate("Nz")
        dBhat_dv = Bhat.differentiate("Nz")
    else:
        raise ValueError(use_slice)

    gradB = np.sqrt(dB_du**2 + dB_dv**2)
    gradV = np.sqrt(dV_du**2 + dV_dv**2)
    gradP = np.sqrt(dP_du**2 + dP_dv**2)
    gradJ = np.sqrt(dJ_du**2 + dJ_dv**2)

    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)

    last_im = ax1.pcolormesh(x_plot, y_plot, gradB, shading="auto", cmap="cividis", vmin=0, vmax=1)
    cbar = fig1.colorbar(last_im, ax=ax1, location="right", shrink=0.9)
    cbar.set_label(r"nT/km")
    ax1.set_title(r"$\nabla$|B|")

    last_im = ax2.pcolormesh(x_plot, y_plot, gradV, shading="auto", cmap="plasma", vmin=0, vmax=5)
    cbar = fig1.colorbar(last_im, ax=ax2, location="right", shrink=0.9)
    cbar.set_label(r"1/s")
    ax2.set_title(r"$\nabla$|V|")

    last_im = ax3.pcolormesh(x_plot, y_plot, gradP, shading="auto", cmap="cool", vmin=0, vmax=100000)
    cbar = fig1.colorbar(last_im, ax=ax3, location="right", shrink=0.9)
    cbar.set_label(r"km$^{-4}$")
    ax3.set_title(r"$\nabla$|N|")

    last_im = ax4.pcolormesh(x_plot, y_plot, gradJ, shading="auto", cmap="viridis", vmin=0, vmax=10000)
    cbar = fig1.colorbar(last_im, ax=ax4, location="right", shrink=0.9)
    cbar.set_label(r"nA/km$^3$")
    ax4.set_title(r"$\nabla$|J|")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_aspect("equal")

    fig1.suptitle(r"CPS Base")
    fig1.show()
    fig1.savefig("/Users/danywaller/Projects/mercury/extreme/slice_bowshock/CPS_Base_gradients.png", dpi=300)

    rotation_strength = (dBhat_du**2 + dBhat_dv**2).sum("comp") ** 0.5

    bmag_threshold_bs = th["Bgradmax_bs"] * np.nanmax(gradB)
    vmag_threshold_bs = th["Vgradmax_bs"] * np.nanmax(gradV)
    jmag_threshold_bs = th["Jgradmax_bs"] * np.nanmax(gradJ)
    den_threshold_bs  = th["Pgradmax_bs"] * np.nanmax(gradP)

    bmag_threshold_mp = th["Bgradmax_mp"] * np.nanmax(gradB)
    # vmag_threshold_mp = th["Vgradmax_mp"] * np.nanmax(gradV)
    jmag_threshold_mp = th["Jgradmax_mp"] * np.nanmax(gradJ)
    den_threshold_mp = th["Pgradmax_mp"] * np.nanmax(gradP)

    rot_threshold_mp = th["rotmax_mp"] * np.nanmax(rotation_strength)

    s = use_slice.lower().strip()

    if s in ("xy", "xz"):
        if 0:
            # XY/XZ logic
            magnetopause_mask = (
                    (gradJ > jmag_threshold) &
                    (gradP > den_threshold) & (dP_du > 0) &
                    (gradV < vmag_threshold) & (dV_du < 0) &
                    (rotation_strength > rot_threshold_mp)
            )
            bowshock_mask = (
                    (gradJ > jmag_threshold) &
                    (gradP > den_threshold) & (dP_du < 0) &
                    (rotation_strength < rot_threshold_mp)
            )
        # XY/XZ logic

        bowshock_mask = (
                (gradV > vmag_threshold_bs) & (dV_du > 0) &
                (gradJ > jmag_threshold_bs) &
                (gradP > den_threshold_bs) & (dP_du < 0)
        )

        magnetopause_mask = (
                (gradJ > jmag_threshold_mp) & (dJ_du > 0) &
                (gradP > den_threshold_mp) & (dP_du > 0) &
                (gradB > bmag_threshold_mp) & (dB_du < 0)
        )

        bowshock_mask = bowshock_mask & (~magnetopause_mask)

    elif s == "yz":
        # ------------------------------------------------------------
        # YZ dayside view (looking along −X):
        #   horizontal axis = Y
        #   vertical axis   = Z
        #
        # The planet is centered at Y=0, so "inward" direction flips sign:
        #   - On Y>0 side, inward is toward decreasing Y
        #   - On Y<0 side, inward is toward increasing Y
        #
        # Therefore use a symmetric inward-density-gradient proxy:
        #   dP_inward = -sign(Y) * dP/dY
        # ------------------------------------------------------------

        # Horizontal derivative is d/du = d/dY (Ny)
        dP_h = dP_du  # already computed as differentiate("Ny")

        # Build Y–Z coordinate grid (in R_M)
        y_axis = x_plot  # coords_for_slice returns (Y, Z) for yz
        z_axis = y_plot
        Yg, Zg = np.meshgrid(y_axis, z_axis, indexing="xy")
        r = np.sqrt(Yg ** 2 + Zg ** 2)

        # Exclude inside the planetary body
        outside_body = r >= 1.0

        # Rotation field as numpy for fast masking
        gradB = gradB.values
        gradJ = gradJ.values
        gradP = gradP.values
        rot = rotation_strength.values

        # ------------------------------------------------------------
        # Unified candidate gate on strong |J| and density gradients
        # ------------------------------------------------------------
        candidate = (
                (gradJ > jmag_threshold_bs) &
                (gradP > den_threshold_bs) &
                outside_body &
                np.isfinite(rot)
        )

        # cand = candidate.values.astype(bool)
        cand = candidate.astype(bool)

        if debug:
            south = (Zg < -1.5) & (np.abs(Yg) < 2.0) & outside_body

            def cnt(mask):
                return int(np.count_nonzero(mask))

            print(
                f"max gradJ={np.nanmax(gradJ):.3g} thr={jmag_threshold_bs:.3g} "
                f"max gradP={np.nanmax(gradP):.3g} thr={den_threshold_bs:.3g} "
                f"max gradB={np.nanmax(gradB):.3g} thr={bmag_threshold_bs:.3g} "
                f"max rot={np.nanmax(rotation_strength):.3g} thr={rot_threshold_mp:.3g} "
                f"cand={cnt(cand)} cand_south={cnt(cand & south)}\n")

        if np.count_nonzero(cand) == 0:
            bowshock_mask = candidate * False
            magnetopause_mask = candidate * False

        else:
            # --------------------------------------------------------
            # Symmetric inward density-gradient proxy
            # --------------------------------------------------------
            dPdy = dP_h.values
            # sign(Y): +1 (Y>0), -1 (Y<0); ignore a thin band near Y=0
            sY = np.sign(Yg)
            eps_rm = 0.001
            y0_band = np.abs(Yg) <= eps_rm
            dP_inward = -sY * dPdy

            # Initial split (symmetric)
            bs_pre = cand & (~y0_band) & (dP_inward > 0.0) & (gradB > bmag_threshold_bs)
            mp_pre = cand & (~y0_band) & (dP_inward < 0.0)

            # --------------------------------------------------------
            # MP: strong rotation; BS: weak rotation
            # --------------------------------------------------------
            mp_gate = rot > float(rot_threshold_mp)
            bs_gate = rot < float(rot_threshold_mp)

            # Secondary split (rotation gates)
            bs_post = bs_pre & bs_gate  # inward increase, weak rotation
            mp_post = mp_pre & mp_gate  # inward decrease, strong rotation

            # Convert back to xarray and enforce exclusivity
            magnetopause_mask = xr.DataArray(mp_pre, coords=gradV.coords, dims=gradV.dims)
            bowshock_mask = xr.DataArray(bs_pre, coords=gradV.coords, dims=gradV.dims)
            bowshock_mask = bowshock_mask & (~magnetopause_mask)
    else:
        raise ValueError(use_slice)

    # Exclusion region (only meaningful when X is in-plane)
    if s in ("xy", "xz"):
        x_bad = x_plot < 0.75
        y_bad = (y_plot > -1.2) & (y_plot < 1.2)
        exclude_region = y_bad[:, None] & x_bad[None, :]
        bowshock_mask = bowshock_mask & (~exclude_region)
        magnetopause_mask = magnetopause_mask & (~exclude_region)

    bg_map = {
        "Bmag": Bmag,
        "Jmag": Jmag,
        "Pmag": Pmag,
    }

    if plot_id not in bg_map:
        raise ValueError(f"Invalid plot_id='{plot_id}'. Options: {list(bg_map)}")

    plot_bg = bg_map[plot_id].values

    if plot_id == "Pmag":
        plot_bg = plot_bg * 1e-6  # convert back to cm^-3

    return x_plot, y_plot, plot_bg, bowshock_mask, magnetopause_mask


def slice_axes_dims(use_slice: str):
    """
    For stacking masks consistently (2D):
      xy -> (Ny, Nx)
      xz -> (Nz, Nx)
      yz -> (Nz, Ny)
    """
    s = use_slice.lower().strip()
    if s == "xy":
        return ("Ny", "Nx")
    if s == "xz":
        return ("Nz", "Nx")
    if s == "yz":
        return ("Nz", "Ny")
    raise ValueError(use_slice)


def occupancy_and_bands(stack_bool: np.ndarray, thresholds=(0.25, 0.125, 0.0625)):
    """
    stack_bool: (T, H, W) boolean
    returns occupancy p (H,W) in [0,1] and masks at thresholds.
    """
    p = stack_bool.mean(axis=0)
    q1_thr, med_thr, q3_thr = thresholds
    q1mask  = (p >= q1_thr)
    medmask = (p >= med_thr)
    q3mask  = (p >= q3_thr)
    return p, q1mask, medmask, q3mask


def max_axis_distance(mask, x, y, width_km=100):
    """
    From a median contour mask, find the point within +/- width_km
    of axis=0 with maximum radial distance.
    Returns (x, axis, r) as Python floats.
    """
    pts = np.argwhere(mask)

    best = None
    best_r = -np.inf

    for iy, ix in pts:
        xv = float(x[ix])
        av = float(y[iy])   # Y (XY) or Z (XZ)

        if abs(av) <= (width_km / 2440.0):
            r = float(np.hypot(xv, av))  # guaranteed scalar

            if r > best_r:
                best_r = r
                best = (xv, av, r)

    return best