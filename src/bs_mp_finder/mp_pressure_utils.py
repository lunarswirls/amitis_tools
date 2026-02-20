#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr

MU0 = 4e-7 * np.pi          # N/A^2 (H/m)
MP  = 1.67262192369e-27     # kg (proton mass)
NT_TO_T = 1e-9
CM3_TO_M3 = 1e6
KMS_TO_MS = 1e3

def compute_mp_mask_pressure_balance(
    f_3d: str,
    debug: bool = False,
    RM_M: float = 2440.0e3,
    r_min_rm: float = 1.0,
    r_max_rm: float = 4.0,
    rel_tol: float = 0.15,
    abs_tol_pa: float = 0.0,
):
    """
    Magnetopause boundary from pressure balance:
        P_B = B^2/(2 mu0)  and  P_dyn = rho * V^2

    Returns:
      x_coords, y_coords, z_coords, PB_pa, Pdyn_pa, mp_mask_da
    """
    ds = xr.open_dataset(f_3d)

    # ---- coordinates (same as your fetch_coords, compact) ----
    xmin, xmax, dx = float(ds.full_xmin), float(ds.full_xmax), float(ds.full_dx)
    ymin, ymax, dy = float(ds.full_ymin), float(ds.full_ymax), float(ds.full_dy)
    zmin, zmax, dz = float(ds.full_zmin), float(ds.full_zmax), float(ds.full_dz)

    x_coords = np.arange(xmin, xmax, dx) / RM_M
    y_coords = np.arange(ymin, ymax, dy) / RM_M
    z_coords = np.arange(zmin, zmax, dz) / RM_M

    # ---- fields (transpose Nz, Ny, Nx -> Nx, Ny, Nz) ----
    def t3(v):  # helper
        return np.transpose(v, (2, 1, 0))

    BX = t3(ds["Bx_tot"].isel(time=0).values)  # nT
    BY = t3(ds["By_tot"].isel(time=0).values)
    BZ = t3(ds["Bz_tot"].isel(time=0).values)

    # species velocities (km/s)
    vxp = t3(ds["vx01"].isel(time=0).values + ds["vx02"].isel(time=0).values)
    vyp = t3(ds["vy01"].isel(time=0).values + ds["vy02"].isel(time=0).values)
    vzp = t3(ds["vz01"].isel(time=0).values + ds["vz02"].isel(time=0).values)

    vxa = t3(ds["vx03"].isel(time=0).values + ds["vx04"].isel(time=0).values)
    vya = t3(ds["vy03"].isel(time=0).values + ds["vy04"].isel(time=0).values)
    vza = t3(ds["vz03"].isel(time=0).values + ds["vz04"].isel(time=0).values)

    # densities (cm^-3)
    np_cm3 = t3(ds["den01"].isel(time=0).values + ds["den02"].isel(time=0).values)
    na_cm3 = t3(ds["den03"].isel(time=0).values + ds["den04"].isel(time=0).values)

    ds.close()

    # ---- exclusion volume ----
    Xg, Yg, Zg = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    Rg = np.sqrt(Xg**2 + Yg**2 + Zg**2)
    outside_body = (Rg >= r_min_rm) & (Rg < r_max_rm)

    # ---- P_B (Pa) ----
    B_T = np.sqrt(BX**2 + BY**2 + BZ**2) * NT_TO_T
    PB_pa = (B_T**2) / (2.0 * MU0)

    # ---- P_dyn,sw (Pa) ----
    # Convert to SI
    np_m3 = np_cm3 * CM3_TO_M3
    na_m3 = na_cm3 * CM3_TO_M3

    # Mass density: rho = m_p (n_p + 4 n_alpha)
    rho = MP * (np_m3 + 4.0 * na_m3)  # kg/m^3

    # Mass-weighted bulk velocity (optional but sensible)
    # u = (m_p n_p v_p + 4 m_p n_a v_a) / (m_p n_p + 4 m_p n_a)
    #   = (n_p v_p + 4 n_a v_a) / (n_p + 4 n_a)
    denom = (np_m3 + 4.0 * na_m3)
    denom_safe = np.where(denom > 0, denom, np.nan)

    ux = (np_m3 * vxp + 4.0 * na_m3 * vxa) / denom_safe
    uy = (np_m3 * vyp + 4.0 * na_m3 * vya) / denom_safe
    uz = (np_m3 * vzp + 4.0 * na_m3 * vza) / denom_safe

    Umag = np.sqrt(ux**2 + uy**2 + uz**2) * KMS_TO_MS  # m/s
    Pdyn_pa = rho * (Umag**2)  # Pa

    # ---- define "equal" on-grid (relative + optional absolute tol) ----
    # |PB - Pdyn| <= rel_tol * max(PB, Pdyn) + abs_tol
    denom_bal = np.maximum(PB_pa, Pdyn_pa)
    denom_bal = np.where(np.isfinite(denom_bal) & (denom_bal > 0), denom_bal, np.nan)

    equal_mask = np.abs(PB_pa - Pdyn_pa) <= (rel_tol * denom_bal + abs_tol_pa)
    mp_mask = equal_mask & outside_body

    # mask out invalids
    PB_pa = np.where(outside_body, PB_pa, np.nan)
    Pdyn_pa = np.where(outside_body, Pdyn_pa, np.nan)

    if debug:
        frac = np.nanmean(mp_mask.astype(float))
        print(f"PB (Pa) min/med/max: {np.nanmin(PB_pa):.3e}, {np.nanmedian(PB_pa):.3e}, {np.nanmax(PB_pa):.3e}")
        print(f"Pdyn (Pa) min/med/max: {np.nanmin(Pdyn_pa):.3e}, {np.nanmedian(Pdyn_pa):.3e}, {np.nanmax(Pdyn_pa):.3e}")
        print(f"MP boundary fraction (within tol): {frac:.3e}")

    mp_mask_da = xr.DataArray(
        mp_mask.astype(np.uint8),
        coords={"Nx": x_coords, "Ny": y_coords, "Nz": z_coords},
        dims=["Nx", "Ny", "Nz"],
        name="mp_mask_pressure_balance"
    )

    return x_coords, y_coords, z_coords, PB_pa, Pdyn_pa, mp_mask_da
