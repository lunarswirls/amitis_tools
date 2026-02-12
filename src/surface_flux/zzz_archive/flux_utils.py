#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def compute_radial_flux(ds, x, y, z, sum_fields="all"):
    """
    Compute radial particle flux F_r = n (v_bulk · r_hat) where v_bulk is the
    density-weighted bulk velocity

    :param ds: xarray.Dataset
    :param x: numpy array
    :param y: numpy array
    :param z: numpy array
    :param sum_fields: string, default="all" returns protons+alphas, "protons" or "alphas" return respective species

    :return flux: radial particle flux F_r in cm^-2 s^-1
    :return v_dot_r: radial velocity in km/s
    """

    if sum_fields == "all":
        # densities converted from cm^-3 to km^-3
        den01 = ds["den01"].isel(time=0).values * 1e15
        den02 = ds["den02"].isel(time=0).values * 1e15
        den03 = ds["den03"].isel(time=0).values * 1e15
        den04 = ds["den04"].isel(time=0).values * 1e15

        # sum all densities to get total density [km^-3]
        den_tot = (den01 + den02 + den03 + den04)

        # density-weighted bulk velocity [km/s]
        vx_bulk = np.zeros_like(den_tot)
        vy_bulk = np.zeros_like(den_tot)
        vz_bulk = np.zeros_like(den_tot)

        # mask where total density is > 0
        mask = den_tot > 0

        vx_bulk[mask] = (den01[mask] * ds["vx01"].isel(time=0).values[mask] +
                         den02[mask] * ds["vx02"].isel(time=0).values[mask] +
                         den03[mask] * ds["vx03"].isel(time=0).values[mask] +
                         den04[mask] * ds["vx04"].isel(time=0).values[mask]) / den_tot[mask]

        vy_bulk[mask] = (den01[mask] * ds["vy01"].isel(time=0).values[mask] +
                         den02[mask] * ds["vy02"].isel(time=0).values[mask] +
                         den03[mask] * ds["vy03"].isel(time=0).values[mask] +
                         den04[mask] * ds["vy04"].isel(time=0).values[mask]) / den_tot[mask]

        vz_bulk[mask] = (den01[mask] * ds["vz01"].isel(time=0).values[mask] +
                         den02[mask] * ds["vz02"].isel(time=0).values[mask] +
                         den03[mask] * ds["vz03"].isel(time=0).values[mask] +
                         den04[mask] * ds["vz04"].isel(time=0).values[mask]) / den_tot[mask]

        # build position grids (Nz, Ny, Nx)
        Zg, Yg, Xg = np.meshgrid(z, y, x, indexing="ij")

        r_mag = np.sqrt(Xg ** 2 + Yg ** 2 + Zg ** 2)
        mask_r = r_mag > 0

        # Inward radial unit vector
        nx = np.zeros_like(r_mag)
        ny = np.zeros_like(r_mag)
        nz = np.zeros_like(r_mag)

        nx[mask_r] = -Xg[mask_r] / r_mag[mask_r]
        ny[mask_r] = -Yg[mask_r] / r_mag[mask_r]
        nz[mask_r] = -Zg[mask_r] / r_mag[mask_r]

        # radial velocity [km/s]
        v_dot_r = vx_bulk * nx + vy_bulk * ny + vz_bulk * nz
        flux = den_tot * v_dot_r

        # convert from km^-2 s^-1 to cm^-2 s^-1
        flux *= 1e-10

        return flux, v_dot_r
    elif sum_fields == "protons":
        # densities converted from cm^-3 to km^-3
        den01 = ds["den01"].isel(time=0).values * 1e15
        den02 = ds["den02"].isel(time=0).values * 1e15

        # sum all densities to get total density [km^-3]
        den_tot = (den01 + den02)

        # density-weighted bulk velocity [km/s]
        vx_bulk = np.zeros_like(den_tot)
        vy_bulk = np.zeros_like(den_tot)
        vz_bulk = np.zeros_like(den_tot)

        # mask where total density is > 0
        mask = den_tot > 0

        vx_bulk[mask] = (den01[mask] * ds["vx01"].isel(time=0).values[mask] +
                         den02[mask] * ds["vx02"].isel(time=0).values[mask]) / den_tot[mask]

        vy_bulk[mask] = (den01[mask] * ds["vy01"].isel(time=0).values[mask] +
                         den02[mask] * ds["vy02"].isel(time=0).values[mask]) / den_tot[mask]

        vz_bulk[mask] = (den01[mask] * ds["vz01"].isel(time=0).values[mask] +
                         den02[mask] * ds["vz02"].isel(time=0).values[mask]) / den_tot[mask]

        # build position grids (Nz, Ny, Nx)
        Zg, Yg, Xg = np.meshgrid(z, y, x, indexing="ij")

        r_mag = np.sqrt(Xg ** 2 + Yg ** 2 + Zg ** 2)
        mask_r = r_mag > 0

        # Inward radial unit vector
        nx = np.zeros_like(r_mag)
        ny = np.zeros_like(r_mag)
        nz = np.zeros_like(r_mag)

        nx[mask_r] = -Xg[mask_r] / r_mag[mask_r]
        ny[mask_r] = -Yg[mask_r] / r_mag[mask_r]
        nz[mask_r] = -Zg[mask_r] / r_mag[mask_r]

        # radial velocity [km/s]
        v_dot_r = vx_bulk * nx + vy_bulk * ny + vz_bulk * nz
        flux = den_tot * v_dot_r

        # convert from km^-2 s^-1 to cm^-2 s^-1
        flux *= 1e-10

        return flux, v_dot_r

    if sum_fields == "alphas":
        # densities converted from cm^-3 to km^-3
        den03 = ds["den03"].isel(time=0).values * 1e15
        den04 = ds["den04"].isel(time=0).values * 1e15

        # sum all densities to get total density [km^-3]
        den_tot = (den03 + den04)

        # density-weighted bulk velocity [km/s]
        vx_bulk = np.zeros_like(den_tot)
        vy_bulk = np.zeros_like(den_tot)
        vz_bulk = np.zeros_like(den_tot)

        # mask where total density is > 0
        mask = den_tot > 0

        vx_bulk[mask] = (den03[mask] * ds["vx03"].isel(time=0).values[mask] +
                         den04[mask] * ds["vx04"].isel(time=0).values[mask]) / den_tot[mask]

        vy_bulk[mask] = (den03[mask] * ds["vy03"].isel(time=0).values[mask] +
                         den04[mask] * ds["vy04"].isel(time=0).values[mask]) / den_tot[mask]

        vz_bulk[mask] = (den03[mask] * ds["vz03"].isel(time=0).values[mask] +
                         den04[mask] * ds["vz04"].isel(time=0).values[mask]) / den_tot[mask]

        # build position grids (Nz, Ny, Nx)
        Zg, Yg, Xg = np.meshgrid(z, y, x, indexing="ij")

        r_mag = np.sqrt(Xg ** 2 + Yg ** 2 + Zg ** 2)
        mask_r = r_mag > 0

        # Inward radial unit vector
        nx = np.zeros_like(r_mag)
        ny = np.zeros_like(r_mag)
        nz = np.zeros_like(r_mag)

        nx[mask_r] = -Xg[mask_r] / r_mag[mask_r]
        ny[mask_r] = -Yg[mask_r] / r_mag[mask_r]
        nz[mask_r] = -Zg[mask_r] / r_mag[mask_r]

        # radial velocity [km/s]
        v_dot_r = vx_bulk * nx + vy_bulk * ny + vz_bulk * nz
        flux = den_tot * v_dot_r

        # convert from km^-2 s^-1 to cm^-2 s^-1
        flux *= 1e-10

        return flux, v_dot_r