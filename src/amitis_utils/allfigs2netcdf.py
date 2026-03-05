#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import os

# cases = ["RPN_HNHV", "CPN_HNHV", "CPS_HNHV", "RPS_HNHV"]
# cases = ["RPN_Base", "RPS_Base", "CPN_Base", "CPS_Base"]
cases = ["RPS_Base"]

for case in cases:
    try:
        # --------------------------
        # Variables to merge
        # --------------------------
        vars_to_merge = [
            'den01', 'vx01', 'vy01', 'vz01',
            'den02', 'vx02', 'vy02', 'vz02',
            'den03', 'vx03', 'vy03', 'vz03',
            'den04', 'vx04', 'vy04', 'vz04',
            'Bx', 'Bx_tot', 'By', 'By_tot', 'Bz', 'Bz_tot',
            'Jx', 'Jy', 'Jz', 'Ex', 'Ey', 'Ez'
        ]

        if "Base" in case:
            base_path = f"/Volumes/T9/mercury/extreme/{case}/plane_product/"
            output_path = f"/Volumes/T9/mercury/extreme/{case}/plane_product/cube/"
            simsteps = list(range(85000, 104000 + 1, 1000))
        elif "HNHV" in case:
            base_path = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/"
            output_path = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/cube/"
            simsteps = list(range(115000, 350000 + 1, 1000))

        os.makedirs(output_path, exist_ok=True)

        RM = 2440.0
        axis_limits = [-4 * RM, 4 * RM]

        # --------------------------
        # Helper: load non-empty variables and trim ±4RM
        # --------------------------
        def load_plane_trim(path):
            try:
                ds = xr.open_dataset(path)
            except Exception as e:
                print(f"Failed to open {path}: {e}")
                return None

            try:
                ds_filtered = ds[[v for v in ds.data_vars if v in vars_to_merge and ds[v].size > 0]]

                # set magnetic field units
                mag_fields = ['Bx', 'Bx_tot', 'By', 'By_tot', 'Bz', 'Bz_tot']
                for var in mag_fields:
                    if var in ds_filtered:
                        ds_filtered[var].attrs['units'] = 'nT'

                # Trim coordinates to ±4RM
                for dim in ['Nx', 'Ny', 'Nz']:
                    if dim in ds_filtered.coords:
                        mask = (ds_filtered[dim] >= axis_limits[0]) & (ds_filtered[dim] <= axis_limits[1])
                        ds_filtered = ds_filtered.sel({dim: ds_filtered[dim].values[mask]})

                return ds_filtered

            except Exception as e:
                print(f"Failed processing dataset {path}: {e}")
                return None

        # --------------------------
        # Loop over timesteps
        # --------------------------
        base_time = np.datetime64('2000-01-01T00:00:00')

        for simstep in simsteps:
            print(f"\nProcessing simstep {"%06d" % simstep}")

            try:
                paths = {
                    'xy': os.path.join(base_path, f'all_xy/Amitis_{case}_{"%06d" % simstep}_xy_comp.nc'),
                    'xz': os.path.join(base_path, f'all_xz/Amitis_{case}_{"%06d" % simstep}_xz_comp.nc'),
                    'yz': os.path.join(base_path, f'all_yz/Amitis_{case}_{"%06d" % simstep}_yz_comp.nc')
                }

                ds_xy = load_plane_trim(paths['xy'])
                ds_xz = load_plane_trim(paths['xz'])
                ds_yz = load_plane_trim(paths['yz'])

                # If any plane failed, skip timestep
                if ds_xy is None or ds_xz is None or ds_yz is None:
                    print(f"Skipping simstep {simstep} due to missing plane.")
                    continue

                # Full coordinates
                Nx = np.unique(np.concatenate([ds_xy['Nx'].values, ds_xz['Nx'].values, ds_yz['Nx'].values]))
                Ny = np.unique(np.concatenate([ds_xy['Ny'].values, ds_xy['Ny'].values, ds_yz['Ny'].values]))
                Nz = np.unique(np.concatenate([ds_xz['Nz'].values, ds_yz['Nz'].values, ds_xy['Nz'].values]))

                time_len = 1
                merged_ds_dict = {
                    var: np.full((time_len, len(Nz), len(Ny), len(Nx)), np.nan)
                    for var in vars_to_merge
                }

                # Insert planes
                for var in vars_to_merge:
                    try:
                        if var in ds_xy.data_vars:
                            z_idx = [np.argmin(np.abs(Nz - z)) for z in ds_xy['Nz'].values]
                            y_idx = [np.argmin(np.abs(Ny - y)) for y in ds_xy['Ny'].values]
                            x_idx = [np.argmin(np.abs(Nx - x)) for x in ds_xy['Nx'].values]
                            merged_ds_dict[var][0][np.ix_(z_idx, y_idx, x_idx)] = ds_xy[var].values[0]

                        if var in ds_xz.data_vars:
                            z_idx = [np.argmin(np.abs(Nz - z)) for z in ds_xz['Nz'].values]
                            y_idx = [np.argmin(np.abs(Ny - y)) for y in ds_xz['Ny'].values]
                            x_idx = [np.argmin(np.abs(Nx - x)) for x in ds_xz['Nx'].values]
                            merged_ds_dict[var][0][np.ix_(z_idx, y_idx, x_idx)] = ds_xz[var].values[0]

                        if var in ds_yz.data_vars:
                            z_idx = [np.argmin(np.abs(Nz - z)) for z in ds_yz['Nz'].values]
                            y_idx = [np.argmin(np.abs(Ny - y)) for y in ds_yz['Ny'].values]
                            x_idx = [np.argmin(np.abs(Nx - x)) for x in ds_yz['Nx'].values]
                            merged_ds_dict[var][0][np.ix_(z_idx, y_idx, x_idx)] = ds_yz[var].values[0]

                    except Exception as e:
                        print(f"Failed merging variable {var} at simstep {simstep}: {e}")
                        continue

                # Create dataset
                merged_ds = xr.Dataset(
                    {var: (['time', 'Nz', 'Ny', 'Nx'], merged_ds_dict[var])
                     for var in vars_to_merge},
                    coords={
                        'time': [base_time + np.timedelta64(simstep, 's')],
                        'Nx': Nx,
                        'Ny': Ny,
                        'Nz': Nz,
                    }
                )

                # Save file
                output_file = os.path.join(
                    output_path,
                    f"Amitis_{case}_{simstep}_merged_4RM.nc"
                )

                comp = {var: {'zlib': True, 'complevel': 4}
                        for var in merged_ds.data_vars}

                merged_ds.to_netcdf(output_file, encoding=comp)
                print(f"Saved trimmed merged cube to {output_file}")

            except Exception as e:
                print(f"Unexpected failure at simstep {simstep}: {e}")
                continue

    except Exception as e:
        print(f"Case-level failure for {case}: {e}")
        continue