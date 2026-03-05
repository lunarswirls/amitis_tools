#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import src.bs_mp_finder.mp_pressure_utils as boundary_utils

# ----------------------------
# SETTINGS
# ----------------------------
debug = False

case = "RPS"
mode = "HNHV"
sim_steps = range(85000, 104000 + 1, 1000)

out_dir = f"/Users/danywaller/Projects/mercury/extreme/magnetopause_3D_timeseries/{case}_{mode}/"
os.makedirs(out_dir, exist_ok=True)

# mask finder params
rmin_rm = 1.0
rmax_rm = 2.5
rel_tol = 0.1
abs_tol_mp_pressure = 0.0

# time conversion
step_to_seconds = 0.002

# output
out_nc = os.path.join(out_dir, f"{case}_{mode}_mp_mask_timeseries.nc")

# ----------------------------
# FILE BUILDER
# ----------------------------
def build_file_path(sim_step: int) -> str:
    if sim_step <= 115000:
        fmode = "Base"
        input_folder = f"/Volumes/T9/mercury/extreme/{case}_Base/plane_product/cube/"
    else:
        fmode = "HNHV"
        input_folder = f"/Volumes/T9/mercury/extreme/High_{mode}/{case}_{mode}/plane_product/cube/"
    return os.path.join(input_folder, f"Amitis_{case}_{fmode}_{sim_step:06d}_merged_4RM.nc")

# ----------------------------
# MAIN LOOP: compute masks
# ----------------------------
mp_masks = []
t_seconds = []
step_kept = None

x_ref = y_ref = z_ref = None

for sim_step in sim_steps:
    f_3d = build_file_path(sim_step)
    if not os.path.exists(f_3d):
        print("File not found:", f_3d)
        continue

    tsec = sim_step * step_to_seconds
    print(f"Processing step={sim_step}  t={tsec:.3f} s")

    x, y, z, PB_pa, Pdyn_pa, mp_mask_da = boundary_utils.compute_mp_mask_pressure_balance(
        f_3d,
        debug=debug,
        r_min_rm=rmin_rm,
        r_max_rm=rmax_rm,
        rel_tol=rel_tol,
        abs_tol_pa=abs_tol_mp_pressure,
    )

    # ensure consistent grid across time
    if x_ref is None:
        x_ref, y_ref, z_ref = np.array(x), np.array(y), np.array(z)
    else:
        if (len(x_ref) != len(x)) or (len(y_ref) != len(y)) or (len(z_ref) != len(z)):
            raise ValueError("Grid size changed across timesteps; cannot stack into one file.")
        if (not np.allclose(x_ref, x)) or (not np.allclose(y_ref, y)) or (not np.allclose(z_ref, z)):
            raise ValueError("Grid coordinates changed across timesteps; cannot stack into one file.")

    mp_mask_bool = (mp_mask_da.values > 0)

    # store as uint8 (0/1) for compactness
    mp_masks.append(mp_mask_bool.astype(np.uint8))
    t_seconds.append(float(tsec))

# stack into (time, x, y, z)
if len(mp_masks) == 0:
    raise RuntimeError("No masks computed (all files missing?)")

mp_mask_4d = np.stack(mp_masks, axis=0)

# ----------------------------
# BUILD DATASET
# ----------------------------
ds_new = xr.Dataset(
    data_vars=dict(
        mp_mask=(("time", "x", "y", "z"), mp_mask_4d),
    ),
    coords=dict(
        time=("time", np.array(t_seconds, dtype=np.float64)),
        x=("x", np.array(x_ref, dtype=np.float32)),
        y=("y", np.array(y_ref, dtype=np.float32)),
        z=("z", np.array(z_ref, dtype=np.float32)),
    ),
    attrs=dict(
        case=case,
        mode=mode,
        time_units="seconds",
        description="Magnetopause mask timeseries from pressure-balance method; mp_mask is 0/1 uint8.",
        rmin_rm=float(rmin_rm),
        rmax_rm=float(rmax_rm),
        rel_tol=float(rel_tol),
        abs_tol_mp_pressure=float(abs_tol_mp_pressure),
        step_to_seconds=float(step_to_seconds),
    )
)

ds_new["mp_mask"].attrs.update(
    dict(
        long_name="magnetopause_mask",
        values="0=outside, 1=inside",
    )
)

# ----------------------------
# MERGE WITH EXISTING FILE
# ----------------------------
if os.path.exists(out_nc):
    print("Existing file found — appending new timesteps")

    ds_old = xr.open_dataset(out_nc)
    ds_old.load()
    ds_old.close()

    if not (
        np.allclose(ds_old.x.values, ds_new.x.values)
        and np.allclose(ds_old.y.values, ds_new.y.values)
        and np.allclose(ds_old.z.values, ds_new.z.values)
    ):
        raise ValueError("Existing file grid differs from new data.")

    new_times = ds_new.time.values
    old_times = ds_old.time.values
    keep_mask = ~np.isin(new_times, old_times)

    ds_new = ds_new.isel(time=keep_mask)

    if ds_new.sizes["time"] == 0:
        print("No new timesteps to append.")
        ds_out = ds_old
    else:
        ds_out = xr.concat([ds_old, ds_new], dim="time")
        ds_out = ds_out.sortby("time")

# ----------------------------
# SAVE DATASET
# ----------------------------
comp = dict(zlib=True, complevel=4)
encoding = {
    "mp_mask": {**comp, "dtype": "uint8"},
    "time": {"dtype": "float64"},
    "x": {"dtype": "float32"},
    "y": {"dtype": "float32"},
    "z": {"dtype": "float32"},
}

tmp_nc = out_nc + ".tmp"

ds_out.to_netcdf(
    tmp_nc,
    format="NETCDF4",
    engine="netcdf4",
    encoding=encoding
)

os.replace(tmp_nc, out_nc)

print("Saved:", out_nc)