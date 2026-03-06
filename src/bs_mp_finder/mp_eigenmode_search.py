#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Method
------------
1) Builds a fixed dayside (theta,phi) grid and samples r(theta,phi,t) from mp_mask
   using the "outermost True along ray" method.
2) Converts to stand-off Δr = r - 1.
3) For each time window:
   - pre:  t < 250 s
   - post: t > 600 s
   Computes:
     a) presence fraction map: fraction of times Δr >= R_PRESENT
     b) mean stand-off map: mean(Δr) where the boundary is sufficiently present
     c) dominant-frequency FFT amplitude & phase maps of δ(Δr)(t) on the retained patch
4) Plots everything in latitude/longitude (deg), with subsolar meridian at lon=0.

Conventions
-----------
- Solar wind is along -X, so Sun/subsolar direction is +X.
  With phi defined from +X toward +Y, lon = phi and lon=0 is subsolar.
- theta is colatitude => latitude = 90° - theta°
- FFT constraints: Nyquist = 1/(2*dt_snap), df ~ 1/(Nt*dt_snap)

Notes on missing data
---------------------
- np.nanmean warns for all-NaN slices; avoid that by only calling nanmean on "good" pixels.
- FFT cannot handle NaNs; fill gaps in time by linear interpolation before FFT.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import src.bs_mp_finder.mp_pressure_utils as boundary_utils

# ----------------------------
# SETTINGS
# ----------------------------
debug = False

case = "CPN"
mode = "HNHV"
sim_steps = list(range(95000, 350000 + 1, 1000))
dt_sim = 0.002
out_dir = f"/Users/danywaller/Projects/mercury/extreme/magnetopause_eigenmode/{case}_{mode}/"
os.makedirs(out_dir, exist_ok=True)

RMIN_RM = 1.0
RMAX_RM = 4.0
REL_TOL = 0.15
ABS_TOL_PA = 0.0

N_THETA = 61
N_PHI = 121
EPS_ANG = 1e-6

DR_OVERRIDE = None

T_PRE_MAX = 240.0
T_POST_MIN = 600.0

# Presence threshold in stand-off Δr = r-1 (RM)
R_PRESENT = 0.0  # include "at surface"

MIN_PRES_FRAC = 0.6  # pixel must satisfy presence >= this fraction of frames in the window

FMIN_PICK = 0.0
USE_HANN = True

PHASE_CMAP = "twilight"
AMP_CMAP = "viridis"
PRES_CMAP = "magma"
MEAN_CMAP = "cividis"


# ----------------------------
# FILE PATH
# ----------------------------
def build_file_path(sim_step: int) -> str:
    if sim_step < 115000:
        input_folder = f"/Volumes/data_backup/mercury/extreme/{case}_Base/plane_product/object/"
        return os.path.join(input_folder, f"Amitis_{case}_Base_{sim_step:06d}_xz_comp.nc")
    input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/plane_product/object/"
    return os.path.join(input_folder, f"Amitis_{case}_{mode}_{sim_step:06d}_xz_comp.nc")


# ----------------------------
# THETA/PHI + RAYS
# ----------------------------
def make_dayside_theta_phi(n_theta=N_THETA, n_phi=N_PHI, eps=EPS_ANG):
    theta = np.linspace(eps, np.pi - eps, n_theta)                 # colatitude
    phi   = np.linspace(-0.5*np.pi + eps, 0.5*np.pi - eps, n_phi)  # dayside
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    return theta, phi, TH, PH


def rhat_from_theta_phi(TH, PH):
    # x=r sinθ cosφ, y=r sinθ sinφ, z=r cosθ
    sx = np.sin(TH) * np.cos(PH)
    sy = np.sin(TH) * np.sin(PH)
    sz = np.cos(TH)
    return sx, sy, sz


def precompute_ray_indices(x, y, z, TH, PH, r_min=RMIN_RM, r_max=RMAX_RM, dr=None):
    if dr is None:
        dx = float(np.min(np.diff(x)))
        dy = float(np.min(np.diff(y)))
        dz = float(np.min(np.diff(z)))
        dr = min(dx, dy, dz)

    r_samp = np.arange(r_min, r_max + 0.5*dr, dr)

    sx, sy, sz = rhat_from_theta_phi(TH, PH)
    X = r_samp[:, None, None] * sx[None, :, :]
    Y = r_samp[:, None, None] * sy[None, :, :]
    Z = r_samp[:, None, None] * sz[None, :, :]

    ix = np.searchsorted(x, X)
    iy = np.searchsorted(y, Y)
    iz = np.searchsorted(z, Z)

    def nearest_idx(grid, idx, coord):
        idx0 = np.clip(idx, 1, len(grid) - 1)
        left = grid[idx0 - 1]
        right = grid[idx0]
        choose_left = (np.abs(coord - left) <= np.abs(coord - right))
        return np.where(choose_left, idx0 - 1, idx0)

    ixn = nearest_idx(x, ix, X).astype(np.int32)
    iyn = nearest_idx(y, iy, Y).astype(np.int32)
    izn = nearest_idx(z, iz, Z).astype(np.int32)

    valid = (ixn >= 0) & (ixn < len(x)) & (iyn >= 0) & (iyn < len(y)) & (izn >= 0) & (izn < len(z))
    return r_samp, ixn, iyn, izn, valid, float(dr)


def r_outermost_true(mp_mask, r_samp, ix, iy, iz, valid):
    m = np.zeros_like(valid, dtype=bool)
    v = valid
    m[v] = mp_mask[ix[v], iy[v], iz[v]]

    any_true = np.any(m, axis=0)
    m_rev = m[::-1, :, :]
    j_from_end = np.argmax(m_rev, axis=0)
    j_last = (len(r_samp) - 1) - j_from_end

    r_map = np.full(any_true.shape, np.nan, dtype=float)
    r_map[any_true] = r_samp[j_last[any_true]]
    return r_map


# ----------------------------
# UTIL: fill NaNs in time (for FFT)
# ----------------------------
def fill_nans_time_linear(arr_t, good_mask):
    Nt = arr_t.shape[0]
    tt = np.arange(Nt)
    filled = np.zeros_like(arr_t, dtype=float)
    for it in range(arr_t.shape[1]):
        for ip in range(arr_t.shape[2]):
            if not good_mask[it, ip]:
                continue
            s = arr_t[:, it, ip]
            m = np.isfinite(s)
            if m.sum() < 3:
                continue
            if m.all():
                filled[:, it, ip] = s
            else:
                filled[:, it, ip] = np.interp(tt, tt[m], s[m])
    return filled


# ----------------------------
# LAT/LON + PLOTTING
# ----------------------------
def theta_phi_to_latlon_deg(theta, phi):
    # lat = 90° - theta° (theta is colatitude)
    lat_deg = 90.0 - np.degrees(theta)
    # lon = phi in this convention (phi from +X toward +Y)
    lon_deg = np.degrees(phi)
    return lat_deg, lon_deg


def imshow_map_latlon(lat_deg, lon_deg, M, title, png_path, cmap="viridis",
                      vmin=None, vmax=None, cbar_label=None):
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.2), dpi=170, constrained_layout=True)
    im = ax.imshow(
        M, origin="lower", aspect="auto",
        extent=[lon_deg[0], lon_deg[-1], lat_deg[0], lat_deg[-1]],
        cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.axvline(0.0, color="k", lw=1.0, alpha=0.6)  # subsolar meridian (Sun along +X)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    if cbar_label:
        cbar.set_label(cbar_label)
    fig.savefig(png_path)
    plt.close(fig)


def plot_amp_phase_latlon(lat_deg, lon_deg, amp, phase, title, png_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=170, constrained_layout=True)

    im0 = axes[0].imshow(
        amp, origin="lower", aspect="auto",
        extent=[lon_deg[0], lon_deg[-1], lat_deg[0], lat_deg[-1]],
        cmap=AMP_CMAP
    )
    axes[0].set_title("Amplitude |X(f)|")
    axes[0].set_xlabel("Longitude (°)")
    axes[0].set_ylabel("Latitude (°)")
    axes[0].axvline(0.0, color="k", lw=1.0, alpha=0.6)
    fig.colorbar(im0, ax=axes[0], shrink=0.9)

    im1 = axes[1].imshow(
        phase, origin="lower", aspect="auto",
        extent=[lon_deg[0], lon_deg[-1], lat_deg[0], lat_deg[-1]],
        cmap=PHASE_CMAP, vmin=-np.pi, vmax=np.pi
    )
    axes[1].set_title("Phase arg(X(f))")
    axes[1].set_xlabel("Longitude (°)")
    axes[1].set_ylabel("Latitude (°)")
    axes[1].axvline(0.0, color="k", lw=1.0, alpha=0.6)
    cbar = fig.colorbar(im1, ax=axes[1], shrink=0.9)
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    fig.suptitle(title)
    fig.savefig(png_path)
    plt.close(fig)


# ----------------------------
# SCRIPT START
# ----------------------------
print("=== MP dayside oscillation maps (lat/lon) ===")
print(f"Upstream flow v_sw = [-400,0,0] km/s => Sun direction +X => lon=0° is subsolar.")
print(f"case={case} mode={mode}")
print(f"sim_steps: {sim_steps[0]}..{sim_steps[-1]} step={sim_steps[1]-sim_steps[0]} (N={len(sim_steps)})")
print(f"Presence threshold (stand-off): Δr=r-1 >= {R_PRESENT} RM, MIN_PRES_FRAC={MIN_PRES_FRAC}")

theta, phi, TH, PH = make_dayside_theta_phi()
lat_deg, lon_deg = theta_phi_to_latlon_deg(theta, phi)

sx, sy, sz = rhat_from_theta_phi(TH, PH)
print(f"Dayside rays check: frac(sx>0)={np.mean(sx>0):.3f} (should be 1.0 for dayside)")

# Storage
t_list, step_list, r_maps = [], [], []
cached = False
x_cache = y_cache = z_cache = None
ray_cache = None
n_missing = 0

for n, sim_step in enumerate(sim_steps):
    f3d = build_file_path(sim_step)
    if not os.path.exists(f3d):
        print(f"[WARN] missing ({n+1}/{len(sim_steps)}): {f3d}")
        n_missing += 1
        continue

    tsec = sim_step * dt_sim
    print(f"\n--- ({n+1}/{len(sim_steps)}) sim_step={sim_step} t={tsec:.3f}s ---")
    print(f"File: {os.path.basename(f3d)}")

    x, y, z, PB_pa, Pdyn_pa, mp_mask_da = boundary_utils.compute_mp_mask_pressure_balance(
        f3d,
        debug=debug,
        r_min_rm=RMIN_RM,
        r_max_rm=RMAX_RM,
        rel_tol=REL_TOL,
        abs_tol_pa=ABS_TOL_PA,
    )
    mp_mask = (mp_mask_da.values > 0)

    dxs = np.diff(x); dys = np.diff(y); dzs = np.diff(z)
    print(f"Grid: Nx,Ny,Nz = {len(x)},{len(y)},{len(z)} | "
          f"dx~{float(np.median(dxs)):.5f} dy~{float(np.median(dys)):.5f} dz~{float(np.median(dzs)):.5f}")

    if not cached:
        print("Building ray cache...")
        r_samp, ix, iy, iz, valid, dr = precompute_ray_indices(
            x, y, z, TH, PH, r_min=RMIN_RM, r_max=RMAX_RM, dr=DR_OVERRIDE
        )
        ray_cache = (r_samp, ix, iy, iz, valid, dr)
        x_cache, y_cache, z_cache = x, y, z
        cached = True
        print(f"Ray cache built: Nr={len(r_samp)}, dr={dr:.5f} RM, valid_frac={np.mean(valid):.3f}")
    else:
        grid_changed = (len(x) != len(x_cache)) or (len(y) != len(y_cache)) or (len(z) != len(z_cache)) \
                       or (not np.allclose(x, x_cache)) or (not np.allclose(y, y_cache)) or (not np.allclose(z, z_cache))
        if grid_changed:
            print("[INFO] Grid changed -> refreshing ray cache...")
            r_samp, ix, iy, iz, valid, dr = precompute_ray_indices(
                x, y, z, TH, PH, r_min=RMIN_RM, r_max=RMAX_RM, dr=DR_OVERRIDE
            )
            ray_cache = (r_samp, ix, iy, iz, valid, dr)
            x_cache, y_cache, z_cache = x, y, z
            print(f"Ray cache refreshed: Nr={len(r_samp)}, dr={dr:.5f} RM, valid_frac={np.mean(valid):.3f}")

    r_samp, ix, iy, iz, valid, dr = ray_cache
    r_map = r_outermost_true(mp_mask, r_samp, ix, iy, iz, valid)

    print(f"r_map: finite_frac={np.isfinite(r_map).mean():.3f}, "
          f"rmin={np.nanmin(r_map):.3f}, rmax={np.nanmax(r_map):.3f}")

    t_list.append(float(tsec))
    step_list.append(int(sim_step))
    r_maps.append(r_map.astype(np.float32))

print(f"\nDone reading timesteps. Missing files: {n_missing}")
if len(t_list) < 8:
    raise RuntimeError("Too few valid timesteps.")

t = np.array(t_list, dtype=float)
steps_ok = np.array(step_list, dtype=int)
r_maps = np.stack(r_maps, axis=0)

dt_snap = float(np.median(np.diff(t)))
nyq = 0.5 / dt_snap
print(f"Effective dt_snap ~ {dt_snap:.3f} s => Nyquist ~ {nyq:.4f} Hz.")

pre_idx = np.where(t < T_PRE_MAX)[0]
post_idx = np.where(t > T_POST_MIN)[0]
print(f"Pre window frames: {pre_idx.size} (t<{T_PRE_MAX})")
print(f"Post window frames: {post_idx.size} (t>{T_POST_MIN})")

# phase reference at (lat=0, lon=0)
it_ref = int(np.argmin(np.abs(lat_deg - 0.0)))
ip_ref = int(np.argmin(np.abs(lon_deg - 0.0)))
print(f"Phase reference point ~ lat={lat_deg[it_ref]:.2f}°, lon={lon_deg[ip_ref]:.2f}°")


def analyze_window(window_name, idx):
    print(f"\n=== Analyzing window: {window_name} ===")
    tw = t[idx]
    rw_r = r_maps[idx, :, :]      # r
    rw = rw_r - 1.0               # stand-off Δr

    Nt = rw.shape[0]
    Trec = Nt * dt_snap
    df = 1.0 / Trec if Trec > 0 else np.nan  # df ~ 1/T

    pres = (np.isfinite(rw)) & (rw >= R_PRESENT)
    pres_count = pres.sum(axis=0)
    pres_frac = pres_count / float(Nt)

    MIN_PRESENT = max(8, int(MIN_PRES_FRAC * Nt))
    good = pres_count >= MIN_PRESENT

    # Summary print of what maps mean for this window
    good_frac = float(good.mean())
    print(f"[{window_name}] Window length: Nt={Nt}, dt={dt_snap:.3f}s, T={Trec:.1f}s => df~{df:.4f} Hz.")
    print(f"[{window_name}] Presence criterion: Δr>= {R_PRESENT} R_M in >= {MIN_PRESENT}/{Nt} frames ({MIN_PRES_FRAC*100:.0f}%).")
    print(f"[{window_name}] Interpreting maps:")
    print(f"  - presence_frac(lat,lon): fraction of times MP stand-off meets the criterion.")
    print(f"  - mean_standoff(lat,lon): time-mean Δr where MP is present often enough (good pixels).")
    print(f"  - amp/phase maps: dominant-frequency oscillation of Δr about its mean (on good pixels), phase referenced to (lat=0°, lon=0°).")
    print(f"[{window_name}] Coverage: good pixels = {good.sum()}/{good.size} = {good_frac*100:.2f}% of dayside grid.")

    mean_standoff = np.full((rw.shape[1], rw.shape[2]), np.nan, dtype=float)
    # avoid empty-slice warning by only computing where good
    if good.any():
        mean_standoff[good] = np.nanmean(rw[:, good], axis=0)

    drw = rw - mean_standoff[None, :, :]
    drw[:, ~good] = np.nan

    # Fill NaNs for FFT
    drw_filled = fill_nans_time_linear(drw, good)

    # FFT
    if USE_HANN:
        w = np.hanning(Nt)[:, None, None]
        X = np.fft.rfft(drw_filled * w, axis=0)
    else:
        X = np.fft.rfft(drw_filled, axis=0)

    f = np.fft.rfftfreq(Nt, d=dt_snap)
    A = np.abs(X)
    A[:, ~good] = 0.0

    # Pick dominant bin excluding DC
    band = np.where(f >= FMIN_PICK)[0]
    band = band[band != 0]
    if band.size == 0 or (not good.any()):
        print(f"[{window_name}] No valid frequency bins or no good pixels; skipping FFT plots.")
        return np.nan, -1, good_frac

    power = np.sum(A[band]**2, axis=(1, 2))
    k0 = band[int(np.argmax(power))]
    f0 = float(f[k0])
    print(f"[{window_name}] Dominant oscillation frequency (FFT peak over good pixels): f0={f0:.6f} Hz (bin {k0}).")

    # Complex map at f0 + phase reference
    Xk = X[k0, :, :]
    ref = Xk[it_ref, ip_ref]
    if np.isfinite(ref.real) and np.isfinite(ref.imag) and (np.abs(ref) > 0):
        Xk = Xk * np.exp(-1j * np.angle(ref))

    amp_map = np.abs(Xk)
    phase_map = np.angle(Xk)

    # mask outside good region
    amp_plot = amp_map.copy(); amp_plot[~good] = np.nan
    phs_plot = phase_map.copy(); phs_plot[~good] = np.nan

    # Save NPZ
    npz_path = os.path.join(out_dir, f"{case}_{mode}_{window_name}_dayside_patch_fft_latlon.npz")
    np.savez_compressed(
        npz_path,
        case=case, mode=mode, window=window_name,
        lat_deg=lat_deg, lon_deg=lon_deg,
        theta=theta, phi=phi,
        t=tw, sim_steps=steps_ok[idx],
        dt_snap=dt_snap,
        r_map=rw_r, standoff=rw,
        presence_frac=pres_frac, good_mask=good,
        mean_standoff=mean_standoff,
        f=f, dom_bin=int(k0), dom_f=f0,
        amp_map=amp_map, phase_map=phase_map
    )
    print(f"[{window_name}] Wrote: {npz_path}")

    # Save plots
    pres_png = os.path.join(out_dir, f"{case}_{mode}_{window_name}_presence_frac_latlon.png")
    imshow_map_latlon(lat_deg, lon_deg, pres_frac,
                      f"{case} {mode} {window_name}: presence fraction (Δr≥{R_PRESENT})",
                      pres_png, cmap=PRES_CMAP, vmin=0.0, vmax=1.0, cbar_label="Fraction")
    print(f"[{window_name}] Wrote: {pres_png}")

    mean_png = os.path.join(out_dir, f"{case}_{mode}_{window_name}_mean_standoff_latlon.png")
    imshow_map_latlon(lat_deg, lon_deg, mean_standoff,
                      f"{case} {mode} {window_name}: mean stand-off Δr (good pixels)",
                      mean_png, cmap=MEAN_CMAP, cbar_label="Δr (R$_M$)")
    print(f"[{window_name}] Wrote: {mean_png}")

    amp_phase_png = os.path.join(out_dir, f"{case}_{mode}_{window_name}_amp_phase_f{f0:.4f}Hz_latlon.png")
    plot_amp_phase_latlon(lat_deg, lon_deg, amp_plot, phs_plot,
                          f"{case} {mode} {window_name}: dominant f={f0:.4f} Hz (patch-mode)",
                          amp_phase_png)
    print(f"[{window_name}] Wrote: {amp_phase_png}")

    return f0, k0, good_frac


f_pre, k_pre, good_pre = analyze_window("pre", pre_idx)
f_post, k_post, good_post = analyze_window("post", post_idx)

# End-of-run summary print
print("\n=== RUN SUMMARY ===")
print("Plot types per window (pre and post):")
print("  1) presence_frac(lat,lon): where the MP boundary proxy exists often enough in time.")
print("  2) mean_standoff(lat,lon): average Δr=r-1 on the retained patch (shows how close MP sits to surface).")
print("  3) amp/phase(lat,lon) at f0: spatial pattern of the strongest oscillatory component of Δr about its mean,")
print("     with phase referenced to subsolar equator (lat=0°, lon=0°).")
print("Interpretation difference:")
print("  - pre: typically a larger detached patch exists, so amp/phase maps represent oscillation of a free boundary over that patch.")
print("  - post: if MP collapses to the surface, the detached patch shrinks; amp/phase then describe oscillations only where Δr meets the presence criterion.")
print(f"Numerics: dt_snap={dt_snap:.3f}s => Nyquist={0.5/dt_snap:.3f} Hz.")
print(f"Dominant f0 (pre) = {f_pre} Hz (bin {k_pre}) --> Dominant f0 (post) = {f_post} Hz (bin {k_post}).")

txt_path = os.path.join(out_dir, f"{case}_{mode}_dominant_freq_summary_patch_latlon.txt")
with open(txt_path, "w") as ftxt:
    ftxt.write(f"Effective dt_snap = {dt_snap:.6f} s\n")
    ftxt.write(f"Presence threshold: Δr>= {R_PRESENT} RM\n")
    ftxt.write(f"MIN_PRES_FRAC = {MIN_PRES_FRAC}\n")
    ftxt.write(f"Pre:  t < {T_PRE_MAX} s, N={pre_idx.size}, good_frac={good_pre:.4f}, dominant f={f_pre} Hz (bin {k_pre})\n")
    ftxt.write(f"Post: t > {T_POST_MIN} s, N={post_idx.size}, good_frac={good_post:.4f}, dominant f={f_post} Hz (bin {k_post})\n")
print(f"Saved summary: {txt_path}")
print("All done.")

