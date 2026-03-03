#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from src.field_topology.topology_utils import trace_field_line_rk_shell

# --------------------------
# USER SETTINGS
# --------------------------
case = "RPS_Base"
step = 115000

input_folder = f"/Volumes/data_backup/mercury/extreme/{case}/plane_product/cube/"
ncfile = os.path.join(input_folder, f"Amitis_{case}_{step:06d}_merged_4RM.nc")

output_folder = f"/Users/danywaller/Projects/mercury/extreme/FAC_detection/{case}/"
os.makedirs(output_folder, exist_ok=True)

Rsurface = 2440.0  # km
Rcore = 2080.0     # km

shell_dr_km = 200.0
sigma_smooth_cells = 1.0

# ring binning (degrees)
dlat = 0.5
lat_min_search = 20.0
lat_max_search = 86.0
contract_frac = 0.5
max_band_halfwidth = 6.0

# LT sectors (hours)
LT_DAWN = (3.0, 9.0)
LT_DUSK = (15.0, 21.0)

# detection thresholds relative to hemisphere background (P95(|Jr|))
alpha_R1 = 2.0
alpha_R2 = 1.2
alpha_NBZ = 1.2

# seeds and tracing
n_seeds_per_branch = 12
lon_window_deg = 5.0
max_steps = 10000
h_step = 25.0

# plotting
line_width = 4

# core sphere
core_inner_offset_km = 0.0
core_sphere_color = "cornflowerblue"
core_sphere_opacity = 1.0


# --------------------------
# PLOT HELPERS
# --------------------------
def mercury_sphere_traces(R=1.0, showlegend=True):
    theta = np.linspace(0, np.pi, 80)
    phi = np.linspace(0, 2*np.pi, 160)
    theta, phi = np.meshgrid(theta, phi)

    xs = R * np.sin(theta) * np.cos(phi)
    ys = R * np.sin(theta) * np.sin(phi)
    zs = R * np.cos(theta)

    mask_pos = xs >= 0
    mask_neg = xs <= 0

    sphere_day = go.Surface(
        x=np.where(mask_pos, xs, np.nan),
        y=np.where(mask_pos, ys, np.nan),
        z=np.where(mask_pos, zs, np.nan),
        surfacecolor=np.where(mask_pos, 0.0, np.nan),
        colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
        showscale=False,
        hoverinfo="skip",
        name="Mercury dayside (+X)",
        showlegend=showlegend,
        opacity=1.0,
    )

    sphere_night = go.Surface(
        x=np.where(mask_neg, xs, np.nan),
        y=np.where(mask_neg, ys, np.nan),
        z=np.where(mask_neg, zs, np.nan),
        surfacecolor=np.where(mask_neg, 0.0, np.nan),
        colorscale=[[0, "black"], [1, "black"]],
        showscale=False,
        hoverinfo="skip",
        name="Mercury nightside (-X)",
        showlegend=showlegend,
        opacity=0.98,
    )

    return sphere_day, sphere_night


def solid_sphere_trace(R, color="cornflowerblue", opacity=0.35, name="Sphere", showlegend=True):
    theta = np.linspace(0, np.pi, 70)
    phi = np.linspace(0, 2*np.pi, 140)
    theta, phi = np.meshgrid(theta, phi)

    xs = R * np.sin(theta) * np.cos(phi)
    ys = R * np.sin(theta) * np.sin(phi)
    zs = R * np.cos(theta)

    return go.Surface(
        x=xs, y=ys, z=zs,
        surfacecolor=np.zeros_like(xs),
        colorscale=[[0, color], [1, color]],
        showscale=False,
        hoverinfo="skip",
        name=name,
        showlegend=showlegend,
        opacity=opacity
    )


def contract_band(lat_centers, score, peak_idx, frac=0.5, max_halfwidth_deg=12.0):
    peak = float(score[peak_idx])
    if not np.isfinite(peak) or peak <= 0:
        return None

    thr = frac * peak
    lat0 = float(lat_centers[peak_idx])

    i0 = int(peak_idx)
    i1 = int(peak_idx)

    while i0 - 1 >= 0 and score[i0 - 1] >= thr and abs(float(lat_centers[i0 - 1]) - lat0) <= max_halfwidth_deg:
        i0 -= 1
    while i1 + 1 < len(score) and score[i1 + 1] >= thr and abs(float(lat_centers[i1 + 1]) - lat0) <= max_halfwidth_deg:
        i1 += 1

    return (float(lat_centers[i0]), float(lat_centers[i1]), peak, lat0)


def color_for_label(lbl):
    if "_R1_" in lbl:
        return "red"
    if "_R2_" in lbl:
        return "dodgerblue"
    if "_NBZ_" in lbl:
        return "gold"
    return "gray"


def pick_lon_seeds(mask_branch, lon_deg, absval, x, y, z, n_seeds=12, lon_window=10.0):
    idx_all = np.array(np.nonzero(mask_branch)).T
    if idx_all.shape[0] == 0:
        return np.empty((0, 3), dtype=float)

    lon_vals = lon_deg[mask_branch]
    lon_min = float(np.nanpercentile(lon_vals, 5))
    lon_max = float(np.nanpercentile(lon_vals, 95))

    if (not np.isfinite(lon_min)) or (not np.isfinite(lon_max)) or lon_max <= lon_min:
        scores = absval[mask_branch]
        k = min(n_seeds, scores.size)
        top = np.argsort(scores)[-k:]
        idx_sel = idx_all[top]
        return np.column_stack([x[idx_sel[:, 0]], y[idx_sel[:, 1]], z[idx_sel[:, 2]]]).astype(float)

    lon_targets = np.linspace(lon_min, lon_max, n_seeds)
    seeds = []

    for lt in lon_targets:
        dlon = np.abs(((lon_deg - lt + 180.0) % 360.0) - 180.0)
        wmask = mask_branch & (dlon <= lon_window)
        if not np.any(wmask):
            continue
        scores = absval[wmask]
        best = int(np.argmax(scores))
        idx = np.array(np.nonzero(wmask)).T[best]
        seeds.append([x[idx[0]], y[idx[1]], z[idx[2]]])

    return np.array(seeds, dtype=float)


def move_seed_to_radius(seed_xyz, target_radius_km):
    r = float(np.linalg.norm(seed_xyz))
    if not np.isfinite(r) or r <= 0:
        return seed_xyz
    return (seed_xyz / r) * float(target_radius_km)


# --------------------------
# LOAD DATA
# --------------------------
ds = xr.open_dataset(ncfile)

x = ds["Nx"].values
y = ds["Ny"].values
z = ds["Nz"].values

RM_KM = 2440.0
xmax = float(np.nanmax(np.abs(x)))
print("x range:", float(np.nanmin(x)), float(np.nanmax(x)), " |max|=", xmax)

# Heuristic: if grid extents are ~10-ish, it's probably in RM, not km.
grid_in_RM = xmax < 50.0
print("grid_in_RM =", grid_in_RM)

if grid_in_RM:
    # Convert grid to km so your Rcore/Rsurface/h_step in km make sense
    x = x * RM_KM
    y = y * RM_KM
    z = z * RM_KM
    print("Converted grid to km.")

Jx = np.transpose(ds["Jx"].isel(time=0).values, (2, 1, 0))
Jy = np.transpose(ds["Jy"].isel(time=0).values, (2, 1, 0))
Jz = np.transpose(ds["Jz"].isel(time=0).values, (2, 1, 0))

Bx = np.transpose(ds["Bx_tot"].isel(time=0).values, (2, 1, 0)) + np.transpose(ds["Bx"].isel(time=0).values, (2, 1, 0))
By = np.transpose(ds["By_tot"].isel(time=0).values, (2, 1, 0)) + np.transpose(ds["By"].isel(time=0).values, (2, 1, 0))
Bz = np.transpose(ds["Bz_tot"].isel(time=0).values, (2, 1, 0)) + np.transpose(ds["Bz"].isel(time=0).values, (2, 1, 0))

ds.close()
print("Loaded data")

# grid spacing for epsilon offset at core boundary
dx = float(np.nanmedian(np.abs(np.diff(x)))) if len(x) > 1 else 0.0
dy = float(np.nanmedian(np.abs(np.diff(y)))) if len(y) > 1 else 0.0
dz = float(np.nanmedian(np.abs(np.diff(z)))) if len(z) > 1 else 0.0
dmin = float(np.nanmin([v for v in [dx, dy, dz] if v > 0])) if any(v > 0 for v in [dx, dy, dz]) else 1.0
core_start_eps = 0.5 * dmin  # start just outside the core boundary


# --------------------------
# COORDINATE FIELDS (BROADCAST-SAFE)
# --------------------------
X = x[:, None, None]
Y = y[None, :, None]
Z = z[None, None, :]

R = np.sqrt(X*X + Y*Y + Z*Z)
shell_dr = float(shell_dr_km)
shell_mask = (R >= Rsurface) & (R <= Rsurface + shell_dr)

rhatx = np.zeros_like(R, dtype=float)
rhaty = np.zeros_like(R, dtype=float)
rhatz = np.zeros_like(R, dtype=float)
np.divide(X, R, out=rhatx, where=shell_mask)
np.divide(Y, R, out=rhaty, where=shell_mask)
np.divide(Z, R, out=rhatz, where=shell_mask)

lat = np.full(R.shape, np.nan, dtype=float)
tmp = np.zeros_like(R, dtype=float)
np.divide(Z, R, out=tmp, where=shell_mask)
lat[shell_mask] = np.degrees(np.arcsin(np.clip(tmp[shell_mask], -1.0, 1.0)))

# 2D lon/LT then broadcast to 3D
X2 = x[:, None]
Y2 = y[None, :]
phi2 = np.degrees(np.arctan2(Y2, X2))

lon2 = (phi2 + 360.0) % 360.0
LT2 = (phi2 / 15.0 + 12.0) % 24.0

lon_full = np.broadcast_to(lon2[:, :, None], R.shape)
LT_full = np.broadcast_to(LT2[:, :, None], R.shape)

lon = lon_full.copy()
LT = LT_full.copy()
lon[~shell_mask] = np.nan
LT[~shell_mask] = np.nan


# --------------------------
# INWARD/OUTWARD CURRENT ON SHELL
# --------------------------
if sigma_smooth_cells and sigma_smooth_cells > 0:
    Jx_s = gaussian_filter(Jx, sigma=sigma_smooth_cells, mode="nearest")
    Jy_s = gaussian_filter(Jy, sigma=sigma_smooth_cells, mode="nearest")
    Jz_s = gaussian_filter(Jz, sigma=sigma_smooth_cells, mode="nearest")
else:
    Jx_s, Jy_s, Jz_s = Jx, Jy, Jz

Jr = np.zeros_like(R, dtype=float)
Jr[shell_mask] = (
    Jx_s[shell_mask]*rhatx[shell_mask] +
    Jy_s[shell_mask]*rhaty[shell_mask] +
    Jz_s[shell_mask]*rhatz[shell_mask]
)
absJr = np.abs(Jr)


# --------------------------
# RING SCORES + SYSTEM CLASSIFICATION
# --------------------------
lat_edges = np.arange(0.0, 90.0 + dlat, dlat)
lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

def ring_scores_for_hemi(hemi="N"):
    hemi_mask = shell_mask & (lat > 0) if hemi == "N" else shell_mask & (lat < 0)
    if not np.any(hemi_mask):
        return None

    bg = float(np.percentile(absJr[hemi_mask], 95))
    SR1 = np.zeros_like(lat_centers, dtype=float)
    Sopp = np.zeros_like(lat_centers, dtype=float)

    for i in range(len(lat_centers)):
        lo = float(lat_edges[i])
        hi = float(lat_edges[i + 1])
        ring = hemi_mask & (np.abs(lat) >= lo) & (np.abs(lat) < hi)

        dawn = ring & (LT >= LT_DAWN[0]) & (LT <= LT_DAWN[1])
        dusk = ring & (LT >= LT_DUSK[0]) & (LT <= LT_DUSK[1])

        v = absJr[dawn & (Jr < 0)]
        dawn_in = float(np.percentile(v, 95)) if v.size else 0.0
        v = absJr[dusk & (Jr > 0)]
        dusk_out = float(np.percentile(v, 95)) if v.size else 0.0

        v = absJr[dawn & (Jr > 0)]
        dawn_out = float(np.percentile(v, 95)) if v.size else 0.0
        v = absJr[dusk & (Jr < 0)]
        dusk_in = float(np.percentile(v, 95)) if v.size else 0.0

        SR1[i] = dawn_in + dusk_out
        Sopp[i] = dawn_out + dusk_in

    return dict(bg=bg, SR1=SR1, Sopp=Sopp)

def find_band(score, lat_lo, lat_hi, alpha, bg):
    m = (lat_centers >= lat_lo) & (lat_centers <= lat_hi)
    if not np.any(m):
        return None
    sub = np.where(m)[0]
    i_peak = int(sub[np.argmax(score[m])])
    if float(score[i_peak]) <= float(alpha) * float(bg):
        return None
    return contract_band(lat_centers, score, i_peak, frac=contract_frac, max_halfwidth_deg=max_band_halfwidth)

def classify_systems(hemi="N"):
    out = dict(R1=None, R2=None, NBZ=None, bg=None)
    rs = ring_scores_for_hemi(hemi)
    if rs is None:
        return out

    bg = rs["bg"]
    out["bg"] = bg

    R1 = find_band(rs["SR1"], lat_min_search, lat_max_search, alpha_R1, bg)
    out["R1"] = R1
    if R1 is None:
        return out

    lat_min_R1, lat_max_R1, _, _ = R1

    out["R2"] = find_band(
        rs["Sopp"],
        max(0.0, lat_min_R1 - 25.0),
        max(0.0, lat_min_R1 - 2.0),
        alpha_R2, bg
    )

    out["NBZ"] = find_band(
        rs["Sopp"],
        min(88.0, lat_max_R1 + 2.0),
        88.0,
        alpha_NBZ, bg
    )

    return out

systems_N = classify_systems("N")
systems_S = classify_systems("S")
print("Systems North:", systems_N)
print("Systems South:", systems_S)


# --------------------------
# BRANCH MASKS + SEED PICKING
# --------------------------
def branch_mask(system_band, hemi, branch_name):
    if system_band is None:
        return np.zeros_like(shell_mask, dtype=bool)

    lat_lo, lat_hi, _, _ = system_band
    hemi_mask = shell_mask & (lat > 0) if hemi == "N" else shell_mask & (lat < 0)
    band = hemi_mask & (np.abs(lat) >= lat_lo) & (np.abs(lat) <= lat_hi)

    dawn = band & (LT >= LT_DAWN[0]) & (LT <= LT_DAWN[1])
    dusk = band & (LT >= LT_DUSK[0]) & (LT <= LT_DUSK[1])

    if branch_name == "R1_dawn_in":
        return dawn & (Jr < 0)
    if branch_name == "R1_dusk_out":
        return dusk & (Jr > 0)
    if branch_name == "OPP_dawn_out":
        return dawn & (Jr > 0)
    if branch_name == "OPP_dusk_in":
        return dusk & (Jr < 0)

    return np.zeros_like(shell_mask, dtype=bool)

seed_sets = []

def add_system_seeds(sys_name, band, hemi):
    if band is None:
        return

    if sys_name == "R1":
        m1 = branch_mask(band, hemi, "R1_dawn_in")
        m2 = branch_mask(band, hemi, "R1_dusk_out")
        s1 = pick_lon_seeds(m1, lon, absJr, x, y, z, n_seeds=n_seeds_per_branch, lon_window=lon_window_deg)
        s2 = pick_lon_seeds(m2, lon, absJr, x, y, z, n_seeds=n_seeds_per_branch, lon_window=lon_window_deg)
        if s1.size: seed_sets.append((f"{hemi}_R1_dawn_in", s1))
        if s2.size: seed_sets.append((f"{hemi}_R1_dusk_out", s2))

    if sys_name in ("R2", "NBZ"):
        m1 = branch_mask(band, hemi, "OPP_dawn_out")
        m2 = branch_mask(band, hemi, "OPP_dusk_in")
        s1 = pick_lon_seeds(m1, lon, absJr, x, y, z, n_seeds=n_seeds_per_branch, lon_window=lon_window_deg)
        s2 = pick_lon_seeds(m2, lon, absJr, x, y, z, n_seeds=n_seeds_per_branch, lon_window=lon_window_deg)
        if s1.size: seed_sets.append((f"{hemi}_{sys_name}_dawn_out", s1))
        if s2.size: seed_sets.append((f"{hemi}_{sys_name}_dusk_in", s2))

add_system_seeds("R1", systems_N["R1"], "N")
add_system_seeds("R2", systems_N["R2"], "N")
add_system_seeds("NBZ", systems_N["NBZ"], "N")
add_system_seeds("R1", systems_S["R1"], "S")
add_system_seeds("R2", systems_S["R2"], "S")
add_system_seeds("NBZ", systems_S["NBZ"], "S")

print("Seed sets:", [(lab, s.shape[0]) for lab, s in seed_sets])


# --------------------------
# TRACE CURRENT LINES (trace from shell seeds, then rotate so plot starts at min(r))
# --------------------------
trajs = []

for label, seeds in seed_sets:
    for seed in seeds:
        traj_fwd, term_fwd, closed_fwd = trace_field_line_rk_shell(
            seed, Jx, Jy, Jz, x, y, z,
            Rcore, Rsurface,              # keep your bounds
            max_steps, h_step
        )
        traj_bwd, term_bwd, closed_bwd = trace_field_line_rk_shell(
            seed, Jx, Jy, Jz, x, y, z,
            Rcore, Rsurface,
            max_steps, -h_step
        )

        if traj_fwd is None or traj_bwd is None:
            continue

        full = np.vstack((traj_bwd[::-1], traj_fwd))
        if full.shape[0] < 3:
            continue

        r = np.linalg.norm(full, axis=1)
        i0 = int(np.nanargmin(r))

        # Start the polyline at closest-approach (core-ward end), then go outward
        full2 = full[i0:, :]

        trajs.append((label, full2))

print("Traced lines:", len(trajs))


# --------------------------
# PLOT: TOGGLEABLE DAY/NIGHT + CORE SPHERE + TRACES
# --------------------------
fig = go.Figure()

# Toggle-able dayside and nightside hemispheres
sphere_day, sphere_night = mercury_sphere_traces(R=Rsurface, showlegend=True)
fig.add_trace(sphere_day)
fig.add_trace(sphere_night)

# Core interior sphere: Rcore - 75 km (cornflowerblue)
core_inner_R = Rcore - float(core_inner_offset_km)
if core_inner_R > 0:
    fig.add_trace(solid_sphere_trace(
        R=core_inner_R,
        color=core_sphere_color,
        opacity=core_sphere_opacity,
        name=f"Core sphere (R={core_inner_R:.0f} km)",
        showlegend=True
    ))

def hemi_from_label(lbl: str) -> str:
    if lbl.startswith("N_"):
        return "N"
    if lbl.startswith("S_"):
        return "S"
    return "?"

def system_from_label(lbl: str) -> str:
    if "_R1_" in lbl:
        return "R1"
    if "_R2_" in lbl:
        return "R2"
    if "_NBZ_" in lbl:
        return "NBZ"
    return "Other"

def legend_group(lbl: str) -> str:
    return f"{system_from_label(lbl)}_{hemi_from_label(lbl)}"

group_seen = {}

# Lines: one legend entry per (system,hemi) group
for lbl, tr in trajs:
    g = legend_group(lbl)
    first = g not in group_seen
    group_seen[g] = True

    fig.add_trace(go.Scatter3d(
        x=tr[:, 0], y=tr[:, 1], z=tr[:, 2],
        mode="lines",
        line=dict(color=color_for_label(lbl), width=line_width),
        legendgroup=g,
        legendgrouptitle_text=g if first else None,
        name=g if first else None,
        showlegend=first,
        hoverinfo="skip"
    ))

# Seed markers (surface seeds)
for lbl, seeds in seed_sets:
    g = legend_group(lbl)
    fig.add_trace(go.Scatter3d(
        x=seeds[:, 0], y=seeds[:, 1], z=seeds[:, 2],
        mode="markers",
        marker=dict(size=3, color=color_for_label(lbl)),
        legendgroup=g,
        showlegend=False,
        hoverinfo="skip"
    ))

fig.update_layout(
    width=1200,
    height=950,
    title=f"Mercury FAC tracing from core boundary — {case} t={step*0.002} s",
    scene=dict(
        xaxis=dict(title="X (sunward) [km]"),
        yaxis=dict(title="Y (duskward) [km]"),
        zaxis=dict(title="Z (north) [km]"),
        aspectmode="cube",
    ),
    legend=dict(
        groupclick="togglegroup",
        x=1.02,
        y=1.0
    )
)

out_html = os.path.join(output_folder, f"FAC_ring_systems_Jtrace_coreStart_{step}.html")
fig.write_html(out_html)
print("Saved:", out_html)
