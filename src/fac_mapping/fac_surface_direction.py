#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --------------------------
# SETTINGS
# --------------------------
case = "CPS"
mode = "HNHV"
sim_steps = list(range(105000, 350000 + 1, 1000))

out_dir = f"/Users/danywaller/Projects/mercury/extreme/FAC_detection/{case}/"
os.makedirs(out_dir, exist_ok=True)

Rsurface = 2440.0
shell_dr_km = 75.0
sigma_smooth_cells = 1.0

dlat = 1.5
dlon = 3.0
lat_min = 0.0
lat_max = 90.0

cmin, cmax = -30.0, 30.0

colorscale_inout = [
    [0.0, "rgb(0,0,255)"],
    [0.5, "rgb(255,255,255)"],
    [1.0, "rgb(255,0,0)"],
]

step_to_seconds = 0.002


# --------------------------
# FILE BUILDER
# --------------------------
def build_file_path(sim_step: int) -> str:
    if sim_step <= 115000:
        fmode = "Base"
        input_folder = f"/Volumes/data_backup/mercury/extreme/High_Base/{case}_Base/plane_product/cube/"
    else:
        fmode = "HNHV"
        input_folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/plane_product/cube/"
    return os.path.join(input_folder, f"Amitis_{case}_{fmode}_{sim_step:06d}_merged_4RM.nc")


# --------------------------
# GEOMETRY PRECOMPUTE
# --------------------------
def load_grid_only(ncfile):
    ds = xr.open_dataset(ncfile)
    x = ds["Nx"].values
    y = ds["Ny"].values
    z = ds["Nz"].values
    ds.close()
    return x, y, z


def precompute_geometry(x, y, z):
    X = x[:, None, None]
    Y = y[None, :, None]
    Z = z[None, None, :]

    R = np.sqrt(X*X + Y*Y + Z*Z)
    shell_mask = (R >= Rsurface) & (R <= Rsurface + float(shell_dr_km))

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

    X2 = x[:, None]
    Y2 = y[None, :]
    phi2 = np.degrees(np.arctan2(Y2, X2))
    lon2 = (phi2 + 360.0) % 360.0
    lon = np.broadcast_to(lon2[:, :, None], R.shape).astype(float)
    lon[~shell_mask] = np.nan

    lon_edges = np.arange(0.0, 360.0 + dlon, dlon)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    alat_edges = np.arange(lat_min, lat_max + dlat, dlat)
    alat_centers = 0.5 * (alat_edges[:-1] + alat_edges[1:])

    colat_edges = (90.0 - alat_edges)[::-1]
    r_colat_centers = (90.0 - alat_centers)[::-1]

    return dict(
        shell_mask=shell_mask,
        rhatx=rhatx, rhaty=rhaty, rhatz=rhatz,
        lat=lat, lon=lon,
        lon_edges=lon_edges, lon_centers=lon_centers,
        alat_edges=alat_edges, alat_centers=alat_centers,
        colat_edges=colat_edges,
        r_colat_centers=r_colat_centers,
    )


# --------------------------
# BINNING
# --------------------------
def binned_map(hemi, shell_mask, lat, lon, Jplot, alat_edges, lon_edges, alat_centers, lon_centers):

    if hemi == "N":
        m = shell_mask & (lat > 0)
    else:
        m = shell_mask & (lat < 0)

    if not np.any(m):
        return np.full((alat_centers.size, lon_centers.size), np.nan)

    lon_v = lon[m]
    alat_v = np.abs(lat[m])
    val_v = Jplot[m]

    ok = np.isfinite(lon_v) & np.isfinite(alat_v) & np.isfinite(val_v)
    lon_v = lon_v[ok]
    alat_v = alat_v[ok]
    val_v = val_v[ok]

    sumw, _, _ = np.histogram2d(alat_v, lon_v, bins=[alat_edges, lon_edges], weights=val_v)
    cnt,  _, _ = np.histogram2d(alat_v, lon_v, bins=[alat_edges, lon_edges])

    out = np.full_like(sumw, np.nan, dtype=float)
    good = cnt > 0
    out[good] = sumw[good] / cnt[good]

    return out


# --------------------------
# POLAR → CARTESIAN GRID
# --------------------------
def polar_to_cart(Z, lon_centers, colat_centers):

    theta = np.radians(lon_centers)
    r = colat_centers

    TH, RR = np.meshgrid(theta, r)

    X = RR * np.cos(TH)
    Y = RR * np.sin(TH)

    return X, Y, Z


# --------------------------
# PER-STEP FAC MAP COMPUTE
# --------------------------
def compute_fac_maps(ncfile, geom):

    ds = xr.open_dataset(ncfile)

    Jx = np.transpose(ds["Jx"].isel(time=0).values, (2, 1, 0))
    Jy = np.transpose(ds["Jy"].isel(time=0).values, (2, 1, 0))
    Jz = np.transpose(ds["Jz"].isel(time=0).values, (2, 1, 0))

    Bx = np.transpose(ds["Bx_tot"].isel(time=0).values, (2, 1, 0))
    By = np.transpose(ds["By_tot"].isel(time=0).values, (2, 1, 0))
    Bz = np.transpose(ds["Bz_tot"].isel(time=0).values, (2, 1, 0))

    ds.close()

    shell_mask = geom["shell_mask"]

    if sigma_smooth_cells and sigma_smooth_cells > 0:
        Jx = gaussian_filter(Jx, sigma=sigma_smooth_cells, mode="nearest")
        Jy = gaussian_filter(Jy, sigma=sigma_smooth_cells, mode="nearest")
        Jz = gaussian_filter(Jz, sigma=sigma_smooth_cells, mode="nearest")
        Bx = gaussian_filter(Bx, sigma=sigma_smooth_cells, mode="nearest")
        By = gaussian_filter(By, sigma=sigma_smooth_cells, mode="nearest")
        Bz = gaussian_filter(Bz, sigma=sigma_smooth_cells, mode="nearest")

    Bmag = np.zeros_like(Jx)
    Bmag[shell_mask] = np.sqrt(Bx[shell_mask]**2 + By[shell_mask]**2 + Bz[shell_mask]**2) + 1e-30

    Jpar = np.zeros_like(Jx)
    Jpar[shell_mask] = (
        (Jx[shell_mask]*Bx[shell_mask] +
         Jy[shell_mask]*By[shell_mask] +
         Jz[shell_mask]*Bz[shell_mask]) / Bmag[shell_mask]
    )

    rhatx, rhaty, rhatz = geom["rhatx"], geom["rhaty"], geom["rhatz"]

    Br = np.zeros_like(Jx)
    Br[shell_mask] = (
        Bx[shell_mask]*rhatx[shell_mask] +
        By[shell_mask]*rhaty[shell_mask] +
        Bz[shell_mask]*rhatz[shell_mask]
    )

    Jpar_r = np.zeros_like(Jx)
    Jpar_r[shell_mask] = Jpar[shell_mask] * (Br[shell_mask] / Bmag[shell_mask])

    Jplot = np.full_like(Jx, np.nan)
    Jplot[shell_mask] = np.sign(Jpar_r[shell_mask]) * np.abs(Jpar[shell_mask])

    mapN = binned_map(
        "N",
        shell_mask, geom["lat"], geom["lon"], Jplot,
        geom["alat_edges"], geom["lon_edges"],
        geom["alat_centers"], geom["lon_centers"],
    )

    mapS = binned_map(
        "S",
        shell_mask, geom["lat"], geom["lon"], Jplot,
        geom["alat_edges"], geom["lon_edges"],
        geom["alat_centers"], geom["lon_centers"],
    )

    zN = mapN[::-1, :]
    zS = mapS[::-1, :]

    return zN, zS


# --------------------------
# MAIN
# --------------------------
first_file = None
for st in sim_steps:
    f = build_file_path(st)
    if os.path.exists(f):
        first_file = f
        break

if first_file is None:
    raise FileNotFoundError("No input files found")

x, y, z = load_grid_only(first_file)
geom = precompute_geometry(x, y, z)

lon_centers = geom["lon_centers"]
colat_centers = geom["r_colat_centers"]

ZN_list, ZS_list, labels = [], [], []
valid_steps = []

for st in sim_steps:

    f = build_file_path(st)
    if not os.path.exists(f):
        print("File not found:", f)
        continue

    tsec = st * step_to_seconds
    lab = f"t={tsec:.3f} s"

    print("Processing:", lab)

    zN, zS = compute_fac_maps(f, geom)

    ZN_list.append(zN)
    ZS_list.append(zS)
    labels.append(lab)
    valid_steps.append(st)

if len(valid_steps) == 0:
    raise RuntimeError("No valid steps processed")


# --------------------------
# FIGURE + SLIDER
# --------------------------
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("North hemisphere", "South hemisphere"),
)

nT = len(valid_steps)

for i in range(nT):

    show = (i == 0)

    XN, YN, ZN = polar_to_cart(ZN_list[i], lon_centers, colat_centers)
    XS, YS, ZS = polar_to_cart(ZS_list[i], lon_centers, colat_centers)

    # Mask NaNs (do not plot empty bins)
    maskN = np.isfinite(ZN)
    maskS = np.isfinite(ZS)

    trN = go.Scattergl(
        x=XN[maskN],
        y=YN[maskN],
        mode="markers",
        marker=dict(
            size=6,
            color=ZN[maskN],
            colorscale=colorscale_inout,
            cmin=cmin,
            cmax=cmax,
            showscale=False
        ),
        showlegend=False,
        visible=show,
    )

    trS = go.Scattergl(
        x=XS[maskS],
        y=YS[maskS],
        mode="markers",
        marker=dict(
            size=6,
            color=ZS[maskS],
            colorscale=colorscale_inout,
            cmin=cmin,
            cmax=cmax,
            showscale=True,
            colorbar=dict(
                title="J∥ (blue=in, red=out) [nA/m²]"
            )
        ),
        showlegend=False,
        visible=show,
    )

    fig.add_trace(trN, row=1, col=1)
    fig.add_trace(trS, row=1, col=2)


steps = []

for i, lab in enumerate(labels):

    vis = [False]*(2*nT)
    vis[2*i] = True
    vis[2*i+1] = True

    steps.append(dict(
        method="update",
        label=lab,
        args=[
            {"visible": vis},
            {"layout": {"title": f"FAC polar maps — {case} — {lab}"}}
        ]
    ))

fig.update_yaxes(scaleanchor="x", scaleratio=1)

fig.update_layout(
    title=f"FAC polar maps — {case} — {labels[0]}",
    sliders=[dict(active=0, steps=steps)],
    width=1500,
    height=720,
)

html_path = os.path.join(out_dir, f"{case}_{mode}_FAC_polar_slider.html")
fig.write_html(html_path, include_plotlyjs="cdn")

print("Saved:", html_path)