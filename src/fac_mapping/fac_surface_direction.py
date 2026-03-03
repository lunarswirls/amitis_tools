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
case = "RPN"
mode = "HNHV"
sim_steps = list(range(105000, 350000 + 1, 1000))

out_dir = f"/Users/danywaller/Projects/mercury/extreme/FAC_detection/{case}/"
os.makedirs(out_dir, exist_ok=True)

Rsurface = 2440.0  # km
shell_dr_km = 200.0
sigma_smooth_cells = 1.0

# Polar binning
dlat = 1.5         # degrees in |lat|
dlon = 3.0         # degrees in lon
lat_min = 0.0
lat_max = 90.0

# Colorbar limits (nA/m^2)
cmin, cmax = -40.0, 40.0

# Diverging colorscale: inward (negative) blue, outward (positive) red
colorscale_inout = [
    [0.0, "rgb(0,0,255)"],
    [0.5, "rgb(255,255,255)"],
    [1.0, "rgb(255,0,0)"],
]

# Slider label time conversion
step_to_seconds = 0.002


# --------------------------
# FILE BUILDER
# --------------------------
def build_file_path(sim_step: int) -> str:
    if sim_step <= 115000:
        fmode = "Base"
        input_folder = f"/Volumes/data_backup/mercury/extreme/{case}_Base/plane_product/cube/"
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

    # lon depends only on x,y -> 2D then broadcast to 3D
    X2 = x[:, None]
    Y2 = y[None, :]
    phi2 = np.degrees(np.arctan2(Y2, X2))  # 0=noon(+X), 90=dusk(+Y), 270=dawn(-Y)
    lon2 = (phi2 + 360.0) % 360.0
    lon = np.broadcast_to(lon2[:, :, None], R.shape).astype(float)
    lon[~shell_mask] = np.nan

    lon_edges = np.arange(0.0, 360.0 + dlon, dlon)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    alat_edges = np.arange(lat_min, lat_max + dlat, dlat)
    alat_centers = 0.5 * (alat_edges[:-1] + alat_edges[1:])

    # Convert |lat| bins to colat bins for polar r (0 at pole, 90 at equator)
    colat_edges = (90.0 - alat_edges)[::-1]  # 0..90 increasing outward
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

    Bmag = np.zeros_like(Jx, dtype=float)
    Bmag[shell_mask] = np.sqrt(Bx[shell_mask]**2 + By[shell_mask]**2 + Bz[shell_mask]**2) + 1e-30

    # J_parallel = (J·B)/|B|
    Jpar = np.zeros_like(Jx, dtype=float)
    Jpar[shell_mask] = (
        (Jx[shell_mask]*Bx[shell_mask] +
         Jy[shell_mask]*By[shell_mask] +
         Jz[shell_mask]*Bz[shell_mask]) / Bmag[shell_mask]
    )

    # Br = B·rhat
    rhatx, rhaty, rhatz = geom["rhatx"], geom["rhaty"], geom["rhatz"]
    Br = np.zeros_like(Jx, dtype=float)
    Br[shell_mask] = (
        Bx[shell_mask]*rhatx[shell_mask] +
        By[shell_mask]*rhaty[shell_mask] +
        Bz[shell_mask]*rhatz[shell_mask]
    )

    # Use sign of Jpar*(Br/|B|) to label inward/outward wrt planet
    Jpar_r = np.zeros_like(Jx, dtype=float)
    Jpar_r[shell_mask] = Jpar[shell_mask] * (Br[shell_mask] / Bmag[shell_mask])

    # Plot quantity: magnitude of Jpar, signed by inward/outward
    Jplot = np.full_like(Jx, np.nan, dtype=float)
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

    # convert from (alat increasing) to (colat increasing): reverse latitude axis
    zN = mapN[::-1, :]
    zS = mapS[::-1, :]
    return zN, zS


# --------------------------
# BARPOLAR "HEATMAP" TRACE
# --------------------------
def barpolar_heatmap_trace(
    Z_colat_lon,
    colat_edges,
    lon_centers,
    dlon,
    cmin, cmax,
    colorscale,
    show_cb=False,
    visible=True,
    cb_title="J∥ (blue=in, red=out) [nA/m²]",
):
    base_r = colat_edges[:-1]
    dr = np.diff(colat_edges)

    theta2d = np.broadcast_to(lon_centers[None, :], Z_colat_lon.shape)
    base2d = np.broadcast_to(base_r[:, None], Z_colat_lon.shape)
    dr2d = np.broadcast_to(dr[:, None], Z_colat_lon.shape)
    width2d = np.full(Z_colat_lon.shape, float(dlon))

    colat_center2d = base2d + 0.5 * dr2d
    alat2d = 90.0 - colat_center2d

    return go.Barpolar(
        theta=theta2d.ravel(),
        r=dr2d.ravel(),
        base=base2d.ravel(),
        width=width2d.ravel(),
        marker=dict(
            color=Z_colat_lon.ravel(),
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=show_cb,
            colorbar=dict(
                title=cb_title,
                tickmode="array",
                tickvals=[-40, -20, 0, 20, 40],
                len=0.85,
                x=1.08
            ) if show_cb else None,
            line=dict(width=0),
        ),
        opacity=1.0,
        visible=visible,
        customdata=alat2d.ravel(),
        hovertemplate="lon=%{theta:.1f}°<br>|lat|=%{customdata:.1f}°<br>J=%{marker.color:.2f}<extra></extra>",
        showlegend=False,
    )


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
    raise FileNotFoundError("No input files found for given sim_steps")

x, y, z = load_grid_only(first_file)
geom = precompute_geometry(x, y, z)

lon_centers = geom["lon_centers"]
colat_edges = geom["colat_edges"]

ZN_list, ZS_list, labels = [], [], []
valid_steps = []

for st in sim_steps:
    f = build_file_path(st)
    if not os.path.exists(f):
        print("File not found:", f)
        continue

    tsec = st * step_to_seconds
    lab = f"t={tsec:.3f} s (step {st})"
    print("Processing:", lab)

    zN, zS = compute_fac_maps(f, geom)
    ZN_list.append(zN)
    ZS_list.append(zS)
    labels.append(lab)
    valid_steps.append(st)

if len(valid_steps) == 0:
    raise RuntimeError("No valid steps processed (all files missing?)")


# --------------------------
# FIGURE + SLIDER
# --------------------------
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "polar"}, {"type": "polar"}]],
    subplot_titles=("North hemisphere", "South hemisphere"),
)

nT = len(valid_steps)

# Add traces: 2 per timestep (N then S)
for i in range(nT):
    show = (i == 0)

    trN = barpolar_heatmap_trace(
        Z_colat_lon=ZN_list[i],
        colat_edges=colat_edges,
        lon_centers=lon_centers,
        dlon=dlon,
        cmin=cmin, cmax=cmax,
        colorscale=colorscale_inout,
        show_cb=False,
        visible=show,
    )
    trS = barpolar_heatmap_trace(
        Z_colat_lon=ZS_list[i],
        colat_edges=colat_edges,
        lon_centers=lon_centers,
        dlon=dlon,
        cmin=cmin, cmax=cmax,
        colorscale=colorscale_inout,
        show_cb=True,
        visible=show,
    )

    fig.add_trace(trN, row=1, col=1)
    fig.add_trace(trS, row=1, col=2)

# Angular labels
tickvals = [0, 90, 180, 270]
ticktext = ["Noon (+X)", "Dusk (+Y)", "Midnight (-X)", "Dawn (-Y)"]

# Radial ticks labeled as |lat|
rticks = [0, 30, 60, 90]
rtext = ["90°", "60°", "30°", "0°"]

for c in (1, 2):
    fig.update_polars(
        angularaxis=dict(
            direction="counterclockwise",
            rotation=0,
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        ),
        radialaxis=dict(
            range=[0, 90],
            tickmode="array",
            tickvals=rticks,
            ticktext=rtext,
            title="|lat|",
            ticks="outside",
        ),
        row=1, col=c
    )

# Make polar bars fill without gaps
fig.update_layout(
    polar=dict(barmode="overlay", bargap=0.0),
    polar2=dict(barmode="overlay", bargap=0.0),
)

# Slider: toggle visibility of the N/S pair for each timestep
steps = []
for i, lab in enumerate(labels):
    vis = [False] * (2 * nT)
    vis[2*i] = True
    vis[2*i + 1] = True
    steps.append(dict(
        method="update",
        label=lab,
        args=[
            {"visible": vis},
            {"title": f"FAC polar maps (filled sectors) — {case} — {lab}"}
        ]
    ))

fig.update_layout(
    title=f"FAC polar maps (filled sectors) — {case} — {labels[0]}",
    sliders=[dict(active=0, steps=steps)],
    width=1500,
    height=720,
)

html_path = os.path.join(out_dir, f"{case}_{mode}_FAC_polar_slider.html")
fig.write_html(html_path, include_plotlyjs="cdn")
print("Saved:", html_path)
