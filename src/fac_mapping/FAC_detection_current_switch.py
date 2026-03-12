#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------
# SETTINGS
# --------------------------
case = "CPS"
mode = "HNHV"
# sim_steps = list(range(105000, 350000 + 1, 1000))
sim_steps = list(range(105000, 115000 + 1, 1000))

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
# FILE PATH BUILDER
# --------------------------
def build_file_path(sim_step):
    if sim_step <= 115000:
        fmode = "Base"
        folder = f"/Volumes/data_backup/mercury/extreme/High_Base/{case}_Base/plane_product/cube/"
    else:
        fmode = "HNHV"
        folder = f"/Volumes/data_backup/mercury/extreme/High_{mode}/{case}_{mode}/plane_product/cube/"
    return os.path.join(folder, f"Amitis_{case}_{fmode}_{sim_step:06d}_merged_4RM.nc")

# --------------------------
# GRID & GEOMETRY
# --------------------------
def load_grid_only(ncfile):
    ds = xr.open_dataset(ncfile)
    x, y, z = ds["Nx"].values, ds["Ny"].values, ds["Nz"].values
    ds.close()
    return x, y, z

def precompute_geometry(x, y, z):
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")  # Nx,Ny,Nz
    R = np.sqrt(X**2 + Y**2 + Z**2)
    shell_mask = (R >= Rsurface) & (R <= Rsurface + shell_dr_km)

    rhatx = np.zeros_like(R); rhaty = np.zeros_like(R); rhatz = np.zeros_like(R)
    np.divide(X, R, out=rhatx, where=shell_mask)
    np.divide(Y, R, out=rhaty, where=shell_mask)
    np.divide(Z, R, out=rhatz, where=shell_mask)

    lat = np.full(R.shape, np.nan)
    lat[shell_mask] = np.degrees(np.arcsin(np.clip(Z[shell_mask]/R[shell_mask], -1.0, 1.0)))

    lon = np.degrees(np.arctan2(Y, X))
    lon = (lon + 360.0) % 360.0

    lon_edges = np.arange(0, 360 + dlon, dlon)
    lon_centers = 0.5*(lon_edges[:-1] + lon_edges[1:])
    alat_edges = np.arange(lat_min, lat_max + dlat, dlat)
    alat_centers = 0.5*(alat_edges[:-1] + alat_edges[1:])
    colat_centers = (90 - alat_centers)[::-1]

    return dict(
        shell_mask=shell_mask,
        rhatx=rhatx, rhaty=rhaty, rhatz=rhatz,
        lat=lat, lon=lon,
        lon_edges=lon_edges, lon_centers=lon_centers,
        alat_edges=alat_edges, alat_centers=alat_centers,
        colat_centers=colat_centers
    )

# --------------------------
# FAC COMPUTATION
# --------------------------
def compute_fac_maps(ncfile, geom):
    ds = xr.open_dataset(ncfile)
    Jx = np.transpose(ds["Jx"].isel(time=0).values,(2,1,0))
    Jy = np.transpose(ds["Jy"].isel(time=0).values,(2,1,0))
    Jz = np.transpose(ds["Jz"].isel(time=0).values,(2,1,0))
    Bx = np.transpose(ds["Bx_tot"].isel(time=0).values,(2,1,0))
    By = np.transpose(ds["By_tot"].isel(time=0).values,(2,1,0))
    Bz = np.transpose(ds["Bz_tot"].isel(time=0).values,(2,1,0))
    ds.close()

    shell = geom["shell_mask"]

    if sigma_smooth_cells>0:
        Jx = gaussian_filter(Jx,sigma_smooth_cells)
        Jy = gaussian_filter(Jy,sigma_smooth_cells)
        Jz = gaussian_filter(Jz,sigma_smooth_cells)
        Bx = gaussian_filter(Bx,sigma_smooth_cells)
        By = gaussian_filter(By,sigma_smooth_cells)
        Bz = gaussian_filter(Bz,sigma_smooth_cells)

    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2) + 1e-30
    Jpar = (Jx*Bx + Jy*By + Jz*Bz)/Bmag

    # Correct FAC definition: along B
    Jplot = Jpar

    mapN = binned_map("N", shell, geom["lat"], geom["lon"], Jplot, geom["alat_edges"], geom["lon_edges"])
    mapS = binned_map("S", shell, geom["lat"], geom["lon"], Jplot, geom["alat_edges"], geom["lon_edges"])

    return mapN, mapS

# --------------------------
# BINNING
# --------------------------
def binned_map(hemi, shell_mask, lat, lon, Jplot, alat_edges, lon_edges):
    if hemi=="N":
        m = shell_mask & (lat>0)
    else:
        m = shell_mask & (lat<0)
    lon_v = lon[m]; lat_v = np.abs(lat[m]); val_v = Jplot[m]
    sumw, _, _ = np.histogram2d(lat_v, lon_v, bins=[alat_edges, lon_edges], weights=val_v)
    cnt, _, _ = np.histogram2d(lat_v, lon_v, bins=[alat_edges, lon_edges])
    out = np.full_like(sumw, np.nan)
    mask = cnt>0
    out[mask] = sumw[mask]/cnt[mask]
    return out[::-1,:]

# --------------------------
# POLAR GRID
# --------------------------
def polar_to_cart(Z, lon_centers, colat_centers):
    theta = np.radians(lon_centers)
    r = colat_centers
    TH, RR = np.meshgrid(theta, r)
    X = RR*np.cos(TH)
    Y = RR*np.sin(TH)
    return X,Y,Z

# --------------------------
# FAC REGION DETECTION
# --------------------------
def detect_fac_regions_sector(Z, lon_centers, alat_centers):
    dawn = (lon_centers >= 0) & (lon_centers < 180)
    dusk = (lon_centers >= 180) & (lon_centers < 360)
    profile_dawn = gaussian_filter1d(np.nanmean(Z[:,dawn],axis=1),2)
    profile_dusk = gaussian_filter1d(np.nanmean(Z[:,dusk],axis=1),2)
    lats = alat_centers[::-1]
    def zero_cross(profile):
        idx = np.where(np.diff(np.sign(profile))!=0)[0]
        return lats[idx]
    cross = np.concatenate([zero_cross(profile_dawn), zero_cross(profile_dusk)])
    cross = np.sort(cross)[::-1]
    regions = {"NBZ":None,"R1":None,"R2":None}
    for lat in cross:
        if lat>80 and regions["NBZ"] is None:
            regions["NBZ"]=lat
        elif 70<lat<=80 and regions["R1"] is None:
            regions["R1"]=lat
        elif 55<lat<=70 and regions["R2"] is None:
            regions["R2"]=lat
    return regions

def add_lat_circle(fig, lat, row, col, color):
    r = 90 - lat
    theta = np.linspace(0,2*np.pi,200)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    fig.add_trace(go.Scatter(x=x,y=y,mode="lines",line=dict(color=color,width=3),showlegend=False),row=row,col=col)

# --------------------------
# MAIN
# --------------------------
first_file=None
for st in sim_steps:
    f = build_file_path(st)
    if os.path.exists(f):
        first_file=f
        break

x,y,z = load_grid_only(first_file)
geom = precompute_geometry(x,y,z)
lon_centers = geom["lon_centers"]
colat_centers = geom["colat_centers"]

ZN_list=[]
ZS_list=[]
labels=[]

for st in sim_steps:
    f = build_file_path(st)
    if not os.path.exists(f):
        continue
    tsec = st*step_to_seconds
    lab = f"t={tsec:.3f}s"
    print("Processing",lab)
    zN,zS = compute_fac_maps(f,geom)
    regionsN = detect_fac_regions_sector(zN,lon_centers,geom["alat_centers"])
    regionsS = detect_fac_regions_sector(zS,lon_centers,geom["alat_centers"])
    print("North:",regionsN)
    print("South:",regionsS)
    ZN_list.append((zN,regionsN))
    ZS_list.append((zS,regionsS))
    labels.append(lab)

# --------------------------
# PLOT
# --------------------------
fig = make_subplots(rows=1,cols=2,subplot_titles=("North","South"))

for i,(zN,regN) in enumerate(ZN_list):
    zS,regS = ZS_list[i]
    show=(i==0)
    XN,YN,ZN = polar_to_cart(zN,lon_centers,colat_centers)
    XS,YS,ZS = polar_to_cart(zS,lon_centers,colat_centers)

    fig.add_trace(go.Scattergl(x=XN.flatten(),y=YN.flatten(),mode="markers",
                marker=dict(size=6,color=ZN.flatten(),colorscale=colorscale_inout,
                            cmin=cmin,cmax=cmax),visible=show),row=1,col=1)
    fig.add_trace(go.Scattergl(x=XS.flatten(),y=YS.flatten(),mode="markers",
                marker=dict(size=6,color=ZS.flatten(),colorscale=colorscale_inout,
                            cmin=cmin,cmax=cmax,showscale=True),visible=show),row=1,col=2)

    # overlay FAC rings
    if regN["R1"]: add_lat_circle(fig,regN["R1"],1,1,"yellow")
    if regN["R2"]: add_lat_circle(fig,regN["R2"],1,1,"orange")
    if regN["NBZ"]: add_lat_circle(fig,regN["NBZ"],1,1,"cyan")
    if regS["R1"]: add_lat_circle(fig,regS["R1"],1,2,"yellow")
    if regS["R2"]: add_lat_circle(fig,regS["R2"],1,2,"orange")
    if regS["NBZ"]: add_lat_circle(fig,regS["NBZ"],1,2,"cyan")

fig.update_yaxes(scaleanchor="x",scaleratio=1)
fig.update_layout(
    width=1500,
    height=750,
    title="FAC Polar Maps with NBZ / R1 / R2 Detection"
)

html_path=os.path.join(out_dir,f"{case}_{mode}_FAC_classification_detection.html")

fig.write_html(html_path, include_plotlyjs="cdn")

print("Saved:",html_path)
