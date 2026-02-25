#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports:
import os
from datetime import datetime
import numpy as np
import xarray as xr
import random
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from src.field_topology.topology_utils import trace_field_line_rk, classify
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines

# --------------------------
# SETTINGS
# --------------------------
debug = False
branch = "HNHV"
case = "PN"
# 115000 (pre) or 142000 (transient) or 174000 (post) or 350000 (new)
step = 350000

# set up variables for each case
case_r = f"R{case}"
case_c = f"C{case}"

if 0:
    if step < 116000:
        input_folder_r = f"/Volumes/data_backup/mercury/extreme/{case_r}_Base/05/out/"
        input_folder_c = f"/Volumes/data_backup/mercury/extreme/{case_c}_Base/05/out/"
        ncfile_r = os.path.join(input_folder_r, f"Amitis_{case_r}_Base_{step}.nc")
        ncfile_c = os.path.join(input_folder_c, f"Amitis_{case_c}_Base_{step}.nc")
    else:
        if 340000 < step < 350000:
            if "HN" in branch:
                input_folder_r = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case_r}_HNHV/10/out/"
                input_folder_c = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case_c}_HNHV/10/out/"
                ncfile_r = os.path.join(input_folder_r, f"Amitis_{case_r}_HNHV_{step}.nc")
                ncfile_c = os.path.join(input_folder_c, f"Amitis_{case_c}_HNHV_{step}.nc")
            elif "LN" in branch:
                input_folder_r = f"/Volumes/data_backup/mercury/extreme/High_LNHV/{case_r}_LNHV/10/out/"
                input_folder_c = f"/Volumes/data_backup/mercury/extreme/High_LNHV/{case_c}_LNHV/10/out/"
                ncfile_r = os.path.join(input_folder_r, f"Amitis_{case_r}_LNHV_{step}.nc")
                ncfile_c = os.path.join(input_folder_c, f"Amitis_{case_c}_LNHV_{step}.nc")
        else:
            if "HN" in branch:
                input_folder_r = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case_r}_HNHV/plane_product/object/"
                input_folder_c = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case_c}_HNHV/plane_product/object/"
                ncfile_r = os.path.join(input_folder_r, f"Amitis_{case_r}_HNHV_{step}_xz_comp.nc")
                ncfile_c = os.path.join(input_folder_c, f"Amitis_{case_c}_HNHV_{step}_xz_comp.nc")

if "HN" in branch:
    input_folder_r = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case_r}_HNHV/plane_product/object/"
    input_folder_c = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case_c}_HNHV/plane_product/object/"
    ncfile_r = os.path.join(input_folder_r, f"Amitis_{case_r}_HNHV_{step}_xz_comp.nc")
    ncfile_c = os.path.join(input_folder_c, f"Amitis_{case_c}_HNHV_{step}_xz_comp.nc")
else:
    input_folder_r = f"/Volumes/data_backup/mercury/extreme/{case_r}_Base/plane_product/object/"
    input_folder_c = f"/Volumes/data_backup/mercury/extreme/{case_c}_Base/plane_product/object/"
    ncfile_r = os.path.join(input_folder_r, f"Amitis_{case_r}_Base_{step}_xz_comp.nc")
    ncfile_c = os.path.join(input_folder_c, f"Amitis_{case_c}_Base_{step}_xz_comp.nc")

output_folder = f"/Users/danywaller/Projects/mercury/extreme/induced_bfield_topology/{case_c}-{case_r}_{branch}/"
os.makedirs(output_folder, exist_ok=True)

# Planet parameters
RM = 2440.0  # km
plot_depth = RM
L = 2.5 * RM         # half-width of cube of interest: [-L, +L]
buf = 0.5 * RM       # buffer to reduce periodic wrap in FFT result

# Seed settings
n_lat = 60
n_lon = n_lat * 2
max_steps = 5000
h_step = 50.0
surface_tol = 75.0
max_lines = 1000

MU0 = 4.0 * np.pi * 1e-7  # H/m

# --------------------------
# CREATE SEEDS ON SPHERE
# --------------------------
lats_surface = np.linspace(-90, 90, n_lat)
lons_surface = np.linspace(-180, 180, n_lon)
seeds = []
for lat in lats_surface:
    for lon in lons_surface:
        phi = np.radians(lat)
        theta = np.radians(lon)
        x_s = plot_depth * np.cos(phi) * np.cos(theta)
        y_s = plot_depth * np.cos(phi) * np.sin(theta)
        z_s = plot_depth * np.sin(phi)
        seeds.append(np.array([x_s, y_s, z_s]))
seeds = np.array(seeds)

# --------------------------
# LOAD VECTOR FIELD FROM NETCDF
# --------------------------
def load_field(ncfile):
    ds = xr.open_dataset(ncfile)
    x = ds["Nx"].values  # [km]
    y = ds["Ny"].values  # [km]
    z = ds["Nz"].values  # [km]

    # Extract fields (drop time dimension) and transpose: Nz, Ny, Nx --> Nx, Ny, Nz
    Jx = np.transpose(ds["Jx"].isel(time=0).values, (2, 1, 0))  # [nA/m^2]
    Jy = np.transpose(ds["Jy"].isel(time=0).values, (2, 1, 0))  # [nA/m^2]
    Jz = np.transpose(ds["Jz"].isel(time=0).values, (2, 1, 0))  # [nA/m^2]
    ds.close()
    return x, y, z, Jx, Jy, Jz

x_km, y_km, z_km, Jx_nA_r, Jy_nA_r, Jz_nA_r = load_field(ncfile_r)

start = datetime.now()
print(f"Loaded resistive core at {str(start)}")

_, _, _, Jx_nA_c, Jy_nA_c, Jz_nA_c = load_field(ncfile_c)

start = datetime.now()
print(f"Loaded conductive core at {str(start)}")

Jx_nA = Jx_nA_c - Jx_nA_r
Jy_nA = Jy_nA_c - Jy_nA_r
Jz_nA = Jz_nA_c - Jz_nA_r

def idx_range(coord_km, lo, hi):
    i0 = int(np.searchsorted(coord_km, lo, side="left"))
    i1 = int(np.searchsorted(coord_km, hi, side="right"))
    i0 = max(i0, 0)
    i1 = min(i1, coord_km.size)
    return slice(i0, i1)

sx = idx_range(x_km, -L-buf, +L+buf)
sy = idx_range(y_km, -L-buf, +L+buf)
sz = idx_range(z_km, -L-buf, +L+buf)

x_sub = x_km[sx]
y_sub = y_km[sy]
z_sub = z_km[sz]

Jx_sub = Jx_nA[sx, sy, sz]
Jy_sub = Jy_nA[sx, sy, sz]
Jz_sub = Jz_nA[sx, sy, sz]

if debug:
    # ---- sanity prints ----
    print("Jx shape:", Jx_nA.shape)
    print("Nx len:", len(x_km))
    print("Ny len:", len(y_km))
    print("Nz len:", len(z_km))

    print("Subgrid shape =", Jx_sub.shape)
    print("x range (km) =", x_sub[0], x_sub[-1])
    print("y range (km) =", y_sub[0], y_sub[-1])
    print("z range (km) =", z_sub[0], z_sub[-1])

    dx_km = np.diff(x_sub); dy_km = np.diff(y_sub); dz_km = np.diff(z_sub)
    print("x_sub[:5] =", x_sub[:5], " uniform?", np.allclose(dx_km, dx_km[0]))
    print("x_sub[:5] =", x_sub[:5], " uniform?", np.allclose(dy_km, dy_km[0]))
    print("x_sub[:5] =", x_sub[:5], " uniform?", np.allclose(dz_km, dz_km[0]))

# --------------------------
# FFT-BASED B FROM J
# --------------------------
def B_from_J_fft_lowmem(x_km, y_km, z_km, Jx_nA, Jy_nA, Jz_nA, use_complex64=True):
    """
    Compute magnetic field B(x,y,z) from a 3-D current density J(x,y,z) using an FFT-based
    spectral solver.
    https://www.sciencedirect.com/science/article/pii/S0021999120301820

    Method (Fourier space):
        B(k) = i * μ0 * (k × J(k)) / |k|^2
    which corresponds to a magnetostatic / Biot–Savart-like relation on a periodic domain.

    Notes / assumptions:
    - The grid is assumed uniform in each direction (dx,dy,dz from mean diff).
    - FFT implies periodic boundary conditions (wrap-around / image currents if not padded).
    - The k=0 (DC) mode is set to zero, enforcing zero-mean B from this solve.
    - If use_complex64=True, uses float32/complex64 intermediates to reduce memory/CPU bandwidth.

    Parameters
    ----------
    x_km, y_km, z_km : 1-D arrays
        Coordinates along each axis in kilometers.
    Jx_nA, Jy_nA, Jz_nA : 3-D arrays, shape (Nx, Ny, Nz)
        Current density components on the grid in nA/m^2.
    use_complex64 : bool
        If True, compute FFTs in complex64 (with float32 inputs) to save memory.

    Returns
    -------
    Bx, By, Bz : 3-D float64 arrays
        Magnetic field components on the grid (SI units: Tesla).
    """
    # Convert coordinates from km to m
    x = x_km * 1e3
    y = y_km * 1e3
    z = z_km * 1e3

    # Convert current density from nA/m^2 to A/m^2
    Jx = Jx_nA * 1e-9
    Jy = Jy_nA * 1e-9
    Jz = Jz_nA * 1e-9

    # Estimate grid spacing (assumes uniform)
    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    dz = float(np.mean(np.diff(z)))

    # Grid sizes from J arrays
    Nx, Ny, Nz = Jx.shape

    # Build 1-D wavenumber vectors (rad/m) for each axis
    # fftfreq gives cycles/m; multiply by 2π to get rad/m
    kx = (2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)).astype(np.float32)
    ky = (2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)).astype(np.float32)
    kz = (2.0 * np.pi * np.fft.fftfreq(Nz, d=dz)).astype(np.float32)

    # Choose dtypes for memory
    ctype = np.complex64 if use_complex64 else np.complex128
    rtype = np.float32 if use_complex64 else np.float64

    # Forward FFT of current density components
    Jxk = np.fft.fftn(Jx.astype(rtype)).astype(ctype)
    Jyk = np.fft.fftn(Jy.astype(rtype)).astype(ctype)
    Jzk = np.fft.fftn(Jz.astype(rtype)).astype(ctype)

    # Compute |k|^2 w/o explicit 3-D allocations
    k2 = (kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2).astype(np.float32)

    # Prevent division by zero at k=0; we will overwrite the DC mode later anyway
    k2[0, 0, 0] = 1.0

    # Common factor: i*μ0/|k|^2
    fac = (1j * MU0) / k2

    # Compute B(k) = fac * (k × J(k)) componentwise
    Bxk = fac * (ky[None, :, None] * Jzk - kz[None, None, :] * Jyk)
    Byk = fac * (kz[None, None, :] * Jxk - kx[:, None, None] * Jzk)
    Bzk = fac * (kx[:, None, None] * Jyk - ky[None, :, None] * Jxk)

    # Enforce zero DC (mean) magnetic field from this computation
    Bxk[0, 0, 0] = 0.0
    Byk[0, 0, 0] = 0.0
    Bzk[0, 0, 0] = 0.0

    # Inverse FFT back to real space, imaginary parts should be numerical noise
    Bx = np.fft.ifftn(Bxk).real.astype(np.float64)
    By = np.fft.ifftn(Byk).real.astype(np.float64)
    Bz = np.fft.ifftn(Bzk).real.astype(np.float64)

    return Bx, By, Bz


bstart = datetime.now()
Bx_sub, By_sub, Bz_sub = B_from_J_fft_lowmem(x_sub, y_sub, z_sub, Jx_sub, Jy_sub, Jz_sub, use_complex64=True)
bend = datetime.now()

print(f"Computed B everywhere with FFT from J at {str(bend)}; elapsed={(bend-bstart)}")

sx2 = idx_range(x_sub, -L, +L)
sy2 = idx_range(y_sub, -L, +L)
sz2 = idx_range(z_sub, -L, +L)

x_box = x_sub[sx2]
y_box = y_sub[sy2]
z_box = z_sub[sz2]
Bx_box = Bx_sub[sx2, sy2, sz2]
By_box = By_sub[sx2, sy2, sz2]
Bz_box = Bz_sub[sx2, sy2, sz2]
Jx_box = Jx_sub[sx2, sy2, sz2]
Jy_box = Jy_sub[sx2, sy2, sz2]
Jz_box = Jz_sub[sx2, sy2, sz2]

# Compute |B| without keeping an extra huge temporary around longer than needed
Bmag = np.sqrt(Bx_box*Bx_box + By_box*By_box + Bz_box*Bz_box)
print("|B| min/max (nT) =", np.nanmin(Bmag)*1e9, np.nanmax(Bmag)*1e9)
del Bmag

# Dobby is a free RAM!!!!!
import gc
del x_km, y_km, z_km
del Jx_nA, Jy_nA, Jz_nA
del Jx_sub, Jy_sub, Jz_sub
del Bx_sub, By_sub, Bz_sub
del x_sub, y_sub, z_sub
del sx, sy, sz, sx2, sy2, sz2
gc.collect()

# --------------------------
# TRACE MAGNETIC FIELD LINES
# --------------------------
lines_by_topo = {"closed": [], "open": []}

for seed in seeds:
    traj_fwd, exit_fwd_y = trace_field_line_rk(seed, Bx_box, By_box, Bz_box, x_box, y_box, z_box, plot_depth, max_steps=max_steps, h=h_step)
    traj_bwd, exit_bwd_y = trace_field_line_rk(seed, Bx_box, By_box, Bz_box, x_box, y_box, z_box, plot_depth, max_steps=max_steps, h=-h_step)
    topo = classify(traj_fwd, traj_bwd, plot_depth + surface_tol, exit_fwd_y, exit_bwd_y)
    if topo in ["closed", "open"]:
        lines_by_topo[topo].append(traj_fwd)
        lines_by_topo[topo].append(traj_bwd)

classtime = datetime.now()
print(f"Classified all B-lines at {str(classtime)}")

del seeds, traj_fwd, exit_fwd_y, traj_bwd, exit_bwd_y, topo

seeds = []
y0 = 0.0  # X–Z plane

# ---- 1. Planet surface seeds (circle in X–Z plane) ----
n_surface = 360
theta = np.linspace(0, 2 * np.pi, n_surface, endpoint=False)
for t in theta:
    x_s = RM * np.cos(t)
    z_s = RM * np.sin(t)
    seeds.append([x_s, y0, z_s])

seeds = np.array(seeds)
print(f"Total new seeds: {len(seeds)}")

# --------------------------
# TRACE MAGNETIC FIELD LINES
# --------------------------
lines_by_topo_lplot = {"closed": [], "open": []}

for seed in seeds:
    traj_fwd, exit_fwd_y = trace_field_line_rk(seed, Bx_box, By_box, Bz_box, x_box, y_box, z_box, plot_depth, max_steps=max_steps, h=h_step)
    traj_bwd, exit_bwd_y = trace_field_line_rk(seed, Bx_box, By_box, Bz_box, x_box, y_box, z_box, plot_depth, max_steps=max_steps, h=-h_step)
    topo = classify(traj_fwd, traj_bwd, plot_depth + surface_tol, exit_fwd_y, exit_bwd_y)
    if topo in ["closed", "open"]:
        lines_by_topo_lplot[topo].append(traj_fwd[:, [0, 2]])
        lines_by_topo_lplot[topo].append(traj_bwd[:, [0, 2]])

classtime = datetime.now()
print(f"Classified X-Z plane B-lines at {str(classtime)}")

# --------------------------
# Smooth for plotting
# --------------------------
def smooth_traj(traj, k=5, order=2):
    if traj.shape[0] < k:
        return traj
    return np.column_stack([savgol_filter(traj[:, i], k, order, mode="interp") for i in range(3)])

# --------------------------
# PLOT 3D FIELD LINES
# --------------------------
colors = {"closed": "blue", "open": "red"}
fig = go.Figure()

# add planet sphere
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 200)
theta, phi = np.meshgrid(theta, phi)

xs = plot_depth * np.sin(theta) * np.cos(phi)
ys = plot_depth * np.sin(theta) * np.sin(phi)
zs = plot_depth * np.cos(theta)

eps = 0
mask_pos = xs >= -eps
mask_neg = xs <=  eps

fig.add_trace(go.Surface(
    x=np.where(mask_pos, xs, np.nan),
    y=np.where(mask_pos, ys, np.nan),
    z=np.where(mask_pos, zs, np.nan),
    surfacecolor=np.ones_like(xs),
    colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
    cmin=0, cmax=1, showscale=False,
    lighting=dict(ambient=1, diffuse=0, specular=0),
    hoverinfo='skip'
))
fig.add_trace(go.Surface(
    x=np.where(mask_neg, xs, np.nan),
    y=np.where(mask_neg, ys, np.nan),
    z=np.where(mask_neg, zs, np.nan),
    surfacecolor=np.zeros_like(xs),
    colorscale=[[0, 'black'], [1, 'black']],
    cmin=0, cmax=1, showscale=False,
    lighting=dict(ambient=1, diffuse=0, specular=0),
    hoverinfo='skip'
))

for topo, lines in lines_by_topo.items():
    first = True
    if len(lines) > max_lines:
        lines_to_plot = random.sample(lines, max_lines)
    else:
        lines_to_plot = lines

    for traj in lines_to_plot:
        traj_s = smooth_traj(traj)

        fig.add_trace(go.Scatter3d(
            x=traj_s[:, 0],
            y=traj_s[:, 1],
            z=traj_s[:, 2],
            mode='lines',
            line=dict(color=colors[topo], width=2),
            name=topo,
            legendgroup=topo,
            showlegend=first
        ))
        first = False

fig.update_layout(
    template="plotly",
    width=1200,
    height=900,
    scene=dict(
        xaxis=dict(title='X [km]', range=[-L, L]),
        yaxis=dict(title='Y [km]', range=[-L, L]),
        zaxis=dict(title='Z [km]', range=[-L, L]),
        aspectmode='cube',
    ),
    legend=dict(groupclick="togglegroup"),
    title=f"{case_c}-{case_r} {branch} Induced B Field Line Topology (FFT): t = {step*0.002} s"
)

out_html = f"{case_c}-{case_r}_{branch}_{step}_Bfft_topology.html"
fig.write_html(os.path.join(output_folder, out_html), include_plotlyjs="cdn")
fig.write_image(os.path.join(output_folder, out_html.replace(".html", ".png")), scale=2)
plottime = datetime.now()
print(f"Saved figure at {str(plottime)}")

# --------------------------
# Matplotlib: 2x3 streamlines on slices
# --------------------------
# nearest indices to 0
ix0 = int(np.argmin(np.abs(x_box)))
iy0 = int(np.argmin(np.abs(y_box)))
iz0 = int(np.argmin(np.abs(z_box)))

def mask_inside_circle(U, V, X, Y, R):
    """
    Mask vector field inside r<R by setting to NaN (streamplot will avoid).
    """
    inside = (X*X + Y*Y) < (R*R)
    Um = U.astype(float).copy()
    Vm = V.astype(float).copy()
    Um[inside] = np.nan
    Vm[inside] = np.nan
    return Um, Vm

def add_panel(ax, x1d, y1d, U_xy, V_xy, R, title, color_by="speed",
              density=1.6, lw=1.0, cmap="viridis",
              Rc=2080, Rc_style=dict(color="hotpink", lw=1.5, ls="-"), min_mag=0.01):
    """
    ...
    Rc: optional second circle radius (km)
    Rc_style: line style dict for Rc circle
    """
    X, Y = np.meshgrid(x1d, y1d, indexing="xy")

    U_m, V_m = U_xy, V_xy

    # Mask low-magnitude regions (prevents streamlines there)
    if min_mag is not None:
        mag = np.hypot(U_m, V_m)
        bad = mag < min_mag
        U_m = U_m.copy()
        V_m = V_m.copy()
        U_m[bad] = np.nan
        V_m[bad] = np.nan

    th = np.linspace(0, 2 * np.pi, 400)

    # planet circle outline (R_M)
    R_M = 1
    ax.plot(R_M * np.cos(th), R_M * np.sin(th), color="mediumorchid", lw=1.5)

    # additional circle outline (Rc)
    R_cmb = Rc/R
    if Rc is not None:
        ax.plot(R_cmb * np.cos(th), R_cmb * np.sin(th), **Rc_style)

    if color_by == "speed":
        C = np.sqrt(U_m*U_m + V_m*V_m)
    else:
        C = None

    sp = ax.streamplot(
        x1d/R, y1d/R, U_m, V_m,
        density=density,
        color=C,
        cmap=cmap if C is not None else None,
        linewidth=lw,
        arrowsize=1.0
    )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x1d[0]/R, x1d[-1]/R)
    ax.set_ylim(y1d[0]/R, y1d[-1]/R)
    ax.grid(False)

    return sp


# Build 2D in-plane component slices
# IMPORTANT: Matplotlib streamplot uses arrays shaped (Ny, Nx), so transpose from (Nx,Ny)/(Nx,Nz)/(Ny,Nz) storage!!

# --- Top row: J streamlines (nA/m^2), in-plane components ---
# XY (z=0): U=Jx, V=Jy on (x,y)
Jx_xy = Jx_box[:, :, iz0].T  # (Ny,Nx)
Jy_xy = Jy_box[:, :, iz0].T  # (Ny,Nx)

# XZ (y=0): U=Jx, V=Jz on (x,z)
Jx_xz = Jx_box[:, iy0, :].T  # (Nz,Nx) -> acts like (Ny,Nx) with y=z
Jz_xz = Jz_box[:, iy0, :].T

# YZ (x=0): U=Jy, V=Jz on (y,z)  (x-axis is y, y-axis is z)
Jy_yz = Jy_box[ix0, :, :].T  # (Nz,Ny)
Jz_yz = Jz_box[ix0, :, :].T

# --- Bottom row: B streamlines (nT), in-plane components ---
Bx_xy = Bx_box[:, :, iz0].T * 1e9
By_xy = By_box[:, :, iz0].T * 1e9

Bx_xz = Bx_box[:, iy0, :].T * 1e9
Bz_xz = Bz_box[:, iy0, :].T * 1e9

By_yz = By_box[ix0, :, :].T * 1e9
Bz_yz = Bz_box[ix0, :, :].T * 1e9

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

# J panels
sp00 = add_panel(axes[0, 0], x_box, y_box, Jx_xy, Jy_xy, RM, "|J| streamlines: XY (z=0)",
                 density=2.2, lw=1.0, cmap="cividis")
sp01 = add_panel(axes[0, 1], x_box, z_box, Jx_xz, Jz_xz, RM, "|J| streamlines: XZ (y=0)",
                 density=2.2, lw=1.0, cmap="cividis")
sp02 = add_panel(axes[0, 2], y_box, z_box, Jy_yz, Jz_yz, RM, "|J| streamlines: YZ (x=0)",
                 density=2.2, lw=1.0, cmap="cividis")

# B panels
sp10 = add_panel(axes[1, 0], x_box, y_box, Bx_xy, By_xy, RM, "|B| streamlines: XY (z=0)",
                 density=2.2, lw=1.0, cmap="viridis")
sp11 = add_panel(axes[1, 1], x_box, z_box, Bx_xz, Bz_xz, RM, "|B| streamlines: XZ (y=0)",
                 density=2.2, lw=1.0, cmap="viridis")
sp12 = add_panel(axes[1, 2], y_box, z_box, By_yz, Bz_yz, RM, "|B| streamlines: YZ (x=0)",
                 density=2.2, lw=1.0, cmap="viridis")

# Axis labels
axes[0, 0].set_xlabel("X [Rₘ]")
axes[0, 0].set_ylabel("Y [Rₘ]")
axes[0, 1].set_xlabel("X [Rₘ]")
axes[0, 1].set_ylabel("Z [Rₘ]")
axes[0, 2].set_xlabel("Y [Rₘ]")
axes[0, 2].set_ylabel("Z [Rₘ]")
axes[1, 0].set_xlabel("X [Rₘ]")
axes[1, 0].set_ylabel("Y [Rₘ]")
axes[1, 1].set_xlabel("X [Rₘ]")
axes[1, 1].set_ylabel("Z [Rₘ]")
axes[1, 2].set_xlabel("Y [Rₘ]")
axes[1, 2].set_ylabel("Z [Rₘ]")

# Optional: colorbars (one for J row, one for B row), based on magnitude coloring
cbar0 = fig.colorbar(sp00.lines, ax=axes[0, :].ravel().tolist(), fraction=0.02, pad=0.02)
cbar0.set_label("[nA/m²]")  # magnitude of (Jx,Jy) etc.
cbar0.mappable.set_clim(0.0, 50.0)

cbar1 = fig.colorbar(sp10.lines, ax=axes[1, :].ravel().tolist(), fraction=0.02, pad=0.02)
cbar1.set_label("[nT]")      # magnitude of (Bx,By) etc.
cbar1.mappable.set_clim(0.0, 30.0)

fig.suptitle(f"{case_c}-{case_r} {branch} streamlines on slices x=0,y=0,z=0; t={step*0.002} s", fontsize=18, y=1.025)

# fig.tight_layout()
out_png = os.path.join(output_folder, f"{case_c}-{case_r}_{branch}_{step}_Jmag_Bmag_streamlines_slices.png")
plt.savefig(out_png, dpi=250, bbox_inches='tight')
# plt.close(fig)
print("Saved:", out_png)

# ======================
# PLOT FIELD LINES (X-Z PLANE)
# ======================
colors = {"closed": "blue", "open": "red"}
fig, ax = plt.subplots(figsize=(7, 7))

# Draw Mercury surface
theta = np.linspace(0, 2 * np.pi, 400)
ax.plot(RM * np.cos(theta), RM * np.sin(theta), "k", lw=2)

# Plot traced field lines
for topo, segments in lines_by_topo_lplot.items():
    if segments:
        lc = LineCollection(segments, colors=colors[topo], linewidths=0.8, alpha=0.5)
        ax.add_collection(lc)

# Legend
legend_handles = [mlines.Line2D([], [], color=c, label=k) for k, c in colors.items() if k in ["closed", "open"]]
ax.legend(handles=legend_handles, loc="upper right")

ax.set_xlabel("X [km]")
ax.set_ylabel("Z [km]")
ax.set_aspect("equal")
ax.set_title(f"{case_c}-{case_r} {branch} Magnetic Field-Line Topology (X-Z Plane); t={step*0.002} s")
ax.set_xlim(-2 * RM, 2 * RM)
ax.set_ylim(-2 * RM, 2 * RM)
plt.tight_layout()
output_topo = os.path.join(output_folder, "2D_topology/")
os.makedirs(output_topo, exist_ok=True)
plt.savefig(os.path.join(output_topo, f"{case_c}-{case_r}_{branch}_bfield_topology_{step}.png"), dpi=150,
            bbox_inches="tight")
print("Saved:\t", os.path.join(output_topo, f"{case_c}-{case_r}_{branch}_bfield_topology_{step}.png"))
plt.close()