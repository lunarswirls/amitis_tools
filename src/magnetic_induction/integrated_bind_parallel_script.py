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

# NEW (non-FFT solve)
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --------------------------
# SETTINGS
# --------------------------
debug = False
plotlog = True
branch = "HNHV"
case = "PN"
# 115000 (pre) or 142000 (transient) or 174000 (post) or 350000 (new)
step = 174000

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
def load_field(ncfile, var_ext):
    ds = xr.open_dataset(ncfile)
    x = ds["Nx"].values  # [km]
    y = ds["Ny"].values  # [km]
    z = ds["Nz"].values  # [km]

    # Extract fields (drop time dimension) and transpose: Nz, Ny, Nx --> Nx, Ny, Nz
    Jx = np.transpose(ds[var_ext[0]].isel(time=0).values, (2, 1, 0))  # [nA/m^2] or [nT]
    Jy = np.transpose(ds[var_ext[1]].isel(time=0).values, (2, 1, 0))  # [nA/m^2] or [nT]
    Jz = np.transpose(ds[var_ext[2]].isel(time=0).values, (2, 1, 0))  # [nA/m^2] or [nT]
    ds.close()
    return x, y, z, Jx, Jy, Jz

# ---- Resistive case ----
x_km, y_km, z_km, Jx_nA_r, Jy_nA_r, Jz_nA_r = load_field(ncfile_r, ["Jx", "Jy", "Jz"])
_, _, _, BxTot_r, ByTot_r, BzTot_r = load_field(ncfile_r, ["Bx_tot", "By_tot", "Bz_tot"])
_, _, _, BxExt_r, ByExt_r, BzExt_r = load_field(ncfile_r, ["Bx", "By", "Bz"])

start = datetime.now()
print(f"Loaded resistive core at {str(start)}")

# ---- Conductive case ----
_, _, _, Jx_nA_c, Jy_nA_c, Jz_nA_c = load_field(ncfile_c, ["Jx", "Jy", "Jz"])
_, _, _, BxTot_c, ByTot_c, BzTot_c = load_field(ncfile_c, ["Bx_tot", "By_tot", "Bz_tot"])
_, _, _, BxExt_c, ByExt_c, BzExt_c = load_field(ncfile_c, ["Bx", "By", "Bz"])

start = datetime.now()
print(f"Loaded conductive core at {str(start)}")

# --------------------------
# CORE FIELD (common dipole)
# --------------------------
# You said you can get the core field as:
#   B_core = B_tot - B_ext
# where B_ext is the non-core part in your outputs.
# If the core dipole is identical between runs, B_core_r and B_core_c should match closely.
Bx_core = BxTot_r - BxExt_r
By_core = ByTot_r - ByExt_r
Bz_core = BzTot_r - BzExt_r

# --------------------------
# DIFFERENCES (conductive - resistive)
# --------------------------
Jx_nA = Jx_nA_c - Jx_nA_r
Jy_nA = Jy_nA_c - Jy_nA_r
Jz_nA = Jz_nA_c - Jz_nA_r

# This contains the signed change in *total* field between runs.
# If the two runs share the same core dipole and the same driver field setup,
# then this is effectively the induced-field contribution that appears due to conductivity.
Bx_nT = BxTot_c - BxTot_r
By_nT = ByTot_c - ByTot_r
Bz_nT = BzTot_c - BzTot_r
# TODO: need signed diffs to see what is being removed


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

Bxdiff_sub = Bx_nT[sx, sy, sz]
Bydiff_sub = By_nT[sx, sy, sz]
Bzdiff_sub = Bz_nT[sx, sy, sz]

# core field on same subgrid
Bxcore_sub = Bx_core[sx, sy, sz]
Bycore_sub = By_core[sx, sy, sz]
Bzcore_sub = Bz_core[sx, sy, sz]

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
    [https://www.sciencedirect.com/science/article/pii/S0021999120301820](https://www.sciencedirect.com/science/article/pii/S0021999120301820)

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
    # coordinates from km to m
    x = x_km * 1e3
    y = y_km * 1e3
    z = z_km * 1e3

    # current density from nA/m^2 to A/m^2
    Jx = Jx_nA * 1e-9
    Jy = Jy_nA * 1e-9
    Jz = Jz_nA * 1e-9

    # grid spacing (assumes uniform)
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

    # dtypes for memory
    ctype = np.complex64 if use_complex64 else np.complex128
    rtype = np.float32 if use_complex64 else np.float64

    # Forward FFT of current density components
    Jxk = np.fft.fftn(Jx.astype(rtype)).astype(ctype)
    Jyk = np.fft.fftn(Jy.astype(rtype)).astype(ctype)
    Jzk = np.fft.fftn(Jz.astype(rtype)).astype(ctype)

    # Compute |k|^2 w/o explicit 3-D allocations
    k2 = (kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2).astype(np.float32)

    # Prevent division by zero at k=0; will overwrite the DC mode later anyway
    k2[0, 0, 0] = 1.0

    # Common factor: i*μ0/|k|^2
    fac = (1j * MU0) / k2

    # Compute B(k) = fac * (k × J(k)) componentwise
    Bxk = fac * (ky[None, :, None] * Jzk - kz[None, None, :] * Jyk)
    Byk = fac * (kz[None, None, :] * Jxk - kx[:, None, None] * Jzk)
    Bzk = fac * (kx[:, None, None] * Jyk - ky[None, :, None] * Jxk)

    # Enforce zero DC (mean) magnetic field
    Bxk[0, 0, 0] = 0.0
    Byk[0, 0, 0] = 0.0
    Bzk[0, 0, 0] = 0.0

    # Inverse FFT back to real space, imaginary parts should be numerical noise
    Bx = np.fft.ifftn(Bxk).real.astype(np.float64)
    By = np.fft.ifftn(Byk).real.astype(np.float64)
    Bz = np.fft.ifftn(Bzk).real.astype(np.float64)

    return Bx, By, Bz


# --------------------------
# POISSON-BASED B FROM J
# --------------------------
def B_from_J_poisson_lowmem(x_km, y_km, z_km, Jx_nA, Jy_nA, Jz_nA,
                            rtol=1e-6, atol=0.0, maxiter=2000, edge_order=1,
                            use_jacobi=True):
    """
    Compute magnetic field B(x,y,z) from a 3-D current density J(x,y,z) without FFT.
    This is a magnetostatic / Biot–Savart-like solve on a non-periodic box.

    Method (real space via vector potential):
        1) Solve Poisson for A:
              ∇² A = -μ0 J
           (solve three scalar Poisson problems for Ax, Ay, Az)
        2) Compute B as:
              B = ∇ × A

    Notes / assumptions:
    - Boundary conditions: this effectively enforces A=0 on the outer box
      (homogeneous Dirichlet).
    - This avoids periodic wrap-around in FFT, but still need padding so that
      induced B in the interior is not dominated by the artificial boundary.

    Solver details:
    - Uses conjugate gradient (CG) on (-∇²)A = μ0 J which is SPD
    - Convergence criterion: ||b - A@x|| <= max(rtol*||b||, atol)
    - Optional Jacobi preconditioner using the diagonal of (-∇²)

    Parameters
    ----------
    x_km, y_km, z_km : 1-D arrays
        Coordinates along each axis in kilometers.
    Jx_nA, Jy_nA, Jz_nA : 3-D arrays, shape (Nx, Ny, Nz)
        Current density components on the grid in nA/m^2.
    rtol, atol : float
        CG stopping criteria
    maxiter : int
        Max CG iterations
    edge_order : int (1 or 2)
        Order used by np.gradient at the boundaries
    use_jacobi : bool
        If True, use a diagonal (Jacobi) preconditioner.

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
    N = Nx * Ny * Nz

    # Build sparse 3-D Laplacian operator via Kronecker sums
    def D2(n, h):
        main = (-2.0) * np.ones(n)
        off  = (1.0) * np.ones(n-1)
        return sp.diags([off, main, off], [-1, 0, 1], shape=(n, n), format="csr") / (h*h)

    Lx = D2(Nx, dx)
    Ly = D2(Ny, dy)
    Lz = D2(Nz, dz)
    Ix = sp.eye(Nx, format="csr")
    Iy = sp.eye(Ny, format="csr")
    Iz = sp.eye(Nz, format="csr")

    Lap = sp.kron(sp.kron(Lx, Iy), Iz, format="csr") \
        + sp.kron(sp.kron(Ix, Ly), Iz, format="csr") \
        + sp.kron(sp.kron(Ix, Iy), Lz, format="csr")

    # Use (-Lap) so the operator is SPD for CG
    Aop = (-1.0) * Lap

    # Jacobi preconditioner: M ≈ inv(Aop)
    M = None
    if use_jacobi:
        d = Aop.diagonal().copy()
        d[d == 0.0] = 1.0

        def Minv(v):
            return v / d

        M = spla.LinearOperator((N, N), matvec=Minv, dtype=np.float64)

    def solve_component(Ji):
        b = (MU0 * Ji).ravel(order="C")
        Ai, info = spla.cg(Aop, b, rtol=rtol, atol=atol, maxiter=maxiter, M=M)
        if info != 0:
            raise RuntimeError(f"CG did not converge (info={info})")
        return Ai.reshape((Nx, Ny, Nz), order="C")

    # Solve Poisson for Ax, Ay, Az
    Ax = solve_component(Jx)
    Ay = solve_component(Jy)
    Az = solve_component(Jz)

    # Curl(A) -> B
    dAz_dy = np.gradient(Az, dy, axis=1, edge_order=edge_order)
    dAy_dz = np.gradient(Ay, dz, axis=2, edge_order=edge_order)
    dAx_dz = np.gradient(Ax, dz, axis=2, edge_order=edge_order)
    dAz_dx = np.gradient(Az, dx, axis=0, edge_order=edge_order)
    dAy_dx = np.gradient(Ay, dx, axis=0, edge_order=edge_order)
    dAx_dy = np.gradient(Ax, dy, axis=1, edge_order=edge_order)

    Bx = (dAz_dy - dAy_dz).astype(np.float64)
    By = (dAx_dz - dAz_dx).astype(np.float64)
    Bz = (dAy_dx - dAx_dy).astype(np.float64)

    return Bx, By, Bz


# --------------------------
# COMPUTE B EVERYWHERE FROM J
# --------------------------
# FFT method (original)
# bstart = datetime.now()
# Bx_sub, By_sub, Bz_sub = B_from_J_fft_lowmem(x_sub, y_sub, z_sub, Jx_sub, Jy_sub, Jz_sub, use_complex64=True)
# bend = datetime.now()
# print(f"Computed B everywhere (FFT) from J at {str(bend)}; elapsed={(bend-bstart)}")

# Poisson method (no FFT)
bstart = datetime.now()
Bx_sub, By_sub, Bz_sub = B_from_J_poisson_lowmem(
    x_sub, y_sub, z_sub, Jx_sub, Jy_sub, Jz_sub,
    rtol=1e-15, atol=0.0, maxiter=2000, edge_order=1, use_jacobi=True
)
bend = datetime.now()
print(f"Computed B everywhere (Poisson) from J at {str(bend)}; elapsed={(bend-bstart)}")

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

Bxdiff_box = Bxdiff_sub[sx2, sy2, sz2]
Bydiff_box = Bydiff_sub[sx2, sy2, sz2]
Bzdiff_box = Bzdiff_sub[sx2, sy2, sz2]

# core field in the same box (nT)
Bxcore_box = Bxcore_sub[sx2, sy2, sz2]
Bycore_box = Bycore_sub[sx2, sy2, sz2]
Bzcore_box = Bzcore_sub[sx2, sy2, sz2]

# --------------------------
# SIGNED INDUCED CONTRIBUTION ALONG CORE
# --------------------------
# This is the key quantity you want: the induced field difference (conductive-resistive)
# projected onto the core-field direction. Positive => reinforces core, negative => opposes.
eps = 1e-30
Bcore_mag = np.sqrt(Bxcore_box*Bxcore_box + Bycore_box*Bycore_box + Bzcore_box*Bzcore_box) + eps
bxhat = Bxcore_box / Bcore_mag
byhat = Bycore_box / Bcore_mag
bzhat = Bzcore_box / Bcore_mag

Bind_parallel = Bxdiff_box*bxhat + Bydiff_box*byhat + Bzdiff_box*bzhat  # [nT]

print("Bind_parallel min/max (nT) =", np.nanmin(Bind_parallel), np.nanmax(Bind_parallel))

# |B| without keeping an extra huge temporary around longer than needed
Bmag = np.sqrt(Bx_box*Bx_box + By_box*By_box + Bz_box*Bz_box)
print("|B| min/max (nT) =", np.nanmin(Bmag)*1e9, np.nanmax(Bmag)*1e9)
del Bmag

# Dobby is a free RAM!!!!!
import gc
del x_km, y_km, z_km
del Jx_nA, Jy_nA, Jz_nA
del Jx_sub, Jy_sub, Jz_sub
del Bx_sub, By_sub, Bz_sub
del Bxdiff_sub, Bydiff_sub, Bzdiff_sub
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

# ---- surface seeds (circle in X–Z plane) ----
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
    title=f"{case_c}-{case_r} {branch} Induced Magnetic Field (Poisson): t = {step*0.002} s"
)

out_html = f"{case_c}-{case_r}_{branch}_{step}_Bind_poisson_topology.html"
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



def add_panel(ax, x1d, y1d, U_xy, V_xy, R, title, density=1.6, lw=1.0, cmap="viridis",
              Rc=2080, Rc_style=dict(color="hotpink", lw=1.5, ls="-"), min_mag=0.01, log_color=False,
              vmin=None, vmax=None):
    """

    Rc: optional second circle radius (km)
    Rc_style: line style dict for Rc circle
    log_color: if True, use log10 of magnitude for streamplot color
    vmin/vmax: optional fixed color limits
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
    R_cmb = Rc / R
    if Rc is not None:
        ax.plot(R_cmb * np.cos(th), R_cmb * np.sin(th), **Rc_style)

    C = np.sqrt(U_m * U_m + V_m * V_m)

    # Apply logarithmic scaling if requested
    if log_color:
        C = np.log10(C)

    sp = ax.streamplot(
        x1d / R, y1d / R, U_m, V_m,
        density=density,
        color=C,
        cmap=cmap,
        linewidth=lw,
        arrowsize=1.0
    )

    if (vmin is not None) and (vmax is not None):
        sp.lines.set_clim(vmin, vmax)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x1d[0] / R, x1d[-1] / R)
    ax.set_ylim(y1d[0] / R, y1d[-1] / R)
    ax.grid(False)

    return sp


# Build 2D in-plane component slices
# IMPORTANT: Matplotlib streamplot uses arrays shaped (Ny, Nx), so transpose from (Nx,Ny)/(Nx,Nz)/(Ny,Nz) storage!!

# --- Top row: Bdiff streamlines (nT), in-plane components ---
# XY (z=0): U=Bx, V=By on (x,y)
Jx_xy = Bxdiff_box[:, :, iz0].T  # (Ny,Nx)
Jy_xy = Bydiff_box[:, :, iz0].T  # (Ny,Nx)

# XZ (y=0): U=Bx, V=Bz on (x,z)
Jx_xz = Bxdiff_box[:, iy0, :].T  # (Nz,Nx) -> acts like (Ny,Nx) with y=z
Jz_xz = Bzdiff_box[:, iy0, :].T

# YZ (x=0): U=By, V=Bz on (y,z)  (x-axis is y, y-axis is z)
Jy_yz = Bydiff_box[ix0, :, :].T  # (Nz,Ny)
Jz_yz = Bzdiff_box[ix0, :, :].T

# --- Bottom row: Bind streamlines (nT), in-plane components ---
Bx_xy = Bx_box[:, :, iz0].T * 1e9
By_xy = By_box[:, :, iz0].T * 1e9

Bx_xz = Bx_box[:, iy0, :].T * 1e9
Bz_xz = Bz_box[:, iy0, :].T * 1e9

By_yz = By_box[ix0, :, :].T * 1e9
Bz_yz = Bz_box[ix0, :, :].T * 1e9

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

# B diff panels
sp00 = add_panel(axes[0, 0], x_box, y_box, Jx_xy, Jy_xy, RM, "ΔB streamlines: XY (z=0)",
                 density=2.2, lw=1.0, cmap="viridis", log_color=plotlog)
sp01 = add_panel(axes[0, 1], x_box, z_box, Jx_xz, Jz_xz, RM, "ΔB streamlines: XZ (y=0)",
                 density=2.2, lw=1.0, cmap="viridis", log_color=plotlog)
sp02 = add_panel(axes[0, 2], y_box, z_box, Jy_yz, Jz_yz, RM, "ΔB streamlines: YZ (x=0)",
                 density=2.2, lw=1.0, cmap="viridis", log_color=plotlog)

# B ind panels
sp10 = add_panel(axes[1, 0], x_box, y_box, Bx_xy, By_xy, RM, "B$_{ind}$ streamlines: XY (z=0)",
                 density=2.2, lw=1.0, cmap="viridis", log_color=plotlog)
sp11 = add_panel(axes[1, 1], x_box, z_box, Bx_xz, Bz_xz, RM, "B$_{ind}$ streamlines: XZ (y=0)",
                 density=2.2, lw=1.0, cmap="viridis", log_color=plotlog)
sp12 = add_panel(axes[1, 2], y_box, z_box, By_yz, Bz_yz, RM, "B$_{ind}$ streamlines: YZ (x=0)",
                 density=2.2, lw=1.0, cmap="viridis", log_color=plotlog)

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

# colorbars (one for B_diff row, one for B_ind row) based on magnitude
cbar0 = fig.colorbar(sp00.lines, ax=axes[0, :].ravel().tolist(), fraction=0.02, pad=0.02)
if plotlog:
    cbar0.set_label("log10(|ΔB|) [nT]")
    cbar0.mappable.set_clim(-1.0, 2.0)
else:
    cbar0.set_label("|ΔB| [nT]")
    cbar0.mappable.set_clim(0.0, 30.0)

cbar1 = fig.colorbar(sp10.lines, ax=axes[1, :].ravel().tolist(), fraction=0.02, pad=0.02)
if plotlog:
    cbar1.set_label("log10(|B$_{ind}$|) [nT]")
    cbar1.mappable.set_clim(-1.0, 2.0)
else:
    cbar1.set_label("|B$_{ind}$| [nT]")
    cbar1.mappable.set_clim(0.0, 30.0)

fig.suptitle(f"{case_c}-{case_r} {branch} streamlines on slices x=0,y=0,z=0; t={step*0.002} s", fontsize=18, y=1.025)

# fig.tight_layout()
out_png = os.path.join(output_folder, f"{case_c}-{case_r}_{branch}_{step}_Bdiff_Bind_streamlines_slices.png")
plt.savefig(out_png, dpi=250, bbox_inches='tight')
# plt.close(fig)
print("Saved:", out_png)

# --------------------------
# NEW FIGURE: signed induced contribution along core direction
# --------------------------
# We plot Bind_parallel on the same three slices. This is a scalar field, so use imshow/pcolormesh.
# Positive => induced field reinforces the core dipole; negative => it opposes it.

# slices of Bind_parallel (nT)
Bp_xy = Bind_parallel[:, :, iz0].T
Bp_xz = Bind_parallel[:, iy0, :].T
Bp_yz = Bind_parallel[ix0, :, :].T

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

# robust symmetric limits
v = np.nanpercentile(np.abs(Bind_parallel), 99.0)
if not np.isfinite(v) or v == 0.0:
    v = np.nanmax(np.abs(Bind_parallel))
    if not np.isfinite(v) or v == 0.0:
        v = 1.0

# XY
im0 = axes2[0].imshow(Bp_xy, origin='lower',
                      extent=[x_box[0]/RM, x_box[-1]/RM, y_box[0]/RM, y_box[-1]/RM],
                      cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest', aspect='equal')
axes2[0].set_title(r"$\Delta\mathbf{B}\cdot\hat{\mathbf{B}}_{core}$ : XY (z=0)")
axes2[0].set_xlabel('X [Rₘ]')
axes2[0].set_ylabel('Y [Rₘ]')

# XZ
im1 = axes2[1].imshow(Bp_xz, origin='lower',
                      extent=[x_box[0]/RM, x_box[-1]/RM, z_box[0]/RM, z_box[-1]/RM],
                      cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest', aspect='equal')
axes2[1].set_title(r"$\Delta\mathbf{B}\cdot\hat{\mathbf{B}}_{core}$ : XZ (y=0)")
axes2[1].set_xlabel('X [Rₘ]')
axes2[1].set_ylabel('Z [Rₘ]')

# YZ
im2 = axes2[2].imshow(Bp_yz, origin='lower',
                      extent=[y_box[0]/RM, y_box[-1]/RM, z_box[0]/RM, z_box[-1]/RM],
                      cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest', aspect='equal')
axes2[2].set_title(r"$\Delta\mathbf{B}\cdot\hat{\mathbf{B}}_{core}$ : YZ (x=0)")
axes2[2].set_xlabel('Y [Rₘ]')
axes2[2].set_ylabel('Z [Rₘ]')

# overlay planet + CMB circles
th = np.linspace(0, 2*np.pi, 400)
Rsurf = 1.0
Rc = 2080.0 / RM
for ax in axes2:
    ax.plot(Rsurf*np.cos(th), Rsurf*np.sin(th), color='k', lw=1.5)
    ax.plot(Rc*np.cos(th), Rc*np.sin(th), color='hotpink', lw=1.2)

cbar = fig2.colorbar(im0, ax=axes2.ravel().tolist(), fraction=0.03, pad=0.02)
cbar.set_label(r"$\Delta\mathbf{B}\cdot\hat{\mathbf{B}}_{core}$ [nT]  ( + reinforces / − opposes )")

fig2.suptitle(f"{case_c}-{case_r} {branch} signed induced contribution along core; t={step*0.002} s", fontsize=16)

out_png2 = os.path.join(output_folder, f"{case_c}-{case_r}_{branch}_{step}_Bind_parallel_slices.png")
plt.savefig(out_png2, dpi=250, bbox_inches='tight')
plt.close(fig2)
print("Saved:", out_png2)


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
ax.set_title(f"{case_c}-{case_r} {branch} Induced Magnetic Field (X-Z Plane); t={step*0.002} s")
ax.set_xlim(-2 * RM, 2 * RM)
ax.set_ylim(-2 * RM, 2 * RM)
plt.tight_layout()
output_topo = os.path.join(output_folder, "2D_topology/")
os.makedirs(output_topo, exist_ok=True)
plt.savefig(os.path.join(output_topo, f"{case_c}-{case_r}_{branch}_bfield_induced_{step}.png"), dpi=150,
            bbox_inches="tight")
print("Saved:\t", os.path.join(output_topo, f"{case_c}-{case_r}_{branch}_bfield_induced_{step}.png"))
plt.close()
