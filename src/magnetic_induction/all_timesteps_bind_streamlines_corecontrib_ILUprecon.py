# -*- coding: utf-8 -*-

import os
from datetime import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import gc

# --------------------------
# SETTINGS
# --------------------------

debug = False
plotlog = True

branch = "HNHV"
case = "PN"

steps = [115000, 142000, 174000, 350000]

case_r = f"R{case}"
case_c = f"C{case}"

RM = 2440.0
L = 2.5 * RM
buf = 0.5 * RM

MU0 = 4.0 * np.pi * 1e-7

# --------------------------
# PATHS
# --------------------------

if "HN" in branch:
    input_folder_r = f"/Volumes/T9/mercury/extreme/High_HNHV/{case_r}_HNHV/plane_product/cube/"
    input_folder_c = f"/Volumes/T9/mercury/extreme/High_HNHV/{case_c}_HNHV/plane_product/cube/"
    filetag = "HNHV"
else:
    input_folder_r = f"/Volumes/T9/mercury/extreme/{case_r}_Base/plane_product/cube/"
    input_folder_c = f"/Volumes/T9/mercury/extreme/{case_c}_Base/plane_product/cube/"
    filetag = "Base"

output_folder = f"/Users/danywaller/Projects/mercury/extreme/induced_bfield_topology/{case_c}-{case_r}_{branch}/"
os.makedirs(output_folder, exist_ok=True)

# --------------------------
# UTILITIES
# --------------------------

def load_field(ncfile, var_ext):
    ds = xr.open_dataset(ncfile)

    x = ds["Nx"].values
    y = ds["Ny"].values
    z = ds["Nz"].values

    Jx = np.transpose(ds[var_ext[0]].isel(time=0).values, (2,1,0))
    Jy = np.transpose(ds[var_ext[1]].isel(time=0).values, (2,1,0))
    Jz = np.transpose(ds[var_ext[2]].isel(time=0).values, (2,1,0))

    ds.close()

    return x,y,z,Jx,Jy,Jz


def idx_range(coord_km, lo, hi):
    i0 = int(np.searchsorted(coord_km, lo, side="left"))
    i1 = int(np.searchsorted(coord_km, hi, side="right"))
    i0 = max(i0,0)
    i1 = min(i1,coord_km.size)
    return slice(i0,i1)


# --------------------------
# POISSON SOLVER CACHE
# --------------------------

poisson_cache = {}

def build_poisson_solver(Nx,Ny,Nz,dx,dy,dz):

    def D2(n,h):
        main = -2*np.ones(n)
        off = np.ones(n-1)
        return sp.diags([off,main,off],[-1,0,1],shape=(n,n))/(h*h)

    Lx = D2(Nx,dx)
    Ly = D2(Ny,dy)
    Lz = D2(Nz,dz)

    Ix = sp.eye(Nx)
    Iy = sp.eye(Ny)
    Iz = sp.eye(Nz)

    Lap = sp.kron(sp.kron(Lx, Iy), Iz) \
          + sp.kron(sp.kron(Ix, Ly), Iz) \
          + sp.kron(sp.kron(Ix, Iy), Lz)

    Aop = (-Lap).tocsc()

    print("Building ILU preconditioner...")

    ilu = spla.spilu(Aop, drop_tol=1e-4, fill_factor=10)

    M = spla.LinearOperator(Aop.shape, ilu.solve)

    return Aop, M


# --------------------------
# POISSON SOLVER
# --------------------------

def B_from_J_poisson_lowmem(x_km,y_km,z_km,Jx_nA,Jy_nA,Jz_nA):

    x = x_km*1e3
    y = y_km*1e3
    z = z_km*1e3

    Jx = Jx_nA*1e-9
    Jy = Jy_nA*1e-9
    Jz = Jz_nA*1e-9

    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))
    dz = np.mean(np.diff(z))

    Nx,Ny,Nz = Jx.shape
    N = Nx*Ny*Nz

    key = (Nx,Ny,Nz,dx,dy,dz)

    if key not in poisson_cache:

        print("Building Poisson matrix...")

        poisson_cache[key] = build_poisson_solver(Nx,Ny,Nz,dx,dy,dz)

    Aop,M = poisson_cache[key]

    # -------- BLOCK RHS SOLVE --------

    B = np.vstack([
        (MU0*Jx).ravel(),
        (MU0*Jy).ravel(),
        (MU0*Jz).ravel()
    ]).T

    X = np.zeros_like(B)

    for i in range(3):
        sol,_ = spla.cg(Aop, B[:,i], maxiter=10000, M=M)
        X[:,i] = sol

    Ax = X[:,0].reshape((Nx,Ny,Nz))
    Ay = X[:,1].reshape((Nx,Ny,Nz))
    Az = X[:,2].reshape((Nx,Ny,Nz))

    # -------- CURL --------

    dAz_dy = np.gradient(Az,dy,axis=1)
    dAy_dz = np.gradient(Ay,dz,axis=2)

    dAx_dz = np.gradient(Ax,dz,axis=2)
    dAz_dx = np.gradient(Az,dx,axis=0)

    dAy_dx = np.gradient(Ay,dx,axis=0)
    dAx_dy = np.gradient(Ax,dy,axis=1)

    Bx = dAz_dy - dAy_dz
    By = dAx_dz - dAz_dx
    Bz = dAy_dx - dAx_dy

    return Bx,By,Bz


# --------------------------
# STREAMLINE PANEL
# --------------------------

def add_panel(ax,x1d,y1d,U,V,R,title):

    C = np.sqrt(U*U+V*V)

    if plotlog:
        C = np.log10(C+1e-12)

    sp_stream = ax.streamplot(
        x1d/R,
        y1d/R,
        U,
        V,
        color=C,
        density=2.0,
        cmap="viridis"
    )

    th = np.linspace(0,2*np.pi,400)

    ax.plot(np.cos(th),np.sin(th),color="mediumorchid")

    Rc = 2080/R
    ax.plot(Rc*np.cos(th),Rc*np.sin(th),color="hotpink")

    ax.set_title(title)
    ax.set_aspect("equal")

    return sp_stream


# --------------------------
# FIGURES
# --------------------------

nrows = len(steps)

fig_stream,ax_stream = plt.subplots(
    nrows,3,figsize=(15,4*nrows),constrained_layout=True
)

fig_parallel,ax_parallel = plt.subplots(
    nrows,3,figsize=(15,4*nrows),constrained_layout=True
)


# --------------------------
# MAIN LOOP
# --------------------------

for row,step in enumerate(steps):

    print("Processing step",step)

    ncfile_r = os.path.join(
        input_folder_r,
        f"Amitis_{case_r}_{filetag}_{step}_merged_4RM.nc"
    )

    ncfile_c = os.path.join(
        input_folder_c,
        f"Amitis_{case_c}_{filetag}_{step}_merged_4RM.nc"
    )

    x_km,y_km,z_km,Jx_r,Jy_r,Jz_r = load_field(ncfile_r,["Jx","Jy","Jz"])
    _,_,_,BxTot_r,ByTot_r,BzTot_r = load_field(ncfile_r,["Bx_tot","By_tot","Bz_tot"])
    _,_,_,BxExt_r,ByExt_r,BzExt_r = load_field(ncfile_r,["Bx","By","Bz"])

    _,_,_,Jx_c,Jy_c,Jz_c = load_field(ncfile_c,["Jx","Jy","Jz"])
    _,_,_,BxTot_c,ByTot_c,BzTot_c = load_field(ncfile_c,["Bx_tot","By_tot","Bz_tot"])
    _,_,_,BxExt_c,ByExt_c,BzExt_c = load_field(ncfile_c,["Bx","By","Bz"])

    Bx_core = BxTot_r - BxExt_r
    By_core = ByTot_r - ByExt_r
    Bz_core = BzTot_r - BzExt_r

    Jx = Jx_c - Jx_r
    Jy = Jy_c - Jy_r
    Jz = Jz_c - Jz_r

    Bx_diff = BxTot_c - BxTot_r
    By_diff = ByTot_c - ByTot_r
    Bz_diff = BzTot_c - BzTot_r

    sx = idx_range(x_km,-L-buf,L+buf)
    sy = idx_range(y_km,-L-buf,L+buf)
    sz = idx_range(z_km,-L-buf,L+buf)

    x_sub = x_km[sx]
    y_sub = y_km[sy]
    z_sub = z_km[sz]

    Bx_sub,By_sub,Bz_sub = B_from_J_poisson_lowmem(
        x_sub,y_sub,z_sub,
        Jx[sx,sy,sz],
        Jy[sx,sy,sz],
        Jz[sx,sy,sz]
    )

    sx2 = idx_range(x_sub,-L,L)
    sy2 = idx_range(y_sub,-L,L)
    sz2 = idx_range(z_sub,-L,L)

    x_box = x_sub[sx2]
    y_box = y_sub[sy2]
    z_box = z_sub[sz2]

    Bx_box = Bx_sub[sx2,sy2,sz2]
    By_box = By_sub[sx2,sy2,sz2]
    Bz_box = Bz_sub[sx2,sy2,sz2]

    Bxcore = Bx_core[sx2,sy2,sz2]
    Bycore = By_core[sx2,sy2,sz2]
    Bzcore = Bz_core[sx2,sy2,sz2]

    Bxdiff = Bx_diff[sx2,sy2,sz2]
    Bydiff = By_diff[sx2,sy2,sz2]
    Bzdiff = Bz_diff[sx2,sy2,sz2]

    Bcore_mag = np.sqrt(Bxcore**2+Bycore**2+Bzcore**2)+1e-30

    bxhat = Bxcore/Bcore_mag
    byhat = Bycore/Bcore_mag
    bzhat = Bzcore/Bcore_mag

    Bind_parallel = Bxdiff*bxhat + Bydiff*byhat + Bzdiff*bzhat

    ix0 = np.argmin(np.abs(x_box))
    iy0 = np.argmin(np.abs(y_box))
    iz0 = np.argmin(np.abs(z_box))

    Bx_xy = Bx_box[:,:,iz0].T*1e9
    By_xy = By_box[:,:,iz0].T*1e9

    Bx_xz = Bx_box[:,iy0,:].T*1e9
    Bz_xz = Bz_box[:,iy0,:].T*1e9

    By_yz = By_box[ix0,:,:].T*1e9
    Bz_yz = Bz_box[ix0,:,:].T*1e9

    sp_stream = add_panel(ax_stream[row,0],x_box,y_box,Bx_xy,By_xy,RM,"XY")
    add_panel(ax_stream[row,1],x_box,z_box,Bx_xz,Bz_xz,RM,"XZ")
    add_panel(ax_stream[row,2],y_box,z_box,By_yz,Bz_yz,RM,"YZ")

    Bp_xy = Bind_parallel[:,:,iz0].T
    Bp_xz = Bind_parallel[:,iy0,:].T
    Bp_yz = Bind_parallel[ix0,:,:].T

    v = np.nanpercentile(np.abs(Bind_parallel),99)

    im = ax_parallel[row,0].imshow(
        Bp_xy,
        origin="lower",
        extent=[x_box[0]/RM,x_box[-1]/RM,y_box[0]/RM,y_box[-1]/RM],
        cmap="RdBu_r",
        vmin=-v,vmax=v
    )

    ax_parallel[row,1].imshow(
        Bp_xz,
        origin="lower",
        extent=[x_box[0]/RM,x_box[-1]/RM,z_box[0]/RM,z_box[-1]/RM],
        cmap="RdBu_r",
        vmin=-v,vmax=v
    )

    ax_parallel[row,2].imshow(
        Bp_yz,
        origin="lower",
        extent=[y_box[0]/RM,y_box[-1]/RM,z_box[0]/RM,z_box[-1]/RM],
        cmap="RdBu_r",
        vmin=-v,vmax=v
    )

    gc.collect()


# --------------------------
# COLORBARS
# --------------------------

fig_stream.colorbar(
    sp_stream.lines,
    ax=ax_stream.ravel().tolist(),
    fraction=0.02,
    pad=0.02
).set_label("|B$_{ind}$|")

fig_parallel.colorbar(
    im,
    ax=ax_parallel.ravel().tolist(),
    fraction=0.02,
    pad=0.02
).set_label("B$_{ind}$ · B$_{d}$")


# --------------------------
# SAVE
# --------------------------

fig_stream.suptitle(f"{case_c}-{case_r} {branch} B$_{{ind}}$ streamlines")
fig_parallel.suptitle(f"{case_c}-{case_r} {branch} B$_{{ind}}$ influence on core")

fig_stream.savefig(
    os.path.join(output_folder,"Bind_streamlines_timesteps.png"),
    dpi=300
)

fig_parallel.savefig(
    os.path.join(output_folder,"Bind_parallel_timesteps.png"),
    dpi=300
)

print("Finished")