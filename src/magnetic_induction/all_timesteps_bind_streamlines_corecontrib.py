#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports:
import os
from datetime import datetime
import numpy as np
import xarray as xr
import gc
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --------------------------
# SETTINGS
# --------------------------
debug = False
branch = "HNHV"
case = "PN"     # 115000 (pre) or 135000 (transient) or 165000 (post) or 350000 (new)
# steps = [115000, 135000, 165000, 350000]
steps = [135000, 165000]

case_r = f"R{case}"
case_c = f"C{case}"

for step in steps:
    if "HN" in branch:
        input_folder_r = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case_r}_HNHV/plane_product/cube/"
        input_folder_c = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case_c}_HNHV/plane_product/cube/"
        ncfile_r = os.path.join(input_folder_r, f"Amitis_{case_r}_HNHV_{step}_merged_4RM.nc")
        ncfile_c = os.path.join(input_folder_c, f"Amitis_{case_c}_HNHV_{step}_merged_4RM.nc")
    else:
        input_folder_r = f"/Volumes/data_backup/mercury/extreme/{case_r}_Base/plane_product/cube/"
        input_folder_c = f"/Volumes/data_backup/mercury/extreme/High_Base/{case_c}_Base/plane_product/cube/"
        ncfile_r = os.path.join(input_folder_r, f"Amitis_{case_r}_Base_{step}_merged_4RM.nc")
        ncfile_c = os.path.join(input_folder_c, f"Amitis_{case_c}_Base_{step}_merged_4RM.nc")

    output_folder = f"/Users/danywaller/Projects/mercury/extreme/induced_bfield_topology/{case_c}-{case_r}_{branch}/"
    os.makedirs(output_folder, exist_ok=True)

    # Planet parameters
    RM = 2440.0  # km
    plot_depth = RM
    L = 4.5 * RM      # half-width of cube of interest: [-L, +L]
    buf = 0.5 * RM    # buffer to reduce edge effects

    MU0 = 4.0 * np.pi * 1e-7  # H/m

    # --------------------------
    # LOAD VECTOR FIELD FROM NETCDF
    # --------------------------
    def load_field(ncfile, var_ext):
        ds = xr.open_dataset(ncfile)
        x = ds["Nx"].values  # [km]
        y = ds["Ny"].values  # [km]
        z = ds["Nz"].values  # [km]

        # Extract fields (drop time dimension) and transpose: Nz, Ny, Nx --> Nx, Ny, Nz
        A0 = np.transpose(ds[var_ext[0]].isel(time=0).values, (2, 1, 0))
        A1 = np.transpose(ds[var_ext[1]].isel(time=0).values, (2, 1, 0))
        A2 = np.transpose(ds[var_ext[2]].isel(time=0).values, (2, 1, 0))

        ds.close()
        return x, y, z, A0, A1, A2

    # ---- Resistive case ----
    x_km, y_km, z_km, Jx_nA_r, Jy_nA_r, Jz_nA_r = load_field(ncfile_r, ["Jx", "Jy", "Jz"])
    _, _, _, BxTot_r, ByTot_r, BzTot_r = load_field(ncfile_r, ["Bx_tot", "By_tot", "Bz_tot"])
    _, _, _, BxExt_r, ByExt_r, BzExt_r = load_field(ncfile_r, ["Bx", "By", "Bz"])
    print(f"Loaded resistive core at {datetime.now()}")

    # ---- Conductive case ----
    _, _, _, Jx_nA_c, Jy_nA_c, Jz_nA_c = load_field(ncfile_c, ["Jx", "Jy", "Jz"])
    _, _, _, BxTot_c, ByTot_c, BzTot_c = load_field(ncfile_c, ["Bx_tot", "By_tot", "Bz_tot"])
    _, _, _, BxExt_c, ByExt_c, BzExt_c = load_field(ncfile_c, ["Bx", "By", "Bz"])
    print(f"Loaded conductive core at {datetime.now()}")

    # --------------------------
    # CORE FIELD (COMMON DIPOLE)
    # --------------------------
    # B_core = B_tot - B_ext
    Bx_core = BxTot_r - BxExt_r
    By_core = ByTot_r - ByExt_r
    Bz_core = BzTot_r - BzExt_r

    # --------------------------
    # J DIFFERENCES (conductive - resistive)
    # --------------------------
    Jx_nA = Jx_nA_c - Jx_nA_r
    Jy_nA = Jy_nA_c - Jy_nA_r
    Jz_nA = Jz_nA_c - Jz_nA_r

    # Fill NaNs, otherwise Poisson func breaks
    Jx_nA = np.where(np.isnan(Jx_nA), 0.0, Jx_nA)
    Jy_nA = np.where(np.isnan(Jy_nA), 0.0, Jy_nA)
    Jz_nA = np.where(np.isnan(Jz_nA), 0.0, Jz_nA)

    # --------------------------
    # B field difference (conductive - resistive)
    # --------------------------
    Bx_diff = BxTot_c - BxTot_r
    By_diff = ByTot_c - ByTot_r
    Bz_diff = BzTot_c - BzTot_r

    # --------------------------
    # SUBGRID SELECTION
    # --------------------------
    def idx_range(coord_km, lo, hi):
        i0 = int(np.searchsorted(coord_km, lo, side="left"))
        i1 = int(np.searchsorted(coord_km, hi, side="right"))
        i0 = max(i0, 0)
        i1 = min(i1, coord_km.size)
        return slice(i0, i1)

    sx = idx_range(x_km, -L - buf, +L + buf)
    sy = idx_range(y_km, -L - buf, +L + buf)
    sz = idx_range(z_km, -L - buf, +L + buf)

    x_sub = x_km[sx]
    y_sub = y_km[sy]
    z_sub = z_km[sz]

    Bxcore_sub = Bx_core[sx, sy, sz]
    Bycore_sub = By_core[sx, sy, sz]
    Bzcore_sub = Bz_core[sx, sy, sz]

    Bxdiff_sub = Bx_diff[sx, sy, sz]
    Bydiff_sub = By_diff[sx, sy, sz]
    Bzdiff_sub = Bz_diff[sx, sy, sz]

    Jx_sub = Jx_nA[sx, sy, sz]
    Jy_sub = Jy_nA[sx, sy, sz]
    Jz_sub = Jz_nA[sx, sy, sz]

    # inner box (no extra buffer)
    sx2 = idx_range(x_sub, -L, +L)
    sy2 = idx_range(y_sub, -L, +L)
    sz2 = idx_range(z_sub, -L, +L)

    x_box = x_sub[sx2]
    y_box = y_sub[sy2]
    z_box = z_sub[sz2]

    Bxcore_box = Bxcore_sub[sx2, sy2, sz2]
    Bycore_box = Bycore_sub[sx2, sy2, sz2]
    Bzcore_box = Bzcore_sub[sx2, sy2, sz2]

    Bxdiff_box = Bxdiff_sub[sx2, sy2, sz2]
    Bydiff_box = Bydiff_sub[sx2, sy2, sz2]
    Bzdiff_box = Bzdiff_sub[sx2, sy2, sz2]

    Jx_box = Jx_sub[sx2, sy2, sz2]
    Jy_box = Jy_sub[sx2, sy2, sz2]
    Jz_box = Jz_sub[sx2, sy2, sz2]


    # --------------------------
    # POISSON-BASED B FROM J
    # --------------------------
    def B_from_J_poisson_lowmem(x_km, y_km, z_km, Jx_nA, Jy_nA, Jz_nA,
                                rtol=1e-6, atol=0.0, maxiter=2000, edge_order=1,
                                use_jacobi=True):
        """
        Compute magnetic field B(x,y,z) from a 3-D current density J(x,y,z) without FFT.
        This is a magnetostatic / Biot–Savart-like solve on a non-periodic box

        Method (real space via vector potential):
            1) Solve Poisson for A:
                  ∇² A = -μ0 J
               (solve three scalar Poisson problems for Ax, Ay, Az)
            2) Compute B as:
                  B = ∇ × A

        Notes / assumptions:
        - Boundary conditions: this effectively enforces A=0 on the outer box
          (homogeneous Dirichlet)
        - This avoids periodic wrap-around in FFT, but still need padding so that
          induced B in the interior is not dominated by the artificial boundary

        Solver details:
        - Uses conjugate gradient (CG) on (-∇²)A = μ0 J which is SPD
        - Convergence criterion: ||b - A@x|| <= max(rtol*||b||, atol)
        - Optional Jacobi preconditioner using the diagonal of (-∇²)

        Parameters
        ----------
        x_km, y_km, z_km : 1-D arrays
            Coordinates along each axis in kilometers
        Jx_nA, Jy_nA, Jz_nA : 3-D arrays, shape (Nx, Ny, Nz)
            Current density components on the grid in nA/m^2
        rtol, atol : float
            CG stopping criteria
        maxiter : int
            Max CG iterations
        edge_order : int (1 or 2)
            Order used by np.gradient at the boundaries
        use_jacobi : bool
            If True, use a diagonal (Jacobi) preconditioner

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

        if debug:
            print("Grid spacing (m): dx,dy,dz =", dx, dy, dz)
            print("Grid sizes: Nx,Ny,Nz,N =", Nx, Ny, Nz, N)
            print("Jx stats: nan/min/max =", np.isnan(Jx).sum(),
                  np.nanmin(Jx), np.nanmax(Jx))

        # Build sparse 3-D Laplacian operator via Kronecker sums
        def D2(n, h):
            main = (-2.0) * np.ones(n)
            off = (1.0) * np.ones(n - 1)
            return sp.diags([off, main, off], [-1, 0, 1], shape=(n, n), format="csr") / (h * h)

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

        if debug:
            print("Aop nnz:", Aop.nnz)
            print("Aop shape:", Aop.shape)
            print("Aop diag stats: min/max =", Aop.diagonal().min(), Aop.diagonal().max())

        # Jacobi preconditioner: M ≈ inv(Aop)
        M = None
        if use_jacobi:
            d = Aop.diagonal().copy()
            d[d == 0.0] = 1.0

            def Minv(v):
                return v / d

            M = spla.LinearOperator((N, N), matvec=Minv, dtype=np.float64)

        def solve_component(name, Ji):
            if debug:
                print(f"Solving component {name} ...")
                print(f"{name} RHS stats: nan/min/max =",
                      np.isnan(Ji).sum(), np.nanmin(Ji), np.nanmax(Ji))
            b = (MU0 * Ji).ravel(order="C")
            Ai, info = spla.cg(Aop, b, rtol=rtol, atol=atol, maxiter=maxiter, M=M)
            if info != 0:
                raise RuntimeError(f"CG did not converge (info={info})")
            return Ai.reshape((Nx, Ny, Nz), order="C")

        # Solve Poisson for Ax, Ay, Az
        Ax = solve_component("Jx", Ji=Jx)
        Ay = solve_component("Jy", Ji=Jy)
        Az = solve_component("Jz", Ji=Jz)

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


    # Poisson method (no FFT)
    bstart = datetime.now()
    Bx_sub, By_sub, Bz_sub = B_from_J_poisson_lowmem(
        x_sub, y_sub, z_sub, Jx_sub, Jy_sub, Jz_sub,
        rtol=1e-6, atol=0.0, maxiter=20000, edge_order=1, use_jacobi=True
    )
    bend = datetime.now()
    print(f"Computed B everywhere (Poisson) from J at {str(bend)}; elapsed={(bend - bstart)}")

    # --------------------------
    # SIGNED INDUCED CONTRIBUTION ALONG CORE
    # --------------------------
    eps = 1e-30
    Bcore_mag = np.sqrt(Bxcore_box**2 + Bycore_box**2 + Bzcore_box**2) + eps
    bxhat = Bxcore_box / Bcore_mag
    byhat = Bycore_box / Bcore_mag
    bzhat = Bzcore_box / Bcore_mag

    # difference field (conductive - resistive) on same box
    Bxdiff_box = Bxdiff_box
    Bydiff_box = Bydiff_box
    Bzdiff_box = Bzdiff_box

    Bx_box = Bx_sub[sx2, sy2, sz2]
    By_box = By_sub[sx2, sy2, sz2]
    Bz_box = Bz_sub[sx2, sy2, sz2]

    # project induced B onto core direction, convert to nT
    Bind_parallel_nT = (Bx_box * bxhat +
                        By_box * byhat +
                        Bz_box * bzhat) * 1e9

    # magnitude of B from Poisson (Bind from J) in nT
    Bmag_bind_nT = np.sqrt(Bx_box**2 + By_box**2 + Bz_box**2) * 1e9

    print("Bind_parallel min/max (nT) =",
          np.nanmin(Bind_parallel_nT), np.nanmax(Bind_parallel_nT))
    print("|B_ind from J| min/max (nT) =",
          np.nanmin(Bmag_bind_nT), np.nanmax(Bmag_bind_nT))

    # clean up heavy arrays not needed in output (optional)
    del (x_km, y_km, z_km,
         Jx_nA, Jy_nA, Jz_nA,
         Jx_sub, Jy_sub, Jz_sub,
         Bxcore_sub, Bycore_sub, Bzcore_sub,
         Bxdiff_sub, Bydiff_sub, Bzdiff_sub)
    gc.collect()

    # --------------------------
    # SAVE TO NETCDF (nT, zlib=4)
    # --------------------------
    # Create xarray Dataset on (Nx, Ny, Nz) grid for inner box
    coords = {
        "x": ("x", x_box),
        "y": ("y", y_box),
        "z": ("z", z_box),
    }

    data_vars = {}

    # core field (nT)
    data_vars["Bx_core_nT"] = (("x", "y", "z"), Bxcore_box)
    data_vars["By_core_nT"] = (("x", "y", "z"), Bycore_box)
    data_vars["Bz_core_nT"] = (("x", "y", "z"), Bzcore_box)

    # difference (conductive - resistive), nT
    data_vars["Bx_diff_nT"] = (("x", "y", "z"), Bxdiff_box)
    data_vars["By_diff_nT"] = (("x", "y", "z"), Bydiff_box)
    data_vars["Bz_diff_nT"] = (("x", "y", "z"), Bzdiff_box)

    # induced field from J solver, nT
    data_vars["Bx_ind_nT"] = (("x", "y", "z"), Bx_box * 1e9)
    data_vars["By_ind_nT"] = (("x", "y", "z"), By_box * 1e9)
    data_vars["Bz_ind_nT"] = (("x", "y", "z"), Bz_box * 1e9)

    # magnitude of induced field from J solver, nT
    data_vars["B_ind_mag_fromJ_nT"] = (("x", "y", "z"), Bmag_bind_nT)

    # signed induced contribution along core, nT
    data_vars["Bind_parallel_nT"] = (("x", "y", "z"), Bind_parallel_nT)

    ds_out = xr.Dataset(data_vars=data_vars, coords=coords)

    comp = dict(zlib=True, complevel=4)
    encoding = {var: {**comp, "dtype": "float32"} for var in ds_out.data_vars}

    outfile = os.path.join(
        output_folder,
        f"{case_c}-{case_r}_{branch}_{step}_Bind_induced_fields_nT.nc"
    )

    ds_out.to_netcdf(outfile, mode="w", format="NETCDF4", encoding=encoding)
    ds_out.close()

    print("Wrote:", outfile)
