# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# [SETTINGS & LOAD DATA - UNCHANGED]
branch = "HNHV"
case = "PN"
steps = [115000, 142000, 174000, 350000]
case_r = f"R{case}"
case_c = f"C{case}"

output_folder = f"/Users/danywaller/Projects/mercury/extreme/induced_bfield_topology/{case_c}-{case_r}_{branch}/"
RM = 2440.0
Rc_norm = 2080.0 / RM

print("Loading processed NetCDF files...")
data = {}
for step in steps:
    infile = os.path.join(output_folder, f"{case_c}-{case_r}_{branch}_{step}_Bind_induced_fields_nT.nc")
    ds = xr.open_dataset(infile)
    data[step] = {
        'x': ds['x'].values, 'y': ds['y'].values, 'z': ds['z'].values,
        'Bx_ind': ds['Bx_ind_nT'].values, 'By_ind': ds['By_ind_nT'].values, 'Bz_ind': ds['Bz_ind_nT'].values,
        'Bind_parallel': ds['Bind_parallel_nT'].values,
    }
    ds.close()
print("All files loaded.")


# [FUNCTIONS - UNCHANGED + AXIS LABELS]
def add_stream_panel(ax, x1d, y1d, U_xy, V_xy, R, title, density=4, min_mag=0.0, plane='XY'):
    X, Y = np.meshgrid(x1d, y1d, indexing="xy")
    U_m, V_m = U_xy, V_xy
    mag = np.hypot(U_m, V_m)
    bad = mag < min_mag
    U_m = U_m.copy()
    V_m = V_m.copy()
    U_m[bad] = np.nan
    V_m[bad] = np.nan

    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="mediumorchid", lw=1.6)
    ax.plot(Rc_norm * np.cos(th), Rc_norm * np.sin(th), color="hotpink", lw=1.6)

    C = np.log10(np.maximum(np.hypot(U_m, V_m), 1e-10))
    sp = ax.streamplot(x1d / R, y1d / R, U_m, V_m, density=density, color=C,
                       cmap="viridis", linewidth=1.0, arrowsize=1.0)
    ax.set_title(title, fontsize=14)
    # Plane-specific axis labels
    if plane == 'XY':
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_xlabel('X [R$_M$]', fontsize=12)
        ax.set_ylabel('Y [R$_M$]', fontsize=12)
    elif plane == 'XZ':
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_xlabel('X [R$_M$]', fontsize=12)
        ax.set_ylabel('Z [R$_M$]', fontsize=12)
    elif plane == 'YZ':
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_xlabel('Y [R$_M$]', fontsize=12)
        ax.set_ylabel('Z [R$_M$]', fontsize=12)
    ax.set_aspect("equal")
    ax.grid(False)
    return sp


def add_bind_panel(ax, x1d, y1d, Bp_slice, R, title, plane='XY'):
    v = np.nanpercentile(np.abs(Bp_slice), 99.0) or np.nanmax(np.abs(Bp_slice)) or 1.0
    im = ax.pcolormesh(x1d / R, y1d / R, Bp_slice, cmap='RdBu_r', vmin=-v, vmax=v, shading='auto')
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color='k', lw=1.5)
    ax.plot(Rc_norm * np.cos(th), Rc_norm * np.sin(th), color='hotpink', lw=1.2)
    ax.set_title(title, fontsize=14)

    # Plane-specific axis labels
    if plane == 'XY':
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_xlabel('X [R$_M$]', fontsize=12)
        ax.set_ylabel('Y [R$_M$]', fontsize=12)
    elif plane == 'XZ':
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_xlabel('X [R$_M$]', fontsize=12)
        ax.set_ylabel('Z [R$_M$]', fontsize=12)
    elif plane == 'YZ':
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_xlabel('Y [R$_M$]', fontsize=12)
        ax.set_ylabel('Z [R$_M$]', fontsize=12)

    ax.set_aspect("equal")
    ax.grid(False)
    return im


# --------------------------
# FIGURE 1: B_ind STREAMLINES
# --------------------------
fig1, axes1 = plt.subplots(4, 3, figsize=(16, 20), constrained_layout=True)
axes1 = axes1.reshape(4, 3)

first_sp = None
for i, step in enumerate(steps):
    row = i
    d = data[step]
    ix0, iy0, iz0 = [np.argmin(np.abs(d[k])) for k in ['x', 'y', 'z']]
    t_sec = step * 0.002

    # XY (col 0) - X/Y labels
    U_xy = d['Bx_ind'][:, :, iz0].T
    V_xy = d['By_ind'][:, :, iz0].T
    sp00 = add_stream_panel(axes1[row, 0], d['x'], d['y'], U_xy, V_xy, RM, f"XY t={t_sec:.1f}s", plane='XY')
    if first_sp is None: first_sp = sp00

    # XZ (col 1) - X/Z labels
    U_xz = d['Bx_ind'][:, iy0, :].T
    V_xz = d['Bz_ind'][:, iy0, :].T
    add_stream_panel(axes1[row, 1], d['x'], d['z'], U_xz, V_xz, RM, f"XZ t={t_sec:.1f}s", plane='XZ')

    # YZ (col 2) - Y/Z labels
    U_yz = d['By_ind'][ix0, :, :].T
    V_yz = d['Bz_ind'][ix0, :, :].T
    add_stream_panel(axes1[row, 2], d['y'], d['z'], U_yz, V_yz, RM, f"YZ t={t_sec:.1f}s", plane='YZ')

# Colorbars for each row's last subplot
for row in range(4):
    cbar = axes1[row, 2].figure.colorbar(first_sp.lines, ax=axes1[row, 2], shrink=0.8, pad=0.05)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('log₁₀(nT)', fontsize=11, labelpad=12)

fig1.suptitle(f"{case_c}-{case_r}: B$_{{ind}}$ Streamlines", fontsize=20, y=1.02)
out1 = os.path.join(output_folder, f"{case_c}-{case_r}_{branch}_B_ind_streamlines_evolution.png")
plt.savefig(out1, dpi=250, bbox_inches='tight')
print(f"Saved: {out1}")

# --------------------------
# FIGURE 2: Bind_parallel
# --------------------------
fig2, axes2 = plt.subplots(4, 3, figsize=(16, 20), constrained_layout=True)
axes2 = axes2.reshape(4, 3)

first_im = None
for i, step in enumerate(steps):
    row = i
    d = data[step]
    ix0, iy0, iz0 = [np.argmin(np.abs(d[k])) for k in ['x', 'y', 'z']]
    t_sec = step * 0.002

    # XY (col 0)
    Bp_xy = d['Bind_parallel'][:, :, iz0].T
    im0 = add_bind_panel(axes2[row, 0], d['x'], d['y'], Bp_xy, RM, f"XY t={t_sec:.1f}s", plane='XY')
    if first_im is None: first_im = im0

    # XZ (col 1)
    Bp_xz = d['Bind_parallel'][:, iy0, :].T
    add_bind_panel(axes2[row, 1], d['x'], d['z'], Bp_xz, RM, f"XZ t={t_sec:.1f}s", plane='XZ')

    # YZ (col 2)
    Bp_yz = d['Bind_parallel'][ix0, :, :].T
    add_bind_panel(axes2[row, 2], d['y'], d['z'], Bp_yz, RM, f"YZ t={t_sec:.1f}s", plane='YZ')

# Colorbars for each row's last subplot
for row in range(4):
    cbar = axes2[row, 2].figure.colorbar(first_im, ax=axes2[row, 2], shrink=0.8, pad=0.05)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('[nT]  ( + reinforces / − opposes )', fontsize=11)

fig2.suptitle(f"{case_c}-{case_r}: Signed B$_{{ind}} \\cdot \\hat{{B}}_{{dipole}}$", fontsize=20, y=1.02)
out2 = os.path.join(output_folder, f"{case_c}-{case_r}_{branch}_Bind_parallel_evolution.png")
plt.savefig(out2, dpi=250, bbox_inches='tight')
print(f"Saved: {out2}")

plt.show()
