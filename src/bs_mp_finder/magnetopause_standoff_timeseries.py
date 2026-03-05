#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import zoom
from skimage.measure import marching_cubes
import plotly.graph_objects as go

# ============================================================
# USER SETTINGS
# ============================================================
debug = False

cases = ["RPN_HNHV", "CPN_HNHV", "RPS_HNHV", "CPS_HNHV"]

for c in cases:
    case = c
    print(f"Running {case.replace("_", " ")} case")

    nc_file = f"/Users/danywaller/Projects/mercury/extreme/magnetopause_3D_timeseries/{case}/{case}_mp_mask_timeseries.nc"
    out_dir = f"/Users/danywaller/Projects/mercury/extreme/magnetopause_3D_timeseries/{case}/mp_mesh_analysis/"
    os.makedirs(out_dir, exist_ok=True)

    INTERP_FACTOR = 3
    N_THETA = 90
    N_PHI = 360

    # Mercury magnetic dipole offset
    Z_OFFSET_RM = 484.0 / 2440.0  # 0.1984

    # ============================================================
    # FIXED MAGNETIC EQUATOR LATITUDE
    # ============================================================

    lat_mag_fixed = np.degrees(np.arcsin(Z_OFFSET_RM))  # ≈ 11.4°
    print("Magnetic equator latitude:", lat_mag_fixed)

    # ============================================================
    # LOAD DATA
    # ============================================================

    ds = xr.open_dataset(nc_file)

    mask_4d = ds["mp_mask"].values
    times = ds["time"].values
    x = ds["x"].values
    y = ds["y"].values
    z = ds["z"].values

    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))
    dz = np.mean(np.diff(z))

    # ============================================================
    # ANGULAR GRID
    # ============================================================

    theta = np.linspace(1e-6, np.pi-1e-6, N_THETA)
    phi = np.linspace(-np.pi/2+1e-6, np.pi/2-1e-6, N_PHI)

    lat_deg = 90.0 - np.degrees(theta)
    lon_deg = np.degrees(phi)

    idx_geo = np.argmin(np.abs(lat_deg - 0.0))
    idx_mag = np.argmin(np.abs(lat_deg - lat_mag_fixed))

    # ============================================================
    # STORAGE
    # ============================================================

    equator_geo_series = []
    equator_mag_series = []
    equator_geo_std = []
    equator_mag_std = []
    standoff_all = []

    # ============================================================
    # TIME LOOP
    # ============================================================

    for it, tsec in enumerate(times):

        print(f"Processing {it+1}/{len(times)}  t={tsec:.2f}s")

        mask = mask_4d[it].astype(np.float32)

        # Interpolate mask
        mask_interp = zoom(mask, INTERP_FACTOR, order=1)

        # Pad mask to prevent surface closing
        mask_interp = np.pad(mask_interp, pad_width=2, mode="constant", constant_values=0)

        # Interpolated voxel spacing
        dx_i = dx / INTERP_FACTOR
        dy_i = dy / INTERP_FACTOR
        dz_i = dz / INTERP_FACTOR

        # Run marching cubes
        verts, faces, _, _ = marching_cubes(
            mask_interp,
            level=0.5,
            spacing=(dx_i, dy_i, dz_i)
        )

        # Shift verts to match original domain
        verts[:, 0] += x.min() - dx_i
        verts[:, 1] += y.min() - dy_i
        verts[:, 2] += z.min() - dz_i

        # ===========================
        # DEBUG: Interactive 3D scatter of mesh
        # ===========================
        if debug:  # only show first frame to debug
            # Option 1: scatter of vertices
            fig_scatter = go.Figure(data=[go.Scatter3d(
                x=verts[:,0],
                y=verts[:,1],
                z=verts[:,2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='orange'
                )
            )])
            fig_scatter.update_layout(
                scene=dict(
                    xaxis_title='X (RM)',
                    yaxis_title='Y (RM)',
                    zaxis_title='Z (RM)',
                ),
                title=f'{case} Mesh Vertices at t={tsec:.2f}s'
            )
            scatter_path = os.path.join(out_dir, f"{case}_mesh_vertices_debug.html")
            fig_scatter.write_html(scatter_path)
            print("Saved mesh scatter:", scatter_path)

            mesh_fig = go.Figure(data=[go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightblue',
                opacity=0.7,
                flatshading=True
            )])

            mesh_fig.update_layout(
                scene=dict(
                    xaxis_title='X (RM)',
                    yaxis_title='Y (RM)',
                    zaxis_title='Z (RM)',
                ),
                title=f'{case} New Surface at t={times[it]:.2f}s'
            )

            mesh_path = os.path.join(out_dir, f"{case}_new_surface.html")
            mesh_fig.write_html(mesh_path)
            print("Saved new surface:", mesh_path)

        # ============================================================
        # Map vertices to theta/phi for dayside standoff
        # ============================================================

        vx, vy, vz = verts[:, 0], verts[:, 1], verts[:, 2]
        r = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        theta_v = np.arccos(vz / r)
        phi_v = np.arctan2(vy, vx)

        # dayside selection: X > 0 and longitude within ±75° of subsolar
        phi_deg_v = np.degrees(phi_v)
        dayside = (vx > 0) & (phi_deg_v >= -75) & (phi_deg_v <= 75)

        r = r[dayside]
        theta_v = theta_v[dayside]
        phi_v = phi_v[dayside]

        theta_idx = np.digitize(theta_v, theta) - 1
        phi_idx = np.digitize(phi_v, phi) - 1

        r_map = np.full((N_THETA, N_PHI), np.nan)

        for k in range(len(r)):
            i = theta_idx[k]
            j = phi_idx[k]
            if 0 <= i < N_THETA and 0 <= j < N_PHI:
                if np.isnan(r_map[i,j]) or r[k] > r_map[i,j]:
                    r_map[i,j] = r[k]

        standoff = r_map   # - 1.0
        standoff_all.append(standoff)

        if debug:
            # Debug: 2D plot of Δr
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.pcolormesh(lon_deg, lat_deg, standoff, shading="auto", cmap="viridis", vmin=1.0, vmax=2.0)
            ax.set_xlim(-90, 90)
            ax.set_ylim(-90, 90)
            ax.set_xlabel("Longitude (deg)")
            ax.set_ylabel("Latitude (deg)")
            ax.axhline(lat_mag_fixed, color="hotpink", linestyle="--", label="Magnetic Equator")
            ax.legend()
            ax.set_title(f"{case.replace('_',' ')} at t={tsec:.2f}s")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Δr ($R_M$)")
            plt.show()

        # Longitude-integrated Δr
        lon_integrated_mean = np.nanmean(standoff, axis=1)
        lon_integrated_std = np.nanstd(standoff, axis=1)
        equator_geo_series.append(lon_integrated_mean[idx_geo])
        equator_mag_series.append(lon_integrated_mean[idx_mag])
        equator_geo_std.append(lon_integrated_std[idx_geo])
        equator_mag_std.append(lon_integrated_std[idx_mag])

    # ============================================================
    # Convert to arrays
    # ============================================================

    equator_geo_series = np.array(equator_geo_series)
    equator_mag_series = np.array(equator_mag_series)
    equator_geo_std = np.array(equator_geo_std)
    equator_mag_std = np.array(equator_mag_std)
    standoff_all = np.array(standoff_all)

    # ============================================================
    # SAVE TIMESERIES TO CSV
    # ============================================================

    csv_path = os.path.join(out_dir, f"{case}_equatorial_standoff_timeseries.csv")

    data_out = np.column_stack([
        times,
        equator_geo_series,
        equator_geo_std,
        equator_mag_series,
        equator_mag_std
    ])

    header = (
        "time_s,"
        "geo_equator_mean_RM,"
        "geo_equator_std_RM,"
        "mag_equator_mean_RM,"
        "mag_equator_std_RM"
    )

    np.savetxt(csv_path, data_out, delimiter=",", header=header, comments="")
    print("Saved CSV:", csv_path)

    # ============================================================
    # ANIMATE 2D Δr MAP
    # ============================================================

    fig, ax = plt.subplots(figsize=(8,5))
    im = ax.pcolormesh(
        lon_deg,
        lat_deg,
        standoff_all[0],
        shading="auto",
        cmap="viridis",
        vmin=1.0,
        vmax=2.0
    )
    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.axhline(lat_mag_fixed, color="hotpink", linestyle="--", label="Magnetic Equator")
    ax.legend()
    title = ax.set_title("")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Δr ($R_M$)")

    def update(frame):
        im.set_array(standoff_all[frame].ravel())
        title.set_text(f"{case.replace('_',' ')} Dayside Magnetopause Standoff\n t = {times[frame]:.2f} s")
        return im, title

    ani = FuncAnimation(fig, update, frames=len(times), interval=200)
    mp4_path = os.path.join(out_dir, f"{case}_delta_r_latlon_animation.mp4")
    ani.save(mp4_path, dpi=200)
    plt.close()
    print("Saved animation:", mp4_path)

    # ============================================================
    # TIMESERIES PLOT
    # ============================================================

    fig, ax1 = plt.subplots(1,1, figsize=(8,8), sharex=True)
    ax1.errorbar(
        times,
        equator_geo_series,
        yerr=equator_geo_std,
        fmt='o-',
        color="mediumorchid",
        capsize=3,
        label="Geographic Equator (0°)"
    )
    ax1.errorbar(
        times,
        equator_mag_series,
        yerr=equator_mag_std,
        fmt='o-',
        color="hotpink",
        capsize=3,
        label=f"Magnetic Equator (+{lat_mag_fixed:.1f}°)"
    )
    ax1.set_ylabel("Δr ($R_M$)")
    ax1.set_xlabel("Time (s)")
    ax1.grid(True)
    ax1.legend()
    ax1.set_title(f"{case.replace('_',' ')} Longitude-Integrated\nDayside Magnetopause Standoff Distance", fontsize=16)
    ax1.set_ylim(1, 1.9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{case}_equatorial_standoff_timeseries.png"))
    plt.close()

    print("Done.")