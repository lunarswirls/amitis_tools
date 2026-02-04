#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from src.field_topology.topology_utils import *

# ======================
# USER PARAMETERS
# ======================
if 0:
    case = "CPN_Base"
    input_folder = f"/Volumes/data_backup/mercury/extreme/{case}/plane_product/object/"
    if "larger" in case:
        fname = case.split("_")[0] + "_" + case.split("_")[1]
    else:
        fname = case
    ncfile = os.path.join(input_folder, f"Amitis_{fname}_115000_xz_comp.nc")

case = "CPN_Base_largerxdomain_smallergridsize"
input_folder = f"/Users/danywaller/Projects/mercury/extreme/{case}/out/"
if "larger" in case:
    fname = case.split("_")[0] + "_" + case.split("_")[1]
else:
    fname = case
ncfile = os.path.join(input_folder, f"Amitis_{fname}_115000.nc")

output_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology_twist/{case}/"
os.makedirs(output_folder, exist_ok=True)

plot_lines = False
plot_3d = False
plot_3d_category = True
RM = 2440.0  # Mercury radius [km]
dx = 200.0
trace_length = 20 * RM
surface_tol = dx

# Seed resolution
n_lat = 90
n_lon = n_lat * 2

max_steps = 100000
h_step = 50.0  # integration step size [km]

# Twist detection parameters
twist_threshold = 0.5  # Low twist threshold (π radians)
moderate_twist = 1.0  # Moderate twist (2π radians)
high_twist = 2.0  # Highly twisted structures (4π radians)
compute_twist = True
compute_open_twist = False  # NEW: Enable twist calculation for open field lines

start = datetime.now()
print(f"Started processing {ncfile} at {str(start)}")
step = int(ncfile.split("/")[-1].split("_")[3].split(".")[0])

# ======================
# LOAD DATA
# ======================
ds = xr.open_dataset(ncfile)
x = ds["Nx"].values
y = ds["Ny"].values
z = ds["Nz"].values

if 0:
    # Extract fields
    Bx = ds["Bx_tot"].isel(time=0).values
    By = ds["By_tot"].isel(time=0).values
    Bz = ds["Bz_tot"].isel(time=0).values

# Extract fields (drop time dimension)
Bx = (ds["Bx"].isel(time=0).values + ds["Bdx"].isel(time=0).values)
By = (ds["By"].isel(time=0).values + ds["Bdy"].isel(time=0).values)
Bz = (ds["Bz"].isel(time=0).values + ds["Bdz"].isel(time=0).values)

# Transpose: Nz, Ny, Nx --> Nx, Ny, Nz
Bx_plane = np.transpose(Bx, (2, 1, 0))
By_plane = np.transpose(By, (2, 1, 0))
Bz_plane = np.transpose(Bz, (2, 1, 0))

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
zmin, zmax = z.min(), z.max()


# ======================
# HELPER FUNCTIONS FOR TWIST CALCULATION
# ======================

def compute_curl_B(Bx, By, Bz, x, y, z):
    """Compute curl of B: ∇ × B"""
    curl_x = np.zeros_like(Bx)
    curl_y = np.zeros_like(By)
    curl_z = np.zeros_like(Bz)

    curl_x[1:-1, 1:-1, 1:-1] = (
            (Bz[1:-1, 2:, 1:-1] - Bz[1:-1, :-2, 1:-1]) / (y[2:] - y[:-2])[:, None] -
            (By[1:-1, 1:-1, 2:] - By[1:-1, 1:-1, :-2]) / (z[2:] - z[:-2])
    )

    curl_y[1:-1, 1:-1, 1:-1] = (
            (Bx[1:-1, 1:-1, 2:] - Bx[1:-1, 1:-1, :-2]) / (z[2:] - z[:-2]) -
            (Bz[2:, 1:-1, 1:-1] - Bz[:-2, 1:-1, 1:-1]) / (x[2:] - x[:-2])[:, None, None]
    )

    curl_z[1:-1, 1:-1, 1:-1] = (
            (By[2:, 1:-1, 1:-1] - By[:-2, 1:-1, 1:-1]) / (x[2:] - x[:-2])[:, None, None] -
            (Bx[1:-1, 2:, 1:-1] - Bx[1:-1, :-2, 1:-1]) / (y[2:] - y[:-2])[:, None]
    )

    return curl_x, curl_y, curl_z


def compute_alpha_field(Bx, By, Bz, curl_x, curl_y, curl_z):
    """Compute force-free parameter α = (∇ × B) · B / |B|²"""
    B_mag_sq = Bx ** 2 + By ** 2 + Bz ** 2
    B_mag_sq = np.where(B_mag_sq < 1e-10, 1e-10, B_mag_sq)
    alpha = (curl_x * Bx + curl_y * By + curl_z * Bz) / B_mag_sq
    return alpha


def interpolate_alpha_along_fieldline(trajectory, alpha_field, x, y, z):
    """Interpolate alpha values along a field line trajectory"""
    interpolator = RegularGridInterpolator((x, y, z), alpha_field,
                                           bounds_error=False, fill_value=0)
    alpha_values = interpolator(trajectory)
    return alpha_values


def compute_twist_number(trajectory, alpha_values, h_step):
    """Compute twist number Tw = (1/2π) ∫ α dl"""
    dl = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    alpha_mid = 0.5 * (alpha_values[:-1] + alpha_values[1:])
    twist_integral = np.sum(alpha_mid * dl)
    twist_number = twist_integral / (2 * np.pi)
    return twist_number


def classify_twist_level(total_twist, twist_threshold, moderate_twist, high_twist):
    """Classify twist into categories"""
    if total_twist >= high_twist:
        return 'high'
    elif total_twist >= moderate_twist:
        return 'moderate'
    elif total_twist >= twist_threshold:
        return 'low'
    else:
        return 'non-twisted'


# ======================
# COMPUTE CURL AND ALPHA FIELD
# ======================
if compute_twist or compute_open_twist:
    print("Computing curl of B field...")
    curl_x, curl_y, curl_z = compute_curl_B(Bx_plane, By_plane, Bz_plane, x, y, z)

    print("Computing force-free parameter α...")
    alpha_field = compute_alpha_field(Bx_plane, By_plane, Bz_plane,
                                      curl_x, curl_y, curl_z)

# ======================
# CREATE SEEDS ON SPHERE
# ======================
lats_surface = np.linspace(-90, 90, n_lat)
lons_surface = np.linspace(-180, 180, n_lon)
seeds = []
for lat in lats_surface:
    for lon in lons_surface:
        phi = np.radians(lat)
        theta = np.radians(lon)
        x_s = RM * np.cos(phi) * np.cos(theta)
        y_s = RM * np.cos(phi) * np.sin(theta)
        z_s = RM * np.sin(phi)
        seeds.append(np.array([x_s, y_s, z_s]))
seeds = np.array(seeds)
print(f"Total seeds: {len(seeds)}")

# ======================
# TRACE FIELD LINES AND COMPUTE TWIST
# ======================
footprints = []
footprints_class = []
lines_by_topo = {"closed": [], "open": [], "solar_wind": []}

# Store twist information by category for CLOSED field lines
twisted_flux_tubes_closed = {
    'low': [],
    'moderate': [],
    'critical': [],
    'high': [],
    'all': []
}

# NEW: Store twist information for OPEN field lines
twisted_flux_tubes_open = {
    'low': [],
    'moderate': [],
    'critical': [],
    'high': [],
    'all': []
}

non_twisted_closed = []
non_twisted_open = []
fieldline_twist_data = []

# For 3D plotting
trajectories_3d = {
    'closed': [],
    'open': [],
    'solar_wind': [],
    'twisted_closed': [],
    'twisted_open': []  # NEW
}

for i, seed in enumerate(seeds):
    # Trace field line
    traj_fwd, exit_fwd_y = trace_field_line_rk(seed, Bx_plane, By_plane, Bz_plane,
                                               x, y, z, RM, max_steps=max_steps, h=h_step)
    traj_bwd, exit_bwd_y = trace_field_line_rk(seed, Bx_plane, By_plane, Bz_plane,
                                               x, y, z, RM, max_steps=max_steps, h=-h_step)

    # Classify topology
    topo = classify(traj_fwd, traj_bwd, RM, exit_fwd_y, exit_bwd_y)

    # Compute twist
    twist_fwd = 0.0
    twist_bwd = 0.0
    total_twist = 0.0
    is_twisted = False
    twist_level = 'non-twisted'

    # NEW: Check if open field line connects to surface
    is_open_connected = False
    if topo == "open":
        r_fwd_end = np.linalg.norm(traj_fwd[-1])
        r_bwd_end = np.linalg.norm(traj_bwd[-1])
        is_open_connected = (r_fwd_end <= RM + surface_tol) or (r_bwd_end <= RM + surface_tol)

    # Compute twist for CLOSED field lines
    if compute_twist and topo == "closed":
        alpha_fwd = interpolate_alpha_along_fieldline(traj_fwd, alpha_field, x, y, z)
        twist_fwd = compute_twist_number(traj_fwd, alpha_fwd, h_step)

        alpha_bwd = interpolate_alpha_along_fieldline(traj_bwd, alpha_field, x, y, z)
        twist_bwd = compute_twist_number(traj_bwd, alpha_bwd, h_step)

        total_twist = abs(twist_fwd) + abs(twist_bwd)
        twist_level = classify_twist_level(total_twist, twist_threshold, moderate_twist, high_twist)

        fieldline_twist_data.append({
            'seed_x': seed[0],
            'seed_y': seed[1],
            'seed_z': seed[2],
            'twist_forward': twist_fwd,
            'twist_backward': twist_bwd,
            'total_twist': total_twist,
            'topology': topo,
            'twist_category': twist_level
        })

        if total_twist >= twist_threshold:
            is_twisted = True
            tube_data = {
                'seed': seed,
                'total_twist': total_twist,
                'trajectory_fwd': traj_fwd,
                'trajectory_bwd': traj_bwd,
                'twist_level': twist_level
            }
            twisted_flux_tubes_closed[twist_level].append(tube_data)
            twisted_flux_tubes_closed['all'].append(tube_data)

            trajectories_3d['twisted_closed'].append({
                'fwd': traj_fwd,
                'bwd': traj_bwd,
                'twist': total_twist,
                'level': twist_level
            })
        else:
            non_twisted_closed.append({
                'trajectory_fwd': traj_fwd,
                'trajectory_bwd': traj_bwd
            })

    # NEW: Compute twist for OPEN field lines connected to surface
    elif compute_open_twist and topo == "open" and is_open_connected:
        # Determine which end connects to surface
        r_fwd_end = np.linalg.norm(traj_fwd[-1])
        r_bwd_end = np.linalg.norm(traj_bwd[-1])

        # Only compute twist along the trajectory that connects to surface
        if r_fwd_end <= RM + surface_tol:
            alpha_fwd = interpolate_alpha_along_fieldline(traj_fwd, alpha_field, x, y, z)
            twist_fwd = compute_twist_number(traj_fwd, alpha_fwd, h_step)
            total_twist = abs(twist_fwd)
            connected_traj = traj_fwd
            disconnected_traj = traj_bwd
        else:  # r_bwd_end connects to surface
            alpha_bwd = interpolate_alpha_along_fieldline(traj_bwd, alpha_field, x, y, z)
            twist_bwd = compute_twist_number(traj_bwd, alpha_bwd, h_step)
            total_twist = abs(twist_bwd)
            connected_traj = traj_bwd
            disconnected_traj = traj_fwd

        twist_level = classify_twist_level(total_twist, twist_threshold, moderate_twist, high_twist)

        fieldline_twist_data.append({
            'seed_x': seed[0],
            'seed_y': seed[1],
            'seed_z': seed[2],
            'twist_forward': twist_fwd,
            'twist_backward': twist_bwd,
            'total_twist': total_twist,
            'topology': topo,
            'twist_category': twist_level
        })

        if total_twist >= twist_threshold:
            is_twisted = True
            tube_data = {
                'seed': seed,
                'total_twist': total_twist,
                'trajectory_fwd': traj_fwd,
                'trajectory_bwd': traj_bwd,
                'twist_level': twist_level,
                'connected_traj': connected_traj,
                'disconnected_traj': disconnected_traj
            }
            twisted_flux_tubes_open[twist_level].append(tube_data)
            twisted_flux_tubes_open['all'].append(tube_data)

            trajectories_3d['twisted_open'].append({
                'fwd': traj_fwd,
                'bwd': traj_bwd,
                'twist': total_twist,
                'level': twist_level
            })
        else:
            non_twisted_open.append({
                'trajectory_fwd': traj_fwd,
                'trajectory_bwd': traj_bwd
            })

    # Store for 2D plotting (X-Z plane projection)
    if topo not in ["TBD"]:
        if topo != "closed" or (topo == "closed" and not is_twisted):
            if topo != "open" or (topo == "open" and not is_twisted):
                lines_by_topo[topo].append(traj_fwd[:, [0, 2]])
                lines_by_topo[topo].append(traj_bwd[:, [0, 2]])

        # Store full 3D trajectories for non-twisted
        if not is_twisted:
            trajectories_3d[topo].append({
                'fwd': traj_fwd,
                'bwd': traj_bwd
            })

    # Compute footprints
    for traj in [traj_fwd, traj_bwd]:
        r_end = traj[-1]
        if np.linalg.norm(r_end) <= RM + surface_tol:
            lat, lon = cartesian_to_latlon(r_end)
            footprints.append((lat, lon))
            footprints_class.append(topo)

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(seeds)} field lines...")

# ======================
# PLOT 1: 2D FIELD LINE TOPOLOGY (X-Z Plane)
# ======================
if plot_lines:
    colors = {"closed": "blue", "open": "red", "solar_wind": "gray",
              "twisted_closed": "goldenrod", "twisted_open": "orange"}
    fig, ax = plt.subplots(figsize=(8, 7))

    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(RM * np.cos(theta), RM * np.sin(theta), "k", lw=2)

    for topo in ["closed", "open", "solar_wind"]:
        if lines_by_topo[topo]:
            lc = LineCollection(lines_by_topo[topo], colors=colors[topo],
                                linewidths=0.8, alpha=0.5)
            ax.add_collection(lc)

    # Plot twisted closed field lines
    if compute_twist and len(twisted_flux_tubes_closed['all']) > 0:
        for tube in twisted_flux_tubes_closed['all']:
            traj_fwd_2d = tube['trajectory_fwd'][:, [0, 2]]
            traj_bwd_2d = tube['trajectory_bwd'][:, [0, 2]]
            ax.plot(traj_fwd_2d[:, 0], traj_fwd_2d[:, 1],
                    color='goldenrod', linewidth=1.5, alpha=0.8)
            ax.plot(traj_bwd_2d[:, 0], traj_bwd_2d[:, 1],
                    color='goldenrod', linewidth=1.5, alpha=0.8)

    # NEW: Plot twisted open field lines
    if compute_open_twist and len(twisted_flux_tubes_open['all']) > 0:
        for tube in twisted_flux_tubes_open['all']:
            traj_fwd_2d = tube['trajectory_fwd'][:, [0, 2]]
            traj_bwd_2d = tube['trajectory_bwd'][:, [0, 2]]
            ax.plot(traj_fwd_2d[:, 0], traj_fwd_2d[:, 1],
                    color='orange', linewidth=1.5, alpha=0.8)
            ax.plot(traj_bwd_2d[:, 0], traj_bwd_2d[:, 1],
                    color='orange', linewidth=1.5, alpha=0.8)

    legend_elements = [
        mlines.Line2D([], [], color='blue', label='Closed (non-twisted)', linewidth=1),
        mlines.Line2D([], [], color='red', label='Open (non-twisted)', linewidth=1),
        mlines.Line2D([], [], color='gray', label='Solar Wind', linewidth=1)
    ]

    if compute_twist and len(twisted_flux_tubes_closed['all']) > 0:
        legend_elements.insert(0, mlines.Line2D([], [], color='goldenrod',
                                                label=f'Twisted Closed (Tw > {twist_threshold})',
                                                linewidth=1.5))
    if compute_open_twist and len(twisted_flux_tubes_open['all']) > 0:
        legend_elements.insert(0, mlines.Line2D([], [], color='orange',
                                                label=f'Twisted Open (Tw > {twist_threshold})',
                                                linewidth=1.5))

    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Z [km]")
    ax.set_aspect("equal")
    ax.set_title(f"{case.replace('_', ' ')} - Magnetic Field Line Topology (X-Z Plane)")
    ax.set_xlim(-2 * RM, 2 * RM)
    ax.set_ylim(-2 * RM, 2 * RM)
    plt.tight_layout()

    output_topo = os.path.join(output_folder, "2D_topology/")
    os.makedirs(output_topo, exist_ok=True)

    plot_name = f"{case}_field_topology_{step}.png"
    plt.savefig(os.path.join(output_topo, plot_name), dpi=150, bbox_inches="tight")
    print(f"Saved topology plot: {os.path.join(output_topo, plot_name)}")
    plt.close()

# ======================
# PLOT 2: 3D INTERACTIVE TWIST VISUALIZATION
# ======================
if plot_3d:
    print("Creating 3D twist visualization...")

    fig_3d = go.Figure()

    # add planet sphere
    theta_sphere = np.linspace(0, np.pi, 100)
    phi_sphere = np.linspace(0, 2 * np.pi, 200)
    theta_sphere, phi_sphere = np.meshgrid(theta_sphere, phi_sphere)

    xs = RM * np.sin(theta_sphere) * np.cos(phi_sphere)
    ys = RM * np.sin(theta_sphere) * np.sin(phi_sphere)
    zs = RM * np.cos(theta_sphere)

    eps = 0
    mask_pos = xs >= -eps
    mask_neg = xs <= eps

    fig_3d.add_trace(go.Surface(
        x=np.where(mask_pos, xs, np.nan),
        y=np.where(mask_pos, ys, np.nan),
        z=np.where(mask_pos, zs, np.nan),
        surfacecolor=np.ones_like(xs),
        colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
        cmin=0, cmax=1, showscale=False,
        lighting=dict(ambient=1, diffuse=0, specular=0),
        hoverinfo='skip'
    ))

    fig_3d.add_trace(go.Surface(
        x=np.where(mask_neg, xs, np.nan),
        y=np.where(mask_neg, ys, np.nan),
        z=np.where(mask_neg, zs, np.nan),
        surfacecolor=np.zeros_like(xs),
        colorscale=[[0, 'black'], [1, 'black']],
        cmin=0, cmax=1, showscale=False,
        lighting=dict(ambient=1, diffuse=0, specular=0),
        hoverinfo='skip'
    ))

    # Combine closed and open twisted field lines for colorbar scaling
    all_twisted = trajectories_3d['twisted_closed'] + trajectories_3d['twisted_open']

    if len(all_twisted) > 0:
        twists = [t['twist'] for t in all_twisted]
        twist_min = twist_threshold
        twist_max = min(max(twists), 2.5)
        norm = mcolors.Normalize(vmin=twist_min, vmax=twist_max)
        cmap = plt.colormaps['plasma']

        # Convert matplotlib colormap to Plotly format
        n_colors = 256
        plotly_colorscale = []
        for i in range(n_colors):
            val = twist_min + (twist_max - twist_min) * i / (n_colors - 1)
            rgba = cmap(norm(val))
            plotly_colorscale.append([i / (n_colors - 1),
                                      f'rgb({int(rgba[0] * 255)},{int(rgba[1] * 255)},{int(rgba[2] * 255)})'])

        # Add colorbar
        fig_3d.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(
                size=0.1,
                color=[twist_min],
                colorscale=plotly_colorscale,
                cmin=twist_min,
                cmax=twist_max,
                colorbar=dict(
                    title="Twist Number<br>(turns)",
                    thickness=20,
                    len=0.7,
                    x=1.02,
                    tickmode='linear',
                    tick0=twist_min,
                    dtick=(twist_max - twist_min) / 5,
                    tickformat='.2f'
                ),
                showscale=True
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Plot twisted CLOSED field lines
        for traj_data in trajectories_3d['twisted_closed']:
            traj_fwd = traj_data['fwd']
            traj_bwd = traj_data['bwd']
            twist_val = traj_data['twist']
            twist_level = traj_data['level']

            rgba = cmap(norm(twist_val))
            color_str = f'rgba({int(rgba[0] * 255)},{int(rgba[1] * 255)},{int(rgba[2] * 255)},{rgba[3]})'

            if twist_level == 'high':
                width = 5
            elif twist_level == 'moderate':
                width = 4
            else:
                width = 3

            fig_3d.add_trace(go.Scatter3d(
                x=traj_fwd[:, 0], y=traj_fwd[:, 1], z=traj_fwd[:, 2],
                mode='lines',
                line=dict(color=color_str, width=width),
                opacity=0.9,
                showlegend=False,
                hovertemplate=f'Type: Closed<br>Twist: {twist_val:.2f} turns<br>Level: {twist_level}<extra></extra>'
            ))

            fig_3d.add_trace(go.Scatter3d(
                x=traj_bwd[:, 0], y=traj_bwd[:, 1], z=traj_bwd[:, 2],
                mode='lines',
                line=dict(color=color_str, width=width),
                opacity=0.9,
                showlegend=False,
                hovertemplate=f'Type: Closed<br>Twist: {twist_val:.2f} turns<br>Level: {twist_level}<extra></extra>'
            ))

        # NEW: Plot twisted OPEN field lines
        for traj_data in trajectories_3d['twisted_open']:
            traj_fwd = traj_data['fwd']
            traj_bwd = traj_data['bwd']
            twist_val = traj_data['twist']
            twist_level = traj_data['level']

            rgba = cmap(norm(twist_val))
            color_str = f'rgba({int(rgba[0] * 255)},{int(rgba[1] * 255)},{int(rgba[2] * 255)},{rgba[3]})'

            if twist_level == 'high':
                width = 5
            elif twist_level == 'moderate':
                width = 4
            else:
                width = 3

            fig_3d.add_trace(go.Scatter3d(
                x=traj_fwd[:, 0], y=traj_fwd[:, 1], z=traj_fwd[:, 2],
                mode='lines',
                line=dict(color=color_str, width=width, dash='dot'),  # Dashed for open
                opacity=0.9,
                showlegend=False,
                hovertemplate=f'Type: Open<br>Twist: {twist_val:.2f} turns<br>Level: {twist_level}<extra></extra>'
            ))

            fig_3d.add_trace(go.Scatter3d(
                x=traj_bwd[:, 0], y=traj_bwd[:, 1], z=traj_bwd[:, 2],
                mode='lines',
                line=dict(color=color_str, width=width, dash='dot'),
                opacity=0.9,
                showlegend=False,
                hovertemplate=f'Type: Open<br>Twist: {twist_val:.2f} turns<br>Level: {twist_level}<extra></extra>'
            ))

        print(f"\nTwist statistics for 3D plot:")
        print(f"  Twist range: {twist_min:.2f} - {twist_max:.2f} turns")
        print(f"  Twisted closed field lines: {len(trajectories_3d['twisted_closed'])}")
        print(f"  Twisted open field lines: {len(trajectories_3d['twisted_open'])}")

    fig_3d.update_layout(
        title=dict(
            text=f"{case.replace('_', ' ')} - 3D Twisted Magnetic Flux Tubes",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title='X [km]', range=[-10 * RM, 4.5 * RM]),
            yaxis=dict(title='Y [km]', range=[-4.5 * RM, 4.5 * RM]),
            zaxis=dict(title='Z [km]', range=[-4.5 * RM, 4.5 * RM]),
            aspectmode='manual',
            aspectratio=dict(x=2.9, y=1.8, z=1.8),  # Matches axis ranges
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=1100,
        height=900,
        showlegend=False,
        margin=dict(r=150)
    )

    output_3d = os.path.join(output_folder, "3D_topology/")
    os.makedirs(output_3d, exist_ok=True)

    html_file = os.path.join(output_3d, f"{case}_3D_twist_{step}.html")
    fig_3d.write_html(html_file)
    print(f"Saved 3D twist plot: {html_file}")

if plot_3d_category:
    print("Creating 3D twist categorized visualization...")

    fig_3d = go.Figure()

    # add planet sphere
    theta_sphere = np.linspace(0, np.pi, 100)
    phi_sphere = np.linspace(0, 2 * np.pi, 200)
    theta_sphere, phi_sphere = np.meshgrid(theta_sphere, phi_sphere)

    xs = RM * np.sin(theta_sphere) * np.cos(phi_sphere)
    ys = RM * np.sin(theta_sphere) * np.sin(phi_sphere)
    zs = RM * np.cos(theta_sphere)

    eps = 0
    mask_pos = xs >= -eps
    mask_neg = xs <= eps

    fig_3d.add_trace(go.Surface(
        x=np.where(mask_pos, xs, np.nan),
        y=np.where(mask_pos, ys, np.nan),
        z=np.where(mask_pos, zs, np.nan),
        surfacecolor=np.ones_like(xs),
        colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
        cmin=0, cmax=1, showscale=False,
        lighting=dict(ambient=1, diffuse=0, specular=0),
        hoverinfo='skip'
    ))

    fig_3d.add_trace(go.Surface(
        x=np.where(mask_neg, xs, np.nan),
        y=np.where(mask_neg, ys, np.nan),
        z=np.where(mask_neg, zs, np.nan),
        surfacecolor=np.zeros_like(xs),
        colorscale=[[0, 'black'], [1, 'black']],
        cmin=0, cmax=1, showscale=False,
        lighting=dict(ambient=1, diffuse=0, specular=0),
        hoverinfo='skip'
    ))

    # Define colors for each twist category
    twist_colors = {
        'low': 'cornflowerblue',
        'moderate': 'darkorange',
        'high': 'firebrick'
    }

    twist_widths = {
        'low': 2,
        'moderate': 4,
        'high': 6
    }

    # Define legend labels with threshold information and physical description
    twist_labels = {
        'low': f'Low ({twist_threshold:.1f}-{moderate_twist:.1f} turns)',
        'moderate': f'Moderate ({moderate_twist:.1f}-{high_twist:.2f} turns)',
        'high': f'High (>{high_twist:.1f} turns)'
    }

    # Combine closed and open twisted field lines
    all_twisted = trajectories_3d['twisted_closed'] + trajectories_3d['twisted_open']

    if len(all_twisted) > 0:
        # Enforce ordering: low > moderate > high
        twist_order = ['low', 'moderate', 'high']

        # Track which categories have been added to legend
        categories_added_closed = set()
        categories_added_open = set()

        # Sort closed field lines by twist level to enforce order
        twisted_closed_sorted = sorted(trajectories_3d['twisted_closed'],
                                       key=lambda x: twist_order.index(x['level']))

        # Sort open field lines by twist level to enforce order
        twisted_open_sorted = sorted(trajectories_3d['twisted_open'],
                                     key=lambda x: twist_order.index(x['level']))

        # Plot twisted CLOSED field lines grouped by category (in order)
        for traj_data in twisted_closed_sorted:
            traj_fwd = traj_data['fwd']
            traj_bwd = traj_data['bwd']
            twist_val = traj_data['twist']
            twist_level = traj_data['level']

            color = twist_colors[twist_level]
            width = twist_widths[twist_level]

            # Show legend only for first occurrence of each category
            show_legend = twist_level not in categories_added_closed
            if show_legend:
                categories_added_closed.add(twist_level)
                legend_name = f'Closed - {twist_labels[twist_level]}'
                legend_group = f'closed_{twist_level}'
                # Assign legend rank to enforce ordering
                legend_rank = twist_order.index(twist_level) * 2  # Even numbers for closed
            else:
                legend_name = None
                legend_group = f'closed_{twist_level}'
                legend_rank = None

            fig_3d.add_trace(go.Scatter3d(
                x=traj_fwd[:, 0], y=traj_fwd[:, 1], z=traj_fwd[:, 2],
                mode='lines',
                line=dict(color=color, width=width),
                opacity=0.9,
                showlegend=show_legend,
                name=legend_name,
                legendgroup=legend_group,
                legendrank=legend_rank,
                hovertemplate=f'Type: Closed<br>Twist: {twist_val:.2f} turns<br>Level: {twist_level}<extra></extra>'
            ))

            fig_3d.add_trace(go.Scatter3d(
                x=traj_bwd[:, 0], y=traj_bwd[:, 1], z=traj_bwd[:, 2],
                mode='lines',
                line=dict(color=color, width=width),
                opacity=0.9,
                showlegend=False,
                legendgroup=legend_group,
                hovertemplate=f'Type: Closed<br>Twist: {twist_val:.2f} turns<br>Level: {twist_level}<extra></extra>'
            ))

        # Plot twisted OPEN field lines grouped by category (in order)
        for traj_data in twisted_open_sorted:
            traj_fwd = traj_data['fwd']
            traj_bwd = traj_data['bwd']
            twist_val = traj_data['twist']
            twist_level = traj_data['level']

            color = twist_colors[twist_level]
            width = twist_widths[twist_level]

            # Show legend only for first occurrence of each category
            show_legend = twist_level not in categories_added_open
            if show_legend:
                categories_added_open.add(twist_level)
                legend_name = f'Open - {twist_labels[twist_level]}'
                legend_group = f'open_{twist_level}'
                # Assign legend rank to enforce ordering (odd numbers for open, after closed)
                legend_rank = twist_order.index(twist_level) * 2 + 1
            else:
                legend_name = None
                legend_group = f'open_{twist_level}'
                legend_rank = None

            fig_3d.add_trace(go.Scatter3d(
                x=traj_fwd[:, 0], y=traj_fwd[:, 1], z=traj_fwd[:, 2],
                mode='lines',
                line=dict(color=color, width=width, dash='dot'),
                opacity=0.9,
                showlegend=show_legend,
                name=legend_name,
                legendgroup=legend_group,
                legendrank=legend_rank,
                hovertemplate=f'Type: Open<br>Twist: {twist_val:.2f} turns<br>Level: {twist_level}<extra></extra>'
            ))

            fig_3d.add_trace(go.Scatter3d(
                x=traj_bwd[:, 0], y=traj_bwd[:, 1], z=traj_bwd[:, 2],
                mode='lines',
                line=dict(color=color, width=width, dash='dot'),
                opacity=0.9,
                showlegend=False,
                legendgroup=legend_group,
                hovertemplate=f'Type: Open<br>Twist: {twist_val:.2f} turns<br>Level: {twist_level}<extra></extra>'
            ))

        # Print statistics
        print(f"\nTwist statistics for 3D plot:")
        print(f"  Twisted closed field lines: {len(trajectories_3d['twisted_closed'])}")
        for level in ['low', 'moderate', 'high']:
            count = len([t for t in trajectories_3d['twisted_closed'] if t['level'] == level])
            if count > 0:
                print(f"    {level.capitalize()}: {count}")

        print(f"  Twisted open field lines: {len(trajectories_3d['twisted_open'])}")
        for level in ['low', 'moderate', 'high']:
            count = len([t for t in trajectories_3d['twisted_open'] if t['level'] == level])
            if count > 0:
                print(f"    {level.capitalize()}: {count}")

    fig_3d.update_layout(
        title=dict(
            text=f"{case.replace('_', ' ')} - 3D Twisted Magnetic Flux Tubes by Category",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title='X [km]', range=[-10 * RM, 4.5 * RM]),
            yaxis=dict(title='Y [km]', range=[-4.5 * RM, 4.5 * RM]),
            zaxis=dict(title='Z [km]', range=[-4.5 * RM, 4.5 * RM]),
            aspectmode='manual',
            aspectratio=dict(x=2.9, y=1.8, z=1.8),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=1100,
        height=900,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1,
            traceorder='normal'  # Respect legendrank ordering
        ),
        margin=dict(r=200)
    )

    output_3d = os.path.join(output_folder, "3D_topology/")
    os.makedirs(output_3d, exist_ok=True)

    html_file = os.path.join(output_3d, f"{case}_3D_twist_categories_{step}.html")
    fig_3d.write_html(html_file)
    print(f"Saved 3D twist category plot: {html_file}")

# ======================
# SAVE TWIST DATA
# ======================
if (compute_twist or compute_open_twist) and len(fieldline_twist_data) > 0:
    df_twist = pd.DataFrame(fieldline_twist_data)

    twist_output = os.path.join(output_folder, "twist_analysis/")
    os.makedirs(twist_output, exist_ok=True)

    twist_file = os.path.join(twist_output, f"{case}_fieldline_twist_{step}.csv")
    df_twist.to_csv(twist_file, index=False)
    print(f"\nSaved twist data: {twist_file}")

    # Print statistics
    print("\n" + "=" * 50)
    print("TWIST STATISTICS")
    print("=" * 50)

    if compute_twist:
        print(f"\nCLOSED FIELD LINES:")
        print(f"  Total analyzed: {len([d for d in fieldline_twist_data if d['topology'] == 'closed'])}")
        print(f"  Low twist: {len(twisted_flux_tubes_closed['low'])}")
        print(f"  Moderate twist: {len(twisted_flux_tubes_closed['moderate'])}")
        print(f"  High twist: {len(twisted_flux_tubes_closed['high'])}")
        print(f"  Total twisted: {len(twisted_flux_tubes_closed['all'])}")

        if len(twisted_flux_tubes_closed['all']) > 0:
            twists = [t['total_twist'] for t in twisted_flux_tubes_closed['all']]
            print(f"  Range: {np.min(twists):.3f} - {np.max(twists):.3f} turns")
            print(f"  Mean: {np.mean(twists):.3f} turns")

    if compute_open_twist:
        print(f"\nOPEN FIELD LINES (connected to surface):")
        print(f"  Total analyzed: {len([d for d in fieldline_twist_data if d['topology'] == 'open'])}")
        print(f"  Low twist: {len(twisted_flux_tubes_open['low'])}")
        print(f"  Moderate twist: {len(twisted_flux_tubes_open['moderate'])}")
        print(f"  High twist: {len(twisted_flux_tubes_open['high'])}")
        print(f"  Total twisted: {len(twisted_flux_tubes_open['all'])}")

        if len(twisted_flux_tubes_open['all']) > 0:
            twists = [t['total_twist'] for t in twisted_flux_tubes_open['all']]
            print(f"  Range: {np.min(twists):.3f} - {np.max(twists):.3f} turns")
            print(f"  Mean: {np.mean(twists):.3f} turns")

print(f"\nCompleted in {datetime.now() - start}")