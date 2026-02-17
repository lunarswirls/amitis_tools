#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyamitis.amitis_observer import *

# -------------------------------
# Configuration
# -------------------------------
case = "RPS_HNHV"
output_folder = f"/Users/danywaller/Projects/mercury/extreme/observer_density_velocity/{case}/"
os.makedirs(output_folder, exist_ok=True)

R_M = 2440.0  # Mercury radius [km]
tolerance = 1.0  # km tolerance for observer selection at surface
high_tolerance = 2440  # km tolerance for observer selection in high altitude shell

# Particle species parameters
species = np.array(['H+', 'H+', 'He++', 'He++'])
sim_ppc = np.array([24, 24, 11, 11])
sim_den = np.array([38.0e6, 76.0e6, 1.0e6, 2.0e6])  # [/m^3]
sim_vel = np.array([400.e3, 700.0e3, 400.e3, 700.0e3])  # [m/s]
species_mass = np.array([1.0, 1.0, 4.0, 4.0])  # [amu]
species_charge = np.array([1.0, 1.0, 2.0, 2.0])  # [e]
e_charge = 1.602176634e-19  # C

# obsDict indices
obsDict = {'time': 0,
           'pBx': 1, 'pBy': 2, 'pBz': 3,
           'pBdx': 4, 'pBdy': 5, 'pBdz': 6,
           'pEx': 7, 'pEy': 8, 'pEz': 9,
           'pJx': 10, 'pJy': 11, 'pJz': 12}

idx = 13
for i in range(1, 5):
    for field in ['rho', 'jix', 'jiy', 'jiz']:
        obsDict[f'p{field}{i:02}'] = idx
        idx += 1

# -------------------------------
# Load AMITIS observer data
# -------------------------------
print("Loading AMITIS observer data...")
amitis_path = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/concat_obs"
amitis = amitis_observer(amitis_path, f"Observer_{case}")
amitis.collect_all_data()

observers, probes, obsDict_loaded, numFields = amitis.load_collected_data()

print(f"Loaded {len(observers)} observers")
print(f"Probe data shape: {probes.shape}")
print(f"Number of fields per observer: {numFields}")

# -------------------------------
# Filter observers by radial distance
# -------------------------------
print("\nFiltering observers by radial distance...")

obs_pos_km = observers / 1000.0  # [km]
radial_distances = np.sqrt(np.sum(obs_pos_km ** 2, axis=1))

# Surface observers (R_M)
mask_surface = (radial_distances >= R_M - tolerance) & (radial_distances <= R_M + tolerance)
indices_surface = np.where(mask_surface)[0]

# 1 R_M above surface (2*R_M)
R_high = 2 * R_M
mask_high = (radial_distances >= R_high - high_tolerance) & (radial_distances <= R_high + high_tolerance)
indices_high = np.where(mask_high)[0]

print(f"\n--- Surface (R = {R_M} km) ---")
print(f"  Total observers: {len(indices_surface)}")

print(f"\n--- 1 R_M above (R = {R_high} km) ---")
print(f"  Total observers: {len(indices_high)}")

obs_surface = obs_pos_km[indices_surface]
obs_high = obs_pos_km[indices_high]

day_mask_surface = obs_surface[:, 0] > 0
night_mask_surface = obs_surface[:, 0] < 0

day_mask_high = obs_high[:, 0] > 0
night_mask_high = obs_high[:, 0] < 0

print(f"\nSurface: {np.sum(day_mask_surface)} day-side, {np.sum(night_mask_surface)} night-side")
print(f"High altitude: {np.sum(day_mask_high)} day-side, {np.sum(night_mask_high)} night-side")

# -------------------------------
# Extract time series
# -------------------------------
print("\nExtracting time series from observers...")

time_idx = obsDict['time']
timestamps = probes[0, :, time_idx]  # [s]
n_steps = len(timestamps)

print(f"Number of time steps: {n_steps}")
print(f"Total time span: {timestamps[-1] - timestamps[0]:.1f} s = {(timestamps[-1] - timestamps[0]) / 60:.2f} min")
print(f"Time step: {np.mean(np.diff(timestamps)):.3f} s")

# -------------------------------
# Extract particle species data
# -------------------------------
print("\nProcessing particle species data...")

density_data = {'surface': {}, 'high': {}}
velocity_data = {'surface': {}, 'high': {}}

for sp_idx in range(4):
    sp_num = sp_idx + 1
    print(f"\nProcessing species {sp_num}: {species[sp_idx]}")

    rho_idx = obsDict[f'prho{sp_num:02}']
    jix_idx = obsDict[f'pjix{sp_num:02}']
    jiy_idx = obsDict[f'pjiy{sp_num:02}']
    jiz_idx = obsDict[f'pjiz{sp_num:02}']

    eps = 1e-30

    # --- SURFACE DATA ---
    rho_surf = probes[indices_surface, :, rho_idx]
    jix_surf = probes[indices_surface, :, jix_idx]
    jiy_surf = probes[indices_surface, :, jiy_idx]
    jiz_surf = probes[indices_surface, :, jiz_idx]

    den_surf = rho_surf / (species_charge[sp_idx] * e_charge) * 1e-6

    vx_surf = (jix_surf / (rho_surf + eps)) * 1e-3
    vy_surf = (jiy_surf / (rho_surf + eps)) * 1e-3
    vz_surf = (jiz_surf / (rho_surf + eps)) * 1e-3
    vmag_surf = np.sqrt(vx_surf ** 2 + vy_surf ** 2 + vz_surf ** 2)

    density_data['surface'][sp_num] = den_surf
    velocity_data['surface'][sp_num] = {
        'vx': vx_surf, 'vy': vy_surf, 'vz': vz_surf, 'vmag': vmag_surf
    }

    # --- HIGH ALTITUDE DATA ---
    rho_high = probes[indices_high, :, rho_idx]
    jix_high = probes[indices_high, :, jix_idx]
    jiy_high = probes[indices_high, :, jiy_idx]
    jiz_high = probes[indices_high, :, jiz_idx]

    den_high = rho_high / (species_charge[sp_idx] * e_charge) * 1e-6

    vx_high = (jix_high / (rho_high + eps)) * 1e-3
    vy_high = (jiy_high / (rho_high + eps)) * 1e-3
    vz_high = (jiz_high / (rho_high + eps)) * 1e-3
    vmag_high = np.sqrt(vx_high ** 2 + vy_high ** 2 + vz_high ** 2)

    density_data['high'][sp_num] = den_high
    velocity_data['high'][sp_num] = {
        'vx': vx_high, 'vy': vy_high, 'vz': vz_high, 'vmag': vmag_high
    }

    print(f"  Surface: mean density = {np.mean(den_surf):.2e} #/cm^3, mean |v| = {np.mean(vmag_surf):.1f} km/s")
    print(f"  High: mean density = {np.mean(den_high):.2e} #/cm^3, mean |v| = {np.mean(vmag_high):.1f} km/s")

# -------------------------------
# Calculate TOTAL density (sum of all species)
# -------------------------------
print("\nCalculating total density (sum of all species)...")

den_total_surf = np.zeros_like(density_data['surface'][1])
den_total_high = np.zeros_like(density_data['high'][1])

for sp_num in range(1, 5):
    den_total_surf += density_data['surface'][sp_num]
    den_total_high += density_data['high'][sp_num]

print(f"Total Surface: mean density = {np.mean(den_total_surf):.2e} #/cm^3")
print(f"Total High: mean density = {np.mean(den_total_high):.2e} #/cm^3")

# Store total density
density_data['surface']['total'] = den_total_surf
density_data['high']['total'] = den_total_high

# -------------------------------
# Compute averaged time series for day/night
# -------------------------------
print("\nComputing averaged time series for plotting...")

species_density_day = {}
species_density_night = {}
species_velocity_day = {}
species_velocity_night = {}

for sp_num in range(1, 5):
    # Average density over all observers on day/night side
    species_density_day[sp_num] = np.mean(density_data['surface'][sp_num][day_mask_surface, :], axis=0)
    species_density_night[sp_num] = np.mean(density_data['surface'][sp_num][night_mask_surface, :], axis=0)

    # Average velocity components
    species_velocity_day[sp_num] = {
        'vx': np.mean(velocity_data['surface'][sp_num]['vx'][day_mask_surface, :], axis=0),
        'vy': np.mean(velocity_data['surface'][sp_num]['vy'][day_mask_surface, :], axis=0),
        'vz': np.mean(velocity_data['surface'][sp_num]['vz'][day_mask_surface, :], axis=0),
        'vmag': np.mean(velocity_data['surface'][sp_num]['vmag'][day_mask_surface, :], axis=0)
    }
    species_velocity_night[sp_num] = {
        'vx': np.mean(velocity_data['surface'][sp_num]['vx'][night_mask_surface, :], axis=0),
        'vy': np.mean(velocity_data['surface'][sp_num]['vy'][night_mask_surface, :], axis=0),
        'vz': np.mean(velocity_data['surface'][sp_num]['vz'][night_mask_surface, :], axis=0),
        'vmag': np.mean(velocity_data['surface'][sp_num]['vmag'][night_mask_surface, :], axis=0)
    }

# -------------------------------
# STATIC MATPLOTLIB PLOTS
# -------------------------------

# PLOT 1: Particle Density vs Time (Day vs Night)
print("\nCreating density plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for sp_idx in range(4):
    sp_num = sp_idx + 1
    row = sp_idx // 2
    col = sp_idx % 2
    ax = axes[row, col]

    ax.plot(timestamps, species_density_day[sp_num], 'b-',
            label='Day-side', linewidth=2, alpha=0.8)
    ax.plot(timestamps, species_density_night[sp_num], 'r--',
            label='Night-side', linewidth=2, alpha=0.8)

    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Density [#/cm³]', fontsize=12)
    ax.set_yscale('log')
    ax.set_title(f'Species {sp_num}: {species[sp_idx]}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)

fig.suptitle(f'{case.replace("_", " ")} Surface Particle Density vs Time: Day-side vs Night-side', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'{case}_particle_density_timeseries.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# PLOT 2: Velocity Magnitude vs Time (Day vs Night)
print("Creating velocity magnitude plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for sp_idx in range(4):
    sp_num = sp_idx + 1
    row = sp_idx // 2
    col = sp_idx % 2
    ax = axes[row, col]

    ax.plot(timestamps, species_velocity_day[sp_num]['vmag'], 'b-',
            label='Day-side', linewidth=2, alpha=0.8)
    ax.plot(timestamps, species_velocity_night[sp_num]['vmag'], 'r--',
            label='Night-side', linewidth=2, alpha=0.8)

    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('|v| [km/s]', fontsize=12)
    ax.set_title(f'Species {sp_num}: {species[sp_idx]}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)

fig.suptitle(f'{case.replace("_", " ")} Surface Velocity Magnitude vs Time: Day-side vs Night-side', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'{case}_velocity_magnitude_timeseries.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# PLOT 3: Velocity Components for each species (Day-side)
print("Creating velocity component plots (day-side)...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for sp_idx in range(4):
    sp_num = sp_idx + 1
    row = sp_idx // 2
    col = sp_idx % 2
    ax = axes[row, col]

    vel = species_velocity_day[sp_num]
    ax.plot(timestamps, vel['vx'], 'r-', label='vx', linewidth=1.5, alpha=0.8)
    ax.plot(timestamps, vel['vy'], 'g-', label='vy', linewidth=1.5, alpha=0.8)
    ax.plot(timestamps, vel['vz'], 'b-', label='vz', linewidth=1.5, alpha=0.8)
    ax.plot(timestamps, vel['vmag'], 'k--', label='|v|', linewidth=2, alpha=0.8)

    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Velocity [km/s]', fontsize=12)
    ax.set_title(f'Species {sp_num} (Day-side): {species[sp_idx]}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)

fig.suptitle(f'{case.replace("_", " ")} Surface Velocity Components vs Time: Day-side', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'{case}_velocity_components_dayside.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# PLOT 4: Velocity Components for each species (Night-side)
print("Creating velocity component plots (night-side)...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for sp_idx in range(4):
    sp_num = sp_idx + 1
    row = sp_idx // 2
    col = sp_idx % 2
    ax = axes[row, col]

    vel = species_velocity_night[sp_num]
    ax.plot(timestamps, vel['vx'], 'r-', label='vx', linewidth=1.5, alpha=0.8)
    ax.plot(timestamps, vel['vy'], 'g-', label='vy', linewidth=1.5, alpha=0.8)
    ax.plot(timestamps, vel['vz'], 'b-', label='vz', linewidth=1.5, alpha=0.8)
    ax.plot(timestamps, vel['vmag'], 'k--', label='|v|', linewidth=2, alpha=0.8)

    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Velocity [km/s]', fontsize=12)
    ax.set_title(f'Species {sp_num} (Night-side): {species[sp_idx]}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)

fig.suptitle(f'{case.replace("_", " ")} Surface Velocity Components vs Time: Night-side', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'{case}_velocity_components_nightside.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# PLOT 5: Comprehensive Surface Plot - Density, Velocity Mag, Velocity Components (Day & Night)
print("Creating comprehensive surface time series plot...")
fig, axes = plt.subplots(4, 4, figsize=(24, 14))

for sp_idx in range(4):
    sp_num = sp_idx + 1

    # Column 1: Density
    ax_den = axes[sp_idx, 0]
    den_surf_day = np.mean(density_data['surface'][sp_num][day_mask_surface, :], axis=0)
    den_surf_night = np.mean(density_data['surface'][sp_num][night_mask_surface, :], axis=0)

    ax_den.plot(timestamps, den_surf_day, 'b-', linewidth=2, alpha=0.8, label='Day')
    ax_den.plot(timestamps, den_surf_night, 'r-', linewidth=2, alpha=0.8, label='Night')
    ax_den.set_ylabel('Density [#/cm³]', fontsize=11)
    ax_den.set_yscale('log')
    ax_den.set_title(f'Species {sp_num}: {species[sp_idx]} Density', fontsize=12, fontweight='bold')
    ax_den.legend(fontsize=9, loc='best')
    ax_den.grid(True, alpha=0.3)
    if sp_idx == 3:
        ax_den.set_xlabel('Time [s]', fontsize=12)

    # Column 2: Velocity Magnitude
    ax_vel = axes[sp_idx, 1]
    vel_surf_day = np.mean(velocity_data['surface'][sp_num]['vmag'][day_mask_surface, :], axis=0)
    vel_surf_night = np.mean(velocity_data['surface'][sp_num]['vmag'][night_mask_surface, :], axis=0)

    ax_vel.plot(timestamps, vel_surf_day, 'b-', linewidth=2, alpha=0.8, label='Day')
    ax_vel.plot(timestamps, vel_surf_night, 'r-', linewidth=2, alpha=0.8, label='Night')
    ax_vel.set_ylabel('|v| [km/s]', fontsize=11)
    ax_vel.set_title(f'Species {sp_num}: {species[sp_idx]} Velocity Mag', fontsize=12, fontweight='bold')
    ax_vel.legend(fontsize=9, loc='best')
    ax_vel.grid(True, alpha=0.3)
    if sp_idx == 3:
        ax_vel.set_xlabel('Time [s]', fontsize=12)

    # Column 3: Velocity Components (Day-side)
    ax_comp_day = axes[sp_idx, 2]
    vel_day = species_velocity_day[sp_num]
    ax_comp_day.plot(timestamps, vel_day['vx'], 'r-', label='vx', linewidth=1.5, alpha=0.8)
    ax_comp_day.plot(timestamps, vel_day['vy'], 'g-', label='vy', linewidth=1.5, alpha=0.8)
    ax_comp_day.plot(timestamps, vel_day['vz'], 'b-', label='vz', linewidth=1.5, alpha=0.8)
    ax_comp_day.plot(timestamps, vel_day['vmag'], 'k--', label='|v|', linewidth=2, alpha=0.8)
    ax_comp_day.set_ylabel('Velocity [km/s]', fontsize=11)
    ax_comp_day.set_title(f'Species {sp_num}: {species[sp_idx]} Components (Day)', fontsize=12, fontweight='bold')
    ax_comp_day.legend(fontsize=8, loc='best', ncol=2)
    ax_comp_day.grid(True, alpha=0.3)
    if sp_idx == 3:
        ax_comp_day.set_xlabel('Time [s]', fontsize=12)

    # Column 4: Velocity Components (Night-side)
    ax_comp_night = axes[sp_idx, 3]
    vel_night = species_velocity_night[sp_num]
    ax_comp_night.plot(timestamps, vel_night['vx'], 'r-', label='vx', linewidth=1.5, alpha=0.8)
    ax_comp_night.plot(timestamps, vel_night['vy'], 'g-', label='vy', linewidth=1.5, alpha=0.8)
    ax_comp_night.plot(timestamps, vel_night['vz'], 'b-', label='vz', linewidth=1.5, alpha=0.8)
    ax_comp_night.plot(timestamps, vel_night['vmag'], 'k--', label='|v|', linewidth=2, alpha=0.8)
    ax_comp_night.set_ylabel('Velocity [km/s]', fontsize=11)
    ax_comp_night.set_title(f'Species {sp_num}: {species[sp_idx]} Components (Night)', fontsize=12, fontweight='bold')
    ax_comp_night.legend(fontsize=8, loc='best', ncol=2)
    ax_comp_night.grid(True, alpha=0.3)
    if sp_idx == 3:
        ax_comp_night.set_xlabel('Time [s]', fontsize=12)

fig.suptitle(f'{case.replace("_", " ")} Surface Plasma Parameters vs Time', fontsize=18, y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'{case}_surface_comprehensive_timeseries.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------
# CREATE INTERACTIVE PLOTLY FIGURE WITH TIME SLIDER
# -------------------------------
print("\nCreating interactive Plotly density visualizations...")

# Sample timesteps for slider
skip = max(1, n_steps // 5000)  # ~5000 frames max
time_indices = np.arange(0, n_steps, skip)


# Function to create interactive plot
def create_density_plot(den_surf, den_high, title, output_filename):
    """Create interactive density plot with time slider"""

    # Determine color scale
    vmin = max(1e-3, np.percentile(den_surf, 1))
    vmax = np.percentile(den_surf, 99)

    # Create sphere for day/night visualization
    plot_depth = R_M
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2 * np.pi, 100)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    xs = plot_depth * np.sin(theta_grid) * np.cos(phi_grid)
    ys = plot_depth * np.sin(theta_grid) * np.sin(phi_grid)
    zs = plot_depth * np.cos(theta_grid)

    mask_pos = xs >= 0
    mask_neg = xs <= 0

    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Surface', 'High Altitude'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        horizontal_spacing=0.08
    )

    # === SURFACE SUBPLOT (col=1) ===
    # Day hemisphere
    fig.add_trace(go.Surface(
        x=np.where(mask_pos, xs, np.nan),
        y=np.where(mask_pos, ys, np.nan),
        z=np.where(mask_pos, zs, np.nan),
        surfacecolor=np.ones_like(xs),
        colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
        showscale=False,
        lighting=dict(ambient=1, diffuse=0, specular=0),
        hoverinfo='skip',
        name='Day'
    ), row=1, col=1)

    # Night hemisphere
    fig.add_trace(go.Surface(
        x=np.where(mask_neg, xs, np.nan),
        y=np.where(mask_neg, ys, np.nan),
        z=np.where(mask_neg, zs, np.nan),
        surfacecolor=np.zeros_like(xs),
        colorscale=[[0, 'black'], [1, 'black']],
        showscale=False,
        lighting=dict(ambient=1, diffuse=0, specular=0),
        hoverinfo='skip',
        name='Night'
    ), row=1, col=1)

    # Observer scatter
    fig.add_trace(go.Scatter3d(
        x=obs_surface[:, 0],
        y=obs_surface[:, 1],
        z=obs_surface[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=den_surf[:, time_indices[0]],
            colorscale='Plasma',
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(
                title='Density<br>[#/cm³]',
                x=0.46,
                len=0.75,
                thickness=15
            ),
            showscale=True
        ),
        hovertemplate='X: %{x:.1f} km<br>Y: %{y:.1f} km<br>Z: %{z:.1f} km<br>Density: %{marker.color:.2e}<extra></extra>',
        name='Observers'
    ), row=1, col=1)

    # === HIGH ALTITUDE SUBPLOT (col=2) ===
    # Day hemisphere
    fig.add_trace(go.Surface(
        x=np.where(mask_pos, xs, np.nan),
        y=np.where(mask_pos, ys, np.nan),
        z=np.where(mask_pos, zs, np.nan),
        surfacecolor=np.ones_like(xs),
        colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
        showscale=False,
        lighting=dict(ambient=1, diffuse=0, specular=0),
        hoverinfo='skip',
        name='Day'
    ), row=1, col=2)

    # Night hemisphere
    fig.add_trace(go.Surface(
        x=np.where(mask_neg, xs, np.nan),
        y=np.where(mask_neg, ys, np.nan),
        z=np.where(mask_neg, zs, np.nan),
        surfacecolor=np.zeros_like(xs),
        colorscale=[[0, 'black'], [1, 'black']],
        showscale=False,
        lighting=dict(ambient=1, diffuse=0, specular=0),
        hoverinfo='skip',
        name='Night'
    ), row=1, col=2)

    # Observer scatter
    fig.add_trace(go.Scatter3d(
        x=obs_high[:, 0],
        y=obs_high[:, 1],
        z=obs_high[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=den_high[:, time_indices[0]],
            colorscale='Plasma',
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(
                title='Density<br>[#/cm³]',
                x=1.02,
                len=0.75,
                thickness=15
            ),
            showscale=True
        ),
        hovertemplate='X: %{x:.1f} km<br>Y: %{y:.1f} km<br>Z: %{z:.1f} km<br>Density: %{marker.color:.2e}<extra></extra>',
        name='Observers'
    ), row=1, col=2)

    # Create frames for animation
    frames = []
    for frame_idx, t_idx in enumerate(time_indices):
        t = timestamps[t_idx]

        frame = go.Frame(
            data=[
                go.Surface(),  # Day hemisphere surface (unchanged)
                go.Surface(),  # Night hemisphere surface (unchanged)
                go.Scatter3d(
                    marker=dict(color=den_surf[:, t_idx])
                ),  # Updated surface scatter
                go.Surface(),  # Day hemisphere high (unchanged)
                go.Surface(),  # Night hemisphere high (unchanged)
                go.Scatter3d(
                    marker=dict(color=den_high[:, t_idx])
                )  # Updated high altitude scatter
            ],
            name=str(frame_idx),
            traces=[0, 1, 2, 3, 4, 5]
        )
        frames.append(frame)

    fig.frames = frames

    # Update layout - NO PLAY/PAUSE BUTTONS, slider only
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        height=700,
        showlegend=False,
        scene=dict(
            xaxis=dict(title='X [km]', range=[-R_M * 1.3, R_M * 1.3]),
            yaxis=dict(title='Y [km]', range=[-R_M * 1.3, R_M * 1.3]),
            zaxis=dict(title='Z [km]', range=[-R_M * 1.3, R_M * 1.3]),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        scene2=dict(
            xaxis=dict(title='X [km]', range=[-R_high * 1.2, R_high * 1.2]),
            yaxis=dict(title='Y [km]', range=[-R_high * 1.2, R_high * 1.2]),
            zaxis=dict(title='Z [km]', range=[-R_high * 1.2, R_high * 1.2]),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        # Slider only - no updatemenus (play/pause buttons)
        sliders=[dict(
            active=0,
            yanchor='top',
            y=0,
            xanchor='left',
            x=0.05,
            len=0.9,
            ticklen=5,
            tickcolor='gray',
            pad=dict(t=50, b=10),
            steps=[
                dict(
                    method='animate',
                    args=[[str(i)], dict(
                        frame=dict(duration=0, redraw=True),
                        mode='immediate',
                        transition=dict(duration=0)
                    )],
                    label=f'{timestamps[t_idx]:.1f}'
                )
                for i, t_idx in enumerate(time_indices)
            ],
            currentvalue=dict(
                prefix='Time: ',
                suffix=' s',
                visible=True,
                xanchor='center',
                font=dict(size=14, color='black')
            )
        )]
    )

    # Save as HTML
    output_path = os.path.join(output_folder, output_filename)
    fig.write_html(output_path, include_plotlyjs='cdn', auto_play=False)
    print(f"    Saved: {output_path}")


# Create plots for individual species
for sp_idx in range(4):
    sp_num = sp_idx + 1
    print(f"  Creating interactive plot for species {sp_num}: {species[sp_idx]}")

    create_density_plot(
        density_data['surface'][sp_num],
        density_data['high'][sp_num],
        f'Species {sp_num}: {species[sp_idx]} | Density Evolution',
        f'{case}_density_interactive_species{sp_num:02}.html'
    )

# Create plot for TOTAL density (all species combined)
print(f"  Creating interactive plot for TOTAL density (all species)")
create_density_plot(
    density_data['surface']['total'],
    density_data['high']['total'],
    f'{case.replace("_", " ")} Total Density',
    f'{case}_density_interactive_total.html'
)

print(f"\n{'=' * 70}")
print("ANALYSIS COMPLETE!")
print(f"{'=' * 70}")
print(f"\nResults saved to: {output_folder}")
print("\nStatic PNG plots:")
print(f"  - {case}_particle_density_timeseries.png")
print(f"  - {case}_velocity_magnitude_timeseries.png")
print(f"  - {case}_velocity_components_dayside.png")
print(f"  - {case}_velocity_components_nightside.png")
print(
    f"  - {case}_surface_comprehensive_timeseries.png  <-- 4 columns: Density, |v|, Components (Day), Components (Night)")
print("\nInteractive HTML files:")
for sp_num in range(1, 5):
    print(f"  - {case}_density_interactive_species{sp_num:02}.html")
print(f"  - {case}_density_interactive_total.html")
print(f"\n{'=' * 70}")
