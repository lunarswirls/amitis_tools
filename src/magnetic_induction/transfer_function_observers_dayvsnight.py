#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from pyamitis.amitis_observer import *

# -------------------------------
# Configuration
# -------------------------------
case = "CPN_HNHV"
output_folder = f"/Users/danywaller/Projects/mercury/extreme/magnetic_induction/{case}_dayside_nightside/"
os.makedirs(output_folder, exist_ok=True)

R_M = 2440.0  # Mercury radius [km]
R_core = 2080.0  # Mercury core radius [km]
tolerance = 75.0  # km tolerance for observer selection

# Mercury conductivity structure
eta_mantle = 1.0e6  # Ohm-m (resistivity of mantle/shell)
eta_core = 1.0e-5  # Ohm-m (resistivity of core)
sigma_mantle = 1.0 / eta_mantle  # S/m
sigma_core = 1.0 / eta_core  # S/m
mu_0 = 4 * np.pi * 1e-7  # H/m

# obsDict indices
obsDict = {'time': 0,
           'pBx': 1,
           'pBy': 2,
           'pBz': 3,
           'pBdx': 4,
           'pBdy': 5,
           'pBdz': 6,
           'pEx': 7,
           'pEy': 8,
           'pEz': 9,
           'pJx': 10,
           'pJy': 11,
           'pJz': 12}

# -------------------------------
# Load AMITIS observer data
# -------------------------------
print("Loading AMITIS observer data...")
amitis_path = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/concat_obs"
amitis = amitis_observer(amitis_path, f"Observer_{case}")

observers, probes, obsDict_loaded, numFields = amitis.load_collected_data()

print(f"Loaded {len(observers)} observers")
print(f"Probe data shape: {probes.shape}")

# -------------------------------
# Filter observers by radial distance AND hemisphere
# -------------------------------
print("\nFiltering observers by radial distance and hemisphere...")

# Calculate radial distance for each observer (in km)
radial_distances = np.sqrt(observers[:, 0] ** 2 + observers[:, 1] ** 2 + observers[:, 2] ** 2) / 1000.0

# Get X coordinates (in km)
X_coords = observers[:, 0] / 1000.0

# --- SURFACE OBSERVERS (R_M) ---
mask_R_M = (radial_distances >= R_M - tolerance) & (radial_distances <= R_M + tolerance)
observer_indices_R_M = np.where(mask_R_M)[0]

# Separate day-side and night-side at surface
mask_R_M_day = mask_R_M & (X_coords > 0)
mask_R_M_night = mask_R_M & (X_coords < 0)

observer_indices_R_M_day = np.where(mask_R_M_day)[0]
observer_indices_R_M_night = np.where(mask_R_M_night)[0]

# --- CMB OBSERVERS (R_core) ---
mask_R_core = (radial_distances >= R_core - tolerance) & (radial_distances <= R_core + tolerance)
observer_indices_R_core = np.where(mask_R_core)[0]

# Separate day-side and night-side at CMB
mask_R_core_day = mask_R_core & (X_coords > 0)
mask_R_core_night = mask_R_core & (X_coords < 0)

observer_indices_R_core_day = np.where(mask_R_core_day)[0]
observer_indices_R_core_night = np.where(mask_R_core_night)[0]

print(f"\n--- Surface (R_M = {R_M} km) ---")
print(f"  Day-side (X>0): {len(observer_indices_R_M_day)} observers")
print(f"  Night-side (X<0): {len(observer_indices_R_M_night)} observers")
print(f"  Total: {len(observer_indices_R_M)} observers")

print(f"\n--- CMB (R_core = {R_core} km) ---")
print(f"  Day-side (X>0): {len(observer_indices_R_core_day)} observers")
print(f"  Night-side (X<0): {len(observer_indices_R_core_night)} observers")
print(f"  Total: {len(observer_indices_R_core)} observers")

# -------------------------------
# Extract time series for day-side and night-side
# -------------------------------
print("\nExtracting time series from observers...")

# Get time array from first observer
time_idx = obsDict['time']
timestamps = probes[0, :, time_idx]  # [s]
n_steps = len(timestamps)

print(f"Number of time steps: {n_steps}")
print(f"Total time span: {timestamps[-1] - timestamps[0]:.1f} s = {(timestamps[-1] - timestamps[0]) / 60:.2f} min")
print(f"Time step: {np.mean(np.diff(timestamps)):.3f} s")

# Extract magnetic field components
Bx_idx = obsDict['pBx']
By_idx = obsDict['pBy']
Bz_idx = obsDict['pBz']

# --- DAY-SIDE TIME SERIES ---
# Surface day-side
surface_day_Bx = np.mean(probes[observer_indices_R_M_day, :, Bx_idx], axis=0) * 1.0e9  # [nT]
surface_day_By = np.mean(probes[observer_indices_R_M_day, :, By_idx], axis=0) * 1.0e9  # [nT]
surface_day_Bz = np.mean(probes[observer_indices_R_M_day, :, Bz_idx], axis=0) * 1.0e9  # [nT]
surface_day_B = np.column_stack([surface_day_Bx, surface_day_By, surface_day_Bz])

# CMB day-side
CMB_day_Bx = np.mean(probes[observer_indices_R_core_day, :, Bx_idx], axis=0) * 1.0e9  # [nT]
CMB_day_By = np.mean(probes[observer_indices_R_core_day, :, By_idx], axis=0) * 1.0e9  # [nT]
CMB_day_Bz = np.mean(probes[observer_indices_R_core_day, :, Bz_idx], axis=0) * 1.0e9  # [nT]
CMB_day_B = np.column_stack([CMB_day_Bx, CMB_day_By, CMB_day_Bz])

# --- NIGHT-SIDE TIME SERIES ---
# Surface night-side
surface_night_Bx = np.mean(probes[observer_indices_R_M_night, :, Bx_idx], axis=0) * 1.0e9  # [nT]
surface_night_By = np.mean(probes[observer_indices_R_M_night, :, By_idx], axis=0) * 1.0e9  # [nT]
surface_night_Bz = np.mean(probes[observer_indices_R_M_night, :, Bz_idx], axis=0) * 1.0e9  # [nT]
surface_night_B = np.column_stack([surface_night_Bx, surface_night_By, surface_night_Bz])

# CMB night-side
CMB_night_Bx = np.mean(probes[observer_indices_R_core_night, :, Bx_idx], axis=0) * 1.0e9  # [nT]
CMB_night_By = np.mean(probes[observer_indices_R_core_night, :, By_idx], axis=0) * 1.0e9  # [nT]
CMB_night_Bz = np.mean(probes[observer_indices_R_core_night, :, Bz_idx], axis=0) * 1.0e9  # [nT]
CMB_night_B = np.column_stack([CMB_night_Bx, CMB_night_By, CMB_night_Bz])

print(f"\nDay-side surface B field shape: {surface_day_B.shape}")
print(f"Day-side CMB B field shape: {CMB_day_B.shape}")
print(f"Night-side surface B field shape: {surface_night_B.shape}")
print(f"Night-side CMB B field shape: {CMB_night_B.shape}")

# -------------------------------
# FFT for both hemispheres
# -------------------------------
print("\nPerforming FFT...")
actual_dt = np.mean(np.diff(timestamps))
print(f"   Actual time step (dt): {actual_dt:.3f} s")
print(f"   Sampling frequency: {1 / actual_dt:.3f} Hz")
print(f"   Nyquist frequency: {1 / (2 * actual_dt):.3f} Hz")

# Frequency array
freqs = fftfreq(n_steps, actual_dt)  # [Hz]
omega = 2 * np.pi * freqs  # [rad/s]

# Only keep positive frequencies
positive_freq_mask = freqs > 0
freqs_pos = freqs[positive_freq_mask]
omega_pos = omega[positive_freq_mask]

# --- DAY-SIDE FFT ---
fft_surface_day_Bx = fft(surface_day_B[:, 0])
fft_surface_day_By = fft(surface_day_B[:, 1])
fft_surface_day_Bz = fft(surface_day_B[:, 2])

fft_CMB_day_Bx = fft(CMB_day_B[:, 0])
fft_CMB_day_By = fft(CMB_day_B[:, 1])
fft_CMB_day_Bz = fft(CMB_day_B[:, 2])

# --- NIGHT-SIDE FFT ---
fft_surface_night_Bx = fft(surface_night_B[:, 0])
fft_surface_night_By = fft(surface_night_B[:, 1])
fft_surface_night_Bz = fft(surface_night_B[:, 2])

fft_CMB_night_Bx = fft(CMB_night_B[:, 0])
fft_CMB_night_By = fft(CMB_night_B[:, 1])
fft_CMB_night_Bz = fft(CMB_night_B[:, 2])

# --- DAY-SIDE PSD ---
psd_surface_day_Bx = np.abs(fft_surface_day_Bx[positive_freq_mask]) ** 2
psd_surface_day_By = np.abs(fft_surface_day_By[positive_freq_mask]) ** 2
psd_surface_day_Bz = np.abs(fft_surface_day_Bz[positive_freq_mask]) ** 2

psd_CMB_day_Bx = np.abs(fft_CMB_day_Bx[positive_freq_mask]) ** 2
psd_CMB_day_By = np.abs(fft_CMB_day_By[positive_freq_mask]) ** 2
psd_CMB_day_Bz = np.abs(fft_CMB_day_Bz[positive_freq_mask]) ** 2

# --- NIGHT-SIDE PSD ---
psd_surface_night_Bx = np.abs(fft_surface_night_Bx[positive_freq_mask]) ** 2
psd_surface_night_By = np.abs(fft_surface_night_By[positive_freq_mask]) ** 2
psd_surface_night_Bz = np.abs(fft_surface_night_Bz[positive_freq_mask]) ** 2

psd_CMB_night_Bx = np.abs(fft_CMB_night_Bx[positive_freq_mask]) ** 2
psd_CMB_night_By = np.abs(fft_CMB_night_By[positive_freq_mask]) ** 2
psd_CMB_night_Bz = np.abs(fft_CMB_night_Bz[positive_freq_mask]) ** 2

# -------------------------------
# Transfer Functions
# -------------------------------
print("\nCalculating transfer functions...")

mantle_thickness = (R_M - R_core) * 1e3  # convert km to m


def analytical_transfer_function(omega, sigma_mantle, sigma_core, h_mantle):
    """Calculate electromagnetic transfer function from surface to CMB"""
    delta_mantle = np.sqrt(2 / (mu_0 * np.abs(omega) * sigma_mantle))
    attenuation = np.exp(-h_mantle / delta_mantle)
    phase_delay = -h_mantle / delta_mantle
    T = attenuation * np.exp(1j * phase_delay)
    return T


# Calculate analytical transfer function
T_analytical = analytical_transfer_function(omega_pos, sigma_mantle, sigma_core, mantle_thickness)

# Empirical transfer functions
eps = 1e-30

# Day-side
T_day_Bx = fft_CMB_day_Bx[positive_freq_mask] / (fft_surface_day_Bx[positive_freq_mask] + eps)
T_day_By = fft_CMB_day_By[positive_freq_mask] / (fft_surface_day_By[positive_freq_mask] + eps)
T_day_Bz = fft_CMB_day_Bz[positive_freq_mask] / (fft_surface_day_Bz[positive_freq_mask] + eps)

# Night-side
T_night_Bx = fft_CMB_night_Bx[positive_freq_mask] / (fft_surface_night_Bx[positive_freq_mask] + eps)
T_night_By = fft_CMB_night_By[positive_freq_mask] / (fft_surface_night_By[positive_freq_mask] + eps)
T_night_Bz = fft_CMB_night_Bz[positive_freq_mask] / (fft_surface_night_Bz[positive_freq_mask] + eps)

# -------------------------------
# Save results
# -------------------------------
print("\nSaving results...")

# Save transfer function data (day-side)
df_transfer_day = pd.DataFrame({
    'frequency_Hz': freqs_pos,
    'period_s': 1.0 / freqs_pos,
    'omega_rad_s': omega_pos,
    'T_analytical_amplitude': np.abs(T_analytical),
    'T_analytical_phase': np.angle(T_analytical),
    'T_day_Bx_amplitude': np.abs(T_day_Bx),
    'T_day_Bx_phase': np.angle(T_day_Bx),
    'T_day_By_amplitude': np.abs(T_day_By),
    'T_day_By_phase': np.angle(T_day_By),
    'T_day_Bz_amplitude': np.abs(T_day_Bz),
    'T_day_Bz_phase': np.angle(T_day_Bz),
})
df_transfer_day.to_csv(os.path.join(output_folder, f'{case}_transfer_function_dayside.csv'), index=False)

# Save transfer function data (night-side)
df_transfer_night = pd.DataFrame({
    'frequency_Hz': freqs_pos,
    'period_s': 1.0 / freqs_pos,
    'omega_rad_s': omega_pos,
    'T_analytical_amplitude': np.abs(T_analytical),
    'T_analytical_phase': np.angle(T_analytical),
    'T_night_Bx_amplitude': np.abs(T_night_Bx),
    'T_night_Bx_phase': np.angle(T_night_Bx),
    'T_night_By_amplitude': np.abs(T_night_By),
    'T_night_By_phase': np.angle(T_night_By),
    'T_night_Bz_amplitude': np.abs(T_night_Bz),
    'T_night_Bz_phase': np.angle(T_night_Bz),
})
df_transfer_night.to_csv(os.path.join(output_folder, f'{case}_transfer_function_nightside.csv'), index=False)

# Save PSD data
df_psd = pd.DataFrame({
    'frequency_Hz': freqs_pos,
    'period_s': 1.0 / freqs_pos,
    'psd_surface_day_Bx': psd_surface_day_Bx,
    'psd_surface_day_By': psd_surface_day_By,
    'psd_surface_day_Bz': psd_surface_day_Bz,
    'psd_CMB_day_Bx': psd_CMB_day_Bx,
    'psd_CMB_day_By': psd_CMB_day_By,
    'psd_CMB_day_Bz': psd_CMB_day_Bz,
    'psd_surface_night_Bx': psd_surface_night_Bx,
    'psd_surface_night_By': psd_surface_night_By,
    'psd_surface_night_Bz': psd_surface_night_Bz,
    'psd_CMB_night_Bx': psd_CMB_night_Bx,
    'psd_CMB_night_By': psd_CMB_night_By,
    'psd_CMB_night_Bz': psd_CMB_night_Bz,
})
df_psd.to_csv(os.path.join(output_folder, f'{case}_power_spectral_density_comparison.csv'), index=False)

# Save time series
df_timeseries = pd.DataFrame({
    'time': timestamps,
    'surface_day_Bx': surface_day_B[:, 0],
    'surface_day_By': surface_day_B[:, 1],
    'surface_day_Bz': surface_day_B[:, 2],
    'CMB_day_Bx': CMB_day_B[:, 0],
    'CMB_day_By': CMB_day_B[:, 1],
    'CMB_day_Bz': CMB_day_B[:, 2],
    'surface_night_Bx': surface_night_B[:, 0],
    'surface_night_By': surface_night_B[:, 1],
    'surface_night_Bz': surface_night_B[:, 2],
    'CMB_night_Bx': CMB_night_B[:, 0],
    'CMB_night_By': CMB_night_B[:, 1],
    'CMB_night_Bz': CMB_night_B[:, 2],
})
df_timeseries.to_csv(os.path.join(output_folder, f'{case}_timeseries_B_field_comparison.csv'), index=False)

# -------------------------------
# Time series comparison plot: Day-side vs Night-side
# -------------------------------
fig, axes = plt.subplots(3, 2, figsize=(16, 10))

components = ['Bx', 'By', 'Bz']
colors_surface = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
colors_CMB = ['#d62728', '#9467bd', '#8c564b']  # Red, Purple, Brown

for i, component in enumerate(components):
    # Day-side (left column)
    ax_day = axes[i, 0]
    ax_day.plot(timestamps, surface_day_B[:, i],
                color=colors_surface[i], linewidth=2,
                label=f'Surface', linestyle='-', alpha=0.8)
    ax_day.plot(timestamps, CMB_day_B[:, i],
                color=colors_CMB[i], linewidth=2,
                label=f'CMB', linestyle='--', alpha=0.8)
    ax_day.set_ylabel(f'{component} [nT]', fontsize=14)
    ax_day.legend(loc='best', fontsize=11)
    ax_day.grid(True, alpha=0.3, linestyle='--')
    ax_day.tick_params(labelsize=11)

    if i == 0:
        ax_day.set_title('Day-side (X > 0)', fontsize=14, fontweight='bold')

    # Night-side (right column)
    ax_night = axes[i, 1]
    ax_night.plot(timestamps, surface_night_B[:, i],
                  color=colors_surface[i], linewidth=2,
                  label=f'Surface', linestyle='-', alpha=0.8)
    ax_night.plot(timestamps, CMB_night_B[:, i],
                  color=colors_CMB[i], linewidth=2,
                  label=f'CMB', linestyle='--', alpha=0.8)
    ax_night.set_ylabel(f'{component} [nT]', fontsize=14)
    ax_night.legend(loc='best', fontsize=11)
    ax_night.grid(True, alpha=0.3, linestyle='--')
    ax_night.tick_params(labelsize=11)

    if i == 0:
        ax_night.set_title('Night-side (X < 0)', fontsize=14, fontweight='bold')

# Add xlabel to bottom plots
axes[2, 0].set_xlabel('Time [s]', fontsize=14)
axes[2, 1].set_xlabel('Time [s]', fontsize=14)

fig.suptitle('Magnetic Field Components: Day-side vs Night-side', fontsize=16, y=0.995)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'{case}_B_field_timeseries_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------
# Transfer function comparison plot: 2x2 grid
# -------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1: Time series Bz comparison
ax = axes[0, 0]
ax.plot(timestamps, surface_day_B[:, 2], 'b-', label='Day Surface', linewidth=1.5, alpha=0.8)
ax.plot(timestamps, CMB_day_B[:, 2], 'r-', label='Day CMB', linewidth=1.5, alpha=0.8)
ax.plot(timestamps, surface_night_B[:, 2], 'b--', label='Night Surface', linewidth=1.5, alpha=0.8)
ax.plot(timestamps, CMB_night_B[:, 2], 'r--', label='Night CMB', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Time [s]', fontsize=12)
ax.set_ylabel('Bz [nT]', fontsize=12)
ax.set_title('Magnetic Field (Bz)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 2: Power spectral density comparison
ax = axes[0, 1]
ax.loglog(1 / freqs_pos, psd_surface_day_Bz, 'b-', label='Day Surface Bz', linewidth=1.5)
ax.loglog(1 / freqs_pos, psd_CMB_day_Bz, 'r-', label='Day CMB Bz', linewidth=1.5)
ax.loglog(1 / freqs_pos, psd_surface_night_Bz, 'b--', label='Night Surface Bz', linewidth=1.5)
ax.loglog(1 / freqs_pos, psd_CMB_night_Bz, 'r--', label='Night CMB Bz', linewidth=1.5)
ax.set_xlabel('Period [s]', fontsize=12)
ax.set_ylabel('PSD [nT²]', fontsize=12)
ax.set_title('Power Spectral Density', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 3: Transfer function amplitude comparison
ax = axes[1, 0]
ax.loglog(1 / freqs_pos, np.abs(T_analytical), 'k-', label='Analytical', linewidth=2)
ax.loglog(1 / freqs_pos, np.abs(T_day_Bz), 'b-', label='Day Bz', linewidth=1.5, alpha=0.8)
ax.loglog(1 / freqs_pos, np.abs(T_night_Bz), 'r--', label='Night Bz', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Period [s]', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_title('Transfer Function Amplitude', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 4: Transfer function phase comparison
ax = axes[1, 1]
ax.semilogx(1 / freqs_pos, np.angle(T_analytical, deg=True), 'k-', label='Analytical', linewidth=2)
ax.semilogx(1 / freqs_pos, np.angle(T_day_Bz, deg=True), 'b-', label='Day Bz', linewidth=1.5, alpha=0.8)
ax.semilogx(1 / freqs_pos, np.angle(T_night_Bz, deg=True), 'r--', label='Night Bz', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Period [s]', fontsize=12)
ax.set_ylabel('Phase [degrees]', fontsize=12)
ax.set_title('Transfer Function Phase', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig.suptitle(f"{case.replace('_', ' ')}: Day-side vs Night-side Comparison", fontsize=16, y=0.995)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'{case}_transfer_function_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------
# Additional plot: All components transfer function comparison
# -------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

components = ['Bx', 'By', 'Bz']
T_day = [T_day_Bx, T_day_By, T_day_Bz]
T_night = [T_night_Bx, T_night_By, T_night_Bz]

for i, component in enumerate(components):
    # Amplitude
    ax_amp = axes[0, i]
    ax_amp.loglog(1 / freqs_pos, np.abs(T_analytical), 'k-', label='Analytical', linewidth=2)
    ax_amp.loglog(1 / freqs_pos, np.abs(T_day[i]), 'b-', label='Day-side', linewidth=1.5, alpha=0.8)
    ax_amp.loglog(1 / freqs_pos, np.abs(T_night[i]), 'r--', label='Night-side', linewidth=1.5, alpha=0.8)
    ax_amp.set_xlabel('Period [s]', fontsize=12)
    ax_amp.set_ylabel('Amplitude', fontsize=12)
    ax_amp.set_title(f'{component} Transfer Function Amplitude', fontsize=12, fontweight='bold')
    ax_amp.legend(fontsize=10)
    ax_amp.grid(True, alpha=0.3)

    # Phase
    ax_phase = axes[1, i]
    ax_phase.semilogx(1 / freqs_pos, np.angle(T_analytical, deg=True), 'k-', label='Analytical', linewidth=2)
    ax_phase.semilogx(1 / freqs_pos, np.angle(T_day[i], deg=True), 'b-', label='Day-side', linewidth=1.5, alpha=0.8)
    ax_phase.semilogx(1 / freqs_pos, np.angle(T_night[i], deg=True), 'r--', label='Night-side', linewidth=1.5,
                      alpha=0.8)
    ax_phase.set_xlabel('Period [s]', fontsize=12)
    ax_phase.set_ylabel('Phase [degrees]', fontsize=12)
    ax_phase.set_title(f'{component} Transfer Function Phase', fontsize=12, fontweight='bold')
    ax_phase.legend(fontsize=10)
    ax_phase.grid(True, alpha=0.3)

fig.suptitle(f"{case.replace('_', ' ')}: Transfer Function Components", fontsize=16, y=0.995)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'{case}_transfer_function_all_components.png'),
            dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAnalysis complete! Results saved to {output_folder}")
print(f"  - {case}_transfer_function_dayside.csv")
print(f"  - {case}_transfer_function_nightside.csv")
print(f"  - {case}_power_spectral_density_comparison.csv")
print(f"  - {case}_timeseries_B_field_comparison.csv")
print(f"  - {case}_B_field_timeseries_comparison.png")
print(f"  - {case}_transfer_function_comparison.png")
print(f"  - {case}_transfer_function_all_components.png")
