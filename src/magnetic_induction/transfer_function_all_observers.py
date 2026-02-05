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
output_folder = f"/Users/danywaller/Projects/mercury/extreme/magnetic_induction/{case}/"
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
           'pBx' : 1,
           'pBy' : 2,
           'pBz' : 3,
           'pBdx': 4,
           'pBdy': 5,
           'pBdz': 6,
           'pEx' : 7,
           'pEy' : 8,
           'pEz' : 9,
           'pJx' : 10,
           'pJy' : 11,
           'pJz' : 12 }

# -------------------------------
# Load AMITIS observer data
# -------------------------------
print("Loading AMITIS observer data...")
amitis_path = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/concat_obs"
amitis = amitis_observer(amitis_path, f"Observer_{case}")
# amitis.collect_all_data()

observers, probes, obsDict_loaded, numFields = amitis.load_collected_data()

print(f"Loaded {len(observers)} observers")
print(f"Probe data shape: {probes.shape}")

# -------------------------------
# Filter observers by radial distance
# -------------------------------
print("\nFiltering observers by radial distance...")

# Calculate radial distance for each observer (in km)
radial_distances = np.sqrt(observers[:, 0]**2 + observers[:, 1]**2 + observers[:, 2]**2) / 1000.0

# Filter observers within 75 km of R_M (surface)
observers_R_M = observers[(radial_distances >= R_M - tolerance) &
                          (radial_distances <= R_M + tolerance)]
observer_indices_R_M = np.where((radial_distances >= R_M - tolerance) &
                                (radial_distances <= R_M + tolerance))[0]

# Filter observers within 75 km of R_core (CMB)
observers_R_core = observers[(radial_distances >= R_core - tolerance) &
                             (radial_distances <= R_core + tolerance)]
observer_indices_R_core = np.where((radial_distances >= R_core - tolerance) &
                                   (radial_distances <= R_core + tolerance))[0]

print(f"Observers near surface (R_M = {R_M} km): {len(observers_R_M)}")
print(f"  Radial range: [{radial_distances[observer_indices_R_M].min():.1f}, {radial_distances[observer_indices_R_M].max():.1f}] km")
print(f"Observers near CMB (R_core = {R_core} km): {len(observers_R_core)}")
print(f"  Radial range: [{radial_distances[observer_indices_R_core].min():.1f}, {radial_distances[observer_indices_R_core].max():.1f}] km")

# -------------------------------
# Extract time series at surface and CMB
# -------------------------------
print("\nExtracting time series from observers...")

# Get time array from first observer
time_idx = obsDict['time']
timestamps = probes[0, :, time_idx]  # [s]
n_steps = len(timestamps)

print(f"Number of time steps: {n_steps}")
print(f"Total time span: {timestamps[-1] - timestamps[0]:.1f} s = {(timestamps[-1] - timestamps[0]) / 60:.2f} min")
print(f"Time step: {np.mean(np.diff(timestamps)):.3f} s")

# Extract magnetic field components at surface (average over all surface observers)
Bx_idx = obsDict['pBx']
By_idx = obsDict['pBy']
Bz_idx = obsDict['pBz']

# Surface time series (average over all surface observers)
surface_Bx = np.mean(probes[observer_indices_R_M, :, Bx_idx], axis=0)*1.0e9  # [nT]
surface_By = np.mean(probes[observer_indices_R_M, :, By_idx], axis=0)*1.0e9  # [nT]
surface_Bz = np.mean(probes[observer_indices_R_M, :, Bz_idx], axis=0)*1.0e9  # [nT]
surface_B_timeseries = np.column_stack([surface_Bx, surface_By, surface_Bz])

# CMB time series (average over all CMB observers)
CMB_Bx = np.mean(probes[observer_indices_R_core, :, Bx_idx], axis=0)*1.0e9  # [nT]
CMB_By = np.mean(probes[observer_indices_R_core, :, By_idx], axis=0)*1.0e9  # [nT]
CMB_Bz = np.mean(probes[observer_indices_R_core, :, Bz_idx], axis=0)*1.0e9  # [nT]
CMB_B_timeseries = np.column_stack([CMB_Bx, CMB_By, CMB_Bz])

print(f"Surface B field shape: {surface_B_timeseries.shape}")
print(f"CMB B field shape: {CMB_B_timeseries.shape}")

# -------------------------------
# Frequency array
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

print(f"   Frequency resolution: {freqs_pos[0]:.6f} Hz = {1 / freqs_pos[0]:.2f} s period")
print(f"   Maximum frequency: {freqs_pos[-1]:.6f} Hz = {1 / freqs_pos[-1]:.6f} s period")

# -------------------------------
# FFT of magnetic field components
# -------------------------------
# Surface
fft_surface_Bx = fft(surface_B_timeseries[:, 0])
fft_surface_By = fft(surface_B_timeseries[:, 1])
fft_surface_Bz = fft(surface_B_timeseries[:, 2])

# CMB
fft_CMB_Bx = fft(CMB_B_timeseries[:, 0])
fft_CMB_By = fft(CMB_B_timeseries[:, 1])
fft_CMB_Bz = fft(CMB_B_timeseries[:, 2])

# Power spectral density (magnitude squared)
psd_surface_Bx = np.abs(fft_surface_Bx[positive_freq_mask]) ** 2
psd_surface_By = np.abs(fft_surface_By[positive_freq_mask]) ** 2
psd_surface_Bz = np.abs(fft_surface_Bz[positive_freq_mask]) ** 2

psd_CMB_Bx = np.abs(fft_CMB_Bx[positive_freq_mask]) ** 2
psd_CMB_By = np.abs(fft_CMB_By[positive_freq_mask]) ** 2
psd_CMB_Bz = np.abs(fft_CMB_Bz[positive_freq_mask]) ** 2

# -------------------------------
# Transfer Functions
# -------------------------------
print("\nCalculating transfer functions...")

mantle_thickness = (R_M - R_core) * 1e3  # convert km to m


# Analytical transfer function for each frequency
def analytical_transfer_function(omega, sigma_mantle, sigma_core, h_mantle):
    """
    Calculate electromagnetic transfer function from surface to CMB

    Parameters:
    -----------
    omega : array
        Angular frequency [rad/s]
    sigma_mantle : float
        Mantle conductivity [S/m]
    sigma_core : float
        Core conductivity [S/m]
    h_mantle : float
        Mantle thickness [m]

    Returns:
    --------
    T : complex array
        Transfer function (complex, includes amplitude and phase)
    """
    # Skin depth in mantle
    delta_mantle = np.sqrt(2 / (mu_0 * np.abs(omega) * sigma_mantle))

    # Attenuation through mantle
    attenuation = np.exp(-h_mantle / delta_mantle)

    # Phase delay through mantle
    phase_delay = -h_mantle / delta_mantle

    # Transfer function (complex)
    T = attenuation * np.exp(1j * phase_delay)

    return T


# Calculate analytical transfer function
T_analytical = analytical_transfer_function(omega_pos, sigma_mantle, sigma_core, mantle_thickness)

# Empirical transfer function from simulation data
# T_empirical = B_CMB / B_surface (for each component)
eps = 1e-30  # Small number to avoid division by zero
T_empirical_Bx = fft_CMB_Bx[positive_freq_mask] / (fft_surface_Bx[positive_freq_mask] + eps)
T_empirical_By = fft_CMB_By[positive_freq_mask] / (fft_surface_By[positive_freq_mask] + eps)
T_empirical_Bz = fft_CMB_Bz[positive_freq_mask] / (fft_surface_Bz[positive_freq_mask] + eps)

# -------------------------------
# Save results
# -------------------------------
print("\nSaving results...")

# Save transfer function data
df_transfer = pd.DataFrame({
    'frequency_Hz': freqs_pos,
    'period_s': 1.0 / freqs_pos,
    'omega_rad_s': omega_pos,
    'T_analytical_amplitude': np.abs(T_analytical),
    'T_analytical_phase': np.angle(T_analytical),
    'T_empirical_Bx_amplitude': np.abs(T_empirical_Bx),
    'T_empirical_Bx_phase': np.angle(T_empirical_Bx),
    'T_empirical_By_amplitude': np.abs(T_empirical_By),
    'T_empirical_By_phase': np.angle(T_empirical_By),
    'T_empirical_Bz_amplitude': np.abs(T_empirical_Bz),
    'T_empirical_Bz_phase': np.angle(T_empirical_Bz),
})
df_transfer.to_csv(os.path.join(output_folder, f'{case}_transfer_function.csv'), index=False)

# Save power spectral densities
df_psd = pd.DataFrame({
    'frequency_Hz': freqs_pos,
    'period_s': 1.0 / freqs_pos,
    'psd_surface_Bx': psd_surface_Bx,
    'psd_surface_By': psd_surface_By,
    'psd_surface_Bz': psd_surface_Bz,
    'psd_CMB_Bx': psd_CMB_Bx,
    'psd_CMB_By': psd_CMB_By,
    'psd_CMB_Bz': psd_CMB_Bz,
})
df_psd.to_csv(os.path.join(output_folder, f'{case}_power_spectral_density.csv'), index=False)

# Save time series
df_timeseries = pd.DataFrame({
    'time': timestamps,
    'surface_Bx': surface_B_timeseries[:, 0],
    'surface_By': surface_B_timeseries[:, 1],
    'surface_Bz': surface_B_timeseries[:, 2],
    'CMB_Bx': CMB_B_timeseries[:, 0],
    'CMB_By': CMB_B_timeseries[:, 1],
    'CMB_Bz': CMB_B_timeseries[:, 2],
})
df_timeseries.to_csv(os.path.join(output_folder, f'{case}_timeseries_B_field.csv'), index=False)

# -------------------------------
# Create 3x1 time series plot (removed density)
# -------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

components = ['Bx', 'By', 'Bz']
colors_surface = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
colors_CMB = ['#d62728', '#9467bd', '#8c564b']  # Red, Purple, Brown

# Plot magnetic field components
for i, (ax, component) in enumerate(zip(axes, components)):
    # Plot surface
    ax.plot(timestamps, surface_B_timeseries[:, i],
            color=colors_surface[i], linewidth=2,
            label=f'Surface (R={R_M} km)', linestyle='-', alpha=0.8)

    # Plot CMB
    ax.plot(timestamps, CMB_B_timeseries[:, i],
            color=colors_CMB[i], linewidth=2,
            label=f'CMB (R={R_core} km)', linestyle='--', alpha=0.8)

    ax.set_ylabel(f'{component} [nT]', fontsize=14)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)

    # Add title only to top subplot
    if i == 0:
        ax.set_title('Magnetic Field Components: Surface vs Core-Mantle Boundary',
                     fontsize=14, pad=15)

# Only add xlabel to bottom plot
axes[-1].set_xlabel('Time [s]', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'{case}_B_field_timeseries.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------
# Transfer function analysis plot
# -------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1: Time series mag component (just Bz)
ax = axes[0, 0]
ax.plot(timestamps, surface_B_timeseries[:, 2], 'b-', label='Surface', linewidth=1.5)
ax.plot(timestamps, CMB_B_timeseries[:, 2], 'r-', label='CMB', linewidth=1.5)
ax.set_xlabel('Time [s]', fontsize=12)
ax.set_ylabel('Bz [nT]', fontsize=12)
ax.set_title('Magnetic Field (Bz)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2: Power spectral density
ax = axes[0, 1]
ax.loglog(1 / freqs_pos, psd_surface_Bx, 'b-', label='Surface Bx', linewidth=1.5)
ax.loglog(1 / freqs_pos, psd_CMB_Bx, 'r-', label='CMB Bx', linewidth=1.5)
ax.loglog(1 / freqs_pos, psd_surface_By, 'b--', label='Surface By', linewidth=1.5)
ax.loglog(1 / freqs_pos, psd_CMB_By, 'r--', label='CMB By', linewidth=1.5)
ax.loglog(1 / freqs_pos, psd_surface_Bz, 'b:', label='Surface Bz', linewidth=1.5)
ax.loglog(1 / freqs_pos, psd_CMB_Bz, 'r:', label='CMB Bz', linewidth=1.5)
ax.set_xlabel('Period [s]', fontsize=12)
ax.set_ylabel('PSD [nT²]', fontsize=12)
ax.set_title('Power Spectral Density', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 3: Transfer function amplitude
ax = axes[1, 0]
ax.loglog(1 / freqs_pos, np.abs(T_analytical), 'k-', label='Analytical', linewidth=2)
ax.loglog(1 / freqs_pos, np.abs(T_empirical_Bx), 'b--', label='Empirical Bx', linewidth=1.5, alpha=0.7)
ax.loglog(1 / freqs_pos, np.abs(T_empirical_By), 'g--', label='Empirical By', linewidth=1.5, alpha=0.7)
ax.loglog(1 / freqs_pos, np.abs(T_empirical_Bz), 'r--', label='Empirical Bz', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Period [s]', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_title('Transfer Function: Surface to CMB', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 4: Transfer function phase
ax = axes[1, 1]
ax.semilogx(1 / freqs_pos, np.angle(T_analytical, deg=True), 'k-', label='Analytical', linewidth=2)
ax.semilogx(1 / freqs_pos, np.angle(T_empirical_Bx, deg=True), 'b--', label='Empirical Bx', linewidth=1.5, alpha=0.7)
ax.semilogx(1 / freqs_pos, np.angle(T_empirical_By, deg=True), 'g--', label='Empirical By', linewidth=1.5, alpha=0.7)
ax.semilogx(1 / freqs_pos, np.angle(T_empirical_Bz, deg=True), 'r--', label='Empirical Bz', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Period [s]', fontsize=12)
ax.set_ylabel('Phase [degrees]', fontsize=12)
ax.set_title('Transfer Function Phase', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle(f"{case.replace('_', ' ')} from {timestamps[0]:.1f} to {timestamps[-1]:.1f} s\nAll observers (mean)", fontsize=16, y=0.995)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'{case}_transfer_function_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAnalysis complete! Results saved to {output_folder}")
print(f"  - {case}_transfer_function.csv")
print(f"  - {case}_power_spectral_density.csv")
print(f"  - {case}_timeseries_B_field.csv")
print(f"  - {case}_B_field_timeseries.png")
print(f"  - {case}_transfer_function_analysis.png")
