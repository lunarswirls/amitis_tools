#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# -------------------------------
# Configuration
# -------------------------------
case = "CPN_HNHV"
output_folder = f"/Users/danywaller/Projects/mercury/extreme/magnetic_induction/{case}/"
os.makedirs(output_folder, exist_ok=True)

R_M = 2440.0  # Mercury radius [km]
R_core = 2080.0  # Mercury core radius [km]
LAT_BINS = 180  # Surface latitude bins
LON_BINS = 360  # Surface longitude bins

# Time step
dt = 2.0  # seconds (data every 2 seconds = 1000 simulation steps)

# Mercury conductivity structure
eta_mantle = 1.0e6  # Ohm-m (resistivity of mantle/shell)
eta_core = 1.0e-5  # Ohm-m (resistivity of core)
sigma_mantle = 1.0 / eta_mantle  # S/m
sigma_core = 1.0 / eta_core  # S/m
mu_0 = 4 * np.pi * 1e-7  # H/m

input_folder1 = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}/plane_product/object/"

sim_steps = list(range(115000, 200000 + 1, 1000))
n_steps = len(sim_steps)

# actual time array based on dt
timestamps = np.array(sim_steps) * 0.002  # [s]

# -------------------------------
# Extract time series at surface and CMB
# -------------------------------
if "CPN" in case:
    print(f"Case: Conductive Core under Planetward/Northward IMF")
elif "CPS" in case:
    print(f"Case: Conductive Core under Planetward/Southward IMF")
print(f"   Number of steps: {n_steps}")
print(f"   Total time span: {timestamps[-1] - timestamps[0]} s = {(timestamps[-1] - timestamps[0]) / 60:.2f} min\n")

# Storage for time series arrays
surface_B_timeseries = []
surface_density_timeseries = []
CMB_B_timeseries = []

for i, sim_step in enumerate(sim_steps):
    nc_file = os.path.join(input_folder1, f"Amitis_{case}_{sim_step:06d}_xz_comp.nc")

    if not os.path.exists(nc_file):
        print(f"Warning: {nc_file} not found, skipping...")
        continue

    ds = xr.open_dataset(nc_file)

    x = ds["Nx"].values  # [km]
    y = ds["Ny"].values  # [km]
    z = ds["Nz"].values  # [km]

    # Original order is Z, Y, X -> transpose to X, Y, Z
    BX = np.transpose(ds["Bx_tot"].isel(time=0).values, (2, 1, 0))  # [nT]
    BY = np.transpose(ds["By_tot"].isel(time=0).values, (2, 1, 0))  # [nT]
    BZ = np.transpose(ds["Bz_tot"].isel(time=0).values, (2, 1, 0))  # [nT]

    # Total density (sum of all species)
    den = (ds["den01"].isel(time=0).values +
           ds["den02"].isel(time=0).values +
           ds["den03"].isel(time=0).values +
           ds["den04"].isel(time=0).values)
    den = np.transpose(den, (2, 1, 0))  # [cm^-3]

    # Create interpolators
    interp_BX = RegularGridInterpolator((x, y, z), BX, bounds_error=False, fill_value=np.nan)
    interp_BY = RegularGridInterpolator((x, y, z), BY, bounds_error=False, fill_value=np.nan)
    interp_BZ = RegularGridInterpolator((x, y, z), BZ, bounds_error=False, fill_value=np.nan)
    interp_den = RegularGridInterpolator((x, y, z), den, bounds_error=False, fill_value=np.nan)

    # Sample at surface (R_M) - use spherical grid
    n_sample = 100  # Number of sample points
    theta_sample = np.linspace(0, np.pi, n_sample)
    phi_sample = np.linspace(0, 2 * np.pi, n_sample)

    # Surface sampling
    B_surface = []
    den_surface = []
    for theta in theta_sample[::1]:  # Subsample
        for phi in phi_sample[::1]:
            x_s = R_M * np.sin(theta) * np.cos(phi)
            y_s = R_M * np.sin(theta) * np.sin(phi)
            z_s = R_M * np.cos(theta)

            bx = interp_BX([x_s, y_s, z_s])[0]
            by = interp_BY([x_s, y_s, z_s])[0]
            bz = interp_BZ([x_s, y_s, z_s])[0]
            density = interp_den([x_s, y_s, z_s])[0]

            if not np.isnan(bx):
                B_surface.append([bx, by, bz])
            if not np.isnan(density):
                den_surface.append(density)

    surface_B_timeseries.append(np.nanmean(B_surface, axis=0) if B_surface else [np.nan, np.nan, np.nan])
    surface_density_timeseries.append(np.nanmean(den_surface) if den_surface else np.nan)

    # CMB sampling
    B_CMB = []
    for theta in theta_sample[::1]:
        for phi in phi_sample[::1]:
            x_c = R_core * np.sin(theta) * np.cos(phi)
            y_c = R_core * np.sin(theta) * np.sin(phi)
            z_c = R_core * np.cos(theta)

            bx = interp_BX([x_c, y_c, z_c])[0]
            by = interp_BY([x_c, y_c, z_c])[0]
            bz = interp_BZ([x_c, y_c, z_c])[0]

            if not np.isnan(bx):
                B_CMB.append([bx, by, bz])

    CMB_B_timeseries.append(np.nanmean(B_CMB, axis=0) if B_CMB else [np.nan, np.nan, np.nan])

    print(f"Processed step {sim_step} ({i + 1}/{n_steps}), time = {timestamps[i]:.1f} s")
    ds.close()

surface_B_timeseries = np.array(surface_B_timeseries)  # Shape: (n_steps, 3)
CMB_B_timeseries = np.array(CMB_B_timeseries)  # Shape: (n_steps, 3)
surface_density_timeseries = np.array(surface_density_timeseries)  # Shape: (n_steps,)

# -------------------------------
# frequency array
# -------------------------------
print("\nPerforming FFT...")
print(f"   Time step (dt): {dt} s")
print(f"   Sampling frequency: {1 / dt} Hz")
print(f"   Nyquist frequency: {1 / (2 * dt)} Hz")

n = len(timestamps)

# Frequency array
freqs = fftfreq(n, dt)  # [Hz]
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
    'surface_density': surface_density_timeseries,
})
df_timeseries.to_csv(os.path.join(output_folder, f'{case}_timeseries_B_field.csv'), index=False)

# -------------------------------
# Create 4x1 time series plot
# -------------------------------
fig, axes = plt.subplots(4, 1, figsize=(12, 12))

components = ['Bx', 'By', 'Bz']
colors_surface = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
colors_CMB = ['#d62728', '#9467bd', '#8c564b']  # Red, Purple, Brown

# Plot magnetic field components (first 3 subplots)
for i, (ax, component) in enumerate(zip(axes[:3], components)):
    # Plot surface
    ax.plot(timestamps, surface_B_timeseries[:, i],
            color=colors_surface[i], linewidth=2,
            label=f'Surface', linestyle='-', marker='o', markersize=4)

    # Plot CMB
    ax.plot(timestamps, CMB_B_timeseries[:, i],
            color=colors_CMB[i], linewidth=2,
            label=f'CMB', linestyle='--', marker='s', markersize=4)

    ax.set_ylabel(f'{component} [nT]', fontsize=14)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)

    # Add title only to top subplot
    if i == 0:
        ax.set_title('Magnetic Field Components and Plasma Density: Surface vs Core-Mantle Boundary',
                     fontsize=14, pad=15)

# Fourth subplot: Total density at surface
ax = axes[3]
ax.plot(timestamps, surface_density_timeseries,
        color='#e377c2', linewidth=2.5,
        label='Surface Total Density', linestyle='-', marker='D', markersize=5)

ax.set_ylabel('Density [cm^-3]', fontsize=14)
ax.set_xlabel('Time [s]', fontsize=14)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'{case}_B_field_density_timeseries.png'),
            dpi=300, bbox_inches='tight')

# -------------------------------
# transfer function analysis plot
# -------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1: Time series mag component (just Bz)
ax = axes[0, 0]
ax.plot(timestamps, surface_B_timeseries[:, 2], 'b-', label='Surface', linewidth=1.5)
ax.plot(timestamps, CMB_B_timeseries[:, 2], 'r-', label='CMB', linewidth=1.5)
ax.set_xlabel('Time [s]', fontsize=12)
ax.set_ylabel('Bz [nT]', fontsize=12)
ax.set_title('Magnetic Field', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2: Power spectral density
ax = axes[0, 1]
ax.loglog(1 / freqs_pos, psd_surface_Bz, 'b-', label='Surface Bz', linewidth=1.5)
ax.loglog(1 / freqs_pos, psd_CMB_Bz, 'r-', label='CMB Bz', linewidth=1.5)
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

fig.suptitle(f"{case.replace("_", " ")} from {timestamps[0]} to {timestamps[-1]} s", fontsize=16, y=0.97)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'{case}_transfer_function_analysis.png'), dpi=300, bbox_inches='tight')

print(f"\nAnalysis complete! Results saved to {output_folder}")
print(f"  - {case}_transfer_function.csv")
print(f"  - {case}_power_spectral_density.csv")
print(f"  - {case}_timeseries_B_field.csv")
print(f"  - {case}_B_field_density_timeseries.png")
print(f"  - {case}_transfer_function_analysis.png")