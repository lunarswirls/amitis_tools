#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Magnetic Field Line Twist Analysis for Mercury's Magnetosphere
===============================================================

Calculate twist for CLOSED magnetic field lines with β-dependent reliability.

Key Formula: Tw = (1/2π) ∫ α dl  where α = (∇×B)·B / |B|²

IMPORTANT NOTES:
----------------
- Twist calculation ONLY for CLOSED field lines (open lines = physically invalid)
- β-dependent reliability classification (high/moderate/low confidence)
- For β ~ 0.3 (Mercury plasma sheet), twist includes ~25% pressure contribution

Author: Dany Waller
Date: February 2026
"""

import os
import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator
from numba import jit
from src.field_topology.topology_utils import trace_field_line_rk, classify

# ======================
# USER PARAMETERS
# ======================
case = "CPN_Base_largerxdomain_smallergridsize"
input_folder = f"/Users/danywaller/Projects/mercury/extreme/{case}/out/"

# Parse case name for file naming
if "larger" in case:
    fname = case.split("_")[0] + "_" + case.split("_")[1]
else:
    fname = case

ncfile = os.path.join(input_folder, f"Amitis_{fname}_115000.nc")
output_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology_twist/{case}/"
os.makedirs(output_folder, exist_ok=True)

# Mercury parameters
RM = 2440.0  # Mercury radius [km]
dx = 200.0  # Surface tolerance for closed field line detection [km]

# Field line tracing parameters
n_lat = 90  # Latitude resolution
n_lon = n_lat * 2  # Longitude resolution (2:1 aspect ratio)
max_steps = 100000  # Maximum RK4 integration steps
h_step = 50.0  # Integration step size [km]

# Twist thresholds for categorization
twist_threshold = 0.5  # Low twist threshold (π radians)
moderate_twist = 1.0  # Moderate twist (2π radians)
high_twist = 2.0  # High twist (4π radians)

# Beta-dependent twist reliability thresholds
beta_threshold_high_confidence = 0.1  # β < 0.1: Force-free highly reliable
beta_threshold_moderate_confidence = 1.0  # β < 1.0: Reasonable approximation

# Visualization control
plot_3d_category = True  # Enable 3D visualization

# ======================
# PHYSICAL CONSTANTS
# ======================
k_B = 1.380649e-23  # Boltzmann constant [J/K]
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space [H/m]
k_B_eV = 8.617333262e-5  # Boltzmann constant [eV/K]


# ======================
# TWIST CALCULATION (NUMBA ACCELERATED)
# ======================

@jit(nopython=True, fastmath=True)
def compute_twist_numba(trajectory, alpha_values):
    """
    Compute twist number from force-free parameter α.

    Twist (Tw) measures internal field line winding around the flux rope axis:
        Tw = (1/2π) ∫ α dl

    Parameters
    ----------
    trajectory : ndarray, shape (N, 3)
        Field line trajectory points in [km]
    alpha_values : ndarray, shape (N,)
        Force-free parameter α sampled along trajectory [1/km]

    Returns
    -------
    twist : float
        Twist number [dimensionless]
        - Tw > 0: Right-handed twist
        - Tw < 0: Left-handed twist
        - |Tw| ~ 1.0: Moderate twist (~2π radians)
        - |Tw| > 1.25: Critical for kink instability

    Notes
    -----
    For β ~ 0.3 plasma, α includes ~25% pressure effects.
    Uses trapezoidal integration for numerical accuracy.
    """
    n = len(trajectory)
    if n < 2:
        return 0.0

    twist_integral = 0.0

    # Trapezoidal rule integration
    for i in range(n - 1):
        # Compute line element dl
        dx = trajectory[i + 1, 0] - trajectory[i, 0]
        dy = trajectory[i + 1, 1] - trajectory[i, 1]
        dz = trajectory[i + 1, 2] - trajectory[i, 2]
        dl = np.sqrt(dx * dx + dy * dy + dz * dz)

        # Trapezoidal rule: use midpoint alpha value
        alpha_mid = 0.5 * (alpha_values[i] + alpha_values[i + 1])

        twist_integral += alpha_mid * dl

    return twist_integral / (2.0 * np.pi)


# ======================
# HELPER FUNCTIONS
# ======================
def compute_twist_from_alpha(trajectory, alpha_field, x, y, z):
    """
    Compute twist by interpolating α field and integrating along trajectory.

    Parameters
    ----------
    trajectory : ndarray, shape (N, 3)
        Field line trajectory in [km]
    alpha_field : ndarray, shape (Nx, Ny, Nz)
        Force-free parameter field α = (∇×B)·B / |B|² [1/km]
    x, y, z : ndarray
        Grid coordinates for interpolation [km]

    Returns
    -------
    twist : float
        Twist number [dimensionless]
    """
    # Interpolate alpha values onto trajectory
    interpolator = RegularGridInterpolator(
        (x, y, z), alpha_field,
        bounds_error=False,
        fill_value=0
    )
    alpha_values = interpolator(trajectory)

    # Compute twist integral with Numba
    return compute_twist_numba(trajectory, alpha_values)


def classify_twist_reliability(beta_mean):
    """
    Classify reliability of twist measurement based on β.

    Parameters
    ----------
    beta_mean : float
        Mean plasma β along field line

    Returns
    -------
    reliability : str
        'high': β < 0.1, pressure < 10% - twist uncertainty ~5-10%
        'moderate': 0.1 ≤ β < 1.0, pressure < 50% - twist uncertainty ~25-30%
        'low': β ≥ 1.0 or pressure ≥ 50% - twist uncertainty >50%
    """
    if beta_mean < beta_threshold_high_confidence:
        return 'high'
    elif beta_mean < beta_threshold_moderate_confidence:
        return 'moderate'
    else:
        return 'low'


def classify_twist_level(total_twist):
    """
    Classify twist into categories based on magnitude.

    Parameters
    ----------
    total_twist : float
        Absolute twist value

    Returns
    -------
    level : str
        'low': 0.5-1.0 turns (π to 2π radians)
        'moderate': 1.0-2.0 turns (2π to 4π radians)
        'high': >2.0 turns (>4π radians, highly twisted)
    """
    if total_twist >= high_twist:
        return 'high'
    elif total_twist >= moderate_twist:
        return 'moderate'
    elif total_twist >= twist_threshold:
        return 'low'
    else:
        return 'non-twisted'


# ======================
# FIELD CALCULATION FUNCTIONS
# ======================

def compute_curl_B(Bx, By, Bz, x, y, z):
    """
    Compute curl of magnetic field: ∇ × B

    Uses centered finite differences for numerical derivatives.

    Parameters
    ----------
    Bx, By, Bz : ndarray, shape (Nx, Ny, Nz)
        Magnetic field components [nT]
    x, y, z : ndarray
        Grid coordinates [km]

    Returns
    -------
    curl_x, curl_y, curl_z : ndarray
        Curl components [nT/km]
    """
    curl_x = np.zeros_like(Bx)
    curl_y = np.zeros_like(By)
    curl_z = np.zeros_like(Bz)

    dx_grid = x[1] - x[0]
    dy_grid = y[1] - y[0]
    dz_grid = z[1] - z[0]

    # curl_x = ∂Bz/∂y - ∂By/∂z
    curl_x[1:-1, 1:-1, 1:-1] = (
            (Bz[1:-1, 2:, 1:-1] - Bz[1:-1, :-2, 1:-1]) / (2 * dy_grid) -
            (By[1:-1, 1:-1, 2:] - By[1:-1, 1:-1, :-2]) / (2 * dz_grid)
    )

    # curl_y = ∂Bx/∂z - ∂Bz/∂x
    curl_y[1:-1, 1:-1, 1:-1] = (
            (Bx[1:-1, 1:-1, 2:] - Bx[1:-1, 1:-1, :-2]) / (2 * dz_grid) -
            (Bz[2:, 1:-1, 1:-1] - Bz[:-2, 1:-1, 1:-1]) / (2 * dx_grid)
    )

    # curl_z = ∂By/∂x - ∂Bx/∂y
    curl_z[1:-1, 1:-1, 1:-1] = (
            (By[2:, 1:-1, 1:-1] - By[:-2, 1:-1, 1:-1]) / (2 * dx_grid) -
            (Bx[1:-1, 2:, 1:-1] - Bx[1:-1, :-2, 1:-1]) / (2 * dy_grid)
    )

    return curl_x, curl_y, curl_z


def compute_alpha_field(Bx, By, Bz, curl_x, curl_y, curl_z):
    """
    Compute force-free parameter α = (∇ × B) · B / |B|²

    Parameters
    ----------
    Bx, By, Bz : ndarray
        Magnetic field components [nT]
    curl_x, curl_y, curl_z : ndarray
        Curl components [nT/km]

    Returns
    -------
    alpha : ndarray
        Force-free parameter [1/km]

    Notes
    -----
    In non-force-free plasma (β > 0.1), this measures field-aligned
    current structure including pressure effects.
    """
    B_mag_sq = Bx ** 2 + By ** 2 + Bz ** 2
    B_mag_sq = np.where(B_mag_sq < 1e-10, 1e-10, B_mag_sq)
    alpha = (curl_x * Bx + curl_y * By + curl_z * Bz) / B_mag_sq
    return alpha


# ======================
# LOAD DATA
# ======================
print("=" * 70)
print(f"Magnetic Field Line Twist Analysis (CLOSED LINES ONLY)")
if "CPN" in case:
    print(f"Case: Conductive Core under Planetward/Northward IMF")
elif "CPS" in case:
    print(f"Case: Conductive Core under Planetward/Southward IMF")
print("=" * 70)
print(f"\nLoading data from {ncfile}")

ds = xr.open_dataset(ncfile)

x = ds["Nx"].values  # [km]
y = ds["Ny"].values  # [km]
z = ds["Nz"].values  # [km]

print(f"Grid size: {len(x)} × {len(y)} × {len(z)}")
print(f"Domain: X=[{x.min():.0f}, {x.max():.0f}] km")
print(f"        Y=[{y.min():.0f}, {y.max():.0f}] km")
print(f"        Z=[{z.min():.0f}, {z.max():.0f}] km")

# ======================
# CREATE SEEDS ON MERCURY'S SURFACE
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
print(f"\nCreated {len(seeds)} seed points ({n_lat} × {n_lon})")

# ======================
# COMPUTE THERMAL PRESSURE
# ======================
print(f"\n{'=' * 70}")
print("Computing Thermal Pressure")
print("=" * 70)

first_species = True

# Species temperatures from simulation input file
s_dict = {
    '1': 1.4e5,  # H+ solar wind protons [K]
    '2': 1.4e5,
    '3': 5.6e5,  # He++ alphas [K]
    '4': 5.6e5,
}

for s in ['1', '2', '3', '4']:
    T_K = s_dict[s]
    n_cm3 = ds[f'den0{s}'].isel(time=0).values
    n_transposed_cm3 = np.transpose(n_cm3, (2, 1, 0))
    n_transposed = n_transposed_cm3 * 1e6  # Convert cm^-3 to m^-3

    if n_transposed.max() < 1e3:  # Less than 0.001 cm^-3
        print(f"Species 0{s}: INACTIVE")
        continue

    # Compute pressure: P = n * k_B * T [Pa]
    P_species_Pa = n_transposed * k_B * T_K
    P_species_nPa = P_species_Pa * 1e9

    if first_species:
        P_total_Pa = P_species_Pa.copy()
        P_total_nPa = P_species_nPa.copy()
        first_species = False
    else:
        P_total_Pa += P_species_Pa
        P_total_nPa += P_species_nPa

    print(f"Species 0{s}: T={T_K:.2e} K, P_range=[{P_species_nPa.min():.3f}, {P_species_nPa.max():.3f}] nPa")

print(f"\nTotal thermal pressure: [{P_total_nPa.min():.3f}, {P_total_nPa.max():.3f}] nPa")

# ======================
# PRESSURE GRADIENTS
# ======================
print(f"\n{'=' * 70}")
print("Computing Pressure Gradients")
print("=" * 70)

dnx = (x[1] - x[0]) * 1e3  # Convert km to m
dny = (y[1] - y[0]) * 1e3
dnz = (z[1] - z[0]) * 1e3

grad_P_x = np.gradient(P_total_Pa, dnx, axis=0)  # [Pa/m]
grad_P_y = np.gradient(P_total_Pa, dny, axis=1)
grad_P_z = np.gradient(P_total_Pa, dnz, axis=2)

grad_P_magnitude = np.sqrt(grad_P_x ** 2 + grad_P_y ** 2 + grad_P_z ** 2)

print(f"|∇P| range: [{grad_P_magnitude.min():.2e}, {grad_P_magnitude.max():.2e}] Pa/m")

# ======================
# MAGNETIC FIELD
# ======================
print(f"\n{'=' * 70}")
print("Loading Magnetic Field")
print("=" * 70)

Bx_plane = np.transpose(ds["Bx"].isel(time=0).values, (2, 1, 0))  # [nT]
By_plane = np.transpose(ds["By"].isel(time=0).values, (2, 1, 0))
Bz_plane = np.transpose(ds["Bz"].isel(time=0).values, (2, 1, 0))

# Convert nT to Tesla
Bx_T = Bx_plane * 1e-9
By_T = By_plane * 1e-9
Bz_T = Bz_plane * 1e-9

B_magnitude_nT = np.sqrt(Bx_plane ** 2 + By_plane ** 2 + Bz_plane ** 2)
B_magnitude_T = B_magnitude_nT * 1e-9

print(f"|B| range: [{B_magnitude_nT.min():.2f}, {B_magnitude_nT.max():.2f}] nT")

# ======================
# CURRENT DENSITY
# ======================
print(f"\n{'=' * 70}")
print("Loading Current Density")
print("=" * 70)

Jx_nA = np.transpose(ds['Jx'].isel(time=0).values, (2, 1, 0))  # [nA/m^2]
Jy_nA = np.transpose(ds['Jy'].isel(time=0).values, (2, 1, 0))
Jz_nA = np.transpose(ds['Jz'].isel(time=0).values, (2, 1, 0))

# Convert to A/m^2
Jx = Jx_nA * 1e-9
Jy = Jy_nA * 1e-9
Jz = Jz_nA * 1e-9

J_magnitude_nA = np.sqrt(Jx_nA ** 2 + Jy_nA ** 2 + Jz_nA ** 2)

print(f"|J| range: [{J_magnitude_nA.min():.2f}, {J_magnitude_nA.max():.2f}] nA/m^2")

# ======================
# LORENTZ FORCE: J × B
# ======================
print(f"\n{'=' * 70}")
print("Computing Lorentz Force")
print("=" * 70)

JcrossB_x = Jy * Bz_T - Jz * By_T  # [N/m^3]
JcrossB_y = Jz * Bx_T - Jx * Bz_T
JcrossB_z = Jx * By_T - Jy * Bx_T

JcrossB_magnitude = np.sqrt(JcrossB_x ** 2 + JcrossB_y ** 2 + JcrossB_z ** 2)

print(f"|J×B| range: [{JcrossB_magnitude.min():.2e}, {JcrossB_magnitude.max():.2e}] N/m^3")

# ======================
# PLASMA BETA
# ======================
print(f"\n{'=' * 70}")
print("Computing Plasma β")
print("=" * 70)

magnetic_pressure_Pa = B_magnitude_T ** 2 / (2 * mu_0)
magnetic_pressure_nPa = magnetic_pressure_Pa * 1e9

# Avoid division by zero
magnetic_pressure_Pa_safe = np.where(magnetic_pressure_Pa < 1e-30, 1e-30, magnetic_pressure_Pa)
plasma_beta = P_total_Pa / magnetic_pressure_Pa_safe

print(f"Plasma β:")
print(f"  Min: {plasma_beta.min():.2e}")
print(f"  Max: {plasma_beta.max():.2e}")
print(f"  Mean: {plasma_beta.mean():.2e}")
print(f"  Median: {np.median(plasma_beta):.2e}")

# Beta distribution statistics
beta_very_low = np.sum(plasma_beta < 0.1) / plasma_beta.size * 100
beta_low = np.sum((plasma_beta >= 0.1) & (plasma_beta < 1.0)) / plasma_beta.size * 100
beta_moderate = np.sum((plasma_beta >= 1.0) & (plasma_beta < 10.0)) / plasma_beta.size * 100
beta_high = np.sum(plasma_beta >= 10.0) / plasma_beta.size * 100

print(f"\nβ distribution:")
print(f"  β < 0.1 (magnetic-dominated):      {beta_very_low:.1f}%")
print(f"  0.1 ≤ β < 1.0 (plasma sheet):      {beta_low:.1f}%")
print(f"  1.0 ≤ β < 10 (pressure-dominated): {beta_moderate:.1f}%")
print(f"  β ≥ 10 (highly non-force-free):    {beta_high:.1f}%")

# ======================
# COMPUTE ALPHA FIELD
# ======================
print(f"\n{'=' * 70}")
print("Computing Force-Free Parameter α")
print("=" * 70)

print("Computing curl of B...")
curl_x, curl_y, curl_z = compute_curl_B(Bx_plane, By_plane, Bz_plane, x, y, z)

print("Computing α = (∇×B)·B / |B|²...")
alpha_field = compute_alpha_field(Bx_plane, By_plane, Bz_plane, curl_x, curl_y, curl_z)

print(f"α field statistics:")
print(f"  Min: {alpha_field.min():.3e} km^-1")
print(f"  Max: {alpha_field.max():.3e} km^-1")
print(f"  Mean: {alpha_field.mean():.3e} km^-1")
print(f"\nNOTE: For β ~ 0.3 plasma, α includes ~25% pressure effects")

# ======================
# CREATE COMBINED INTERPOLATOR
# ======================
print(f"\n{'=' * 70}")
print("Creating Combined Interpolator")
print("=" * 70)

# Stack fields for efficient interpolation
combined_fields = np.stack([
    plasma_beta,
], axis=-1)  # Shape: (Nx, Ny, Nz, 1)

interp_combined = RegularGridInterpolator(
    (x, y, z),
    combined_fields,
    bounds_error=False,
    fill_value=0
)

print("Created interpolator for beta")


def analyze_force_balance_along_fieldline(trajectory):
    """
    Analyze force balance parameters along a field line trajectory.

    Parameters
    ----------
    trajectory : ndarray, shape (N, 3)
        Field line trajectory points in [km]

    Returns
    -------
    stats : dict
        Dictionary containing beta and pressure statistics
    """
    # Single interpolation call returns all 4 fields
    all_vals = interp_combined(trajectory)  # Shape: (N_points, 4)

    beta_vals = all_vals[:, 0]

    n_points = len(beta_vals)

    return {
        'beta_mean': np.mean(beta_vals),
        'beta_max': np.max(beta_vals),
        'beta_min': np.min(beta_vals),
    }


# ======================
# WARM UP NUMBA JIT
# ======================
print(f"\n{'=' * 70}")
print("Compiling Numba Functions")
print("=" * 70)

dummy_trajectory = np.random.rand(100, 3).astype(np.float64)
dummy_alpha = np.random.rand(100).astype(np.float64)
_ = compute_twist_numba(dummy_trajectory, dummy_alpha)

print("JIT compilation complete!")

# ======================
# TRACE FIELD LINES AND COMPUTE TWIST
# ======================
print(f"\n{'=' * 70}")
print("Tracing Field Lines and Computing Twist")
print("=" * 70)
print(f"Configuration:")
print(f"  Seeds: {len(seeds)}")
print(f"  Max integration steps: {max_steps}")
print(f"  Step size: {h_step} km")
print(f"  Minimum twist for analysis: {twist_threshold}")
print(f"\nReliability thresholds:")
print(f"  High confidence: β < {beta_threshold_high_confidence}")
print(f"  Moderate confidence: β < {beta_threshold_moderate_confidence}")
print(f"  Low confidence: β ≥ {beta_threshold_moderate_confidence}")

fieldline_data = []
classification_counts = {'closed': 0, 'open': 0, 'solar_wind': 0}
twist_reliability_counts = {'high': 0, 'moderate': 0, 'low': 0}
twist_level_counts = {'non-twisted': 0, 'low': 0, 'moderate': 0, 'high': 0}

start_time = time.time()

for i, seed in enumerate(seeds):
    # Trace field line forward and backward
    traj_fwd, exit_fwd_y = trace_field_line_rk(
        seed, Bx_plane, By_plane, Bz_plane,
        x, y, z, RM, max_steps=max_steps, h=h_step
    )
    traj_bwd, exit_bwd_y = trace_field_line_rk(
        seed, Bx_plane, By_plane, Bz_plane,
        x, y, z, RM, max_steps=max_steps, h=-h_step
    )

    # Classify topology
    field_line_type = classify(traj_fwd, traj_bwd, RM, exit_fwd_y, exit_bwd_y)
    classification_counts[field_line_type] += 1

    # ONLY ANALYZE CLOSED FIELD LINES FOR TWIST
    if field_line_type != "closed":
        continue

    # Combine trajectories
    full_trajectory = np.vstack([traj_bwd[::-1], traj_fwd])

    # Analyze force balance
    force_stats = analyze_force_balance_along_fieldline(full_trajectory)

    # Classify twist reliability based on β
    twist_reliability = classify_twist_reliability(force_stats['beta_mean'])
    twist_reliability_counts[twist_reliability] += 1

    # Compute twist
    twist = compute_twist_from_alpha(full_trajectory, alpha_field, x, y, z)

    # Classify twist level
    twist_level = classify_twist_level(abs(twist))
    twist_level_counts[twist_level] += 1

    # Store results
    fieldline_data.append({
        'seed_x': seed[0],
        'seed_y': seed[1],
        'seed_z': seed[2],
        'field_line_type': field_line_type,
        'twist': twist,
        'abs_twist': abs(twist),
        'twist_level': twist_level,
        'trajectory_length': len(full_trajectory),
        'beta_mean': force_stats['beta_mean'],
        'beta_max': force_stats['beta_max'],
        'beta_min': force_stats['beta_min'],
        'twist_reliability': twist_reliability
    })

    # Progress reporting
    if (i + 1) % 100 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        remaining = (len(seeds) - i - 1) / rate if rate > 0 else 0
        print(f"Processed {i + 1}/{len(seeds)} seeds | Rate: {rate:.1f} seeds/s | ETA: {remaining / 60:.1f} min")

total_time = time.time() - start_time

# ======================
# PRINT SUMMARY STATISTICS
# ======================
print(f"\n{'=' * 70}")
print("FIELD LINE TRACING COMPLETE")
print("=" * 70)
print(f"Total time: {total_time / 60:.2f} minutes")
print(f"Average rate: {len(seeds) / total_time:.1f} seeds/second")

print(f"\nField line classification:")
print(
    f"  Closed:      {classification_counts['closed']:5d} ({100 * classification_counts['closed'] / len(seeds):5.1f}%)")
print(f"  Open:        {classification_counts['open']:5d} ({100 * classification_counts['open'] / len(seeds):5.1f}%)")
print(
    f"  Solar wind:  {classification_counts['solar_wind']:5d} ({100 * classification_counts['solar_wind'] / len(seeds):5.1f}%)")

print(f"\nTwist reliability for closed field lines:")
total_closed = classification_counts['closed']
if total_closed > 0:
    print(
        f"  High confidence (β < 0.1):         {twist_reliability_counts['high']:5d} ({100 * twist_reliability_counts['high'] / total_closed:5.1f}%)")
    print(
        f"  Moderate confidence (0.1 ≤ β < 1): {twist_reliability_counts['moderate']:5d} ({100 * twist_reliability_counts['moderate'] / total_closed:5.1f}%)")
    print(
        f"  Low confidence (β ≥ 1):            {twist_reliability_counts['low']:5d} ({100 * twist_reliability_counts['low'] / total_closed:5.1f}%)")

print(f"\nTwist level distribution:")
if total_closed > 0:
    print(
        f"  Non-twisted (|Tw| < 0.5):  {twist_level_counts['non-twisted']:5d} ({100 * twist_level_counts['non-twisted'] / total_closed:5.1f}%)")
    print(
        f"  Low (0.5-1.0 turns):       {twist_level_counts['low']:5d} ({100 * twist_level_counts['low'] / total_closed:5.1f}%)")
    print(
        f"  Moderate (1.0-2.0 turns):  {twist_level_counts['moderate']:5d} ({100 * twist_level_counts['moderate'] / total_closed:5.1f}%)")
    print(
        f"  High (>2.0 turns):         {twist_level_counts['high']:5d} ({100 * twist_level_counts['high'] / total_closed:5.1f}%)")

# ======================
# SAVE RESULTS TO CSV
# ======================
df = pd.DataFrame(fieldline_data)
output_csv = os.path.join(output_folder, f"twist_analysis_{case}.csv")
df.to_csv(output_csv, index=False)
print(f"\nSaved CSV: {output_csv}")

# ======================
# DETAILED STATISTICS
# ======================
if len(df) > 0:
    print("\n" + "=" * 70)
    print("TWIST STATISTICS")
    print("=" * 70)

    print(f"\nTotal closed field lines analyzed: {len(df)}")

    print(f"\nTwist statistics:")
    print(f"  Mean: {df['twist'].mean():.3f}")
    print(f"  Std:  {df['twist'].std():.3f}")
    print(f"  Range: [{df['twist'].min():.3f}, {df['twist'].max():.3f}]")
    print(f"  Mean |twist|: {df['abs_twist'].mean():.3f}")

    # Statistics by reliability
    print(f"\n{'=' * 70}")
    print("STATISTICS BY RELIABILITY")
    print("=" * 70)

    for reliability in ['high', 'moderate', 'low']:
        subset = df[df['twist_reliability'] == reliability]
        if len(subset) > 0:
            print(f"\n{reliability.upper()} confidence ({len(subset)} lines):")
            print(f"  Mean β: {subset['beta_mean'].mean():.3f}")
            print(f"  Mean |twist|: {subset['abs_twist'].mean():.3f}")
            print(f"  Twist range: [{subset['twist'].min():.3f}, {subset['twist'].max():.3f}]")

    # Statistics by β regime
    print(f"\n{'=' * 70}")
    print("STATISTICS BY β REGIME")
    print("=" * 70)

    low_beta = df[df['beta_mean'] < 0.1]
    mid_beta = df[(df['beta_mean'] >= 0.1) & (df['beta_mean'] < 1.0)]
    high_beta = df[df['beta_mean'] >= 1.0]

    print(f"\nField line distribution:")
    print(f"  Low β (< 0.1):  {len(low_beta):4d} lines ({100 * len(low_beta) / len(df):5.1f}%)")
    print(f"  Mid β (0.1-1):  {len(mid_beta):4d} lines ({100 * len(mid_beta) / len(df):5.1f}%)")
    print(f"  High β (> 1.0): {len(high_beta):4d} lines ({100 * len(high_beta) / len(df):5.1f}%)")

    for subset, name in [(low_beta, "Low β (< 0.1)"),
                         (mid_beta, "Mid β (0.1-1.0)"),
                         (high_beta, "High β (> 1.0)")]:
        if len(subset) > 0:
            print(f"\n  {name}:")
            print(f"    Mean |twist|: {subset['abs_twist'].mean():.3f}")
            print(f"    Twist range: [{subset['twist'].min():.3f}, {subset['twist'].max():.3f}]")

# ======================
# CREATE 2D PLOTS
# ======================
if len(df) > 0:
    print(f"\n{'=' * 70}")
    print("Creating 2D Visualizations")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Twist distribution
    axes[0, 0].hist(df['twist'], bins=40, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0, 0].set_xlabel('Twist (Tw)', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Twist Distribution (Closed Field Lines)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axvline(x=df['twist'].mean(), color='r', linestyle='--',
                       label=f'Mean = {df["twist"].mean():.2f}', linewidth=2)
    axes[0, 0].legend()

    # Plot 2: |Twist| vs β (colored by reliability)
    reliability_colors = {'high': 'green', 'moderate': 'orange', 'low': 'red'}
    for reliability in ['high', 'moderate', 'low']:
        subset = df[df['twist_reliability'] == reliability]
        if len(subset) > 0:
            axes[0, 1].scatter(
                subset['beta_mean'], subset['abs_twist'],
                c=reliability_colors[reliability], s=20, alpha=0.6,
                label=f'{reliability.capitalize()} confidence'
            )

    axes[0, 1].set_xlabel('Mean β', fontsize=11)
    axes[0, 1].set_ylabel('|Twist|', fontsize=11)
    axes[0, 1].set_title('|Twist| vs β (colored by reliability)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xscale('log')
    axes[0, 1].axvline(x=0.1, color='k', linestyle='--', alpha=0.5, label='β = 0.1')
    axes[0, 1].axvline(x=1.0, color='k', linestyle=':', alpha=0.5, label='β = 1.0')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Twist by level
    twist_counts = [
        twist_level_counts['low'],
        twist_level_counts['moderate'],
        twist_level_counts['high']
    ]
    twist_labels_plot = ['Low\n(0.5-1.0)', 'Moderate\n(1.0-2.0)', 'High\n(>2.0)']
    colors_plot = ['cornflowerblue', 'darkorange', 'firebrick']

    axes[1, 0].bar(twist_labels_plot, twist_counts, color=colors_plot, alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Twisted Field Lines by Level', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: β distribution
    axes[1, 1].hist(df['beta_mean'], bins=50, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 1].set_xlabel('Mean β', fontsize=11)
    axes[1, 1].set_ylabel('Count', fontsize=11)
    axes[1, 1].set_title('β Distribution (Closed Field Lines)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xscale('log')
    axes[1, 1].axvline(x=0.1, color='k', linestyle='--', alpha=0.7, linewidth=2, label='β = 0.1')
    axes[1, 1].axvline(x=1.0, color='k', linestyle=':', alpha=0.7, linewidth=2, label='β = 1.0')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plot_file = os.path.join(output_folder, f"twist_analysis_{case}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 2D plots: {plot_file}")

# ======================
# CREATE 3D VISUALIZATION
# ======================
if plot_3d_category and len(df) > 0:
    print(f"\n{'=' * 70}")
    print("Creating 3D Visualization")
    print("=" * 70)

    # Collect trajectories for visualization
    trajectories_3d = {
        'twisted_closed': []
    }

    print(f"Collecting trajectories for 3D plot...")
    count = 0
    max_viz = 200  # Limit for performance

    for idx, row in df.iterrows():
        # Only visualize significantly twisted lines
        if abs(row['twist']) < twist_threshold:
            continue

        seed = np.array([row['seed_x'], row['seed_y'], row['seed_z']])

        # Re-trace this field line
        traj_fwd, _ = trace_field_line_rk(
            seed, Bx_plane, By_plane, Bz_plane,
            x, y, z, RM, max_steps=max_steps, h=h_step
        )
        traj_bwd, _ = trace_field_line_rk(
            seed, Bx_plane, By_plane, Bz_plane,
            x, y, z, RM, max_steps=max_steps, h=-h_step
        )

        trajectories_3d['twisted_closed'].append({
            'fwd': traj_fwd,
            'bwd': traj_bwd,
            'twist': row['twist'],
            'level': row['twist_level'],
            'reliability': row['twist_reliability'],
            'beta': row['beta_mean']
        })

        count += 1
        if count >= max_viz:
            break

    print(f"Collected {count} twisted field lines for 3D visualization")

    # Create figure
    fig_3d = go.Figure()

    # Add planet sphere
    theta_sphere = np.linspace(0, np.pi, 100)
    phi_sphere = np.linspace(0, 2 * np.pi, 200)
    theta_sphere, phi_sphere = np.meshgrid(theta_sphere, phi_sphere)

    xs = RM * np.sin(theta_sphere) * np.cos(phi_sphere)
    ys = RM * np.sin(theta_sphere) * np.sin(phi_sphere)
    zs = RM * np.cos(theta_sphere)

    eps = 0
    mask_pos = xs >= -eps
    mask_neg = xs <= eps

    # Day side (light grey)
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

    # Night side (black)
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

    # Define colors for twist levels
    twist_colors = {
        'low': 'cornflowerblue',
        'moderate': 'darkorange',
        'high': 'firebrick'
    }

    twist_labels = {
        'low': f'Low ({twist_threshold:.1f}-{moderate_twist:.1f} turns, π rad)',
        'moderate': f'Moderate ({moderate_twist:.1f}-{high_twist:.1f} turns, 2π rad)',
        'high': f'High (>{high_twist:.1f} turns, highly twisted)'
    }

    # Reliability line styles
    reliability_linewidth = {
        'high': 6,
        'moderate': 3,
        'low': 2
    }

    reliability_labels = {
        'high': 'β<0.1 (high confidence)',
        'moderate': '0.1≤β<1 (moderate)',
        'low': 'β≥1 (low confidence)'
    }

    # Track legend entries
    categories_added = {}
    twist_order = ['low', 'moderate', 'high']
    reliability_order = ['high', 'moderate', 'low']

    # Sort trajectories by twist level first, then reliability
    trajectories_sorted = sorted(
        trajectories_3d['twisted_closed'],
        key=lambda x: (twist_order.index(x['level']), reliability_order.index(x['reliability']))
    )

    # Plot field lines
    legend_rank_counter = 0

    for traj_data in trajectories_sorted:
        traj_fwd = traj_data['fwd']
        traj_bwd = traj_data['bwd']
        twist_val = traj_data['twist']
        twist_level = traj_data['level']
        reliability = traj_data['reliability']
        beta_val = traj_data['beta']

        color = twist_colors[twist_level]
        width = reliability_linewidth[reliability]

        # Legend logic
        category_key = f'{twist_level}_{reliability}'
        show_legend = category_key not in categories_added

        if show_legend:
            categories_added[category_key] = True
            legend_name = f'{twist_labels[twist_level]} - {reliability_labels[reliability]}'
            legend_group = category_key
            legend_rank = legend_rank_counter
            legend_rank_counter += 1
        else:
            legend_name = None
            legend_group = category_key
            legend_rank = None

        # Forward trajectory
        fig_3d.add_trace(go.Scatter3d(
            x=traj_fwd[:, 0], y=traj_fwd[:, 1], z=traj_fwd[:, 2],
            mode='lines',
            line=dict(color=color, width=width, dash="solid"),
            opacity=0.8,
            showlegend=show_legend,
            name=legend_name,
            legendgroup=legend_group,
            legendrank=legend_rank,
            hovertemplate=(
                f'Twist: {twist_val:.2f} turns<br>'
                f'β: {beta_val:.2f}<br>'
                f'Level: {twist_level}<br>'
                f'Reliability: {reliability}'
                f'<extra></extra>'
            )
        ))

        # Backward trajectory
        fig_3d.add_trace(go.Scatter3d(
            x=traj_bwd[:, 0], y=traj_bwd[:, 1], z=traj_bwd[:, 2],
            mode='lines',
            line=dict(color=color, width=width, dash="solid"),
            opacity=0.8,
            showlegend=False,
            legendgroup=legend_group,
            hovertemplate=(
                f'Twist: {twist_val:.2f} turns<br>'
                f'β: {beta_val:.2f}<br>'
                f'Level: {twist_level}<br>'
                f'Reliability: {reliability}'
                f'<extra></extra>'
            )
        ))

    # Print 3D statistics
    print(f"\n3D visualization statistics:")
    for twist_level in twist_order:
        count_level = sum(1 for t in trajectories_3d['twisted_closed'] if t['level'] == twist_level)
        if count_level > 0:
            print(f"  {twist_level.capitalize()} twist: {count_level} lines")
            for reliability in reliability_order:
                count_rel = sum(1 for t in trajectories_3d['twisted_closed']
                                if t['level'] == twist_level and t['reliability'] == reliability)
                if count_rel > 0:
                    print(f"    {reliability} confidence: {count_rel}")

    # Update layout
    fig_3d.update_layout(
        title=dict(
            text=f"{case.replace('_', ' ')}<br>Twisted Magnetic Flux Tubes (Closed Only)",
            x=0.5,
            xanchor='center',
            font=dict(size=14)
        ),
        scene=dict(
            xaxis=dict(title='X [km]', range=[-12 * RM, 4.5 * RM]),
            yaxis=dict(title='Y [km]', range=[-4.5 * RM, 4.5 * RM]),
            zaxis=dict(title='Z [km]', range=[-4.5 * RM, 4.5 * RM]),
            aspectmode='manual',
            aspectratio=dict(x=2.9, y=1.8, z=1.8),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=1200,
        height=900,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            traceorder='normal',
            font=dict(size=9)
        ),
        margin=dict(r=280, l=50, t=100, b=50)
    )

    # Save HTML
    output_3d_folder = os.path.join(output_folder, "3D_topology/")
    os.makedirs(output_3d_folder, exist_ok=True)

    html_file = os.path.join(output_3d_folder, f"{case}_3D_twist_reliability_{fname}_115000.html")
    fig_3d.write_html(html_file)
    print(f"\nSaved 3D visualization: {html_file}")

# ======================
# FINAL SUMMARY
# ======================
print(f"\n{'=' * 70}")
print("ANALYSIS COMPLETE!")
print("=" * 70)
print(f"Results saved to: {output_folder}")
print(f"  - CSV file: twist_analysis_{case}.csv")
if len(df) > 0:
    print(f"  - 2D plots: twist_analysis_{case}.png")
    if plot_3d_category:
        print(f"  - 3D visualization: 3D_topology/{case}_3D_twist_reliability_{fname}_115000.html")
print(f"{'=' * 70}\n")
