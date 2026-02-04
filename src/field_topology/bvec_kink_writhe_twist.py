#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Winding Number Analysis for Mercury's Magnetosphere

Calculate winding number (linking number) decomposed into twist and writhe
for magnetic flux ropes, tracking twist-to-writhe conversion during kink instability.

Based on Călugăreanu's theorem: Lk = Tw + Wr
where:
    Lk = linking number (conserved topological invariant)
    Tw = twist (internal field line winding)
    Wr = writhe (geometric axis deformation/kinking)

Key optimizations:
    - Numba JIT compilation for writhe calculation (~20-50x speedup)
    - Parallel execution for double integral
    - Combined field interpolation
    - Adaptive axis subsampling

References:
    https://doi.org/10.1098/rspa.2005.1527 (Călugăreanu theorem)
    https://doi.org/10.1098/rspa.2020.0483 (Magnetic winding)
"""
import os
import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy import stats
from numba import jit, prange
from src.field_topology.topology_utils import trace_field_line_rk, classify

# ======================
# USER PARAMETERS
# ======================
case = "CPS_Base_largerxdomain_smallergridsize"
input_folder = f"/Users/danywaller/Projects/mercury/extreme/{case}/out/"

# Parse case name for file naming
if "larger" in case:
    fname = case.split("_")[0] + "_" + case.split("_")[1]
else:
    fname = case

ncfile = os.path.join(input_folder, f"Amitis_{fname}_115000.nc")
output_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology_kinks/{case}/"
os.makedirs(output_folder, exist_ok=True)

# Mercury parameters
RM = 2440.0  # Mercury radius [km]
dx = 75.0  # Surface tolerance for closed field line detection [km]

# Field line tracing parameters
n_lat = 90  # Latitude resolution (60 = 3,600 seeds; 90 = 16,200 seeds)
n_lon = n_lat * 2  # Longitude resolution (2:1 aspect ratio)
max_steps = 100000  # Maximum RK4 integration steps
h_step = 50.0  # Integration step size [km]

# Kink detection thresholds where writhe above this indicates kinking [dimensionless]
writhe_threshold_incipient = 0.1  # Early kinking
writhe_threshold_moderate = 0.5  # Clear kink deformation
writhe_threshold_strong = 1.0  # Fully developed single helix
writhe_threshold_extreme = 1.5  # Multi-turn helix

writhe_threshold = writhe_threshold_incipient
initial_twist_min = 0.5  # Minimum twist to track (0.5 ≈ π radians)

# Optimization parameters
max_axis_points = 500  # Subsample axis to this length for writhe calculation

# ======================
# PHYSICAL CONSTANTS
# ======================
k_B = 1.380649e-23  # Boltzmann constant [J/K]
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space [H/m]
k_B_eV = 8.617333262e-5  # Boltzmann constant [eV/K]


# ======================
# OPTIMIZED HELPER FUNCTIONS (NUMBA ACCELERATED)
# ======================

@jit(nopython=True, fastmath=True)
def compute_flux_rope_axis_numba(trajectory, window_size):
    """
    Compute flux rope axis curve using moving average smoothing.

    The axis represents the central "spine" of the flux rope, obtained by
    smoothing the field line trajectory. This is used for writhe calculation.

    Parameters
    ----------
    trajectory : ndarray, shape (N, 3)
        Field line trajectory points in [km]
    window_size : int
        Smoothing window size (larger = smoother axis)

    Returns
    -------
    axis : ndarray, shape (M, 3)
        Smoothed axis curve where M = N - window_size + 1

    Notes
    -----
    Uses simple moving average convolution for computational efficiency.
    """
    n_points = len(trajectory)
    if n_points < window_size:
        return trajectory.copy()

    n_output = n_points - window_size + 1
    axis = np.empty((n_output, 3), dtype=np.float64)

    # Compute moving average for each dimension
    for dim in range(3):
        for i in range(n_output):
            axis[i, dim] = np.mean(trajectory[i:i + window_size, dim])

    return axis


@jit(nopython=True, fastmath=True, parallel=True)
def compute_writhe_numba(axis_curve):
    """
    Compute writhe using the Gauss double integral formula.

    Writhe (Wr) measures the geometric deformation/kinking of the flux rope
    axis. It's computed via:

        Wr = (1/4π) ∫∫ (r₁ - r₂) · (dr₁ × dr₂) / |r₁ - r₂|³

    where the integral is over all pairs of axis segments.

    Parameters
    ----------
    axis_curve : ndarray, shape (N, 3)
        Flux rope axis curve points in [km]

    Returns
    -------
    writhe : float
        Writhe value [dimensionless]
        - Wr ≈ 0: Straight or weakly deformed axis
        - |Wr| > 0.1: Significant kinking (geometric instability)
        - |Wr| > 1: Highly kinked structure

    Notes
    -----
    This is the computational bottleneck (O(N²) complexity).
    Optimizations:
        - Numba JIT compilation: ~20x speedup
        - Parallel execution: Additional 2-4x on multi-core CPU
        - fastmath=True: Relaxed IEEE compliance for speed

    The parallel loop distributes outer loop iterations across CPU cores.
    """
    n = len(axis_curve)
    if n < 3:
        return 0.0

    writhe = 0.0
    inv_4pi = 1.0 / (4.0 * np.pi)

    # Parallel outer loop over first curve segment
    for i in prange(n - 1):
        r1 = axis_curve[i]
        dr1 = axis_curve[i + 1] - r1

        local_sum = 0.0  # Thread-local accumulator

        # Inner loop over second curve segment (must start at i+2 to avoid singularities)
        for j in range(i + 2, n - 1):
            r2 = axis_curve[j]
            dr2 = axis_curve[j + 1] - r2

            # Vector from r2 to r1
            r12 = r1 - r2
            r12_norm_sq = r12[0] ** 2 + r12[1] ** 2 + r12[2] ** 2

            # Skip if points are too close (avoid numerical instability)
            if r12_norm_sq < 1e-12:
                continue

            r12_norm = np.sqrt(r12_norm_sq)

            # Cross product: dr1 × dr2
            cross_x = dr1[1] * dr2[2] - dr1[2] * dr2[1]
            cross_y = dr1[2] * dr2[0] - dr1[0] * dr2[2]
            cross_z = dr1[0] * dr2[1] - dr1[1] * dr2[0]

            # Dot product: r12 · (dr1 × dr2)
            dot = r12[0] * cross_x + r12[1] * cross_y + r12[2] * cross_z

            # Add to integrand
            local_sum += dot / (r12_norm_sq * r12_norm)

        writhe += local_sum

    return writhe * inv_4pi


@jit(nopython=True, fastmath=True)
def compute_twist_numba(trajectory, alpha_values):
    """
    Compute twist number from force-free parameter α.

    Twist (Tw) measures internal field line winding around the flux rope axis:

        Tw = (1/2π) ∫ α dl

    where α = (∇×B)·B / |B|² is the force-free parameter and dl is the
    line element along the trajectory.

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
    Uses trapezoidal integration along the field line.
    """
    n = len(trajectory)
    if n < 2:
        return 0.0

    twist_integral = 0.0

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
# WRAPPER FUNCTIONS (INTERFACE TO NUMBA)
# ======================

def compute_flux_rope_axis(trajectory):
    """
    Compute and subsample flux rope axis for writhe calculation.

    Parameters
    ----------
    trajectory : ndarray, shape (N, 3)
        Field line trajectory in [km]

    Returns
    -------
    axis : ndarray, shape (M, 3)
        Smoothed and subsampled axis curve
        M ≤ max_axis_points for computational efficiency

    Notes
    -----
    Subsampling to max_axis_points reduces writhe computation from
    O(N²) to O(max_axis_points²) with minimal accuracy loss.
    """
    # Smooth with moving average (window = 5% of trajectory length)
    window = max(5, len(trajectory) // 20)
    axis = compute_flux_rope_axis_numba(trajectory, window)

    # Subsample if too long (maintains accuracy while reducing computation)
    if len(axis) > max_axis_points:
        indices = np.linspace(0, len(axis) - 1, max_axis_points, dtype=int)
        axis = axis[indices]

    return axis


def compute_writhe(axis_curve):
    """
    Compute writhe using Numba-accelerated double integral.

    Parameters
    ----------
    axis_curve : ndarray, shape (N, 3)
        Flux rope axis curve in [km]

    Returns
    -------
    writhe : float
        Writhe value [dimensionless]
    """
    return compute_writhe_numba(axis_curve)


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

    Notes
    -----
    Uses scipy's RegularGridInterpolator for α values, then Numba
    for the line integral computation.
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


# ======================
# STANDARD FIELD CALCULATION FUNCTIONS
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
    curl_x, curl_y, curl_z : ndarray, shape (Nx, Ny, Nz)
        Curl components [nT/km]

    Notes
    -----
    Boundary points (first and last in each dimension) are set to zero.
    Interior points use centered difference: df/dx ≈ (f[i+1] - f[i-1])/(2*dx)
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

    In a force-free field, ∇ × B = α B, where α is constant along
    field lines. Non-zero α indicates twisted magnetic field structures.

    Parameters
    ----------
    Bx, By, Bz : ndarray, shape (Nx, Ny, Nz)
        Magnetic field components [nT]
    curl_x, curl_y, curl_z : ndarray, shape (Nx, Ny, Nz)
        Curl of B components [nT/km]

    Returns
    -------
    alpha : ndarray, shape (Nx, Ny, Nz)
        Force-free parameter [1/km]

    Notes
    -----
    Small |B|² values are clamped to avoid division by zero.
    α > 0: Right-handed twist; α < 0: Left-handed twist
    """
    B_mag_sq = Bx ** 2 + By ** 2 + Bz ** 2
    B_mag_sq = np.where(B_mag_sq < 1e-10, 1e-10, B_mag_sq)
    alpha = (curl_x * Bx + curl_y * By + curl_z * Bz) / B_mag_sq
    return alpha


def compute_linking_number(twist, writhe):
    """
    Compute linking number from Călugăreanu's theorem: Lk = Tw + Wr

    The linking number is a topological invariant that is conserved
    during ideal MHD evolution (no reconnection). During kink instability,
    twist converts to writhe while Lk remains constant.

    Parameters
    ----------
    twist : float
        Twist number [dimensionless]
    writhe : float
        Writhe number [dimensionless]

    Returns
    -------
    linking_number : float
        Linking number [dimensionless]

    Notes
    -----
    Initially for straight flux tube: Lk ≈ Tw, Wr ≈ 0
    During kinking: Tw decreases, Wr increases, Lk = constant
    """
    return twist + writhe


# ======================
# LOAD DATA
# ======================
print("=" * 70)
print(f"Twist + Writhe + Kink Analysis")
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
        # Convert spherical to Cartesian coordinates
        phi = np.radians(lat)
        theta = np.radians(lon)
        x_s = RM * np.cos(phi) * np.cos(theta)
        y_s = RM * np.cos(phi) * np.sin(theta)
        z_s = RM * np.sin(phi)
        seeds.append(np.array([x_s, y_s, z_s]))

seeds = np.array(seeds)
print(f"\nCreated seed points")
print(f"  Latitude resolution: {n_lat} points")
print(f"  Longitude resolution: {n_lon} points")
print(f"  Total seeds: {n_lat} × {n_lon} = {len(seeds)}")

# ======================
# COMPUTE THERMAL PRESSURE
# ======================
first_species = True

# Species temperatures from simulation input file
s_dict = {
    '1': 1.4e5,  # H+ solar wind protons [K]
    '2': 1.4e5,  # Additional H+ (if used) [K]
    '3': 5.6e5,  # Hot electrons or He++ alphas [K]
    '4': 5.6e5,  # Additional species (if used) [K]
}

for s in ['1', '2', '3', '4']:
    # Get temperature [K]
    T_K = s_dict[s]

    # Get density field [cm^-3]
    n_cm3 = ds[f'den0{s}'].isel(time=0).values
    n_transposed_cm3 = np.transpose(n_cm3, (2, 1, 0))

    # Convert cm^-3 to m^-3
    n_transposed = n_transposed_cm3 * 1e6  # 1 cm^-3 = 10^6 m^-3

    # Check if species is active
    if n_transposed.max() < 1e3:  # Less than 0.001 cm^-3
        print(f"\nSpecies 0{s}: INACTIVE (density < 0.001 cm^-3)")
        continue

    # Compute pressure: P = n * k_B * T [Pa]
    P_species_Pa = n_transposed * k_B * T_K
    P_species_nPa = P_species_Pa * 1e9  # Convert to nPa

    if first_species:
        P_total_Pa = P_species_Pa.copy()
        P_total_nPa = P_species_nPa.copy()
        first_species = False
    else:
        P_total_Pa += P_species_Pa
        P_total_nPa += P_species_nPa

    print(f"\nSpecies 0{s}:")
    print(f"  Temperature: {T_K:.2e} K ({T_K * k_B_eV:.2f} eV)")
    print(f"  Density range: [{n_transposed.min():.2e}, {n_transposed.max():.2e}] m^-3")
    print(f"  Pressure range: [{P_species_nPa.min():.3f}, {P_species_nPa.max():.3f}] nPa")

print(f"\n{'=' * 70}")
print(f"TOTAL THERMAL PRESSURE:")
print(f"  Min: {P_total_nPa.min():.3f} nPa")
print(f"  Max: {P_total_nPa.max():.3f} nPa")
print(f"  Mean: {P_total_nPa.mean():.3f} nPa")
print(f"{'=' * 70}\n")

# ======================
# PRESSURE GRADIENTS
# ======================
dnx = (x[1] - x[0]) * 1e3  # Convert km to m
dny = (y[1] - y[0]) * 1e3
dnz = (z[1] - z[0]) * 1e3

grad_P_x = np.gradient(P_total_Pa, dnx, axis=0)  # [Pa/m]
grad_P_y = np.gradient(P_total_Pa, dny, axis=1)
grad_P_z = np.gradient(P_total_Pa, dnz, axis=2)

grad_P_magnitude = np.sqrt(grad_P_x ** 2 + grad_P_y ** 2 + grad_P_z ** 2)

print(f"Pressure gradient magnitude:")
print(f"  Min: {grad_P_magnitude.min():.2e} Pa/m")
print(f"  Max: {grad_P_magnitude.max():.2e} Pa/m")
print(f"  Mean: {grad_P_magnitude.mean():.2e} Pa/m\n")

# ======================
# MAGNETIC FIELD
# ======================
Bx_plane = np.transpose(ds["Bx"].isel(time=0).values, (2, 1, 0))  # [nT]
By_plane = np.transpose(ds["By"].isel(time=0).values, (2, 1, 0))
Bz_plane = np.transpose(ds["Bz"].isel(time=0).values, (2, 1, 0))

# Convert nT to Tesla for pressure calculations
Bx_T = Bx_plane * 1e-9
By_T = By_plane * 1e-9
Bz_T = Bz_plane * 1e-9

B_magnitude_nT = np.sqrt(Bx_plane ** 2 + By_plane ** 2 + Bz_plane ** 2)
B_magnitude_T = B_magnitude_nT * 1e-9

print(f"Magnetic field magnitude:")
print(f"  Min: {B_magnitude_nT.min():.2f} nT")
print(f"  Max: {B_magnitude_nT.max():.2f} nT")
print(f"  Mean: {B_magnitude_nT.mean():.2f} nT\n")

# ======================
# CURRENT DENSITY
# ======================
Jx_nA = np.transpose(ds['Jx'].isel(time=0).values, (2, 1, 0))  # [nA/m^2]
Jy_nA = np.transpose(ds['Jy'].isel(time=0).values, (2, 1, 0))
Jz_nA = np.transpose(ds['Jz'].isel(time=0).values, (2, 1, 0))

# Convert nA/m^2 to A/m^2 for force calculations
Jx = Jx_nA * 1e-9  # [A/m^2]
Jy = Jy_nA * 1e-9
Jz = Jz_nA * 1e-9

J_magnitude_nA = np.sqrt(Jx_nA ** 2 + Jy_nA ** 2 + Jz_nA ** 2)

print(f"Current density magnitude:")
print(f"  Min: {J_magnitude_nA.min():.2f} nA/m^2")
print(f"  Max: {J_magnitude_nA.max():.2f} nA/m^2")
print(f"  Mean: {J_magnitude_nA.mean():.2f} nA/m^2\n")

# ======================
# J × B LORENTZ FORCE
# ======================
JcrossB_x = Jy * Bz_T - Jz * By_T  # [N/m^3]
JcrossB_y = Jz * Bx_T - Jx * Bz_T
JcrossB_z = Jx * By_T - Jy * Bx_T

JcrossB_magnitude = np.sqrt(JcrossB_x ** 2 + JcrossB_y ** 2 + JcrossB_z ** 2)

# Express in nPa/R_M for Mercury magnetosphere context
R_M_meters = RM * 1e3
JcrossB_nPa_per_RM = JcrossB_magnitude * R_M_meters * 1e9

print(f"Lorentz force |J × B|:")
print(f"  Min: {JcrossB_magnitude.min():.2e} N/m^3 ({JcrossB_nPa_per_RM.min():.3f} nPa/R_M)")
print(f"  Max: {JcrossB_magnitude.max():.2e} N/m^3 ({JcrossB_nPa_per_RM.max():.3f} nPa/R_M)")
print(f"  Mean: {JcrossB_magnitude.mean():.2e} N/m^3 ({JcrossB_nPa_per_RM.mean():.3f} nPa/R_M)\n")

# ======================
# NET FORCE (J × B - ∇P)
# ======================
net_force_x = JcrossB_x - grad_P_x
net_force_y = JcrossB_y - grad_P_y
net_force_z = JcrossB_z - grad_P_z

net_force_magnitude = np.sqrt(net_force_x ** 2 + net_force_y ** 2 + net_force_z ** 2)
net_force_nPa_per_RM = net_force_magnitude * R_M_meters * 1e9

print(f"Net force |J × B - ∇P|:")
print(f"  Min: {net_force_magnitude.min():.2e} N/m^3")
print(f"  Max: {net_force_magnitude.max():.2e} N/m^3 ({net_force_nPa_per_RM.max():.3f} nPa/R_M)")
print(f"  Mean: {net_force_magnitude.mean():.2e} N/m^3 ({net_force_nPa_per_RM.mean():.3f} nPa/R_M)\n")

# ======================
# PLASMA BETA
# ======================
magnetic_pressure_Pa = B_magnitude_T ** 2 / (2 * mu_0)
magnetic_pressure_nPa = magnetic_pressure_Pa * 1e9

# Avoid division by zero
magnetic_pressure_Pa_safe = np.where(magnetic_pressure_Pa < 1e-30, 1e-30, magnetic_pressure_Pa)
plasma_beta = P_total_Pa / magnetic_pressure_Pa_safe

print(f"\n{'=' * 70}")
print(f"Plasma β = P_thermal / P_magnetic")
print(f"{'=' * 70}")
print(f"Magnetic pressure:")
print(f"  Min: {magnetic_pressure_nPa.min():.3f} nPa")
print(f"  Max: {magnetic_pressure_nPa.max():.3f} nPa")
print(f"  Mean: {magnetic_pressure_nPa.mean():.3f} nPa")

print(f"\nPlasma β:")
print(f"  Min: {plasma_beta.min():.2e}")
print(f"  Max: {plasma_beta.max():.2e}")
print(f"  Mean: {plasma_beta.mean():.2e}")
print(f"  Median: {np.median(plasma_beta):.2e}")

# Statistics on beta distribution (mutually exclusive)
beta_very_low = np.sum(plasma_beta < 0.1) / plasma_beta.size * 100
beta_low = np.sum((plasma_beta >= 0.1) & (plasma_beta < 1.0)) / plasma_beta.size * 100
beta_moderate = np.sum((plasma_beta >= 1.0) & (plasma_beta < 10.0)) / plasma_beta.size * 100
beta_high = np.sum(plasma_beta >= 10.0) / plasma_beta.size * 100

print(f"\nPlasma β distribution:")
print(f"  β < 0.1 (magnetic-dominated):     {beta_very_low:.1f}%")
print(f"  0.1 ≤ β < 1.0 (plasma sheet):     {beta_low:.1f}%")
print(f"  1.0 ≤ β < 10 (pressure-dominated): {beta_moderate:.1f}%")
print(f"  β ≥ 10 (highly non-force-free):    {beta_high:.1f}%")
# print(f"  Total: {beta_very_low + beta_low + beta_moderate + beta_high:.1f}%")

# ======================
# IMPROVED FORCE BALANCE METRIC
# ======================
# Dimensionless pressure contribution to force balance
force_sum = JcrossB_magnitude + grad_P_magnitude + 1e-20  # Avoid zero
pressure_ratio = grad_P_magnitude / force_sum

print(f"\n{'=' * 70}")
print(f"Pressure force fraction |∇P| / (|J×B| + |∇P|)")
print(f"{'=' * 70}")
print(f"Pressure ratio:")
print(f"  Mean: {pressure_ratio.mean():.2e}")
print(f"  Median: {np.median(pressure_ratio):.2e}")
print(f"  Max: {pressure_ratio.max():.2e}")

# Statistics on force-free quality
ff_excellent = np.sum(pressure_ratio < 0.01) / pressure_ratio.size * 100
ff_good = np.sum(pressure_ratio < 0.1) / pressure_ratio.size * 100
ff_fair = np.sum((pressure_ratio >= 0.1) & (pressure_ratio < 0.5)) / pressure_ratio.size * 100
ff_poor = np.sum(pressure_ratio >= 0.5) / pressure_ratio.size * 100

print(f"\nForce-free approximation quality:")
print(f"  Excellent (pressure < 1% of forces): {ff_excellent:.1f}%")
print(f"  Good (pressure < 10% of forces): {ff_good:.1f}%")
print(f"  Fair (10% ≤ pressure < 50%): {ff_fair:.1f}%")
print(f"  Poor (pressure ≥ 50% of forces): {ff_poor:.1f}%")

if 0:
    print(f"\nInterpretation:")
    if ff_good > 90:
        print("  EXCELLENT: Domain is strongly force-free - twist analysis highly reliable")
    elif ff_good > 70:
        print("  GOOD: Mostly force-free - twist analysis reliable in most regions")
    elif ff_good > 50:
        print("  FAIR: Moderately force-free - filter results by β and pressure ratio")
    else:
        print("  POOR: Non-force-free - use β-dependent analysis and interpret cautiously")

# ======================
# DOUBLE-GRADIENT INSTABILITY CHECK
# ======================
grad_Bz_x = np.gradient(Bz_plane, dnx, axis=0)
grad_Bx_z = np.gradient(Bx_plane, dnz, axis=2)
instability_param = grad_Bz_x * grad_Bx_z

unstable_fraction = np.sum(instability_param < 0) / instability_param.size * 100

print(f"\n{'=' * 70}")
print(f"Double-Gradient Instability")
print(f"{'=' * 70}")
print(f"Fraction with ∇_x(B_z) * ∇_z(B_x) < 0: {unstable_fraction:.1f}%")
if unstable_fraction > 50:
    print("  HIGH: Strong instability!")
elif unstable_fraction > 20:
    print("  MODERATE: Partial instability - localized unstable regions")
else:
    print("  LOW: Mostly stable current sheet configuration")

# ======================
# COMPUTE ALPHA FIELD (FORCE-FREE PARAMETER)
# ======================
print(f"\n{'=' * 70}")
print("Force-Free Parameter")
print(f"{'=' * 70}")

print("Computing curl of B field...")
curl_x, curl_y, curl_z = compute_curl_B(Bx_plane, By_plane, Bz_plane, x, y, z)

print("Computing α = (∇×B)·B / |B|²...")
alpha_field = compute_alpha_field(Bx_plane, By_plane, Bz_plane, curl_x, curl_y, curl_z)

print(f"α field statistics:")
print(f"  Min: {alpha_field.min():.3e} km^-1")
print(f"  Max: {alpha_field.max():.3e} km^-1")
print(f"  Mean: {alpha_field.mean():.3e} km^-1")

# ======================
# CREATE COMBINED INTERPOLATOR
# ======================
print(f"\n{'=' * 70}")
print("CREATING VECTORIZED INTERPOLATOR")
print(f"{'=' * 70}")

# Stack all fields into a single array for efficient interpolation
combined_fields = np.stack([
    plasma_beta,
    pressure_ratio,
    P_total_nPa,
    magnetic_pressure_nPa
], axis=-1)  # Shape: (Nx, Ny, Nz, 4)

interp_combined = RegularGridInterpolator(
    (x, y, z),
    combined_fields,
    bounds_error=False,
    fill_value=0
)

print("Created combined interpolator for force balance analysis")
print("  Fields: [beta, pressure_ratio, P_thermal, P_magnetic]")


def analyze_force_balance_along_fieldline(trajectory):
    """
    Analyze force balance parameters along a field line trajectory.

    Uses vectorized interpolation to sample all 4 fields simultaneously

    Parameters
    ----------
    trajectory : ndarray, shape (N, 3)
        Field line trajectory points in [km]

    Returns
    -------
    stats : dict
        Dictionary containing:
        - beta_mean, beta_max, beta_min: Plasma beta statistics
        - pressure_mean_nPa, pressure_max_nPa: Thermal pressure [nPa]
        - mag_pressure_mean_nPa: Magnetic pressure [nPa]
        - pressure_ratio_mean, pressure_ratio_max: Pressure contribution to forces
        - high_beta_fraction: Fraction of trajectory with β > 1
        - non_force_free_fraction: Fraction with pressure_ratio > 0.5
    """
    # Single interpolation call returns all 4 fields
    all_vals = interp_combined(trajectory)  # Shape: (N_points, 4)

    beta_vals = all_vals[:, 0]
    pressure_ratio_vals = all_vals[:, 1]
    pressure_vals = all_vals[:, 2]
    mag_pressure_vals = all_vals[:, 3]

    n_points = len(beta_vals)

    return {
        'beta_mean': np.mean(beta_vals),
        'beta_max': np.max(beta_vals),
        'beta_min': np.min(beta_vals),
        'pressure_mean_nPa': np.mean(pressure_vals),
        'pressure_max_nPa': np.max(pressure_vals),
        'mag_pressure_mean_nPa': np.mean(mag_pressure_vals),
        'pressure_ratio_mean': np.mean(pressure_ratio_vals),
        'pressure_ratio_max': np.max(pressure_ratio_vals),
        'high_beta_fraction': np.count_nonzero(beta_vals > 1.0) / n_points,
        'non_force_free_fraction': np.count_nonzero(pressure_ratio_vals > 0.5) / n_points
    }

# ======================
# WARM UP NUMBA JIT (COMPILE FUNCTIONS)
# ======================

# Trigger JIT compilation with dummy data
dummy_curve = np.random.rand(100, 3).astype(np.float64)
dummy_alpha = np.random.rand(100).astype(np.float64)

_ = compute_writhe_numba(dummy_curve)
_ = compute_twist_numba(dummy_curve, dummy_alpha)
_ = compute_flux_rope_axis_numba(dummy_curve, 5)

# ======================
# ANALYZE FLUX ROPES (MAIN LOOP)
# ======================
print(f"\n{'=' * 70}")
print("TRACING AND ANALYZING FIELD LINES")
print(f"{'=' * 70}")
print("JIT compilation complete!")
print(f"Configuration:")
print(f"  Max integration steps: {max_steps}")
print(f"  Step size: {h_step} km")
print(f"  Writhe threshold: {writhe_threshold}")
print(f"  Minimum twist: {initial_twist_min}")
print(f"  Max axis points for writhe: {max_axis_points}")
# print(f"{'=' * 70}\n")

flux_rope_data = []
classification_counts = {'closed': 0, 'open': 0, 'solar_wind': 0}
start_time = time.time()

for i, seed in enumerate(seeds):
    # Trace field line forward and backward from seed point
    traj_fwd, exit_fwd_y = trace_field_line_rk(
        seed, Bx_plane, By_plane, Bz_plane,
        x, y, z, RM, max_steps=max_steps, h=h_step
    )
    traj_bwd, exit_bwd_y = trace_field_line_rk(
        seed, Bx_plane, By_plane, Bz_plane,
        x, y, z, RM, max_steps=max_steps, h=-h_step
    )

    # Classify field line using topology_utils.classify
    field_line_type = classify(traj_fwd, traj_bwd, RM, exit_fwd_y, exit_bwd_y)
    classification_counts[field_line_type] += 1

    # Only analyze closed field lines for winding number
    if field_line_type != "closed":
        continue

    # Combine forward and backward trajectories
    full_trajectory = np.vstack([traj_bwd[::-1], traj_fwd])

    # Compute flux rope axis (smoothed and subsampled)
    axis_curve = compute_flux_rope_axis(full_trajectory)

    # Compute twist number (Numba-accelerated)
    twist = compute_twist_from_alpha(full_trajectory, alpha_field, x, y, z)

    # Compute writhe (Numba-accelerated)
    writhe = compute_writhe(axis_curve)

    # Compute linking number (conserved topological invariant)
    linking_number = compute_linking_number(twist, writhe)

    # Analyze force balance along field line
    force_stats = analyze_force_balance_along_fieldline(full_trajectory)

    # Pressure-modified writhe threshold
    # High beta reduces stability, so reduce threshold
    effective_threshold = writhe_threshold * (1 + force_stats['beta_mean']) ** (-0.5)

    # Compute twist and writhe fractions of linking number
    twist_fraction = twist / linking_number if abs(linking_number) > 0.01 else 0
    writhe_fraction = writhe / linking_number if abs(linking_number) > 0.01 else 0

    # Flag if kink instability is active
    kink_active = abs(writhe) > effective_threshold and abs(twist) > initial_twist_min

    # Store results
    flux_rope_data.append({
        'seed_x': seed[0],
        'seed_y': seed[1],
        'seed_z': seed[2],
        'field_line_type': field_line_type,
        'twist': twist,
        'writhe': writhe,
        'linking_number': linking_number,
        'twist_fraction': twist_fraction,
        'writhe_fraction': writhe_fraction,
        'kink_active': kink_active,
        'axis_length': len(axis_curve),
        'trajectory_length': len(full_trajectory),
        'beta_mean': force_stats['beta_mean'],
        'beta_max': force_stats['beta_max'],
        'pressure_mean_nPa': force_stats['pressure_mean_nPa'],
        'pressure_ratio_mean': force_stats['pressure_ratio_mean'],  # CHANGED
        'high_beta_fraction': force_stats['high_beta_fraction'],
        'non_force_free_fraction': force_stats['non_force_free_fraction'],  # CHANGED
    })

    # Progress reporting
    if (i + 1) % 100 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        remaining = (len(seeds) - i - 1) / rate if rate > 0 else 0
        print(f"Processed {i + 1}/{len(seeds)} seeds | ETA: {remaining / 60:.1f} min")

total_time = time.time() - start_time
print(f"\n{'=' * 70}")
print(f"FIELD LINE TRACING COMPLETE")
print(f"{'=' * 70}")
print(f"Total time: {total_time / 60:.2f} minutes")
print(f"Average rate: {len(seeds) / total_time:.1f} seeds/second")
print(f"\nField line classification:")
print(f"  Closed:      {classification_counts['closed']:5d} ({100 * classification_counts['closed'] / len(seeds):5.1f}%)")
print(f"  Open:        {classification_counts['open']:5d} ({100 * classification_counts['open'] / len(seeds):5.1f}%)")
print(f"  Solar wind:  {classification_counts['solar_wind']:5d} ({100 * classification_counts['solar_wind'] / len(seeds):5.1f}%)")

# ======================
# SAVE RESULTS
# ======================
# print(f"\n{'=' * 70}")
# print("SAVING RESULTS")
# print(f"{'=' * 70}")

df = pd.DataFrame(flux_rope_data)
output_csv = os.path.join(output_folder, f"winding_number_analysis_{case}.csv")
df.to_csv(output_csv, index=False)
# print(f"Saved CSV: {output_csv}")
# print(f"  Rows: {len(df)}")
# print(f"  Columns: {len(df.columns)}")

# ======================
# SUMMARY STATISTICS
# ======================
print("\n" + "=" * 70)
print("Twist + Writhe + Kink Analysis Summary")
print("=" * 70)

if len(df) > 0:
    print(f"\nTotal closed field lines analyzed: {len(df)}")

    print(f"\nLinking number (Lk) statistics:")
    print(f"  Mean: {df['linking_number'].mean():.3f}")
    print(f"  Std:  {df['linking_number'].std():.3f}")
    print(f"  Range: [{df['linking_number'].min():.3f}, {df['linking_number'].max():.3f}]")

    print(f"\nTwist (Tw) statistics:")
    print(f"  Mean: {df['twist'].mean():.3f}")
    print(f"  Std:  {df['twist'].std():.3f}")
    print(f"  Range: [{df['twist'].min():.3f}, {df['twist'].max():.3f}]")

    print(f"\nWrithe (Wr) statistics:")
    print(f"  Mean: {df['writhe'].mean():.3f}")
    print(f"  Std:  {df['writhe'].std():.3f}")
    print(f"  Range: [{df['writhe'].min():.3f}, {df['writhe'].max():.3f}]")

    kink_count = df['kink_active'].sum()
    print(f"\nKink instability detection:")
    print(f"  Active kink (|Wr| > {writhe_threshold}): {kink_count}/{len(df)} ({100 * kink_count / len(df):.1f}%)")

    if kink_count > 0:
        kink_df = df[df['kink_active']]
        print(f"\n  For kinked flux ropes:")
        print(f"    Average twist fraction: {kink_df['twist_fraction'].mean():.3f}")
        print(f"    Average writhe fraction: {kink_df['writhe_fraction'].mean():.3f}")
        print(f"    Twist-to-writhe conversion: {kink_df['writhe_fraction'].mean() * 100:.1f}% of Lk is now writhe")

    # ======================
    # FORCE-FREE VALIDITY ASSESSMENT
    # ======================
    print(f"\n{'=' * 70}")
    print("FORCE-FREE APPROXIMATION VALIDITY FOR CLOSED FIELD LINES")
    print(f"{'=' * 70}")

    good_ff = df[df['pressure_ratio_mean'] < 0.1]
    print(f"\nForce-free quality along closed field lines:")
    print(
        f"  Approximately force-free (pressure < 10%): {len(good_ff)}/{len(df)} ({100 * len(good_ff) / len(df):.1f}%)")
    print(f"  Mean β: {df['beta_mean'].mean():.2f}")
    print(f"  Mean pressure ratio: {df['pressure_ratio_mean'].mean():.2f}")

    if len(good_ff) > 0:
        print(f"\n  Statistics for force-free flux ropes:")
        print(f"    Mean β: {good_ff['beta_mean'].mean():.2f}")
        print(f"    Mean |twist|: {good_ff['twist'].abs().mean():.2f}")
        print(f"    Mean |writhe|: {good_ff['writhe'].abs().mean():.2f}")

    # ======================
    # PRESSURE-MODIFIED KINK ANALYSIS
    # ======================
    print("\n" + "=" * 70)
    print("PRESSURE EFFECTS ON KINK INSTABILITY")
    print("=" * 70)

    # Separate by beta regime
    low_beta = df[df['beta_mean'] < 0.1]
    mid_beta = df[(df['beta_mean'] >= 0.1) & (df['beta_mean'] < 1.0)]
    high_beta = df[df['beta_mean'] >= 1.0]

    print(f"\nField line distribution by β regime:")
    print(f"  Low β (< 0.1):  {len(low_beta):4d} lines ({100 * len(low_beta) / len(df):5.1f}%)")
    print(f"  Mid β (0.1-1):  {len(mid_beta):4d} lines ({100 * len(mid_beta) / len(df):5.1f}%)")
    print(f"  High β (> 1.0): {len(high_beta):4d} lines ({100 * len(high_beta) / len(df):5.1f}%)")

    # Kink rates by beta regime
    for subset, name in [(low_beta, "Low β (< 0.1)"),
                         (mid_beta, "Mid β (0.1-1.0)"),
                         (high_beta, "High β (> 1.0)")]:
        if len(subset) > 0:
            kink_frac = subset['kink_active'].sum() / len(subset) * 100
            mean_twist = subset['twist'].abs().mean()
            mean_writhe = subset['writhe'].abs().mean()
            print(f"\n  {name}:")
            print(f"    Kink fraction: {kink_frac:.1f}%")
            print(f"    Mean |twist|: {mean_twist:.3f}")
            print(f"    Mean |writhe|: {mean_writhe:.3f}")

    # Correlation analysis
    if kink_count > 5:
        print(f"\n{'=' * 70}")
        print("CORRELATION ANALYSIS")
        print(f"{'=' * 70}")

        kinked = df[df['kink_active']]
        non_kinked = df[~df['kink_active']]

        print(f"\nKinked vs non-kinked field lines:")
        print(f"  Mean β (kinked):     {kinked['beta_mean'].mean():.3f}")
        print(f"  Mean β (non-kinked): {non_kinked['beta_mean'].mean():.3f}")
        print(f"\n  Mean |twist| (kinked):     {kinked['twist'].abs().mean():.3f}")
        print(f"  Mean |twist| (non-kinked): {non_kinked['twist'].abs().mean():.3f}")

        # Pearson correlation between beta and kinking
        corr, pval = stats.pearsonr(df['beta_mean'], df['kink_active'])
        print(f"\n  Pearson correlation (β vs kink activity):")
        print(f"    r = {corr:.3f}")
        print(f"    p-value = {pval:.3e}")
        if pval < 0.05:
            print(f"    *** Statistically significant correlation ***")

else:
    print("\nWARNING: No closed field lines found!")
    print("This may indicate:")
    print("  - Domain does not contain closed field region")
    print("  - Seeds placed outside closed field line region")
    print("  - Integration parameters need adjustment")

# ======================
# VISUALIZATION
# ======================
# print(f"\n{'=' * 70}")
# print("CREATING VISUALIZATIONS")
# print(f"{'=' * 70}")

if len(df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ==================
    # Plot 1: Twist vs Writhe (Călugăreanu decomposition)
    # ==================
    scatter1 = axes[0, 0].scatter(
        df['twist'], df['writhe'],
        c=df['linking_number'],
        s=20, alpha=0.6, cmap='viridis'
    )
    axes[0, 0].set_xlabel('Twist (Tw)', fontsize=11)
    axes[0, 0].set_ylabel('Writhe (Wr)', fontsize=11)
    axes[0, 0].set_title('Călugăreanu Decomposition: Lk = Tw + Wr', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    axes[0, 0].axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('Linking Number (Lk)', fontsize=10)

    # ==================
    # Plot 2: Linking number distribution
    # ==================
    axes[0, 1].hist(df['linking_number'], bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0, 1].set_xlabel('Linking Number (Lk)', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Linking Number Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].axvline(x=df['linking_number'].mean(), color='r', linestyle='--',
                       label=f'Mean = {df["linking_number"].mean():.2f}', linewidth=2)
    axes[0, 1].legend()

    # ==================
    # Plot 3: Twist/Writhe fractions for kinked ropes
    # ==================
    kink_count = df['kink_active'].sum()
    if kink_count > 0:
        kink_df = df[df['kink_active']].head(50)  # Show first 50 for clarity
        x_pos = np.arange(len(kink_df))

        axes[1, 0].bar(x_pos, kink_df['twist_fraction'].abs(),
                       alpha=0.7, label='Twist fraction', color='steelblue')
        axes[1, 0].bar(x_pos, kink_df['writhe_fraction'].abs(),
                       bottom=kink_df['twist_fraction'].abs(),
                       alpha=0.7, label='Writhe fraction', color='coral')
        axes[1, 0].set_xlabel('Kinked Flux Rope Index', fontsize=11)
        axes[1, 0].set_ylabel('Fraction of |Linking Number|', fontsize=11)
        axes[1, 0].set_title(
            f'Twist-to-Writhe Conversion\n({min(50, kink_count)} of {kink_count} kinked ropes)',
            fontsize=12, fontweight='bold'
        )
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim(0, 3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No kinked flux ropes detected',
                        ha='center', va='center', transform=axes[1, 0].transAxes,
                        fontsize=14, color='red')
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)

    # ==================
    # Plot 4: Writhe vs Twist (colored by beta)
    # ==================
    scatter2 = axes[1, 1].scatter(
        np.abs(df['twist']), np.abs(df['writhe']),
        c=df['beta_mean'], s=20, alpha=0.6,
        cmap='plasma',
        norm=plt.matplotlib.colors.LogNorm(vmin=max(df['beta_mean'].min(), 1e-3))
    )
    axes[1, 1].set_xlabel('|Twist|', fontsize=11)
    axes[1, 1].set_ylabel('|Writhe|', fontsize=11)
    axes[1, 1].set_title('Writhe vs Twist (colored by β)', fontsize=12, fontweight='bold')
    axes[1, 1].axhline(y=writhe_threshold, color='r', linestyle='--',
                       label=f'Writhe threshold = {writhe_threshold}', linewidth=2)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=axes[1, 1])
    cbar2.set_label('Plasma β (mean)', fontsize=10)

    plt.tight_layout()
    plot_file = os.path.join(output_folder, f"winding_number_analysis_{case}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    # print(f"Saved figure: {plot_file}")
    plt.close()

print(f"\n{'=' * 70}")
print("ANALYSIS COMPLETE!")
print(f"{'=' * 70}")
print(f"Results saved to: {output_folder}")
print(f"  - CSV file: winding_number_analysis_{case}.csv")
print(f"  - Figure: winding_number_analysis_{case}.png")
print(f"{'=' * 70}\n")
