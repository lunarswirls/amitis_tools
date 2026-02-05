#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports:
import numpy as np
import pandas as pd


def compute_radial_flux(all_particles_filename, sim_dx, sim_dy, sim_dz,
                        sim_ppc, sim_den, spec_map, R_M, select_R,
                        species="all", n_lat=180, n_lon=360):
    """
    Compute particle counts, density, radial velocity, and surface flux in
    spherical coordinates for particles impacting the surface at r = R_M.

    Parameters
    ----------
    all_particles_filename : str
        Numpy .npz file containing all particles.
    sim_dx, sim_dy, sim_dz : float
        Simulation grid cell resolution [m].
    sim_ppc : array-like
        Macroparticles per simulation grid cell per species.
    sim_den : array-like
        Physical number density per species in the upstream region [m^-3].
    spec_map : array-like
        Name of each species.
    R_M : float
        Planet radius [m].
    select_R : float
        Outer radius for shell selection [m].
    species : {"all", "protons", "alphas"}
        Which species to use.
    n_lat, n_lon : int
        Number of latitude and longitude bins.

    Returns
    -------
    flux_map_cm : ndarray (n_lat, n_lon)
        Surface precipitation flux [cm^-2 s^-1].
    lat_centers : ndarray
        Latitude bin centers [deg].
    lon_centers : ndarray
        Longitude bin centers [deg].
    v_r_map : ndarray (n_lat, n_lon)
        Mass-weighted impact velocity [km/s]. Negative = inward.
    count_map : ndarray (n_lat, n_lon)
        Physical particle count in shell [# particles].
    den_map_cm3 : ndarray (n_lat, n_lon)
        Shell volume-averaged number density [cm^-3].
    """

    # ========== Load particle data ==========
    with np.load(all_particles_filename) as data:
        prx = data["prx"]  # [m]
        pry = data["pry"]  # [m]
        prz = data["prz"]  # [m]
        pvx = data["pvx"]  # [m/s]
        pvy = data["pvy"]  # [m/s]
        pvz = data["pvz"]  # [m/s]
        psid = data["psid"].astype(int)
        num_files = data["num_files"]
        selected_radius = data["selected_radius"]  # [m]

    if selected_radius < select_R:
        raise ValueError(f"Selected radius {selected_radius:e} < select_R {select_R:e}")

    sim_ppc = np.asarray(sim_ppc, dtype=float)
    sim_den = np.asarray(sim_den, dtype=float)

    # Species properties (adjust to match your spec_map ordering)
    species_mass = np.array([1.0, 1.0, 4.0, 4.0])  # [amu]
    species_charge = np.array([1.0, 1.0, 2.0, 2.0])  # [e]

    # ========== Geometric setup ==========
    # Angular binning
    lon_edges = np.linspace(-180.0, 180.0, n_lon + 1)  # [deg]
    lat_edges = np.linspace(-90.0, 90.0, n_lat + 1)  # [deg]
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    dlon = np.radians(lon_edges[1] - lon_edges[0])  # [rad]

    # Surface area per (lat,lon) bin on sphere radius R_M [m^2]
    area = R_M ** 2 * dlon * (np.sin(np.radians(lat_edges[1:])) -
                              np.sin(np.radians(lat_edges[:-1])))
    area = area[:, None]  # shape (n_lat, 1) for broadcasting

    # Shell volume per (lat,lon) bin [m^3]
    dR = select_R - R_M  # [m]
    shell_volume = area * dR  # [m^3]

    # ========== Per-species processing ==========
    def process_species(spec_id):
        """
        Process a single species and return maps.

        Returns
        -------
        macro_count_map : Macroparticle count in shell [# macroparticles]
        count_map : Physical particle count in shell [# particles]
        flux_contribution : Flux contribution Σ(w_i * |v_r,i|) [# particles * m/s]
        momentum_contribution : Mass-weighted momentum Σ(m * w_i * v_r,i) [amu * # particles * m/s]
        mass_count : Mass-weighted count Σ(m * w_i) [amu * # particles]
        """
        print(f"Processing species {spec_map[spec_id]}")

        # ========== 1) Select species ==========
        mask = psid == spec_id
        prx_s = prx[mask]
        pry_s = pry[mask]
        prz_s = prz[mask]
        pvx_s = pvx[mask]
        pvy_s = pvy[mask]
        pvz_s = pvz[mask]

        print(f"  Initial macroparticles: {len(prx_s)}")

        # ========== 2) Shell selection: R_M <= r <= select_R ==========
        r = np.sqrt(prx_s ** 2 + pry_s ** 2 + prz_s ** 2)  # [m]
        shell_mask = (r >= R_M) & (r <= select_R)

        prx_s = prx_s[shell_mask]
        pry_s = pry_s[shell_mask]
        prz_s = prz_s[shell_mask]
        pvx_s = pvx_s[shell_mask]
        pvy_s = pvy_s[shell_mask]
        pvz_s = pvz_s[shell_mask]
        r = r[shell_mask]

        print(f"  After shell selection: {len(r)}")

        if len(r) == 0:
            empty = np.zeros((n_lat, n_lon))
            return empty, empty, empty, empty, empty

        # ========== 3) Compute radial velocity ==========
        r_hat_x = prx_s / r
        r_hat_y = pry_s / r
        r_hat_z = prz_s / r
        v_r = pvx_s * r_hat_x + pvy_s * r_hat_y + pvz_s * r_hat_z  # [m/s]

        # ========== 4) Select inward trajectories ==========
        inward_mask = v_r < 0.0

        prx_s = prx_s[inward_mask]
        pry_s = pry_s[inward_mask]
        prz_s = prz_s[inward_mask]
        pvx_s = pvx_s[inward_mask]
        pvy_s = pvy_s[inward_mask]
        pvz_s = pvz_s[inward_mask]
        v_r = v_r[inward_mask]

        print(f"  After inward selection: {len(v_r)}")
        print(f"  v_r range: [{v_r.min():.2e}, {v_r.max():.2e}] m/s")

        if len(v_r) == 0:
            empty = np.zeros((n_lat, n_lon))
            return empty, empty, empty, empty, empty

        # ========== 5) Ray-sphere intersection to find surface impact points ==========
        r_vec = np.column_stack([prx_s, pry_s, prz_s])  # [m] shape (N, 3)
        v_vec = np.column_stack([pvx_s, pvy_s, pvz_s])  # [m/s] shape (N, 3)

        # Solve |r0 + t*v|^2 = R_M^2
        a = np.sum(v_vec ** 2, axis=1)  # [m^2/s^2]
        b = 2.0 * np.sum(r_vec * v_vec, axis=1)  # [m^2/s]
        c = np.sum(r_vec ** 2, axis=1) - R_M ** 2  # [m^2]

        disc = b ** 2 - 4 * a * c
        valid_intersection = disc > 0.0

        print(f"  After intersection check: {valid_intersection.sum()}")

        if valid_intersection.sum() == 0:
            empty = np.zeros((n_lat, n_lon))
            return empty, empty, empty, empty, empty

        # Apply intersection filter
        r_vec = r_vec[valid_intersection]
        v_vec = v_vec[valid_intersection]
        v_r = v_r[valid_intersection]
        a = a[valid_intersection]
        b = b[valid_intersection]
        disc = disc[valid_intersection]

        # Time to impact [s]
        t_impact = (-b - np.sqrt(disc)) / (2 * a)

        # Impact position on surface [m]
        impact_pos = r_vec + v_vec * t_impact[:, np.newaxis]
        x_impact = impact_pos[:, 0]
        y_impact = impact_pos[:, 1]
        z_impact = impact_pos[:, 2]

        # ========== 6) Convert to geographic coordinates ==========
        lon_impact = np.degrees(np.arctan2(y_impact, x_impact))  # [-180, 180] deg
        lat_impact = np.degrees(np.arcsin(z_impact / R_M))  # [-90, 90] deg

        # ========== 7) Compute physical weights ==========
        # Each macroparticle represents w physical particles
        cell_volume = sim_dx * sim_dy * sim_dz  # [m^3]
        w = (sim_den[spec_id] * cell_volume) / (sim_ppc[spec_id] * num_files)  # [# particles]

        weights = w * np.ones(len(lat_impact))  # [# particles per macroparticle]

        print(f"  Macroparticle weight: {w:.2e} physical particles")
        print(f"  Total macroparticles: {len(lat_impact)}")
        print(f"  Total physical particles: {weights.sum():.2e}\n")

        # ========== 8) Histogram onto surface bins ==========
        # Macroparticle count: number of simulation macroparticles
        macro_count_map, _, _ = np.histogram2d(
            lat_impact, lon_impact,
            bins=[lat_edges, lon_edges],
            weights=None  # [# macroparticles]
        )

        # Physical particle count: total number of physical particles
        count_map, _, _ = np.histogram2d(
            lat_impact, lon_impact,
            bins=[lat_edges, lon_edges],
            weights=weights  # [# particles]
        )

        # Flux contribution: Σ(w_i * |v_r,i|) for computing flux
        # Note: v_r is negative, so use -v_r for magnitude
        flux_contribution, _, _ = np.histogram2d(
            lat_impact, lon_impact,
            bins=[lat_edges, lon_edges],
            weights=weights * (-v_r)  # [# particles * m/s]
        )

        # For velocity: need mass-weighted sums
        m = species_mass[spec_id]  # [amu]

        # Mass-weighted momentum: Σ(m * w_i * v_r,i)
        momentum_contribution, _, _ = np.histogram2d(
            lat_impact, lon_impact,
            bins=[lat_edges, lon_edges],
            weights=m * weights * v_r  # [amu * # particles * m/s]
        )

        # Mass-weighted count: Σ(m * w_i)
        mass_count, _, _ = np.histogram2d(
            lat_impact, lon_impact,
            bins=[lat_edges, lon_edges],
            weights=m * weights  # [amu * # particles]
        )

        return macro_count_map, count_map, flux_contribution, momentum_contribution, mass_count

    # ========== Combine species based on selection ==========
    # Determine which species to process
    if species == "all":
        spec_ids = list(range(len(sim_ppc)))
    elif species == "protons":
        if "Base" in all_particles_filename:
            spec_ids = [0]  # Single proton species
        elif "HNHV" in all_particles_filename:
            spec_ids = [0, 1]  # Two proton species
        else:
            raise ValueError(f"Unknown case: {all_particles_filename}")
    elif species == "alphas":
        if "Base" in all_particles_filename:
            spec_ids = [2]  # Single alpha species
        elif "HNHV" in all_particles_filename:
            spec_ids = [2, 3]  # Two alpha species
        else:
            raise ValueError(f"Unknown case: {all_particles_filename}")
    else:
        raise ValueError(f"Unknown species: {species}")

    # Initialize accumulator maps
    macro_count_total = np.zeros((n_lat, n_lon))  # [# macroparticles]
    count_map_total = np.zeros((n_lat, n_lon))  # [# physical particles]
    flux_contrib_total = np.zeros((n_lat, n_lon))  # [# particles * m/s]
    momentum_total = np.zeros((n_lat, n_lon))  # [amu * # particles * m/s]
    mass_count_total = np.zeros((n_lat, n_lon))  # [amu * # particles]

    # Process each species and accumulate
    for spec_id in spec_ids:
        macro_count, count, flux_contrib, momentum, mass_count = process_species(spec_id)
        macro_count_total += macro_count
        count_map_total += count
        flux_contrib_total += flux_contrib
        momentum_total += momentum
        mass_count_total += mass_count

    # ========== Compute final quantities ==========

    # 1) Shell volume density [m^-3] → [cm^-3]
    # Use physical particle count, not macroparticles
    den_map = np.where(shell_volume > 0, count_map_total / shell_volume, 0.0)  # [m^-3]
    den_map_cm3 = den_map * 1e-6  # [cm^-3]

    # 2) Surface flux [cm^-2 s^-1]
    # Flux = Σ(w_i * |v_r,i|) / area
    flux_map = np.where(area > 0, flux_contrib_total / area, 0.0)  # [m^-2 s^-1]
    flux_map_cm = flux_map * 1e-4  # [cm^-2 s^-1]

    # 3) Mass-weighted average velocity [km/s]
    # v_avg = Σ(m * w_i * v_r,i) / Σ(m * w_i)
    v_r_map = np.zeros((n_lat, n_lon))
    mass_threshold = 1e10  # [amu * # particles] minimum for statistics
    valid_mask = mass_count_total > mass_threshold
    v_r_map[valid_mask] = (momentum_total[valid_mask] /
                           mass_count_total[valid_mask]) * 1e-3  # [km/s]

    # Clip unphysical values (should not occur)
    v_r_map = np.clip(v_r_map, -3000, 0)

    # ========== Debug output ==========
    print("=" * 60)
    print("FINAL MAPS STATISTICS")
    print("=" * 60)
    print(f"Total physical particles: {count_map_total.sum():.2e}")
    print(f"den_map_cm3: [{den_map_cm3[den_map_cm3 > 0].min():.2e}, {den_map_cm3.max():.2e}] cm^-3")
    print(f"flux_map_cm: [{flux_map_cm[flux_map_cm > 0].min():.2e}, {flux_map_cm.max():.2e}] cm^-2 s^-1")
    print(f"v_r_map:     [{v_r_map[valid_mask].min():.2f}, {v_r_map[valid_mask].max():.2f}] km/s")
    print("=" * 60 + "\n")

    return flux_map_cm, lat_centers, lon_centers, v_r_map, macro_count_total, den_map_cm3


def compute_flux_statistics(flux_map, lat_centers, lon_centers, R_M,
                            flux_threshold=None, case_name="Case"):
    """
    Compute summary statistics for a georeferenced flux map.

    Parameters
    ----------
    flux_map : ndarray (n_lat, n_lon)
        Surface flux map [cm^-2 s^-1]
    lat_centers : ndarray
        Latitude bin centers [deg]
    lon_centers : ndarray
        Longitude bin centers [deg]
    R_M : float
        Planet radius [m]
    flux_threshold : float, optional
        Threshold flux value for spatial extent calculation [cm^-2 s^-1]
        If None, uses 10% of peak flux
    case_name : str
        Identifier for this case

    Returns
    -------
    stats : dict
        Dictionary containing computed statistics
    """

    # Initialize statistics dictionary
    stats = {
        'case_name': case_name,
        'total_integrated_flux': 0.0,
        'peak_flux_value': 0.0,
        'peak_flux_lat': 0.0,
        'peak_flux_lon': 0.0,
        'spatial_extent_area': 0.0,
        'spatial_extent_percentage': 0.0,
        'northern_hemisphere_flux': 0.0,
        'southern_hemisphere_flux': 0.0,
        'hemispheric_asymmetry_ratio': 0.0,
        'dayside_flux': 0.0,
        'nightside_flux': 0.0,
        'dayside_nightside_ratio': 0.0,
        'mean_flux': 0.0,
        'median_flux': 0.0,
        'threshold_used': 0.0
    }

    # Handle edge case of empty or all-zero flux map
    if np.all(flux_map == 0) or np.all(np.isnan(flux_map)):
        return stats

    # Compute bin sizes
    dlat = np.radians(lat_centers[1] - lat_centers[0]) if len(lat_centers) > 1 else 0
    dlon = np.radians(lon_centers[1] - lon_centers[0]) if len(lon_centers) > 1 else 0

    # Compute area of each bin [m^2]
    # A = R^2 * dlon * (sin(lat_upper) - sin(lat_lower))
    lat_edges_rad = np.linspace(np.radians(-90), np.radians(90), len(lat_centers) + 1)
    dlon_rad = dlon

    area_per_bin = np.zeros((len(lat_centers), len(lon_centers)))
    for i in range(len(lat_centers)):
        area_i = R_M ** 2 * dlon_rad * (np.sin(lat_edges_rad[i + 1]) - np.sin(lat_edges_rad[i]))
        area_per_bin[i, :] = area_i  # Same area for all longitudes at this latitude

    # Convert flux from cm^-2 s^-1 to m^-2 s^-1 for integration
    flux_map_si = flux_map * 1e4  # cm^-2 s^-1 -> m^-2 s^-1

    # Total integrated flux [particles/s]
    # Integrate flux * area over entire surface
    total_flux = np.nansum(flux_map_si * area_per_bin)  # [particles/s]
    stats['total_integrated_flux'] = total_flux

    # Peak flux and location
    peak_idx = np.unravel_index(np.nanargmax(flux_map), flux_map.shape)
    stats['peak_flux_value'] = flux_map[peak_idx]  # [cm^-2 s^-1]
    stats['peak_flux_lat'] = lat_centers[peak_idx[0]]  # [deg]
    stats['peak_flux_lon'] = lon_centers[peak_idx[1]]  # [deg]

    # Spatial extent above threshold
    if flux_threshold is None:
        flux_threshold = 0.1 * stats['peak_flux_value']  # 10% of peak
    stats['threshold_used'] = flux_threshold

    above_threshold = flux_map >= flux_threshold
    area_above_threshold = np.sum(area_per_bin[above_threshold])  # [m^2]
    total_surface_area = 4 * np.pi * R_M ** 2  # [m^2]

    stats['spatial_extent_area'] = area_above_threshold  # [m^2]
    stats['spatial_extent_percentage'] = (area_above_threshold / total_surface_area) * 100  # [%]

    # Hemispheric asymmetry
    # Northern hemisphere: lat >= 0
    # Southern hemisphere: lat < 0
    northern_mask = lat_centers >= 0
    southern_mask = lat_centers < 0

    # Create 2D masks
    northern_2d = northern_mask[:, np.newaxis] * np.ones_like(flux_map)
    southern_2d = southern_mask[:, np.newaxis] * np.ones_like(flux_map)

    northern_flux = np.nansum(flux_map_si[northern_2d.astype(bool)] *
                              area_per_bin[northern_2d.astype(bool)])
    southern_flux = np.nansum(flux_map_si[southern_2d.astype(bool)] *
                              area_per_bin[southern_2d.astype(bool)])

    stats['northern_hemisphere_flux'] = northern_flux  # [particles/s]
    stats['southern_hemisphere_flux'] = southern_flux  # [particles/s]

    if southern_flux > 0:
        stats['hemispheric_asymmetry_ratio'] = northern_flux / southern_flux
    else:
        stats['hemispheric_asymmetry_ratio'] = np.inf

    # Dayside vs Nightside
    # Dayside: -90 < lon < 90 (+X is sunward)
    # Nightside: lon < -90 or lon > 90
    dayside_mask = np.abs(lon_centers) < 90
    nightside_mask = np.abs(lon_centers) >= 90

    # Create 2D masks
    dayside_2d = np.ones((len(lat_centers), 1)) * dayside_mask[np.newaxis, :]
    nightside_2d = np.ones((len(lat_centers), 1)) * nightside_mask[np.newaxis, :]

    dayside_flux = np.nansum(flux_map_si[dayside_2d.astype(bool)] *
                             area_per_bin[dayside_2d.astype(bool)])
    nightside_flux = np.nansum(flux_map_si[nightside_2d.astype(bool)] *
                               area_per_bin[nightside_2d.astype(bool)])

    stats['dayside_flux'] = dayside_flux  # [particles/s]
    stats['nightside_flux'] = nightside_flux  # [particles/s]

    if nightside_flux > 0:
        stats['dayside_nightside_ratio'] = dayside_flux / nightside_flux
    else:
        stats['dayside_nightside_ratio'] = np.inf

    # Mean and median flux
    # Consider only non-zero flux bins
    nonzero_flux = flux_map[flux_map > 0]
    if len(nonzero_flux) > 0:
        stats['mean_flux'] = np.nanmean(nonzero_flux)  # [cm^-2 s^-1]
        stats['median_flux'] = np.nanmedian(nonzero_flux)  # [cm^-2 s^-1]

    return stats


def create_comparison_table(stats_list, output_csv='flux_statistics_comparison.csv',
                            output_latex='flux_statistics_comparison.tex'):
    """
    Create a formatted comparison table from multiple case statistics.

    Parameters
    ----------
    stats_list : list of dict
        List of statistics dictionaries from compute_flux_statistics
    output_csv : str
        Output CSV filename
    output_latex : str
        Output LaTeX table filename

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the statistics
    """

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(stats_list)

    # Reorder columns for logical presentation
    column_order = [
        'case_name',
        'total_integrated_flux',
        'peak_flux_value',
        'peak_flux_lat',
        'peak_flux_lon',
        'mean_flux',
        'median_flux',
        'spatial_extent_area',
        'spatial_extent_percentage',
        'northern_hemisphere_flux',
        'southern_hemisphere_flux',
        'hemispheric_asymmetry_ratio',
        'dayside_flux',
        'nightside_flux',
        'dayside_nightside_ratio',
        'threshold_used'
    ]

    df = df[column_order]

    # Format for display
    df_display = df.copy()
    df_display['total_integrated_flux'] = df_display['total_integrated_flux'].apply(lambda x: f'{x:.2e}')
    df_display['peak_flux_value'] = df_display['peak_flux_value'].apply(lambda x: f'{x:.2f}')
    df_display['peak_flux_lat'] = df_display['peak_flux_lat'].apply(lambda x: f'{x:.1f}')
    df_display['peak_flux_lon'] = df_display['peak_flux_lon'].apply(lambda x: f'{x:.1f}')
    df_display['mean_flux'] = df_display['mean_flux'].apply(lambda x: f'{x:.2f}')
    df_display['median_flux'] = df_display['median_flux'].apply(lambda x: f'{x:.2f}')
    df_display['spatial_extent_area'] = df_display['spatial_extent_area'].apply(lambda x: f'{x:.2e}')
    df_display['spatial_extent_percentage'] = df_display['spatial_extent_percentage'].apply(lambda x: f'{x:.1f}')
    df_display['northern_hemisphere_flux'] = df_display['northern_hemisphere_flux'].apply(lambda x: f'{x:.2e}')
    df_display['southern_hemisphere_flux'] = df_display['southern_hemisphere_flux'].apply(lambda x: f'{x:.2e}')
    df_display['hemispheric_asymmetry_ratio'] = df_display['hemispheric_asymmetry_ratio'].apply(
        lambda x: f'{x:.2f}' if x != np.inf else 'inf')
    df_display['dayside_flux'] = df_display['dayside_flux'].apply(lambda x: f'{x:.2e}')
    df_display['nightside_flux'] = df_display['nightside_flux'].apply(lambda x: f'{x:.2e}')
    df_display['dayside_nightside_ratio'] = df_display['dayside_nightside_ratio'].apply(
        lambda x: f'{x:.2f}' if x != np.inf else 'inf')
    df_display['threshold_used'] = df_display['threshold_used'].apply(lambda x: f'{x:.2f}')

    # Rename columns for readability
    df_display.columns = [
        'Case',
        'Total Flux [particles/s]',
        'Peak Flux [cm⁻² s⁻¹]',
        'Peak Lat [°]',
        'Peak Lon [°]',
        'Mean Flux [cm⁻² s⁻¹]',
        'Median Flux [cm⁻² s⁻¹]',
        'Precip. Area [m²]',
        'Precip. Area [%]',
        'North Flux [particles/s]',
        'South Flux [particles/s]',
        'North/South Ratio',
        'Day Flux [particles/s]',
        'Night Flux [particles/s]',
        'Day/Night Ratio',
        'Threshold [cm⁻² s⁻¹]'
    ]

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Statistics saved to {output_csv}\n")

    # Print formatted table
    print("=" * 120)
    print("FLUX STATISTICS COMPARISON TABLE")
    print("=" * 120)
    print(df_display.to_string(index=False))
    print("=" * 120)

    # Create a condensed version for LaTeX (most important metrics)
    df_latex = df.copy()

    # Select key columns for LaTeX table (to fit on page)
    latex_columns = [
        'case_name',
        'total_integrated_flux',
        'peak_flux_value',
        'peak_flux_lat',
        'peak_flux_lon',
        'spatial_extent_percentage',
        'hemispheric_asymmetry_ratio',
        'dayside_nightside_ratio'
    ]

    df_latex = df_latex[latex_columns]

    # Helper function to format scientific notation for LaTeX
    def format_sci_latex(value):
        """Convert float to LaTeX scientific notation"""
        sci_str = f"{value:.2e}"
        # Split mantissa and exponent
        parts = sci_str.split('e')
        mantissa = parts[0]
        exponent = int(parts[1])  # Convert to int to remove leading zeros/plus
        return f"${mantissa} \\times 10^{{{exponent}}}$"

    # Build LaTeX table manually for better control
    latex_lines = []
    latex_lines.append("% LaTeX table for flux statistics comparison")
    latex_lines.append("% Copy this into your document or use \\input{" + output_latex + "}")
    latex_lines.append("")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Summary statistics of surface precipitation flux for different simulation cases.}")
    latex_lines.append("\\label{tab:flux_statistics}")
    latex_lines.append("\\begin{tabular}{lccccccc}")
    latex_lines.append("\\hline")
    latex_lines.append("\\hline")

    # Header row
    latex_lines.append("Case & Total Flux & Peak Flux & Peak Lat & Peak Lon & Precip. Area & N/S & Day/Night \\\\")
    latex_lines.append(
        " & [particles~s$^{-1}$] & [cm$^{-2}$~s$^{-1}$] & [$^\\circ$] & [$^\\circ$] & [\\%] & Ratio & Ratio \\\\")
    latex_lines.append("\\hline")

    # Data rows
    for _, row in df_latex.iterrows():
        case = row['case_name']
        total_flux = format_sci_latex(row['total_integrated_flux'])
        peak_flux = format_sci_latex(row['peak_flux_value'])  # Now in scientific notation
        peak_lat = f"{row['peak_flux_lat']:.1f}"
        peak_lon = f"{row['peak_flux_lon']:.1f}"
        precip_pct = f"{row['spatial_extent_percentage']:.1f}"
        ns_ratio = f"{row['hemispheric_asymmetry_ratio']:.2f}" if row[
                                                                      'hemispheric_asymmetry_ratio'] != np.inf else '$\\infty$'
        dn_ratio = f"{row['dayside_nightside_ratio']:.2f}" if row['dayside_nightside_ratio'] != np.inf else '$\\infty$'

        latex_lines.append(
            f"{case} & {total_flux} & {peak_flux} & {peak_lat} & {peak_lon} & {precip_pct} & {ns_ratio} & {dn_ratio} \\\\")

    latex_lines.append("\\hline")
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    # Write LaTeX to file
    with open(output_latex, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"LaTeX table saved to {output_latex}\n")

    # Also print LaTeX to console
    print("=" * 120)
    print("LATEX TABLE OUTPUT")
    print("=" * 120)
    print('\n'.join(latex_lines))
    print("=" * 120 + "\n")

    # ========== CREATE FULL LATEX TABLE (all metrics) ==========

    output_latex_full = output_latex.replace('.tex', '_full.tex')

    latex_full = []
    latex_full.append("% Full LaTeX table with all flux statistics")
    latex_full.append("% This table is wide and may require landscape orientation or small font")
    latex_full.append("")
    latex_full.append("\\begin{table*}[htbp]")
    latex_full.append("\\centering")
    latex_full.append("\\small")  # Use smaller font
    latex_full.append("\\caption{Comprehensive surface precipitation flux statistics for all simulation cases.}")
    latex_full.append("\\label{tab:flux_statistics_full}")
    latex_full.append("\\begin{tabular}{lcccccccc}")
    latex_full.append("\\hline")
    latex_full.append("\\hline")
    latex_full.append(
        "Case & Total Flux & Peak Flux & Peak Location & Mean Flux & Precip. Area & N Hemi & S Hemi & N/S \\\\")
    latex_full.append(
        " & [particles~s$^{-1}$] & [cm$^{-2}$~s$^{-1}$] & (lat, lon) [$^\\circ$] & [cm$^{-2}$~s$^{-1}$] & [\\%] & [particles~s$^{-1}$] & [particles~s$^{-1}$] & Ratio \\\\")
    latex_full.append("\\hline")

    for _, row in df.iterrows():
        case = row['case_name']
        total_flux = format_sci_latex(row['total_integrated_flux'])
        peak_flux = format_sci_latex(row['peak_flux_value'])  # Now in scientific notation
        peak_loc = f"({row['peak_flux_lat']:.1f}, {row['peak_flux_lon']:.1f})"
        mean_flux = format_sci_latex(row['mean_flux'])
        precip_pct = f"{row['spatial_extent_percentage']:.1f}"
        n_flux = format_sci_latex(row['northern_hemisphere_flux'])
        s_flux = format_sci_latex(row['southern_hemisphere_flux'])
        ns_ratio = f"{row['hemispheric_asymmetry_ratio']:.2f}" if row[
                                                                      'hemispheric_asymmetry_ratio'] != np.inf else '$\\infty$'

        latex_full.append(
            f"{case} & {total_flux} & {peak_flux} & {peak_loc} & {mean_flux} & {precip_pct} & {n_flux} & {s_flux} & {ns_ratio} \\\\")

    latex_full.append("\\hline")
    latex_full.append("\\hline")
    latex_full.append("\\end{tabular}")
    latex_full.append("\\end{table*}")

    with open(output_latex_full, 'w') as f:
        f.write('\n'.join(latex_full))

    print(f"Full LaTeX table saved to {output_latex_full}\n")

    return df
