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

    Returns shell volume density [m^-3] and surface quantities
    - Particles selected from shell R_M <= r <= select_R
    - Ray-traced to surface impact points
    - Shell density [m^-3], surface flux [cm^-2 s^-1], impact velocity [km/s]

    Parameters
    ----------
    all_particles_filename : str
        Numpy .npz file containing all particles.
    sim_dx, sim_dy, sim_dz : float
        Simulation grid cell resolution [m].
    sim_ppc : array-like
        Macroparticles per simulation grid cell per species.
    sim_den : array-like
        Physical number density per species in the upstream region [m^-3]
        corresponding to each species' macroparticles.
    spec_map : array-like
        Name of each species
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
        Radial surface flux [cm^-2 s^-1].
    lat_centers : ndarray
        Latitude bin centers [deg].
    lon_centers : ndarray
        Longitude bin centers [deg].
    v_r_map : ndarray (n_lat, n_lon)
        Impact velocity for surface precipitation [km/s].
        Negative = inward.
    count_map : ndarray (n_lat, n_lon)
        Physical particle count per surface bin [# particles].
    n_shell_map : ndarray (n_lat, n_lon)
        Shell volume density [m^-3].
    """

    # Load particle data
    with np.load(all_particles_filename) as data:
        prx = data["prx"]  # [m]
        pry = data["pry"]  # [m]
        prz = data["prz"]  # [m]
        pvx = data["pvx"]  # [m/s]
        pvy = data["pvy"]  # [m/s]
        pvz = data["pvz"]  # [m/s]
        psid = data["psid"].astype(int)  # ensure integer indexing
        num_files = data["num_files"]
        selected_radius = data["selected_radius"]  # [m]

    if selected_radius < select_R:
        raise ValueError(f"Selected radius {selected_radius:e} < select_R {select_R:e}")

    sim_ppc = np.asarray(sim_ppc, dtype=float)
    sim_den = np.asarray(sim_den, dtype=float)

    # Angular binning (spherical surface at R_M)
    lon_edges = np.linspace(-180.0, 180.0, n_lon + 1)  # [deg]
    lat_edges = np.linspace(-90.0, 90.0, n_lat + 1)  # [deg]
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    dlon = np.radians(lon_edges[1] - lon_edges[0])  # [rad]

    # Surface area per (lat,lon) bin on sphere of radius R_M [m^2]
    area = R_M ** 2 * dlon * (np.sin(np.radians(lat_edges[1:])) -
                              np.sin(np.radians(lat_edges[:-1])))
    area = area[:, None]  # broadcast over longitude, shape (n_lat, n_lon)

    # ADDED: Shell volume per (lat,lon) bin for density calculation [m^3]
    dR = select_R - R_M  # [m] shell thickness
    shell_volume = area * dR  # [m^3/bin]

    # Helper: per-species processing, returns surface maps
    def process_species(spec_id):
        print(f"Processing species {spec_map[spec_id]}")

        # Select species
        mask = psid == spec_id
        prx_s, pry_s, prz_s = prx[mask], pry[mask], prz[mask]  # [m]
        pvx_s, pvy_s, pvz_s = pvx[mask], pvy[mask], pvz[mask]  # [m/s]

        print(f"Number of initial particles: {len(pvz_s)}")

        # Shell selection: R_M <= r <= select_R
        r = np.sqrt(prx_s ** 2 + pry_s ** 2 + prz_s ** 2)  # [m]
        shell = (r >= R_M) & (r <= select_R)
        prx_s, pry_s, prz_s = prx_s[shell], pry_s[shell], prz_s[shell]
        pvx_s, pvy_s, pvz_s = pvx_s[shell], pvy_s[shell], pvz_s[shell]
        r = r[shell]

        print(f"Number of particles after shell selection: {len(r)}")

        if len(r) == 0:
            empty = np.zeros((n_lat, n_lon))
            print("\n")
            return empty, empty, empty, empty

        # Radial unit vector and radial velocity [m/s]
        r_hat = np.vstack((prx_s, pry_s, prz_s)) / r
        v_r_s = pvx_s * r_hat[0] + pvy_s * r_hat[1] + pvz_s * r_hat[2]

        # Only particles on impact trajectories (inward)
        inward = v_r_s < 0.0
        prx_s, pry_s, prz_s = prx_s[inward], pry_s[inward], prz_s[inward]
        pvx_s, pvy_s, pvz_s = pvx_s[inward], pvy_s[inward], pvz_s[inward]
        v_r_s = v_r_s[inward]
        r = r[inward]

        print(f"Number of particles after inward trajectory selection: {len(r)}")
        # Print statistics in both m/s and km/s
        print(f"Radial velocity magnitudes [m/s]:")
        print(f"  Min:    {np.min(v_r_s):.2e}")
        print(f"  Median: {np.median(v_r_s):.2e}")
        print(f"  Max:    {np.max(v_r_s):.2e}")
        print("\n")

        if len(v_r_s) == 0:
            empty = np.zeros((n_lat, n_lon))
            return empty, empty, empty, empty

        # ========== Ray–sphere intersection to project impact point to SURFACE ==========

        # Stack positions into a (N, 3) array [m]:
        # r_vec[i] = [x_i, y_i, z_i] is the current position vector of particle i.
        r_vec = np.vstack((prx_s, pry_s, prz_s)).T

        # Stack velocities into a (N, 3) array [m/s]:
        # v_vec[i] = [vx_i, vy_i, vz_i] is the current velocity vector of particle i.
        v_vec = np.vstack((pvx_s, pvy_s, pvz_s)).T

        # Quadratic coefficients for ray–sphere intersection:
        # For each particle, we solve |r(t)|^2 = R_M^2, where r(t) = r0 + t v.
        # This gives a quadratic of the form: a*t^2 + b*t + c = 0.

        # a = |v|^2 (squared speed) for each particle [m^2/s^2].
        a = np.sum(v_vec ** 2, axis=1)

        # b = 2 (r0 · v) for each particle (dot product) [m^2/s].
        b = 2.0 * np.sum(r_vec * v_vec, axis=1)

        # c = |r0|^2 - R_M^2 [m^2].
        c = np.sum(r_vec ** 2, axis=1) - R_M ** 2

        # Discriminant of the quadratic: disc = b^2 - 4 a c [m^4/s^2].
        # If disc < 0, the ray does not intersect the sphere.
        disc = b ** 2 - 4 * a * c

        # Keep only particles whose trajectories intersect the sphere at least once.
        valid = disc > 0.0

        # Filter positions, velocities, and radial velocities to intersecting particles only.
        r_vec = r_vec[valid]
        v_vec = v_vec[valid]
        v_r_s = v_r_s[valid]

        print(f"Number of particles with intersecting trajectory: {len(r_vec)}")
        # Print statistics in both m/s and km/s
        print(f"Radial velocity magnitudes [m/s]:")
        print(f"  Min:    {np.min(v_r_s):.2e}")
        print(f"  Median: {np.median(v_r_s):.2e}")
        print(f"  Max:    {np.max(v_r_s):.2e}")
        print("\n")

        # If nothing intersects, return empty maps (no impacts on the surface).
        if len(v_r_s) == 0:
            empty = np.zeros((n_lat, n_lon))
            return empty, empty, empty, empty

        # Solve for the first intersection time along the ray [s]:
        # t_hit = (-b - sqrt(disc)) / (2 a) is the smaller root (entry point).
        t_hit = (-b[valid] - np.sqrt(disc[valid])) / (2 * a[valid])

        # Compute the impact positions on the sphere [m]:
        # hit[i] = r0_i + t_hit_i * v_i lies on the surface |hit| = R_M.
        hit = r_vec + v_vec * t_hit[:, None]

        # Extract Cartesian coordinates of impact points [m].
        x, y, z = hit[:, 0], hit[:, 1], hit[:, 2]

        # Geographic coordinates of SURFACE IMPACTS (+X sunward, +Z north) [deg]
        lon_s = np.degrees(np.arctan2(y, x))  # [-180, 180]
        lat_s = np.degrees(np.arcsin(z / R_M))  # [-90, 90]

        # ========== Per-particle physical weight [# particles per macroparticle] ==========
        # One macroparticle represents this many real particles.
        # Use upstream density [m^-3] and cell volume [m^3], normalized over num_files.
        cell_volume = sim_dx * sim_dy * sim_dz  # [m^3]
        w_species = (sim_den[spec_id] * cell_volume) / (sim_ppc[spec_id] * num_files)

        weights = w_species * np.ones_like(lat_s)  # [# particles/macroparticle]

        # ========== 1) Count map: physically weighted particle count [# particles/bin] ==========
        count_map, _, _ = np.histogram2d(lat_s, lon_s, bins=[lat_edges, lon_edges], weights=weights)

        # ========== 2) Shell Volume density [m^-3] ==========
        n_shell_map = np.where(shell_volume > 0.0, count_map / shell_volume, 0.0)  # [m^-3]

        # ========== 3) Surface flux [m^-2 s^-1] ==========
        # Flux = sum(n * v_r) / area, where v_r is negative for inward
        nv_r_map, _, _ = np.histogram2d(
            lat_s, lon_s, bins=[lat_edges, lon_edges],
            weights=weights * v_r_s  # [# particles * m/s]
        )
        # flux_map = np.where(area > 0.0, nv_r_map / area, 0.0)  # [m^-2 s^-1]

        # ========== 4) Impact velocity [km/s] (density-weighted average) ==========
        # v_r = sum(n * v_r) / sum(n)
        v_r_hist, _, _ = np.histogram2d(
            lat_s, lon_s, bins=[lat_edges, lon_edges],
            weights=weights * v_r_s  # [# particles * m/s]
        )
        n_hist, _, _ = np.histogram2d(
            lat_s, lon_s, bins=[lat_edges, lon_edges],
            weights=weights  # [# particles]
        )

        macro_hits = count_map / w_species

        vr_map = np.zeros((n_lat, n_lon))
        n_floor = 1e10  # [# particles] minimum for reliable statistics
        valid_bins = n_hist > n_floor
        if np.any(valid_bins):
            v_r_raw = v_r_hist[valid_bins] / n_hist[valid_bins]  # [m/s]
            vr_map[valid_bins] = v_r_raw * 1e-3  # [km/s]
            # Clip unphysical velocities (should not be needed)
            vr_map[valid_bins] = np.clip(vr_map[valid_bins], -3000, 0)

        flx_map = vr_map * 1e3 * n_shell_map  # [m^-2 s^-1]

        return flx_map, vr_map, macro_hits, n_shell_map  # [m^-2 s^-1, km/s, #, m^-3]

    # weighted sum over species
    if species == "all":
        flux_map = np.zeros((n_lat, n_lon))
        v_r_num = np.zeros((n_lat, n_lon))
        v_r_den = np.zeros((n_lat, n_lon))
        count_map = np.zeros((n_lat, n_lon))
        n_shell_map = np.zeros((n_lat, n_lon))

        for spec_id in range(len(sim_ppc)):
            F_s, v_r_s, C_s, n_s = process_species(spec_id)
            flux_map += F_s
            count_map += C_s
            n_shell_map += n_s

            # Density-weighted combination of v_r over species [km/s]:
            # Weight by shell density n_s [m^-3] in each bin
            v_r_num += v_r_s * n_s
            v_r_den += n_s

        v_r_map = np.zeros_like(flux_map)
        mask = v_r_den > 1e3  # [m^-3] floor
        v_r_map[mask] = v_r_num[mask] / v_r_den[mask]

    else:
        if "Base" in all_particles_filename:
            spec_id = 0 if species == "protons" else 2
            flux_map, v_r_map, count_map, n_shell_map = process_species(spec_id)
        elif "HNHV" in all_particles_filename:
            if species == "protons":
                flux_map = np.zeros((n_lat, n_lon))
                v_r_num = np.zeros((n_lat, n_lon))
                v_r_den = np.zeros((n_lat, n_lon))
                count_map = np.zeros((n_lat, n_lon))
                n_shell_map = np.zeros((n_lat, n_lon))

                for spec_id in [0, 1]:
                    F_s, v_r_s, C_s, n_s = process_species(spec_id)
                    flux_map += F_s
                    count_map += C_s
                    n_shell_map += n_s

                    # Density-weighted combination of v_r over species [km/s]:
                    # Weight by shell density n_s [m^-3] in each bin
                    v_r_num += v_r_s * n_s
                    v_r_den += n_s

                v_r_map = np.zeros_like(flux_map)
                mask = v_r_den > 1e3  # [m^-3] floor
                v_r_map[mask] = v_r_num[mask] / v_r_den[mask]
            elif species == "alphas":
                flux_map = np.zeros((n_lat, n_lon))
                v_r_num = np.zeros((n_lat, n_lon))
                v_r_den = np.zeros((n_lat, n_lon))
                count_map = np.zeros((n_lat, n_lon))
                n_shell_map = np.zeros((n_lat, n_lon))

                for spec_id in [2, 3]:
                    F_s, v_r_s, C_s, n_s = process_species(spec_id)
                    flux_map += F_s
                    count_map += C_s
                    n_shell_map += n_s

                    # Density-weighted combination of v_r over species [km/s]:
                    # Weight by shell density n_s [m^-3] in each bin
                    v_r_num += v_r_s * n_s
                    v_r_den += n_s

                v_r_map = np.zeros_like(flux_map)
                mask = v_r_den > 1e3  # [m^-3] floor
                v_r_map[mask] = v_r_num[mask] / v_r_den[mask]

    # Convert flux to [cm^-2 s^-1]
    flux_map_cm = flux_map * 1e-4  # [m^-2 s^-1] → [cm^-2 s^-1]

    # DEBUG OUTPUT STATISTICS
    print("=" * 60)
    print("FINAL MAPS STATISTICS")
    print("=" * 60)
    print(f"n_shell_map: [{np.nanmin(n_shell_map):.2e}, {np.nanmax(n_shell_map):.2e}] m^-3")
    print(f"             [{np.nanmin(n_shell_map)*1e-6:.2e}, {np.nanmax(n_shell_map)*1e-6:.2e}] cm^-3")
    print(f"flux_map_cm: [{np.nanmin(flux_map_cm):.2e}, {np.nanmax(flux_map_cm):.2e}] cm^-2 s^-1")
    print(f"v_r_map:     [{np.nanmin(v_r_map):.2f}, {np.nanmax(v_r_map):.2f}] km/s")
    print("=" * 60 + "\n")

    return flux_map_cm, lat_centers, lon_centers, v_r_map, count_map, n_shell_map


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
