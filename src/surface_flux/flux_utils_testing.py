#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports:
import numpy as np


def compute_radial_flux(all_particles_filename, sim_dx, sim_dy, sim_dz,
                        sim_ppc, sim_den, spec_map, species_mass, species_charge,
                        R_M, select_R, species="all", n_lat=180, n_lon=360, debug=False):
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
    species_mass : array-like
        Mass of each species [amu].
    species_charge : array-like
        Charge of each species [e].
    R_M : float
        Planet radius [m].
    select_R : float
        Outer radius for shell selection [m].
    species : {"all", "protons", "alphas"}
        Which species to use.
    n_lat, n_lon : int
        Number of latitude and longitude bins.
    debug : bool
        Flag to print debugging statements.

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
    mass_flux_map : ndarray (n_lat, n_lon)
        Mass flux [amu cm^-2 s^-1].
    energy_flux_map : ndarray (n_lat, n_lon)
        Energy flux [eV cm^-2 s^-1].
    """

    # Constants
    AMU_TO_KG = 1.66053906660e-27  # [kg/amu]
    J_TO_EV = 6.241509074e18  # [eV/J]

    # ========== Load particle data ==========
    with np.load(all_particles_filename) as data:
        prx = data["rx"]  # [m]
        pry = data["ry"]  # [m]
        prz = data["rz"]  # [m]
        pvx = data["vx"]  # [m/s]
        pvy = data["vy"]  # [m/s]
        pvz = data["vz"]  # [m/s]
        psid = data["sid"].astype(int)
        num_files = 1  # data["num_files"]
        selected_radius = 2480e3  # data["selected_radius"]  # [m]

    if selected_radius < select_R:
        raise ValueError(f"Selected radius {selected_radius:e} < select_R {select_R:e}")

    sim_ppc = np.asarray(sim_ppc, dtype=float)
    sim_den = np.asarray(sim_den, dtype=float)
    species_mass = np.asarray(species_mass, dtype=float)

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
        mass_count : Mass-weighted count Σ(m * w_i) [amu * # particles]
        mass_flux_contribution : Mass flux contribution Σ(m * w_i * |v_r,i|) [amu * # particles * m/s]
        energy_flux_contribution : Energy flux contribution Σ(w_i * KE_i * |v_r,i|) [J * # particles * m/s]
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

        if debug:
            print(f"  Initial macroparticles: {len(prx_s)}")

        if len(prx_s) == 0:
            print(f"No particles matching  {spec_map[spec_id]}! Returning empty map.")
            empty = np.zeros((n_lat, n_lon))
            return empty, empty, empty, empty, empty, empty

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

        if debug:
            print(f"  After shell selection: {len(r)}")

        if len(r) == 0:
            print(f"No particles with r <= {select_R}! Returning empty map.")
            empty = np.zeros((n_lat, n_lon))
            return empty, empty, empty, empty, empty, empty

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

        if debug:
            print(f"  After inward selection: {len(v_r)}")
            print(f"  v_r range: [{v_r.min():.2e}, {v_r.max():.2e}] m/s")

        if len(v_r) == 0:
            print(f"No particles with inward radial velocity! Returning empty map.")
            empty = np.zeros((n_lat, n_lon))
            return empty, empty, empty, empty, empty, empty

        # ========== 5) Ray-sphere intersection to find surface impact points ==========
        r_vec = np.column_stack([prx_s, pry_s, prz_s])  # [m] shape (N, 3)
        v_vec = np.column_stack([pvx_s, pvy_s, pvz_s])  # [m/s] shape (N, 3)

        # Solve |r0 + t*v|^2 = R_M^2
        a = np.sum(v_vec ** 2, axis=1)  # [m^2/s^2]
        b = 2.0 * np.sum(r_vec * v_vec, axis=1)  # [m^2/s]
        c = np.sum(r_vec ** 2, axis=1) - R_M ** 2  # [m^2]

        disc = b ** 2 - 4 * a * c
        valid_intersection = disc > 0.0

        if debug:
            print(f"  After intersection check: {valid_intersection.sum()}")

        if valid_intersection.sum() == 0:
            print(f"No particles with intersecting trajectories! Returning empty map.")
            empty = np.zeros((n_lat, n_lon))
            return empty, empty, empty, empty, empty, empty

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

        if debug:
            # Debug: Check impact position distribution
            print(f"\nDEBUG: Impact position statistics for {spec_map[spec_id]}")
            print(f"  x_impact range: [{x_impact.min():.2e}, {x_impact.max():.2e}] m")
            print(f"  y_impact range: [{y_impact.min():.2e}, {y_impact.max():.2e}] m")
            print(f"  z_impact range: [{z_impact.min():.2e}, {z_impact.max():.2e}] m")
            print(f"  lon_impact range: [{lon_impact.min():.1f}, {lon_impact.max():.1f}]°")
            print(f"  lat_impact range: [{lat_impact.min():.1f}, {lat_impact.max():.1f}]°")

            # Check where most flux is
            hist_lon, bins_lon = np.histogram(lon_impact, bins=36, range=(-180, 180))
            peak_lon_bin = np.argmax(hist_lon)
            peak_lon_center = (bins_lon[peak_lon_bin] + bins_lon[peak_lon_bin + 1]) / 2
            print(f"  Most particles near lon = {peak_lon_center:.1f}°")

        # ========== 7) Compute physical weights and energetics ==========
        # Each macroparticle represents w physical particles
        cell_volume = sim_dx * sim_dy * sim_dz  # [m^3]
        w = (sim_den[spec_id] * cell_volume) / (sim_ppc[spec_id] * num_files)  # [# particles]

        weights = w * np.ones(len(lat_impact))  # [# particles per macroparticle]

        # Mass and velocity properties
        m = species_mass[spec_id]  # [amu]
        v_squared = np.sum(v_vec ** 2, axis=1)  # [m^2/s^2]
        v_r_mag = -v_r  # [m/s] magnitude (v_r is negative)

        # Kinetic energy per particle [J]
        KE_per_particle = 0.5 * m * AMU_TO_KG * v_squared  # [J]

        if debug:
            print(f"  Macroparticle weight: {w:.2e} physical particles")
            print(f"  Total macroparticles: {len(lat_impact)}")
            print(f"  Total physical particles: {weights.sum():.2e}")
            print(f"  Energy range: [{KE_per_particle.min() * J_TO_EV:.2e}, {KE_per_particle.max() * J_TO_EV:.2e}] eV\n")

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

        # Number flux contribution: Σ(w_i * |v_r,i|)
        flux_contribution, _, _ = np.histogram2d(
            lat_impact, lon_impact,
            bins=[lat_edges, lon_edges],
            weights=weights * v_r_mag  # [# particles * m/s]
        )

        # Mass-weighted count: Σ(m * w_i) - for velocity calculation
        mass_count, _, _ = np.histogram2d(
            lat_impact, lon_impact,
            bins=[lat_edges, lon_edges],
            weights=m * weights  # [amu * # particles]
        )

        # Mass flux contribution: Σ(m * w_i * |v_r,i|)
        mass_flux_contribution, _, _ = np.histogram2d(
            lat_impact, lon_impact,
            bins=[lat_edges, lon_edges],
            weights=m * weights * v_r_mag  # [amu * # particles * m/s]
        )

        # Energy flux contribution: Σ(w_i * KE_i * |v_r,i|)
        energy_flux_contribution, _, _ = np.histogram2d(
            lat_impact, lon_impact,
            bins=[lat_edges, lon_edges],
            weights=weights * KE_per_particle * v_r_mag  # [J * # particles * m/s]
        )

        return macro_count_map, count_map, flux_contribution, mass_count, mass_flux_contribution, energy_flux_contribution

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
    mass_count_total = np.zeros((n_lat, n_lon))  # [amu * # particles]
    mass_flux_total = np.zeros((n_lat, n_lon))  # [amu * # particles * m/s]
    energy_flux_total = np.zeros((n_lat, n_lon))  # [J * # particles * m/s]

    # Process each species and accumulate
    for spec_id in spec_ids:
        macro_count, count, flux_contrib, mass_count, mass_flux_contrib, energy_flux_contrib = process_species(spec_id)
        macro_count_total += macro_count
        count_map_total += count
        flux_contrib_total += flux_contrib
        mass_count_total += mass_count
        mass_flux_total += mass_flux_contrib
        energy_flux_total += energy_flux_contrib

    # ========== Compute final quantities ==========

    # 1) Shell volume density [m^-3] → [cm^-3]
    # Use physical particle counts
    den_map = np.where(shell_volume > 0, count_map_total / shell_volume, 0.0)  # [m^-3]
    den_map_cm3 = den_map * 1e-6  # [cm^-3]

    # 2) Mass-weighted average velocity [km/s]
    # v_avg = (mass_flux / mass_count) because mass_flux = Σ(m*w*v) and mass_count = Σ(m*w)
    v_r_map = np.zeros((n_lat, n_lon))
    mass_threshold = 1e10  # [amu * # particles] minimum for statistics
    # valid_mask = mass_count_total > mass_threshold
    # valid_mask = macro_count_total > 2
    # v_r_map[valid_mask] = -(mass_flux_total[valid_mask] / mass_count_total[valid_mask]) * 1e-3  # [km/s], negative for inward
    v_r_map = -(mass_flux_total / mass_count_total) * 1e-3  # [km/s], negative for inward

    # 3) Surface flux [cm^-2 s^-1]
    # Flux = Σ(w_i * |v_r,i|) / area
    # flux_map = np.where(area > 0, flux_contrib_total / area, 0.0)  # [m^-2 s^-1]
    # flux_map_cm = flux_map * 1e-4  # [cm^-2 s^-1]

    flux_map = den_map * np.abs(v_r_map)*1e3  # [m^-2 s^-1]
    flux_map_cm = flux_map * 1e-4  # [cm^-2 s^-1]

    # 4) Mass flux [amu cm^-2 s^-1]
    # Mass flux = Σ(m * w_i * |v_r,i|) / area
    mass_flux_map = np.where(area > 0, mass_flux_total / area, 0.0)  # [amu * # particles * m/s / m^2]
    mass_flux_map = mass_flux_map * 1e-4  # [amu cm^-2 s^-1]

    # 5) Energy flux [eV cm^-2 s^-1]
    # Energy flux = Σ(w_i * KE_i * |v_r,i|) / area
    energy_flux_map_SI = np.where(area > 0, energy_flux_total / area, 0.0)  # [J * # particles * m/s / m^2] = [W/m^2]
    energy_flux_map = energy_flux_map_SI * J_TO_EV * 1e-4  # [eV cm^-2 s^-1]

    if debug:
        # ========== Debug output ==========
        print("=" * 60)
        print("FINAL MAPS STATISTICS")
        print("=" * 60)
        print(f"Total physical particles: {count_map_total.sum():.2e}")
        print(f"den_map_cm3:     [{den_map_cm3[den_map_cm3 > 0].min():.2e}, {den_map_cm3.max():.2e}] cm^-3")
        print(f"flux_map_cm:     [{flux_map_cm[flux_map_cm > 0].min():.2e}, {flux_map_cm.max():.2e}] cm^-2 s^-1")
        # print(f"v_r_map:         [{v_r_map[valid_mask].min():.2f}, {v_r_map[valid_mask].max():.2f}] km/s")
        print(f"v_r_map:         [{v_r_map.min():.2f}, {v_r_map.max():.2f}] km/s")
        print(f"mass_flux_map:   [{mass_flux_map[mass_flux_map > 0].min():.2e}, {mass_flux_map.max():.2e}] amu cm^-2 s^-1")
        print(f"energy_flux_map: [{energy_flux_map[energy_flux_map > 0].min():.2e}, {energy_flux_map.max():.2e}] eV cm^-2 s^-1")
        print("=" * 60 + "\n")

    return flux_map_cm, lat_centers, lon_centers, v_r_map, macro_count_total, den_map_cm3, mass_flux_map, energy_flux_map
