#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports:
import numpy as np


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

        vr_map = np.zeros((n_lat, n_lon))
        n_floor = 1e10  # [# particles] minimum for reliable statistics
        valid_bins = n_hist > n_floor
        if np.any(valid_bins):
            v_r_raw = v_r_hist[valid_bins] / n_hist[valid_bins]  # [m/s]
            vr_map[valid_bins] = v_r_raw * 1e-3  # [km/s]
            # Clip unphysical velocities (should not be needed)
            vr_map[valid_bins] = np.clip(vr_map[valid_bins], -3000, 0)

        flx_map = vr_map * 1e3 * n_shell_map  # [m^-2 s^-1]

        return flx_map, vr_map, count_map, n_shell_map  # [m^-2 s^-1, km/s, #, m^-3]

    # ========== Main logic: sum over species ==========
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
        spec_id = 0 if species == "protons" else 1
        flux_map, v_r_map, count_map, n_shell_map = process_species(spec_id)

    # Convert flux to [cm^-2 s^-1]
    flux_map_cm = flux_map * 1e-4  # [m^-2 s^-1] → [cm^-2 s^-1]

    # ========== FINAL OUTPUT STATISTICS ==========
    print("=" * 60)
    print("FINAL MAPS STATISTICS")
    print("=" * 60)
    print(f"n_shell_map: [{np.nanmin(n_shell_map):.2e}, {np.nanmax(n_shell_map):.2e}] m^-3")
    print(f"             [{np.nanmin(n_shell_map)*1e-6:.2e}, {np.nanmax(n_shell_map)*1e-6:.2e}] cm^-3")
    print(f"flux_map_cm: [{np.nanmin(flux_map_cm):.2e}, {np.nanmax(flux_map_cm):.2e}] cm^-2 s^-1")
    print(f"v_r_map:     [{np.nanmin(v_r_map):.2f}, {np.nanmax(v_r_map):.2f}] km/s")
    print("=" * 60 + "\n")

    return flux_map_cm, lat_centers, lon_centers, v_r_map, count_map, n_shell_map
