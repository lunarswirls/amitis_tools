#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.surface_flux.flux_utils as flux_utils
import src.helper_utils as helper_utils

debug = True

# cases = ["inert_planetward", "inert_sunward"]
cases = ["jeremias_validation"]

for case in cases:
    if "inert_sunward" in case:
        main_path = f"/Users/danywaller/Projects/mercury/inert_small_body_sunward_IMF/"
        case = "SW_IMF"
    elif "inert_planetward" in case:
        main_path = f"/Users/danywaller/Projects/mercury/inert_small_body_planetward_IMF/"
        case = "PW_IMF"
    elif "validation" in case:
        main_path = f"/Volumes/data_backup/2026_02_12_LongPrecipValidation/"
        case = "prec_valid"

    if "inert" in case:
        output_folder = f"/Users/danywaller/Projects/mercury/precipitation_validation_test_cases_1sec_n11_dR100km/"

        plot_meth = "log"  # raw, log, lognorm
        run_species = "all"  # 'all'

        outdir = output_folder + f"{run_species}"
        os.makedirs(outdir, exist_ok=True)

        species = np.array(['H+'])  # The order is important and it should be based on Amitis.inp file
        sim_ppc = np.array([12])  # Number of particles per species, based on Amitis.inp
        sim_den = np.array([40.0e6])  # [/m^3]
        sim_vel = np.array([400.e3])  # [m/s]

        # Species properties
        species_mass = np.array([1.0])  # [amu] proton1
        species_charge = np.array([1.0])  # [e] proton1

        sim_dx = 200.e3  # simulation cell size based on Amitis.inp [m]
        sim_dy = 200.e3  # simulation cell size based on Amitis.inp [m]
        sim_dz = 200.e3  # simulation cell size based on Amitis.inp [m]
        sim_robs = 2440.e3  # obstacle radius based on Amitis.inp [m]

        nlat = 90
        nlon = 180

        select_R = 2480.e3  # the radius of a sphere + 1/2 grid cell above the surface for particle selection [m]

        all_particles_directory = main_path + 'precipitation_1sec_n11/'
        all_particles_filename = all_particles_directory + f"{case}_all_particles_at_surface.npz"
    else:
        output_folder = f"/Users/danywaller/Projects/precipitation_validation_test_case_trange10sec/"
        # output_folder = f"/Users/danywaller/Projects/precipitation_validation_test_case/"

        plot_meth = "log"  # raw, log, lognorm
        run_species = "all"  # 'all'

        outdir = output_folder
        os.makedirs(outdir, exist_ok=True)

        species = np.array(['H+'])  # The order is important and it should be based on Amitis.inp file
        sim_ppc = np.array([15])  # Number of particles per species, based on Amitis.inp
        sim_den = np.array([10.0e6])  # [/m^3]
        sim_vel = np.array([400.e3])  # [m/s]

        dt = 0.001

        # Species properties
        species_mass = np.array([1.0])  # [amu] proton1
        species_charge = np.array([1.0])  # [e] proton1

        sim_dx = 80.e3  # simulation cell size based on Amitis.inp [m]
        sim_dy = 80.e3  # simulation cell size based on Amitis.inp [m]
        sim_dz = 80.e3  # simulation cell size based on Amitis.inp [m]
        sim_robs = 1500e3  # obstacle radius based on Amitis.inp [m]
        select_R = 1580.e3

        nlat = 90
        nlon = 180

        all_particles_directory = main_path + 'precipitation_trange10sec/'
        all_particles_filename = all_particles_directory + f"{case}_all_particles_at_surface.npz"

    flux_cm, lat_centers, lon_centers, v_r_map, count_map, n_shell_map, mass_flux_map, energy_flux_map = \
        flux_utils.compute_radial_flux(
            all_particles_filename=all_particles_filename,
            dt=dt,
            sim_dx=sim_dx, sim_dy=sim_dy, sim_dz=sim_dz,
            sim_ppc=sim_ppc, sim_den=sim_den, spec_map=species,
            species_mass=species_mass, species_charge=species_charge,
            R_M=sim_robs, select_R=select_R,
            species=run_species,
            n_lat=nlat, n_lon=nlon, debug=debug
        )

    n_lat = len(lat_centers)
    n_lon = len(lon_centers)

    # Rebuild bin edges consistent with centers
    lon_edges = np.linspace(-180.0, 180.0, n_lon + 1)
    lat_edges = np.linspace(-90.0, 90.0, n_lat + 1)

    # ========== 2D maps with units ==========
    cnts = count_map.copy()  # [# particles]
    den_cm3 = n_shell_map.copy()  # [cm^-3] shell volume density
    vr = v_r_map.copy()  # [km/s]
    flux = flux_cm.copy()  # [cm^-2 s^-1]
    mass_flux = mass_flux_map.copy()  # [amu cm^-2 s^-1]
    energy_flux = energy_flux_map.copy()  # [eV cm^-2 s^-1]

    vr_abs = np.abs(vr)  # [km/s]
    flux_abs = np.abs(flux)  # [cm^-2 s^-1]
    mass_flux_abs = np.abs(mass_flux)  # [amu cm^-2 s^-1]
    energy_flux_abs = np.abs(energy_flux)  # [eV cm^-2 s^-1]

    # Set low-count pixels to NaN
    mask = count_map <= 1e-30
    cnts[mask] = np.nan
    den_cm3[mask] = np.nan
    vr_abs[mask] = np.nan
    flux_abs[mask] = np.nan
    mass_flux_abs[mask] = np.nan
    energy_flux_abs[mask] = np.nan

    # ========== Logarithmic maps ==========
    log_cnts = helper_utils.safe_log10(cnts)
    log_den  = helper_utils.safe_log10(den_cm3)  # log10(cm^-3)
    log_vel  = helper_utils.safe_log10(vr_abs)   # log10(km/s)
    log_flx  = helper_utils.safe_log10(flux_abs) # log10(cm^-2 s^-1)
    log_mass_flux = helper_utils.safe_log10(mass_flux_abs)  # log10(amu cm^-2 s^-1)
    log_energy_flux = helper_utils.safe_log10(energy_flux_abs)  # log10(eV cm^-2 s^-1)

    # Total upstream number density from quasi-neutrality Σ(Z_i * n_i) [m^-3]
    sim_den_tot = np.sum(species_charge * sim_den)  # [m^-3]

    # Mass-weighted upstream velocity [km/s]
    # v_avg = Σ(m_i * n_i * v_i) / Σ(m_i * n_i)
    mass_weighted_velocity = np.sum(species_mass * sim_den * sim_vel) / np.sum(species_mass * sim_den)
    sim_vel_tot = mass_weighted_velocity * 1e-3  # [m/s] → [km/s]

    # Upstream flux using mass-weighted velocity [cm^-2 s^-1]
    sim_flux_upstream = sim_den_tot * mass_weighted_velocity * 1e-4  # [m^-3 * m/s] → [cm^-2 s^-1]

    # Upstream mass flux [amu cm^-2 s^-1]
    # Mass flux = Σ(m_i * n_i * v_i) * 1e-4 [amu/m^3 * m/s * (m^2/cm^2)]
    sim_mass_flux_upstream = np.sum(species_mass * sim_den * sim_vel) * 1e-4  # [amu cm^-2 s^-1]

    # Upstream energy flux [eV cm^-2 s^-1]
    # Energy flux = Σ(0.5 * m_i * n_i * v_i^3) * conversion
    AMU_TO_KG = 1.66053906660e-27
    J_TO_EV = 6.241509074e18
    sim_energy_flux_upstream = 0.5 * np.sum(species_mass * AMU_TO_KG * sim_den * sim_vel**3) * J_TO_EV * 1e-4  # [eV cm^-2 s^-1]

    # Normalized quantities
    log_den_norm = helper_utils.safe_log10(den_cm3 / (sim_den_tot * 1e-6))  # [cm^-3] / [cm^-3]
    log_vel_norm = helper_utils.safe_log10(vr_abs / sim_vel_tot)  # [km/s] / [km/s]
    log_flx_norm = helper_utils.safe_log10(flux_abs / sim_flux_upstream)  # [cm^-2 s^-1] / [cm^-2 s^-1]
    log_mass_flux_norm = helper_utils.safe_log10(mass_flux_abs / sim_mass_flux_upstream)
    log_energy_flux_norm = helper_utils.safe_log10(energy_flux_abs / sim_energy_flux_upstream)

    # Debug output
    print(f"Upstream normalization values:")
    print(f"  Total density: {sim_den_tot * 1e-6:.1f} cm^-3")
    print(f"  Mass-weighted velocity: {sim_vel_tot:.1f} km/s")
    print(f"  Upstream flux: {sim_flux_upstream:.2e} cm^-2 s^-1")
    print(f"  Upstream mass flux: {sim_mass_flux_upstream:.2e} amu cm^-2 s^-1")
    print(f"  Upstream energy flux: {sim_energy_flux_upstream:.2e} eV cm^-2 s^-1")

    # Define fields for plotting (6 fields in 3x2 layout)
    fields_raw = [
        (cnts, (0, 12), "viridis", "# particles"),
        (den_cm3, (0, 10), "cividis", r"$n$ [cm$^{-3}$]"),
        (vr_abs, (0, 400), "plasma", r"$|v_r|$ [km/s]"),
        (flux_abs, (0, 5e8), "jet", r"$F_r$ [cm$^{-2}$ s$^{-1}$]"),
        (mass_flux_abs, (0, 1e12), "copper", r"$F_{mass}$ [amu cm$^{-2}$ s$^{-1}$]"),
        (energy_flux_abs, (0, 4e13), "inferno", r"$F_{energy}$ [eV cm$^{-2}$ s$^{-1}$]")
    ]

    fields_log = [
        (cnts, (0, 12), "viridis", "# particles"),
        (log_den, (0, 10), "cividis", r"log$_{10}$($n$) [cm$^{-3}$]"),
        (log_vel, (0, 400), "plasma", r"log$_{10}$($|v_r|$) [km s$^{-1}$]"),
        (log_flx, (0, 8.5), "jet", r"log$_{10}$($F_r$) [cm$^{-2}$ s$^{-1}$]"),
        (log_mass_flux, (0, 12), "copper", r"log$_{10}$($F_{mass}$) [amu cm$^{-2}$ s$^{-1}$]"),
        (log_energy_flux, (0, 13), "inferno", r"log$_{10}$($F_{energy}$) [eV cm$^{-2}$ s$^{-1}$]")
    ]

    fields_log_norm = [
        (cnts, (np.nanmin(cnts), np.nanmax(cnts)), "viridis", "# particles"),
        (log_den_norm, (-1, 1), "cividis", r"log$_{10}$($n/n_0$)"),
        (log_vel_norm, (-1.0, 1.0), "plasma", r"log$_{10}$($|v_r|/v_0$)"),
        (log_flx_norm, (2.0, 5), "jet", r"log$_{10}$($F_r/F_0$)"),
        (log_mass_flux_norm, (1, 5), "winter", r"log$_{10}$($F_{mass}/F_{mass,0}$)"),
        (log_energy_flux_norm, (1, 5), "inferno", r"log$_{10}$($F_{energy}/F_{energy,0}$)")
    ]

    if plot_meth == 'raw':
        use_fields = fields_raw
    elif plot_meth == 'log':
        use_fields = fields_log
    elif plot_meth == 'lognorm':
        use_fields = fields_log_norm
    else:
        raise ValueError(f"Plotting method {plot_meth} not recognized! Use one of 'raw', 'log', or 'lognorm'")

    titles = ["Counts", "Shell density", "Radial velocity", "Precipitation", "Mass flux", "Energy flux"]

    # ---- 3. Plot in Hammer projection (3x2 layout) ----
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 13.5),  # Increased height for third row
        subplot_kw={"projection": "hammer"}
    )

    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for ax, (data, clim, cmap, cblabel), title in zip(axes, use_fields, titles):
        ax.set_facecolor("white")
        ax.grid(True, linestyle="dotted", color="gray")

        # IMPORTANT: use edges (length n+1) and data (n_lat, n_lon)
        pcm = ax.pcolormesh(
            np.radians(lon_edges),  # X: shape (n_lon+1,)
            np.radians(lat_edges),  # Y: shape (n_lat+1,)
            data,                   # C: shape (n_lat, n_lon)
            cmap=cmap,
            shading="flat"
        )
        pcm.set_clim(*clim)

        cbar = plt.colorbar(
            pcm,
            ax=ax,
            orientation="horizontal",
            pad=0.05,
            shrink=0.85
        )
        cbar.set_label(cblabel, fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        ax.set_title(title, fontsize=20)

         # Longitude ticks (-170 to 170 every n °)
        lon_ticks_deg = np.arange(-120, 121, 60)
        lon_ticks_rad = np.deg2rad(lon_ticks_deg)

        # Latitude ticks (-90 to 90 every n °)
        lat_ticks_deg = np.arange(-60, 61, 30)
        lat_ticks_rad = np.deg2rad(lat_ticks_deg)

        # Apply to the current axis
        ax.set_xticks(lon_ticks_rad)
        ax.set_yticks(lat_ticks_rad)

        # Label ticks in degrees
        ax.set_xticklabels([f"{int(l)}°" for l in lon_ticks_deg])
        ax.set_yticklabels([f"{int(l)}°" for l in lat_ticks_deg])

    # Generate title based on species selection
    if "SW" in case:
        title_name = "Sunward IMF"
    if "PW" in case:
        title_name = "Planetward IMF"
    else:
        title_name = "Validation"

    stitle = f"{title_name}: One species (H+)"
    plot_fname = f"{case}_cnts_den_vr_precipitation_mass_ener_one_species_{plot_meth}vals"

    fig.suptitle(stitle, fontsize=20, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

    outfile_png = os.path.join(outdir, plot_fname)
    plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
    print("Saved figure:", outfile_png)
    # plt.show()
    plt.close(fig)

    fields_raw = [
        (flux_abs, (0, 2e9), "jet", r"$F_r$ [cm$^{-2}$ s$^{-1}$]"),
        (mass_flux_abs, (0, 1e14), "copper", r"$F_{mass}$ [amu cm$^{-2}$ s$^{-1}$]"),
        (energy_flux_abs, (0, 4e16), "inferno", r"$F_{energy}$ [eV cm$^{-2}$ s$^{-1}$]")
    ]

    if plot_meth == 'raw':
        use_fields = fields_raw
    elif plot_meth == 'log':
        use_fields = fields_log
    elif plot_meth == 'lognorm':
        use_fields = fields_log_norm
    else:
        raise ValueError(f"Plotting method {plot_meth} not recognized! Use one of 'raw', 'log', or 'lognorm'")

    titles = ["Precipitation", "Mass flux", "Energy flux"]

    # ---- 3. Plot in Hammer projection (3x1 layout) ----
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10),  # Increased height for third row
        subplot_kw={"projection": "hammer"}
    )

    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for ax, (data, clim, cmap, cblabel), title in zip(axes, use_fields, titles):
        ax.set_facecolor("white")
        ax.grid(True, linestyle="dotted", color="gray")

        # IMPORTANT: use edges (length n+1) and data (n_lat, n_lon)
        pcm = ax.pcolormesh(
            np.radians(lon_edges),  # X: shape (n_lon+1,)
            np.radians(lat_edges),  # Y: shape (n_lat+1,)
            data,                   # C: shape (n_lat, n_lon)
            cmap=cmap,
            shading="flat"
        )
        pcm.set_clim(*clim)

        cbar = plt.colorbar(
            pcm,
            ax=ax,
            pad=0.01,
            shrink=0.8
        )
        cbar.set_label(cblabel, fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        ax.set_title(title, fontsize=20)

         # Longitude ticks (-170 to 170 every n °)
        lon_ticks_deg = np.arange(-120, 121, 60)
        lon_ticks_rad = np.deg2rad(lon_ticks_deg)

        # Latitude ticks (-90 to 90 every n °)
        lat_ticks_deg = np.arange(-60, 61, 30)
        lat_ticks_rad = np.deg2rad(lat_ticks_deg)

        # Apply to the current axis
        ax.set_xticks(lon_ticks_rad)
        ax.set_yticks(lat_ticks_rad)

        # Label ticks in degrees
        ax.set_xticklabels([f"{int(l)}°" for l in lon_ticks_deg])
        ax.set_yticklabels([f"{int(l)}°" for l in lat_ticks_deg])

    # Generate title based on species selection
    if "SW" in case:
        title_name = "Sunward IMF"
    if "PW" in case:
        title_name = "Planetward IMF"
    else:
        title_name = "Validation"

    # stitle = f"dt = 1 second, n = 28"
    stitle = title_name
    plot_fname = f"{case}_precipitation_mass_ener_one_species_{plot_meth}vals"

    fig.suptitle(
        stitle,
        fontsize=20,
        x=0.55,  # force horizontal center
        y=0.97,
        ha="left"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

    outfile_png = os.path.join(outdir, plot_fname)
    plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
    print("Saved figure:", outfile_png)
    # plt.show()
    plt.close(fig)

    def cos_sza_map(lat_centers_deg, lon_centers_deg, *, subsolar_lat_deg=0.0, subsolar_lon_deg=0.0):
        """
        Returns mu = cos(SZA) on a (n_lat, n_lon) grid defined by 1D centers.
        Spherical law of cosines for the angular distance from subsolar point:
          mu = cos(SZA) = sin(lat)sin(lat_s) + cos(lat)cos(lat_s)cos(dlon)
        """
        lat = np.deg2rad(lat_centers_deg)[:, None]          # (n_lat, 1)
        lon = np.deg2rad(lon_centers_deg)[None, :]          # (1, n_lon)
        lat_s = np.deg2rad(subsolar_lat_deg)
        lon_s = np.deg2rad(subsolar_lon_deg)

        dlon = lon - lon_s
        mu = np.sin(lat) * np.sin(lat_s) + np.cos(lat) * np.cos(lat_s) * np.cos(dlon)
        return mu  # (n_lat, n_lon)

    def weighted_mean(x, w):
        return np.sum(w * x) / np.sum(w)

    def weighted_var(x, w):
        m = weighted_mean(x, w)
        return np.sum(w * (x - m)**2) / np.sum(w)

    def weighted_corr(x, y, w):
        mx, my = weighted_mean(x, w), weighted_mean(y, w)
        cov = np.sum(w * (x - mx) * (y - my)) / np.sum(w)
        vx = weighted_var(x, w)
        vy = weighted_var(y, w)
        if vx <= 0 or vy <= 0:
            return np.nan
        return cov / np.sqrt(vx * vy)

    def fit_power_law_mu(y, mu, w, *, mu_min=1e-3):
        """
        Fit y ≈ A * mu^p on dayside using weighted least squares in log space.
        Returns (A, p, R2_w) where R2_w is weighted R^2 in *linear* y space.
        """
        # keep only finite and positive mu and y
        good = np.isfinite(y) & np.isfinite(mu) & np.isfinite(w) & (mu > mu_min) & (y > 0)
        if np.count_nonzero(good) < 10:
            return np.nan, np.nan, np.nan

        yy = y[good]
        mm = mu[good]
        ww = w[good]

        # log fit: log(y) = log(A) + p log(mu)
        X = np.vstack([np.ones_like(mm), np.log(mm)]).T  # (N,2)
        Y = np.log(yy)

        # weighted normal equations
        W = ww / np.sum(ww)
        XtW = X.T * W
        beta = np.linalg.solve(XtW @ X, XtW @ Y)  # [logA, p]
        logA, p = beta
        A = np.exp(logA)

        # weighted R^2 in linear space
        yhat = A * (mm**p)
        ybar = weighted_mean(yy, ww)
        ss_res = np.sum(ww * (yy - yhat)**2)
        ss_tot = np.sum(ww * (yy - ybar)**2)
        R2w = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return A, p, R2w

    # ---- Cos(SZA) grid ----
    mu = cos_sza_map(lat_centers, lon_centers, subsolar_lat_deg=0.0, subsolar_lon_deg=0.0)

    # ---- Pixel-area weights ~ cos(lat) ----
    lat_rad = np.deg2rad(lat_centers)[:, None]
    area_w = np.cos(lat_rad) * np.ones((len(lat_centers), len(lon_centers)))

    # ---- Dayside mask ----
    dayside = (mu > 0)

    # Choose what to test against cos(SZA)
    tests = {
        "Density": den_cm3,
        "Radial velocity": vr_abs,
        "Volume Flux": flux_abs,
        # "mass_flux_abs": mass_flux_abs,
        # "energy_flux_abs": energy_flux_abs,
        # "flux_norm": flux_abs / sim_flux_upstream,
        # "mass_flux_norm": mass_flux_abs / sim_mass_flux_upstream,
        # "energy_flux_norm": energy_flux_abs / sim_energy_flux_upstream,
    }

    print(f"\n=== {case} Cos(SZA) dayside validation ===")
    for name, ymap in tests.items():
        y = np.array(ymap, float)

        # apply your existing NaNs + dayside
        valid = np.isfinite(y) & np.isfinite(mu) & np.isfinite(area_w) & dayside
        yv  = y[valid]
        muv = mu[valid]
        wv  = area_w[valid]

        # simple weighted correlation against mu (tests ~cos law)
        r_w = weighted_corr(yv, muv, wv)

        # power-law fit y ~ A mu^p (tests departures from pure cosine)
        A, p, R2w = fit_power_law_mu(yv, muv, wv, mu_min=1e-3)

        print(f"{name:16s}  r_w(y,mu)={r_w: .3f}   fit: A={A: .3e}, p={p: .2f}, R2w={R2w: .3f}")

    def binned_stat(x, y, bins, w=None):
        idx = np.digitize(x, bins) - 1
        xc, yc = [], []
        for i in range(len(bins)-1):
            m = idx == i
            if np.count_nonzero(m) < 10:
                continue
            if w is None:
                yc.append(np.nanmedian(y[m]))
            else:
                yc.append(weighted_mean(y[m], w[m]))
            xc.append(0.5*(bins[i] + bins[i+1]))
        return np.array(xc), np.array(yc)


    mu_bins = np.linspace(0, 1, 21)

    for name, ymap in tests.items():
        y = np.array(ymap, float)

        # dayside + finite
        valid = np.isfinite(y) & np.isfinite(mu) & np.isfinite(area_w) & (mu > 0)
        muv, yv, wv = mu[valid], y[valid], area_w[valid]

        # --- compute stats for title ---
        r_w = weighted_corr(yv, muv, wv)
        A, p, R2w = fit_power_law_mu(yv, muv, wv, mu_min=1e-3)

        # binned curve (your existing helper)
        xbin, ybin = binned_stat(muv, yv, mu_bins, w=wv)

        plt.figure(figsize=(6, 4))
        plt.plot(xbin, ybin, marker="o")
        plt.xlabel(r"$\mu=\cos(\mathrm{SZA})$")
        plt.ylabel(name)

        # --- title with fit values ---
        title_fit = (
            f"{title_name}: {name} vs cos(SZA) (dayside)\n"
            rf"$r_w={r_w:.3f}$, $A={A:.3e}$, $p={p:.2f}$, $R^2_w={R2w:.3f}$"
        )
        plt.title(title_fit)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        outfile_png = os.path.join(
            outdir,
            f"{case}_{name.lower().replace(' ', '_')}_cosSZA_dayside.png"
        )
        plt.savefig(outfile_png, dpi=150, bbox_inches="tight")
        print("Saved figure:", outfile_png)
        plt.close()