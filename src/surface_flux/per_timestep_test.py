#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Goal
-------------------------------------
For each NPZ snapshot, compute an *instantaneous inward* particle precipitation
flux onto a spherical obstacle (Mercury) as a lat/lon heatmap, per species and
total, and save each map as a PNG.

Method
-------------------------------------
Estimate the inward *through-surface* number flux density using a thin
shell control volume adjacent to the surface:

  Shell: R < r < R + Δr  (Δr = 0.5*dx)

For a given surface patch (lat/lon bin) with area A_bin, the inward number flux
density Φ (#/m^2/s) is physically:

  Φ = ∫_{v_n<0} f(r,v) (-v_n) d^3v

In a macroparticle representation, each particle i represents W_s real particles
(# per macroparticle). In a thin shell of thickness Δr, a particle with inward
normal speed v_n^- ≡ max(0, -v_n) traverses distance v_n^- * dt toward the surface
during dt. Converting a *near-surface population* to a *through-surface rate*
introduces a 1/Δr factor (a thin-layer approximation):

  contribution to (number rate per area) ≈ (W_i * v_n^-) / Δr

Then for each bin:
  Φ_bin ≈ (1 / A_bin) * Σ_{i in bin & shell} [ W_{sid(i)} * v_n^- / Δr ]

Units check:
  W: #/macroparticle
  v_n^-: m/s
  Δr: m
  => W * v_n^- / Δr = #/s   (a number rate)
  Divide by A_bin (m^2) => #/m^2/s  (flux density)

Then integrate over the whole sphere to get a total inward precipitation rate:
  Rate_total = ∑_bins Φ_bin * A_bin   [#/s]

Plotting
-------------------------------------
Save one PNG per snapshot. All flux-map PNGs share fixed color limits in
log10 space: log10(Φ) in [0, 12]. Compute plot = log10(max(Φ, EPS)) and then
set vmin=0, vmax=12 on that array.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Upstream conditions
# -----------------------
species = np.array(['H+', 'H+', 'He++', 'He++'])
sim_ppc = np.array([24, 24, 11, 11], dtype=float)                    # macroparticles per cell per species
sim_den = np.array([38.0e6, 76.0e6, 1.0e6, 2.0e6], dtype=float)      # upstream density (/m^3)
sim_vel = np.array([400.e3, 700.0e3, 400.e3, 700.0e3], dtype=float)  # m/s
species_mass = np.array([1.0, 1.0, 4.0, 4.0], dtype=float)           # amu
species_charge = np.array([1.0, 1.0, 2.0, 2.0], dtype=float)         # e

# Simulation grid and obstacle radius
sim_dx = 75.e3
sim_dy = 75.e3
sim_dz = 75.e3
sim_robs = 2440.e3

# -----------------------
# SETTINGS
# -----------------------
NPZ_GLOB = "/Volumes/data_backup/mercury/extreme/High_HNHV/CPN_HNHV/10/particles/Subset_*.npz"

OUT_FOLDER = "/Users/danywaller/Projects/mercury/extreme/surface_flux_maps_test/"
os.makedirs(OUT_FOLDER, exist_ok=True)

# Geometry / grid
R_SURF_M = sim_robs
DX_M = sim_dx
DELTA_R_M = 0.5 * DX_M  # shell thickness [m]

# Lat/lon bins (edges)
DLAT = 2.0
DLON = 2.0
LAT_EDGES = np.arange(-90.0, 90.0 + DLAT, DLAT)
LON_EDGES = np.arange(-180.0, 180.0 + DLON, DLON)  # [-180, 180)

# Plot controls
LOG10 = True
EPS = 1e-30     # floor before log10 to avoid -inf
CMAP = "inferno"

# Fixed limits across all output maps in log10(Φ) space
LOG10_VMIN = 0.0
LOG10_VMAX = 12.0

SAVE_PER_SPECIES = True


# -----------------------
# Helpers
# -----------------------
def bin_areas_sphere(R, lat_edges_deg, lon_edges_deg):
    """
    Area of each lat/lon bin on a sphere of radius R.

    For lat edges φ_i (radians) and lon edges λ_j (radians), the spherical
    quadrilateral area is:

      A_ij = R^2 * (λ_{j+1} - λ_j) * (sin φ_{i+1} - sin φ_i)

    Returns A with shape (Nlat, Nlon).
    """
    lat = np.deg2rad(lat_edges_deg)
    lon = np.deg2rad(lon_edges_deg)
    dlon = np.diff(lon)[None, :]  # (1, Nlon)
    sin_dlat = (np.sin(lat[1:]) - np.sin(lat[:-1]))[:, None]  # (Nlat, 1)
    return (R**2) * sin_dlat * dlon


def macro_weights(sim_den_arr, sim_ppc_arr, V_cell):
    """
    Macroparticle weight W_s (# real particles per macroparticle) per species s:

      W_s = n_up,s * V_cell / ppc_s

    where:
      n_up,s  : upstream density for species s (#/m^3)
      V_cell  : cell volume (m^3), here dx*dy*dz
      ppc_s   : macroparticles per cell for species s (dimensionless)
    """
    return sim_den_arr * V_cell / sim_ppc_arr


def flux_maps_snapshot_shell(npz_path, R, delta_r_m, lat_edges, lon_edges, W_by_sid):
    """
    Compute per-species and total inward flux maps from a single snapshot.

    Steps
    -----
    1) Load particle species id sid, positions r=(rx,ry,rz) and velocities v=(vx,vy,vz).
    2) Compute outward unit normal at each particle: n_hat = r / |r|.
    3) Compute outward normal velocity component: v_n = v · n_hat.
       Inward speed is v_n^- = max(0, -v_n).
    4) Select particles in the shell: R < |r| < R+Δr and with v_n^- > 0.
    5) Convert particle positions to (lat, lon) and bin into lat/lon cells.
    6) For each species s and each bin, accumulate the number-rate:
         rate_bin = Σ ( W_s * v_n^- / Δr )    [#/s]
    7) Divide by bin area A_bin to get flux density:
         Φ_bin = rate_bin / A_bin            [#/m^2/s]
    8) Integrated inward rate over entire surface:
         Rate_total = Σ_bins Φ_bin * A_bin   [#/s]

    Returns
    -------
    flux_by_sid : (Ns, Nlat, Nlon) array, #/m^2/s
    flux_all    : (Nlat, Nlon) array, #/m^2/s
    t           : python float, seconds
    total_rate  : python float, #/s
    """
    p = np.load(npz_path)

    rx, ry, rz = p["rx"], p["ry"], p["rz"]
    vx, vy, vz = p["vx"], p["vy"], p["vz"]
    sid = p["sid"].astype(int)

    # extract time
    t = np.asarray(p["time"]).item()

    Ns = len(W_by_sid)
    Nlat = len(lat_edges) - 1
    Nlon = len(lon_edges) - 1

    # Radius and outward unit normal
    r = np.sqrt(rx*rx + ry*ry + rz*rz)
    invr = np.where(r > 0, 1.0 / r, 0.0)
    nx, ny, nz = rx * invr, ry * invr, rz * invr

    # Normal component of velocity (positive = outward, negative = inward)
    vn = vx * nx + vy * ny + vz * nz

    # Inward normal speed (m/s), i.e. magnitude of inward component only
    vn_in = np.maximum(0.0, -vn)

    # Shell selection: near surface and moving inward
    shell = (r > R) & (r < (R + delta_r_m)) & (vn_in > 0.0)

    # Bin areas on the sphere (constant for all snapshots)
    A = bin_areas_sphere(R, lat_edges, lon_edges)

    # If no inward shell particles, return zeros
    if not np.any(shell):
        flux_by_sid = np.zeros((Ns, Nlat, Nlon), float)
        flux_all = np.zeros((Nlat, Nlon), float)
        return flux_by_sid, flux_all, t, 0.0

    # Lat/lon for shell particles in degrees
    # lat = asin(z/r), lon = atan2(y,x)
    lat = np.rad2deg(np.arcsin(rz[shell] / r[shell]))
    lon = np.rad2deg(np.arctan2(ry[shell], rx[shell]))
    lon = (lon + 180.0) % 360.0 - 180.0  # wrap to [-180, 180)

    # Convert lat/lon to bin indices based on edges
    ilat = np.searchsorted(lat_edges, lat, side="right") - 1
    ilon = np.searchsorted(lon_edges, lon, side="right") - 1

    # Keep only points that fall within the defined bin ranges
    ok = (ilat >= 0) & (ilat < Nlat) & (ilon >= 0) & (ilon < Nlon)
    ilat = ilat[ok]
    ilon = ilon[ok]
    sid_s = sid[shell][ok]
    vn_in_s = vn_in[shell][ok]

    # Drop any particles whose sid is outside 0..Ns-1
    valid_sid = (sid_s >= 0) & (sid_s < Ns)
    ilat = ilat[valid_sid]
    ilon = ilon[valid_sid]
    sid_s = sid_s[valid_sid]
    vn_in_s = vn_in_s[valid_sid]

    # rate_by_sid[s, i, j] accumulates (#/s) into bin (i,j) for species s
    # using Σ (W_s * v_n^- / Δr).
    rate_by_sid = np.zeros((Ns, Nlat, Nlon), float)

    for s in range(Ns):
        m = (sid_s == s)
        if not np.any(m):
            continue
        # Each particle adds:
        #   W_s * v_n^- / Δr    [#/s]
        np.add.at(rate_by_sid[s], (ilat[m], ilon[m]),
                  W_by_sid[s] * (vn_in_s[m] / delta_r_m))

    # Convert (#/s) per bin into flux density (#/m^2/s) by dividing by bin area
    flux_by_sid = rate_by_sid / A[None, :, :]
    flux_all = np.sum(flux_by_sid, axis=0)

    # Integrated inward rate over whole surface: Σ Φ_bin A_bin
    total_rate = np.asarray(np.sum(flux_all * A)).item()

    return flux_by_sid, flux_all, t, total_rate


def save_flux_map_png(outpath, lon_edges, lat_edges, flux2d, *,
                      title, log10=True, eps=1e-30, cmap="inferno",
                      cbar_label=None, vmin=None, vmax=None):
    """
    Save lat/lon heatmap.

    If log10=True, plot log10(max(flux, eps)). Then vmin/vmax apply in log10 space.
    vmin/vmax are passed to pcolormesh to enforce consistent colormap limits.
    """
    plot = np.log10(np.maximum(flux2d, eps)) if log10 else flux2d

    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    pm = ax.pcolormesh(lon_edges, lat_edges, plot, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(title)

    cb = fig.colorbar(pm, ax=ax, pad=0.02)
    if cbar_label is None:
        cbar_label = "log10(#/m²/s)" if log10 else "#/m²/s"
    cb.set_label(cbar_label)

    fig.savefig(outpath, dpi=250)
    plt.close(fig)


# -----------------------
# Main
# -----------------------
files = sorted(glob.glob(NPZ_GLOB))
if len(files) == 0:
    raise FileNotFoundError(f"No files matched: {NPZ_GLOB}")

# Macroparticle weights W_s: # real particles represented by one macro in species s
V_cell = sim_dx * sim_dy * sim_dz
W_by_sid = macro_weights(sim_den, sim_ppc, V_cell)
Ns = len(W_by_sid)

times = []
total_rates = []

for i, f in enumerate(files):
    base = os.path.basename(f).replace(".npz", "")
    print(f"[{i+1}/{len(files)}] Processing {base} ...")

    flux_by_sid, flux_all, t, total_rate = flux_maps_snapshot_shell(f, R=R_SURF_M, delta_r_m=DELTA_R_M,
                                                                    lat_edges=LAT_EDGES, lon_edges=LON_EDGES,
                                                                    W_by_sid=W_by_sid)

    times.append(t)
    total_rates.append(total_rate)

    # Total map (all species)
    out_total = os.path.join(OUT_FOLDER, f"{base}_flux_total.png")
    save_flux_map_png(
        out_total, LON_EDGES, LAT_EDGES, flux_all,
        title=f"Total (all) inward surface flux, t={t:.3f} s",
        log10=LOG10, eps=EPS, cmap=CMAP,
        cbar_label="log10(#/m²/s)" if LOG10 else "#/m²/s",
        vmin=LOG10_VMIN if LOG10 else None,
        vmax=LOG10_VMAX if LOG10 else None
    )

    # Per-species maps
    if SAVE_PER_SPECIES:
        for s in range(Ns):
            sid_label = s  # 0-based
            out_s = os.path.join(OUT_FOLDER, f"{base}_flux_sid{sid_label:02d}_{species[s]}.png")
            save_flux_map_png(
                out_s, LON_EDGES, LAT_EDGES, flux_by_sid[s],
                title=f"{species[s]} ({sid_label}) inward surface flux, t={t:.3f} s",
                log10=LOG10, eps=EPS, cmap=CMAP,
                cbar_label="log10(#/m²/s)" if LOG10 else "#/m²/s",
                vmin=LOG10_VMIN if LOG10 else None,
                vmax=LOG10_VMAX if LOG10 else None
            )

# plot integrated inward rate vs time
times = np.asarray(times)
total_rates = np.asarray(total_rates)

# Sort by time just in case
order = np.argsort(times)
times = times[order]
total_rates = total_rates[order]

fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
ax.plot(times, total_rates, lw=2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Integrated inward rate (#/s)")
ax.grid(True, alpha=0.3)
ax.set_title("Total precipitation rate vs time")

out_ts = os.path.join(OUT_FOLDER, "total_surface_precip_rate_vs_time.png")
fig.savefig(out_ts, dpi=250)
plt.show()

print(f"Done. Wrote maps + timeseries to:\n  {OUT_FOLDER}")
