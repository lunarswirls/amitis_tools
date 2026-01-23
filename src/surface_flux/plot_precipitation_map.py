#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, glob
import numpy as np
from netCDF4 import Dataset
from matplotlib.pyplot import figure, subplot, colorbar, show
import cartopy.crs as ccrs
from math import pi

# -----------------------------
# User parameters
# -----------------------------
main_path = '/Volumes/data_backup/mercury/extreme/CPN_Base/05/particles/'
species = np.array(['H+', 'He++'])
sim_ppc = [24, 11]
sim_den = [38.0e6, 1.0e6]
sim_dx = sim_dy = sim_dz = 75.e3
sim_robs = 2440.e3
select_R = 2480.e3
dphi = 2.
dtheta = 2.

# -----------------------------
# Function: combine NetCDF files
# -----------------------------
def combine_netcdfs():
    files = sorted(glob.glob(os.path.join(main_path, "*.nc")))
    print(f"Found {len(files)} NetCDF files.")

    # Initialize empty arrays
    prx, pry, prz = np.array([]), np.array([]), np.array([])
    pvx, pvy, pvz = np.array([]), np.array([]), np.array([])
    psid = np.array([])

    for f in files:
        print("Reading:", f)
        with Dataset(f, 'r') as nc:
            prx  = np.append(prx, nc.variables['rx'][:])
            pry  = np.append(pry, nc.variables['ry'][:])
            prz  = np.append(prz, nc.variables['rz'][:])
            pvx  = np.append(pvx, nc.variables['vx'][:])
            pvy  = np.append(pvy, nc.variables['vy'][:])
            pvz  = np.append(pvz, nc.variables['vz'][:])
            psid = np.append(psid, nc.variables['sid'][:])

    # Filter particles within selection radius
    idx = np.where(prx**2 + pry**2 + prz**2 <= select_R**2)[0]
    prx, pry, prz = prx[idx], pry[idx], prz[idx]
    pvx, pvy, pvz = pvx[idx], pvy[idx], pvz[idx]
    psid = psid[idx]

    print("Total number of particles after selection:", prx.size)

    return prx, pry, prz, pvx, pvy, pvz, psid


# -----------------------------
# Function to calculate moments
# -----------------------------
def calc_moments_at_surface(prx, pry, prz, pvx, pvy, pvz, psid, specie_id):
    # Select particles by species
    idx = np.where((psid == specie_id) &
                   (prx**2 + pry**2 + prz**2 <= select_R**2))[0]

    rx = prx[idx]
    ry = pry[idx]
    rz = prz[idx]
    vx = pvx[idx]
    vy = pvy[idx]
    vz = pvz[idx]

    tnp = rx.size
    print(f"Total number of particles for species {species[specie_id]}: {tnp}")

    # Initialize bins
    nx = int(360/dphi) + 2
    ny = int(180/dtheta) + 2
    cnts = np.zeros((nx, ny))
    velx = np.zeros((nx, ny))
    vely = np.zeros((nx, ny))
    velz = np.zeros((nx, ny))
    velr = np.zeros((nx, ny))
    den  = np.zeros((nx-2, ny-2))

    # Bin particles
    for i in range(tnp):
        rmag = np.sqrt(rx[i]**2 + ry[i]**2 + rz[i]**2)
        theta = np.arccos(rz[i]/rmag) * 180/pi
        phi   = np.arctan2(ry[i], rx[i]) * 180/pi
        if phi < 0: phi += 360

        xi = int(np.floor(phi/dphi - 0.5)) + 1
        yi = int(np.floor(theta/dtheta - 0.5)) + 1

        u = xi + 1
        v = yi + 1

        x = xi + 0.5 - (phi/dphi)
        y = yi + 0.5 - (theta/dtheta)

        r_dot_v = abs(rx[i]*vx[i] + ry[i]*vy[i] + rz[i]*vz[i])/rmag

        # distribute particle to four surrounding cells (bilinear)
        c_list = [(xi, yi, (1-x)*(1-y)), (u, yi, x*(1-y)),
                  (xi, v, (1-x)*y), (u, v, x*y)]

        for ii, jj, c in c_list:
            cnts[ii, jj]  += c
            velx[ii, jj] += c*vx[i]
            vely[ii, jj] += c*vy[i]
            velz[ii, jj] += c*vz[i]
            velr[ii, jj] += c*r_dot_v

    # Correct guard cells (wrap around phi)
    cnts[1,:] += cnts[-1,:]
    velx[1,:] += velx[-1,:]
    vely[1,:] += vely[-1,:]
    velz[1,:] += velz[-1,:]
    velr[1,:] += velr[-1,:]

    cnts = cnts[1:-1, 1:-1]
    velx = velx[1:-1, 1:-1]
    vely = vely[1:-1, 1:-1]
    velz = velz[1:-1, 1:-1]
    velr = velr[1:-1, 1:-1]

    # Weight and density
    weight = ((sim_dx*sim_dy*sim_dz)/sim_ppc[specie_id])
    dr = select_R - sim_robs
    dv = (select_R**2)*dr*(dphi*pi/180)*(dtheta*pi/180)

    for i in range(cnts.shape[0]):
        den[i,:] = cnts[i,:] * sim_den[specie_id] * weight / (dv*np.sin((i+0.5)*dtheta*pi/180))

    # Average velocities
    mask = cnts > 0
    velx[mask] /= cnts[mask]
    vely[mask] /= cnts[mask]
    velz[mask] /= cnts[mask]
    velr[mask] /= cnts[mask]

    vmag = np.sqrt(velx**2 + vely**2 + velz**2)
    flxr = velr * den

    # Return all relevant moments
    return cnts, den, velx, vely, velz, velr, vmag, flxr


def plot_moments_at_surface(cnts, den, velr, flxr, specie_id):
    interp = 'bilinear'

    # Avoid log10(0) warnings
    log_cnts = np.where(cnts>0, np.log10(cnts), np.nan)
    log_den  = np.where(den>0, np.log10(den), np.nan)
    log_velr = np.where(velr>0, np.log10(velr), np.nan)
    log_flxr = np.where(flxr>0, np.log10(flxr), np.nan)

    fig = figure(species[specie_id], figsize=(14,9))

    # # particles
    ax = subplot(221, projection=ccrs.Aitoff())
    ax.gridlines(color='black', linestyle='dotted')
    im = ax.imshow(np.flipud(log_cnts.T), origin="upper", interpolation=interp,
                   extent=(-180,180,-90,90), transform=ccrs.PlateCarree())
    colorbar(im, extend='neither').set_label("# particles")

    # Density
    ax = subplot(222, projection=ccrs.Aitoff())
    ax.gridlines(color='black', linestyle='dotted')
    im = ax.imshow(np.flipud(log_den.T), origin="upper", interpolation=interp,
                   extent=(-180,180,-90,90), transform=ccrs.PlateCarree())
    colorbar(im, extend='min').set_label("Density")

    # Radial velocity
    ax = subplot(223, projection=ccrs.Aitoff())
    ax.gridlines(color='black', linestyle='dotted')
    im = ax.imshow(np.flipud(log_velr.T), origin="upper", interpolation=interp,
                   extent=(-180,180,-90,90), transform=ccrs.PlateCarree())
    colorbar(im, extend='min').set_label("Radial velocity")

    # Flux
    ax = subplot(224, projection=ccrs.Aitoff())
    ax.gridlines(color='black', linestyle='dotted')
    im = ax.imshow(np.flipud(log_flxr.T), origin="upper", interpolation=interp,
                   extent=(-180,180,-90,90), transform=ccrs.PlateCarree())
    colorbar(im, extend='min').set_label("Flux")


# -----------------------------
# Main script
# -----------------------------
if __name__ == '__main__':
    # Combine all NetCDFs into arrays
    prx, pry, prz, pvx, pvy, pvz, psid = combine_netcdfs()

    # Compute moments for species 0
    cnts, den, velx, vely, velz, velr, vmag, flxr = calc_moments_at_surface(
        prx, pry, prz, pvx, pvy, pvz, psid, specie_id=0
    )

    # Plot results
    plot_moments_at_surface(cnts, den, velr, flxr, specie_id=0)
    show()

