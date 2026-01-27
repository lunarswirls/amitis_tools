#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Imports:
import os, glob
from pathlib import Path
from pyamitis.amitis_particle import *

case = "RPS_Base"
main_path = f'/Volumes/data_backup/mercury/extreme/{case}/05/'
output_folder = f"/Users/danywaller/Projects/mercury/extreme/surface_flux/"

species = np.array(['H+', 'He++'])  # The order is important and it should be based on Amitis.inp file
sim_ppc = [24, 11]  # Number of particles per species, based on Amitis.inp
sim_den = [38.0e6, 1.0e6]
sim_vel = [400.e3, 400.e3]

sim_dx = 75.e3  # simulation cell size based on Amitis.inp
sim_dy = 75.e3  # simulation cell size based on Amitis.inp
sim_dz = 75.e3  # simulation cell size based on Amitis.inp
sim_robs = 2440.e3  # obstacle radius based on Amitis.inp

select_R = 2480.e3  # the radius of a sphere above the surface for particle selection
dphi = 2.  # delta_phi   [0, 360] deg   (0:+x, 90:+y,  180:-x, 270:-y)
dtheta = 2.  # delta_theta [0, 180] deg   (0:+z)

sub_filepath = main_path + 'particles/'
sub_filename = f'Subset_{case}'

all_particles_directory = main_path + 'precipitation/'
os.makedirs(all_particles_directory, exist_ok=True)
all_particles_filename = all_particles_directory + "all_particles_at_surface.npz"
moments_filename = all_particles_directory + "moments"

def calc_moments_at_surface(specie_id):
    with np.load(all_particles_filename) as data:
        prx = data["prx"]
        pry = data["pry"]
        prz = data["prz"]
        pvx = data["pvx"]
        pvy = data["pvy"]
        pvz = data["pvz"]
        psid = data["psid"]
        num_files = data["num_files"]
        selected_radius = data["selected_radius"]

    if (selected_radius < select_R):
        print("Selected radius in particle file %e is smaller than the select_R" % (selected_radius))
        raise

    # Select for specie
    id = np.where((psid == specie_id) &
                  (prx ** 2 + pry ** 2 + prz ** 2 <= select_R ** 2))[0]
    prx = prx[id]
    pry = pry[id]
    prz = prz[id]
    pvx = pvx[id]
    pvy = pvy[id]
    pvz = pvz[id]

    tnp = prx.size  # total number of particles
    print("Total number of particles for species %s is %d" % (species[specie_id], tnp))

    # pre-allocate memory + 2 guard cells
    cnts = np.zeros((int(360 / dphi) + 2, int(180 / dtheta) + 2))
    velx = np.zeros((int(360 / dphi) + 2, int(180 / dtheta) + 2))
    vely = np.zeros((int(360 / dphi) + 2, int(180 / dtheta) + 2))
    velz = np.zeros((int(360 / dphi) + 2, int(180 / dtheta) + 2))
    velr = np.zeros((int(360 / dphi) + 2, int(180 / dtheta) + 2))
    den = np.zeros((int(360 / dphi), int(180 / dtheta)))

    # for all selected particles, calculate counts on the surface
    for idx in range(0, tnp):
        if (idx > 0 and idx % 10000 == 0):
            print("%-10d particles out of  %-10d processed!" % (idx, tnp))

        rmag = np.sqrt(prx[idx] ** 2 + pry[idx] ** 2 + prz[idx] ** 2)
        theta = np.arccos(prz[idx] / rmag) * (180. / np.pi)
        phi = np.arctan2(pry[idx], prx[idx]) * (180. / np.pi)
        if (phi < 0):
            phi += 360.

        i = int(np.floor(phi / dphi - 0.5)) + 1
        j = int(np.floor(theta / dtheta - 0.5)) + 1

        u = i + 1
        v = j + 1

        x = i + 0.5 - (phi / dphi)
        y = j + 0.5 - (theta / dtheta)

        if (x < 0 or x > 1 or y < 0 or y > 1):
            print("r=(%+-16.8f, %+-16.8f, %+-16.8f), (x,y)=(%+-16.8f, %+-16.8f) \n", prx[idx], pry[idx], prz[idx], x, y)
            raise ("Error in x or y!")

        c_sum = 0.0
        r_dot_v = abs(prx[idx] * pvx[idx] + pry[idx] * pvy[idx] + prz[idx] * pvz[idx]) / rmag

        c = x * y
        c_sum += c
        cnts[i, j] += c
        velx[i, j] += c * pvx[idx]
        vely[i, j] += c * pvy[idx]
        velz[i, j] += c * pvz[idx]
        velr[i, j] += c * r_dot_v

        c = x * (1.0 - y)
        c_sum += c
        cnts[i, v] += c
        velx[i, v] += c * pvx[idx]
        vely[i, v] += c * pvy[idx]
        velz[i, v] += c * pvz[idx]
        velr[i, v] += c * r_dot_v

        c = (1.0 - x) * y
        c_sum += c
        cnts[u, j] += c
        velx[u, j] += c * pvx[idx]
        vely[u, j] += c * pvy[idx]
        velz[u, j] += c * pvz[idx]
        velr[u, j] += c * r_dot_v

        c = (1.0 - x) * (1.0 - y)
        c_sum += c
        cnts[u, v] += c
        velx[u, v] += c * pvx[idx]
        vely[u, v] += c * pvy[idx]
        velz[u, v] += c * pvz[idx]
        velr[u, v] += c * r_dot_v

        if (c_sum < 0.99999 or c_sum > 1.00001):
            raise ("Error c_sum \n");

    print("All particles processed successfully!")

    # correct for the guard cells
    cnts[1, :] += cnts[int(360 / dphi) + 1, :]
    velx[1, :] += velx[int(360 / dphi) + 1, :]
    vely[1, :] += vely[int(360 / dphi) + 1, :]
    velz[1, :] += velz[int(360 / dphi) + 1, :]
    velr[1, :] += velr[int(360 / dphi) + 1, :]

    cnts[int(360 / dphi), :] += cnts[0, :]
    velx[int(360 / dphi), :] += velx[0, :]
    vely[int(360 / dphi), :] += vely[0, :]
    velz[int(360 / dphi), :] += velz[0, :]
    velr[int(360 / dphi), :] += velr[0, :]

    cnts = cnts[1:int(360 / dphi) + 1, 1:int(180 / dtheta) + 1]
    velx = velx[1:int(360 / dphi) + 1, 1:int(180 / dtheta) + 1]
    vely = vely[1:int(360 / dphi) + 1, 1:int(180 / dtheta) + 1]
    velz = velz[1:int(360 / dphi) + 1, 1:int(180 / dtheta) + 1]
    velr = velr[1:int(360 / dphi) + 1, 1:int(180 / dtheta) + 1]

    # move the map and make the subsolar point in the middle of the map
    tmp = np.zeros((int(360 / dphi), int(180 / dtheta)))
    tmp[0:int(tmp.shape[0] / 2), :] = cnts[int(tmp.shape[0] / 2):int(tmp.shape[0]), :]
    tmp[int(tmp.shape[0] / 2):int(tmp.shape[0]), :] = cnts[0:int(tmp.shape[0] / 2), :]
    cnts = tmp

    tmp = np.zeros((int(360 / dphi), int(180 / dtheta)))
    tmp[0:int(tmp.shape[0] / 2), :] = velx[int(tmp.shape[0] / 2):int(tmp.shape[0]), :]
    tmp[int(tmp.shape[0] / 2):int(tmp.shape[0]), :] = velx[0:int(tmp.shape[0] / 2), :]
    velx = tmp

    tmp = np.zeros((int(360 / dphi), int(180 / dtheta)))
    tmp[0:int(tmp.shape[0] / 2), :] = vely[int(tmp.shape[0] / 2):int(tmp.shape[0]), :]
    tmp[int(tmp.shape[0] / 2):int(tmp.shape[0]), :] = vely[0:int(tmp.shape[0] / 2), :]
    vely = tmp

    tmp = np.zeros((int(360 / dphi), int(180 / dtheta)))
    tmp[0:int(tmp.shape[0] / 2), :] = velz[int(tmp.shape[0] / 2):int(tmp.shape[0]), :]
    tmp[int(tmp.shape[0] / 2):int(tmp.shape[0]), :] = velz[0:int(tmp.shape[0] / 2), :]
    velz = tmp

    tmp = np.zeros((int(360 / dphi), int(180 / dtheta)))
    tmp[0:int(tmp.shape[0] / 2), :] = velr[int(tmp.shape[0] / 2):int(tmp.shape[0]), :]
    tmp[int(tmp.shape[0] / 2):int(tmp.shape[0]), :] = velr[0:int(tmp.shape[0] / 2), :]
    velr = tmp

    # calculate particles weight and account for the spherical coordinate system
    weight = ((sim_dx * sim_dy * sim_dz) / sim_ppc[specie_id]) / num_files
    dr = select_R - sim_robs
    dv = (select_R ** 2) * dr * (dphi * np.pi / 180.) * (dtheta * np.pi / 180.)

    # print( "Weight: %e" %(weight) )
    # convert counts to density
    for i in range(0, int(180. / dtheta)):
        den[:, i] = cnts[:, i] * sim_den[specie_id] * weight / (dv * np.sin((i + 0.5) * dtheta * np.pi / 180.))

    # correct velocity; and account for division by zero when there is no particle
    for i in range(0, int(360 / dphi)):
        for j in range(0, int(180 / dtheta)):
            if (cnts[i, j] > 0):
                velx[i, j] /= cnts[i, j]
                vely[i, j] /= cnts[i, j]
                velz[i, j] /= cnts[i, j]
                velr[i, j] /= cnts[i, j]

                phi = (i * dphi + 0.5) * np.pi / 180.
                theta = (j * dtheta + 0.5) * np.pi / 180.

                vect_normal_x = np.sin(theta) * np.cos(phi)
                vect_normal_y = np.sin(theta) * np.sin(phi)
                vect_normal_z = np.cos(theta)

            else:
                velx[i, j] = 0.0
                vely[i, j] = 0.0
                velz[i, j] = 0.0
                velr[i, j] = 0.0

    vmag = np.sqrt(velx ** 2 + vely ** 2 + velz ** 2)
    flxr = velr * den

    np.savez(moments_filename + "_" + species[specie_id] + ".npz",
             prx=prx, pry=pry, prz=prz,
             pvx=pvx, pvy=pvy, pvz=pvz,
             select_R=select_R, dphi=dphi, dtheta=dtheta,
             specie=species[specie_id],
             sim_den=sim_den[specie_id],
             sim_dx=sim_dx, sim_dy=sim_dy, sim_dz=sim_dz,
             num_files=num_files,
             sim_ppc=sim_ppc[specie_id], sim_robs=sim_robs,
             weight=weight, dr=dr, dv=dv,
             cnts=cnts, den=den,
             velx=velx, vely=vely, velz=velz,
             vmag=vmag, velr=velr, flxr=flxr)

calc_moments_at_surface(0)