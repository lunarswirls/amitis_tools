from pyamitis.amitis_netcdf import *
from pyamitis.amitis_hdf import *

indir = '/Users/danywaller/Projects/mercury/test9_2025dec09/'
sim_steps = [20000, 40000, 60000, 80000, 100000]
compress = True

for sim_step in sim_steps:
    filename = 'Amitis_field_' + "%06d" % (sim_step)

    obj_hdf = amitis_hdf(indir, filename + '.h5')

    # original dimensions  of the simulation domain
    original_domain = obj_hdf.get_hdf_domain()

    #####################################################
    # READ DATASETS AND CONVERT THEIR UNITS
    #####################################################
    bdx = obj_hdf.load_dataset('Bdx', 1.0e9)
    bdy = obj_hdf.load_dataset('Bdy', 1.0e9)
    bdz = obj_hdf.load_dataset('Bdz', 1.0e9)
    bdmag = np.sqrt(bdx**2 + bdy**2 + bdz**2)

    bx  = obj_hdf.load_dataset('Bx', 1.0e9)
    by  = obj_hdf.load_dataset('By', 1.0e9)
    bz  = obj_hdf.load_dataset('Bz', 1.0e9)

    bx  = bx + bdx
    by  = by + bdy
    bz  = bz + bdz

    bmag = np.sqrt(bx**2 + by**2 + bz**2)

    Ex  = obj_hdf.load_dataset('Ex', 1.0e3)   # (V/m) => (mV/m)
    Ey  = obj_hdf.load_dataset('Ey', 1.0e3)
    Ez  = obj_hdf.load_dataset('Ez', 1.0e3)
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    # Charge density and plasma number density
    rho_tot = obj_hdf.load_dataset('rho_tot')    # Total Charge Density: qn
    den_tot = rho_tot*1.e-6 / obj_hdf.get_mean_charge()   # calculating total number density in units of [#/cm^3]

    # Calcluate total plasma velocity
    jix = obj_hdf.load_dataset('jix_tot')   # rho_tot * vx
    jiy = obj_hdf.load_dataset('jiy_tot')   # rho_tot * vy
    jiz = obj_hdf.load_dataset('jiz_tot')   # rho_tot * vz
    vx  = jix*1.e-3 / rho_tot     # (m/s) => (km/s)
    vy  = jiy*1.e-3 / rho_tot
    vz  = jiz*1.e-3 / rho_tot
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)

    # Electric current density from Ampere's law
    Jx = obj_hdf.load_dataset('Jx', 1.0e9)  # (A/m^2) => (nA/m^2)
    Jy = obj_hdf.load_dataset('Jy', 1.0e9)
    Jz = obj_hdf.load_dataset('Jz', 1.0e9)
    Jmag = np.sqrt(Jx**2 + Jy**2 + Jz**2)

    #####################################################
    # WRITE INTO A NETCDF FILE
    #####################################################
    # Open, write, and close netcdf file with real data
    # Comparison with original data file by eye for example with Panoply
    obj_netcdf = amitis_netcdf(obj_hdf.file_path, filename + '.nc', sim_step,
                               original_domain,
                               compression=compress) #, trimmed_domain)
    obj_netcdf.open()
    obj_netcdf.write_hdf_attributes(obj_hdf)

    obj_netcdf.write(bdx     , 'Bdx'     , 'nT'  )
    obj_netcdf.write(bdy     , 'Bdy'     , 'nT'  )
    obj_netcdf.write(bdz     , 'Bdz'     , 'nT'  )
    obj_netcdf.write(bdmag   , 'Bdmag'   , 'nT'  )

    obj_netcdf.write(bx      , 'Bx'      , 'nT'  )
    obj_netcdf.write(by      , 'By'      , 'nT'  )
    obj_netcdf.write(bz      , 'Bz'      , 'nT'  )
    obj_netcdf.write(bmag    , 'Bmag'    , 'nT'  )

    obj_netcdf.write(den_tot , 'den_tot' , 'cm-3')

    obj_netcdf.write(vx      , 'vx_tot'  , 'km/s')
    obj_netcdf.write(vy      , 'vy_tot'  , 'km/s')
    obj_netcdf.write(vz      , 'vz_tot'  , 'km/s')
    obj_netcdf.write(vmag    , 'vmag'    , 'km/s')

    obj_netcdf.write(Jx      , 'Jx'      , 'nA/m^2'  )
    obj_netcdf.write(Jy      , 'Jy'      , 'nA/m^2'  )
    obj_netcdf.write(Jz      , 'Jz'      , 'nA/m^2'  )
    obj_netcdf.write(Jmag    , 'Jmag'    , 'nA/m^2'  )

    obj_netcdf.write(Ex      , 'Ex'      , 'mV/m'  )
    obj_netcdf.write(Ey      , 'Ey'      , 'mV/m'  )
    obj_netcdf.write(Ez      , 'Ez'      , 'mV/m'  )
    obj_netcdf.write(Emag    , 'Emag'    , 'mV/m'  )

    obj_netcdf.close()

    print(f'Done writing {obj_hdf.file_path + filename}.nc')
