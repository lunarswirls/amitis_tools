import pyamitis.version
import matplotlib.pyplot as plt
from pyamitis.amitis_hdf import *

sim_step = 2500
filename = 'Amitis_field_' + "%06d"%(sim_step)
obj_hdf  = amitis_hdf('/Users/shahab/tmp/test_data/',
                        filename + '.h5')

obj_hdf.print_all_attributes()

# Reading Bx dataset from hdf file and convert it from Tesla to nano-Tesla
Bx = obj_hdf.load_dataset('Bx', 1.e9)
By = obj_hdf.load_dataset('By', 1.e9)
Bz = obj_hdf.load_dataset('Bz', 1.e9)

Bx += obj_hdf.load_dataset('Bdx', 1.e9)
By += obj_hdf.load_dataset('Bdy', 1.e9)
Bz += obj_hdf.load_dataset('Bdz', 1.e9)

Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

print(Bmag.shape)


density = obj_hdf.load_dataset('rho_tot', 1./1.602e-19)

nx = obj_hdf.load_attribute('nx')  # Number of grid cells along x
ny = obj_hdf.load_attribute('ny')  # Number of grid cells along y
nz = obj_hdf.load_attribute('nz')  # Number of grid cells along z

# Domain extent
xmin = obj_hdf.load_attribute('xmin') * 0.001
xmax = obj_hdf.load_attribute('xmax') * 0.001

ymin = obj_hdf.load_attribute('ymin') * 0.001
ymax = obj_hdf.load_attribute('ymax') * 0.001

zmin = obj_hdf.load_attribute('zmin') * 0.001
zmax = obj_hdf.load_attribute('zmax') * 0.001

if __name__ == '__main__':       
    fig = plt.figure(obj_hdf.file_name)
    im  = plt.imshow( np.log10(Bmag[:,int(ny/2),:].T), 
                     origin="upper", interpolation='nearest', 
                     extent=[xmin, xmax, zmin, zmax],
                     cmap='Spectral')
    plt.colorbar(im)
    plt.show()

