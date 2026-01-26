import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

R = 2440.e3  # Object radius for normalization

def plot_ellipsoid(ax, center, radii, n_points=20):
    """Plot a single ellipsoid given center and radii."""
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = radii[0] * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radii[1] * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) + center[2]

    ax.plot_surface(x, y, z, color='b', alpha=0.3, edgecolor='k', linewidth=0.5)

def read_ellipsoids_from_file(filename):
    """Read ellipsoid data from a text file."""
    data = np.loadtxt(filename)
    centers = data[:, :3] / R
    radii = data[:, 3:] / R
    return centers, radii

def main(filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    centers, radii = read_ellipsoids_from_file(filename)

    for center, radius in zip(centers, radii):
        plot_ellipsoid(ax, center, radius)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.set_aspect('equal')
    plt.grid('True')
    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    main('/Users/shahab/tmp/Amitis.sub')
