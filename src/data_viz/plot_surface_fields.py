#!/usr/bin/env python
# -*- coding: utf-8 -
desc="""
Map crustal magnetic field magnitude from an AMITIS HDF file
onto the near- and far-side surfaces of the Moon and visualize
results using orthographic map projections.

Main principles preserved:
- Read AMITIS HDF data
- Compute |B| from vector components
- Sample data on a spherical lunar surface
- Separate near-side and far-side mappings
- Produce the same three-panel visualization layout
"""

# ================================
# Imports
# ================================
from pyamitis.amitis_hdf import *
from pyamitis.amitis_colorbar import *

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


# ================================
# Load AMITIS HDF data
# ================================
# Initialize HDF object (path and filename)
obj_hdf = amitis_hdf(
    '/Data/Moon/run005/',
    'Amitis_r005_080000.h5')

# Read magnetic-field components [Tesla] and convert to [nT]
Bx = obj_hdf.load_dataset('Bx', 1.0e9)
By = obj_hdf.load_dataset('By', 1.0e9)
Bz = obj_hdf.load_dataset('Bz', 1.0e9)

# Calculate the magnitude of the magnetic field, |B|
Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

# ================================
# Define lunar surface grid
# ================================
# Lunar radius [m]
r_moon = 1740.0e3

# Surface resolution (1° × 1°)
lat_bins = int(180)   # latitude
lon_bins = int(360)   # longitude

# Spherical coordinates
phi, theta = np.mgrid[
    0:np.pi:lat_bins * 1j,
    0:2 * np.pi:lon_bins * 1j]

# Convert spherical coordinates to Cartesian coordinates
x = r_moon * np.sin(phi) * np.cos(theta)
y = r_moon * np.sin(phi) * np.sin(theta)
z = r_moon * np.cos(phi)


# ================================
# Data mapping utilities
# ================================

def get_data_surface_far_side(data):
    """
    Map simulation data onto the lunar far-side surface.
    The x-coordinate is mirrored to sample the far side.
    """
    mapped_data = np.zeros((lat_bins, lon_bins))

    for i in range(lat_bins - 1):
        for j in range(lon_bins - 1):
            pos = [-x[i, j], y[i, j], z[i, j]]
            mapped_data[i, j] = obj_hdf.get_data( data, pos, interpolation=True )

    return mapped_data


def get_data_surface_near_side(data):
    """
    Map simulation data onto the lunar near-side surface.
    """
    mapped_data = np.zeros((lat_bins, lon_bins))

    for i in range(lat_bins - 1):
        for j in range(lon_bins - 1):
            pos = [x[i, j], y[i, j], z[i, j]]
            mapped_data[i, j] = obj_hdf.get_data( data, pos, interpolation=True )

    return mapped_data


# ================================
# Plotting routine
# ================================

def plot_two_maps():
    """
    Create a three-panel figure showing:
    (a) Near-side orthographic projection
    (b) Far-side orthographic projection
    (c) 2D cylindrical map of the far side
    """

    # ---- Layout parameters ----
    subplt_left   = 0.04
    subplt_right  = 0.94
    subplt_top    = 0.94
    subplt_bottom = 0.05
    subplt_hspace = 0.20
    subplt_vspace = 0.15

    title_fontsize = 26

    # ---- Colormap and colorbar settings ----
    Bmap = myspectral
    Bmap_min = 0.0
    Bmap_max = 3.0

    Bcbar_label = r'$\rm{|B_{crustal}| [nT]}$'
    Bcbar_ticks = np.array([0.0, 1.0, 2.0, 3.0])
    Bcbar_tickslabel = ( r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$' )
    Bcbar_fontsize = 24
    Bcbar_ax_font  = 22

    # Gridline ticks
    x_ticks = np.linspace(-180, 180, 13)
    y_ticks = np.linspace(-80, 80, 9)

    # ---- Map data onto lunar surface ----
    Bmag_nearside = get_data_surface_near_side(Bmag)
    Bmag_farside  = get_data_surface_far_side(Bmag)

    # ---- Create figure ----
    fig = plt.figure(
        obj_hdf.file_name,
        figsize=(13, 9),
        frameon=True )

    # ================================
    # Panel (a): Near side
    # ================================
    ax = plt.subplot(221, projection=ccrs.Orthographic(0, 0))
    plt.subplots_adjust( left=subplt_left, right=subplt_right,
        top=subplt_top, bottom=subplt_bottom,
        wspace=subplt_hspace, hspace=subplt_vspace )

    ax.gridlines(color='black', linestyle='dotted')

    im = ax.imshow(
        np.log10(Bmag_nearside),
        origin='upper',
        interpolation='nearest',
        cmap=Bmap,
        extent=(-180, 180, -90, 90),
        transform=ccrs.PlateCarree() )

    im.set_clim(Bmap_min, Bmap_max)

    gl = ax.gridlines( xlocs=x_ticks, ylocs=y_ticks, linestyle='--', linewidth=1, color='w', draw_labels=True )
	
    gl.xlabels_top    = False
    gl.xlabels_bottom = False
    gl.ylabels_right  = False
    gl.ylabel_style   = { 'color': 'k', 'weight': 'bold', 'fontsize': '14' }

    cbar = plt.colorbar(im, extend='min')
    cbar.set_ticks(Bcbar_ticks)
    cbar.set_label(Bcbar_label, fontsize=Bcbar_fontsize)
    cbar.ax.set_yticklabels(Bcbar_tickslabel)

    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_fontsize(Bcbar_ax_font)

    ax.set_title('Near side', fontsize=title_fontsize)
    ax.text( 0.0, 0.93, 'a)', fontsize=24, color='k',
        transform=plt.gcf().transFigure, clip_on=False )

    # ================================
    # Panel (b): Far side (orthographic)
    # ================================
    ax = plt.subplot(222, projection=ccrs.Orthographic(0, 0))
    plt.subplots_adjust( left=subplt_left, right=subplt_right,
        top=subplt_top, bottom=subplt_bottom,
        wspace=subplt_hspace, hspace=subplt_vspace )

    ax.gridlines(color='black', linestyle='dotted')

    im = ax.imshow(
        np.log10(Bmag_farside),
        origin='upper',
        interpolation='nearest',
        cmap=Bmap,
        extent=(180, -180, -90, 90),
        transform=ccrs.PlateCarree() )

    im.set_clim(Bmap_min, Bmap_max)

    gl = ax.gridlines( xlocs=x_ticks, ylocs=y_ticks, linestyle='--', linewidth=1, color='w', draw_labels=True )
    gl.xlabels_top    = False
    gl.xlabels_bottom = False
    gl.ylabels_right  = False
    gl.ylabel_style   = { 'color': 'k', 'weight': 'bold', 'fontsize': '14' }

    cbar = plt.colorbar(im, extend='min')
    cbar.set_ticks(Bcbar_ticks)
    cbar.set_label(Bcbar_label, fontsize=Bcbar_fontsize)
    cbar.ax.set_yticklabels(Bcbar_tickslabel)

    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_fontsize(Bcbar_ax_font)

    ax.set_title('Far side', fontsize=title_fontsize)
    ax.text( 0.5, 0.93, 'b)', fontsize=24, color='k',
        transform=plt.gcf().transFigure, clip_on=False )

    # Horizontal divider between top and bottom panels
    ax.plot( [0.01, 0.98], [0.495, 0.495],
        color='k', lw=2,
        transform=plt.gcf().transFigure, clip_on=False )

    # ================================
    # Panel (c): 2D far-side map
    # ================================
    ax = plt.subplot(212)
    plt.subplots_adjust(
        left=subplt_left, right=subplt_right,
        top=subplt_top, bottom=subplt_bottom,
        wspace=subplt_hspace, hspace=subplt_vspace )

    im = ax.imshow(
        np.log10(np.fliplr(Bmag_farside)),
        origin='upper',
        interpolation='nearest',
        cmap=Bmap,
        extent=(180, -180, -90, 90) )

    im.set_clim(Bmap_min, Bmap_max)

    cbar = plt.colorbar(im, extend='min')
    cbar.set_ticks(Bcbar_ticks)
    cbar.set_label(Bcbar_label, fontsize=Bcbar_fontsize)
    cbar.ax.set_yticklabels(Bcbar_tickslabel)

    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_fontsize(Bcbar_ax_font)

    ax.text( 0.04, 0.45, 'c)', fontsize=24, color='k',
        transform=plt.gcf().transFigure, clip_on=False )


# ================================
# Script entry point
# ================================
if __name__ == '__main__':
    plot_two_maps()
    plt.show()
