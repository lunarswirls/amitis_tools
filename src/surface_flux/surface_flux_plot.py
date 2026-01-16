#!/usr/bin/env python
# -*- coding: utf-8 -
# Imports:
import os
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from src.surface_flux.flux_utils import compute_radial_flux
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# overrides following subplot setup and makes 1 simple 2d plot for debugging
quickplot = True

# if true, plot near and far side as two subplots
near_far = False

# if true, plot 3D and cylindrical projection on same figure
all_one = False

# choose case
case = "RPN"

# -------------------------------
# Configuration
# -------------------------------
input_folder1  = f"/Users/danywaller/Projects/mercury/extreme/{case}_Base/object/"
input_folder2 = f"/Users/danywaller/Projects/mercury/extreme/{case}_Base/object/"
output_folder = f"/Users/danywaller/Projects/mercury/extreme/{case}_Base/surface_density/"
os.makedirs(output_folder, exist_ok=True)

if 0:
    if case == "RPS" or case == "CPS":
        sim_steps = range(27000, 115000 + 1, 1000)
    else:
        sim_steps = range(98000, 115000 + 1, 1000)

sim_steps = range(115000, 115000 + 1, 1000)

t_index = 0

R_M = 2440.0        # Mercury radius [km]

LAT_BINS = 180      # [degrees]
LON_BINS = 360      # [degrees]

# -------------------------------
# Load grid (once)
# -------------------------------
ds0 = xr.open_dataset(
    os.path.join(input_folder1, f"Amitis_{case}_Base_{sim_steps.start:06d}_xz_comp.nc")  # Amitis_{case}_Base_000000_xz_comp.nc
)

x = ds0["Nx"].values  # [units: km]
y = ds0["Ny"].values  # [units: km]
z = ds0["Nz"].values  # [units: km]

ds0.close()

# -------------------------------
# Time-averaged total radial flux (full volume)
# -------------------------------
flux_sum = None
count = 0

for step in sim_steps:
    nc_file = os.path.join(input_folder1, f"Amitis_{case}_Base_{step:06d}_xz_comp.nc")
    ds = xr.open_dataset(nc_file)

    flux, vr = compute_radial_flux(ds, x, y, z)

    ds.close()

    if flux_sum is None:
        flux_sum = np.zeros_like(flux, dtype=np.float64)

    flux_sum += flux
    count += 1

flux_avg = flux_sum / count

# -------------------------------
# Interpolator (Cartesian space)
# -------------------------------
interp = RegularGridInterpolator(
    (z, y, x),
    flux_avg,
    bounds_error=False,
    fill_value=np.nan
)

# -------------------------------
# Surface grid
# -------------------------------
lat = np.linspace(-90, 90, LAT_BINS)
lon = np.linspace(0, 360, LON_BINS)


def surface_points_from_angles(lat_deg, lon_deg, RM):
    lat_r = np.deg2rad(lat_deg)
    lon_r = np.deg2rad(lon_deg)

    X_s = RM * np.cos(lat_r[:, None]) * np.cos(lon_r[None, :])
    Y_s = RM * np.cos(lat_r[:, None]) * np.sin(lon_r[None, :])
    Z_s = RM * np.sin(lat_r[:, None]) * np.ones_like(lon_r[None, :])

    return X_s, Y_s, Z_s


lat_flipped = lat[::-1]
Xs, Ys, Zs = surface_points_from_angles(lat_flipped, lon, R_M)

# -------------------------------
# Interpolate onto surface
# -------------------------------
points = np.stack((Zs, Ys, Xs), axis=-1).reshape(-1, 3)
flux_surface = interp(points).reshape(LAT_BINS, LON_BINS)
flux_surface = flux_surface[::-1, :]

# -------------------------------
# Normalize geometry for Plotly
# -------------------------------
Xn = Xs / R_M
Yn = Ys / R_M
Zn = Zs / R_M

# -------------------------------
# Calculate min and max for clims
# -------------------------------
c_min = np.nanpercentile(flux_surface, 5)
c_max = np.nanpercentile(flux_surface, 95)

# -------------------------------
# Fine grid interpolation
# -------------------------------
LAT_FINE = 360*3
LON_FINE = 720*3

lat_fine = np.linspace(-90, 90, LAT_FINE)
lon_fine = np.linspace(0, 360, LON_FINE)

interp = RegularGridInterpolator(
    (lat, lon),
    flux_surface,
    bounds_error=False,
    fill_value=np.nan
)

lon_grid_fine, lat_grid_fine = np.meshgrid(lon_fine, lat_fine)
points_fine = np.column_stack((lat_grid_fine.ravel(), lon_grid_fine.ravel()))
flux_fine = interp(points_fine).reshape(LAT_FINE, LON_FINE)
flux_fine = flux_fine[::-1, :]

# Mask non-positive values
flux_surface_masked = np.where(flux_fine > 0, flux_fine, np.nan)

# Log10
log_flux_surface = np.log10(flux_surface_masked)

# -------------------------------
# Flatten
# -------------------------------
x_flat = lon_grid_fine.ravel()
y_flat = lat_grid_fine.ravel()
z_flat = log_flux_surface.ravel()

if quickplot:
    # Apply -180° shift
    lon_grid_fine_shifted = (lon_grid_fine - 180) % 360
    x_flat_shifted = lon_grid_fine_shifted.ravel()

    c_min = np.nanpercentile(log_flux_surface, 5)
    c_max = np.nanpercentile(log_flux_surface, 95)

    # -------------------------------
    # 2D plot of everything
    # -------------------------------
    fig3 = make_subplots(rows=1, cols=1)

    fig3.add_trace(
        go.Scattergl(
            x=x_flat_shifted,
            y=y_flat,
            mode="markers",
            marker=dict(
                size=2,
                color=z_flat,
                colorscale="Viridis",
                cmin=c_min,
                cmax=c_max,
                colorbar=dict(title="log10(F)<br>(cm^-2 s^-1)"),
            ),
        ),
        row=1, col=1
    )

    # Vertical line at 90°
    fig3.add_shape(
        type="line",
        x0=90, x1=90,
        y0=-90, y1=90,
        xref="x", yref="y",
        line=dict(color="white", width=2, dash="dash"),
        row=1, col=1
    )

    # Vertical line at 180°
    fig3.add_shape(
        type="line",
        x0=270, x1=270,
        y0=-90, y1=90,
        xref="x", yref="y",
        line=dict(color="white", width=2, dash="dash"),
        row=1, col=1
    )

    # --- Set 2D axes ---
    fig3.update_xaxes(range=[0, 360], row=1, col=1)
    fig3.update_yaxes(range=[-90, 90], row=1, col=1)
    fig3.update_xaxes(title_text="Longitude [°]", row=1, col=1)
    fig3.update_yaxes(title_text="Latitude [°]", row=1, col=1)

    fig3.update_layout(
        title=f"Mercury Surface Flux (Local noon centered at 180°): {case}",
        height=800,
        width=800,
    )

    # --- Save ---
    outfile_png = os.path.join(output_folder, f"{case}_surface_flux_all_2D.png")
    fig3.write_image(outfile_png, width=1200, height=600, scale=2)
    print("Saved:", outfile_png)
else:
    if all_one:
        if not near_far:
            # Apply -180° shift
            lon_grid_fine_shifted = (lon_grid_fine - 180) % 360
            x_flat_shifted = lon_grid_fine_shifted.ravel()

            # -------------------------------
            # Create subplots: 3 columns, 1 row
            # -------------------------------
            fig = make_subplots(
                rows=1, cols=3,
                specs=[[{"colspan": 2, "type": "scatter"}, None, {"type": "scene"}]],
                subplot_titles=["Local noon centered at 180°", " "],
                horizontal_spacing=0.1
            )

            fig.add_trace(
                go.Scattergl(
                    x=x_flat_shifted,
                    y=y_flat,
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=z_flat,
                        colorscale="Viridis",
                        cmin=c_min,
                        cmax=c_max,
                        showscale=False,
                    ),
                    showlegend=False,
                ),
                row=1, col=1
            )

            # Vertical line at 90°
            fig.add_shape(
                type="line",
                x0=90, x1=90,
                y0=-90, y1=90,
                xref="x", yref="y",
                line=dict(color="white", width=2, dash="dash"),
                row=1, col=1
            )

            # Vertical line at 180°
            fig.add_shape(
                type="line",
                x0=270, x1=270,
                y0=-90, y1=90,
                xref="x", yref="y",
                line=dict(color="white", width=2, dash="dash"),
                row=1, col=1
            )

            # --- Set 2D axes ---
            fig.update_xaxes(range=[0, 360], row=1, col=1)
            fig.update_yaxes(range=[-90, 90], row=1, col=1)
            fig.update_xaxes(title_text="Longitude [°]", row=1, col=1)
            fig.update_yaxes(title_text="Latitude [°]", row=1, col=1)

            # --- Right column: 3D surface ---
            fig.add_trace(
                go.Surface(
                    x=Xn,
                    y=Yn,
                    z=Zn,
                    surfacecolor=log_flux_surface,
                    colorscale="Viridis",
                    cmin=c_min,
                    cmax=c_max,
                    colorbar=dict(title=f"log10(F)<br>(cm^-2 s^-1)", len=0.5,
                                  x=0.65,  # move left of the plotting area
                                  y=0.5,  # center vertically
                                  ),
                ),
                row=1, col=3
            )

            # --- Update 3D scene ---
            fig.update_layout(
                title=f"{case} Surface Flux",
                scene=dict(
                    xaxis_title="X (Rₘ)",
                    yaxis_title="Y (Rₘ)",
                    zaxis_title="Z (Rₘ)",
                    aspectmode="cube",
                ),
                height=600,
                width=1600,
                margin=dict(l=0, r=0, t=75, b=0),
                scene_camera=dict(
                    eye=dict(x=1.75, y=1.75, z=1.4),  # 45° between X and Y, slightly above
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            )

            # --- Save ---
            outfile_html = os.path.join(output_folder, f"{case}_surface_flux_all_3D.html")
            pio.write_html(fig, outfile_html, auto_open=False)
            print("Saved:", outfile_html)

            outfile_png = os.path.join(output_folder, f"{case}_surface_flux_all_3D.png")
            fig.write_image(outfile_png, width=1600, height=600, scale=2)
            print("Saved:", outfile_png)

        if near_far:
            # Apply -90° shift
            lon_grid_fine_shifted = (lon_grid_fine - 90) % 360
            x_flat_shifted = lon_grid_fine_shifted.ravel()

            # Masks
            mask_near = (x_flat_shifted > 180) & (x_flat_shifted <= 360)
            mask_far = (x_flat_shifted >= 0) & (x_flat_shifted <= 180)

            # -------------------------------
            # Create subplots: 2 columns, 2 rows left, 1 big 3D on right
            # -------------------------------
            fig = make_subplots(
                rows=2, cols=2,
                column_widths=[0.5, 0.5],
                row_heights=[0.5, 0.5],
                specs=[[{"type": "scatter"}, {"type": "scene", "rowspan": 2}],
                       [{"type": "scatter"}, None]],
                subplot_titles=["Near-side", " ", "Far-side"],
                vertical_spacing=0.15
            )

            # --- Left column: Near-side ---
            fig.add_trace(
                go.Scattergl(
                    x=x_flat_shifted[mask_near],
                    y=y_flat[mask_near],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=z_flat[mask_near],
                        colorscale="Viridis",
                        cmin=c_min,
                        cmax=c_max,
                        showscale=False,
                    ),
                    showlegend=False,
                ),
                row=1, col=1
            )

            # --- Left column: Far-side ---
            fig.add_trace(
                go.Scattergl(
                    x=x_flat_shifted[mask_far],
                    y=y_flat[mask_far],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=z_flat[mask_far],
                        colorscale="Viridis",
                        cmin=c_min,
                        cmax=c_max,
                        showscale=False,
                    ),
                    showlegend=False,
                ),
                row=2, col=1
            )

            # --- Set 2D axes ---
            fig.update_xaxes(range=[180, 360], row=1, col=1)
            fig.update_xaxes(range=[0, 180], row=2, col=1)
            fig.update_yaxes(range=[-90, 90], row=1, col=1)
            fig.update_yaxes(range=[-90, 90], row=2, col=1)
            fig.update_xaxes(title_text="Longitude [°]", row=1, col=1)
            fig.update_xaxes(title_text="Longitude [°]", row=2, col=1)
            fig.update_yaxes(title_text="Latitude [°]", row=1, col=1)
            fig.update_yaxes(title_text="Latitude [°]", row=2, col=1)

            # --- Right column: 3D surface ---
            fig.add_trace(
                go.Surface(
                    x=Xn,
                    y=Yn,
                    z=Zn,
                    surfacecolor=flux_surface,
                    colorscale="Viridis",
                    cmin=c_min,
                    cmax=c_max,
                    colorbar=dict(title="log10(F)<br>(cm^-2 s^-1)", len=0.5,
                                  x=0.5,  # move left of the plotting area
                                  y=0.5,  # center vertically
                                  ),
                ),
                row=1, col=2
            )

            # --- Update 3D scene ---
            fig.update_layout(
                title=f"Mercury Surface Flux: {case}",
                scene=dict(
                    xaxis_title="X (Rₘ)",
                    yaxis_title="Y (Rₘ)",
                    zaxis_title="Z (Rₘ)",
                    aspectmode="cube",
                ),
                height=800,
                width=1600,
                margin=dict(l=0, r=0, t=75, b=0),
                scene_camera=dict(
                    eye=dict(x=1.75, y=1.75, z=1.4),  # 45° between X and Y, slightly above
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            )

            # --- Save ---
            outfile_html = os.path.join(output_folder, f"{case}_surface_flux_near_far_3D.html")
            pio.write_html(fig, outfile_html, auto_open=False)
            print("Saved:", outfile_html)

            outfile_png = os.path.join(output_folder, f"{case}_surface_flux_near_far_3D.png")
            fig.write_image(outfile_png, width=1600, height=800, scale=2)
            print("Saved:", outfile_png)

    if not all_one:
        # =====================================================
        # Plotly 3D surface
        # =====================================================
        fig = go.Figure(
            go.Surface(
                x=Xn,
                y=Yn,
                z=Zn,
                surfacecolor=log_flux_surface,
                colorscale="Viridis",
                cmin=c_min,
                cmax=c_max,
                colorbar=dict(title="log10(F)<br>(cm^-2 s^-1)"),
            )
        )

        fig.update_layout(
            title=f"Mercury Surface Flux: {case}",
            scene=dict(
                xaxis_title="X (Rₘ)",
                yaxis_title="Y (Rₘ)",
                zaxis_title="Z (Rₘ)",
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # --- Save ---
        outfile = os.path.join(output_folder, f"{case}_surface_flux_3D.html")
        pio.write_html(fig, outfile, auto_open=False)

        print("Saved:", outfile)

        # --- Fine grid for interpolation ---
        LAT_FINE = 360*3
        LON_FINE = 720*3

        lat_fine = np.linspace(-90, 90, LAT_FINE)
        lon_fine = np.linspace(0, 360, LON_FINE)

        # Interpolator: original data is den_surface(lat, lon)
        interp = RegularGridInterpolator(
            (lat, lon),  # original coordinates
            flux_surface,
            bounds_error=False,
            fill_value=np.nan
        )

        # Generate fine meshgrid
        lon_grid_fine, lat_grid_fine = np.meshgrid(lon_fine, lat_fine)
        points_fine = np.column_stack((lat_grid_fine.ravel(), lon_grid_fine.ravel()))

        # Interpolate data
        flux_fine = interp(points_fine).reshape(LAT_FINE, LON_FINE)

        # --- Flatten for plotting ---
        x_flat = lon_grid_fine.ravel()
        y_flat = lat_grid_fine.ravel()
        z_flat = flux_fine.ravel()

        # Apply -90 degree shift (to the east)
        lon_grid_fine_shifted = (lon_grid_fine - 90) % 360
        x_flat_shifted = lon_grid_fine_shifted.ravel()

        # Near side (Sun-facing)
        mask_near = (x_flat_shifted > 180) & (x_flat_shifted <= 360)

        # Far side (opposite Sun)
        mask_far  = (x_flat_shifted >= 0) & (x_flat_shifted <= 180)

        # --- Plot 2D cylindrical projection with subplots ---
        fig2 = make_subplots(
            rows=2, cols=1,
            subplot_titles=["(a) Local noon ", "(b) Local midnight "],
            vertical_spacing=0.15  # increase spacing
        )

        # Near-side
        fig2.add_trace(
            go.Scattergl(
                x=x_flat_shifted[mask_near],
                y=y_flat[mask_near],
                mode="markers",
                marker=dict(
                    size=2,
                    color=z_flat[mask_near],
                    colorscale="Viridis",
                    cmin=c_min,
                    cmax=c_max,
                    colorbar=dict(title="Radial flux<br>(cm^-2 s^-1)", len=0.45),
                ),
                showlegend=False,
            ),
            row=1, col=1
        )

        # Far-side
        fig2.add_trace(
            go.Scattergl(
                x=x_flat_shifted[mask_far],
                y=y_flat[mask_far],
                mode="markers",
                marker=dict(
                    size=2,
                    color=z_flat[mask_far],
                    colorscale="Viridis",
                    cmin=c_min,
                    cmax=c_max,
                    showscale=False,
                ),
                showlegend=False,
            ),
            row=2, col=1
        )

        # --- Set axes limits (crop view) ---
        fig2.update_xaxes(range=[0, 180], row=2, col=1)
        fig2.update_xaxes(range=[180, 360], row=1, col=1)
        fig2.update_yaxes(range=[-90, 90], row=2, col=1)
        fig2.update_yaxes(range=[-90, 90], row=1, col=1)

        # --- Axis labels ---
        for r in (1,2):
            fig2.update_xaxes(title_text="Longitude [°]", row=r, col=1)
            fig2.update_yaxes(title_text="Latitude [°]", row=r, col=1)

        fig2.update_layout(
            title=f"Mercury Surface Flux: {case}",
            height=800,
            width=800,
        )

        # --- Save ---
        outfile2 = os.path.join(output_folder, f"{case}_surface_flux_near_far.html")
        pio.write_html(fig2, outfile2, auto_open=False)
        print("Saved:", outfile2)

        outfile_png = os.path.join(output_folder, f"{case}_surface_flux_near_far.png")
        fig2.write_image(outfile_png, width=900, height=800, scale=2)  # scale=2 for higher resolution
        print("Saved:", outfile_png)