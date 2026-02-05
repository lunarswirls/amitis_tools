#!/usr/bin/env python3
import os
from PIL import Image

# Use settings from mag_calc_plot
base = "CPS_Base"
use_slices = ["xy", "xz", "yz"]  # plot all 3
n_slices = len(use_slices)       # number of requested slices
slice_tag = "_".join(use_slices)
plot_id = "Pmag"   # options: "Bmag", "Jmag", "Pmag"

# indir = f"/Users/danywaller/Projects/mercury/{base}_Base/"  # Directory containing image frames
# plot_folder = os.path.join(indir, f"{plot_id.lower()}/")
# plot_folder_ts = os.path.join(plot_folder, f"timeseries_{slice_tag}/")

# output_file = os.path.join(plot_folder, f"{base}_{plot_id.lower()}_{slice_tag}.gif")

# indir = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology/{base}_Base/"  # Directory containing image frames
# plot_folder_ts = os.path.join(indir, f"topology/")
# output_file = os.path.join(indir, f"{base}_field_topology.gif")

# indir = f"/Users/danywaller/Projects/mercury/extreme/High_HNHV_surface_flux/"
# plot_folder_ts = os.path.join(indir, f"timeseries_{base.lower()}/")
# output_file = os.path.join(indir, f"{base}_HNHV_surface_flux.gif")

indir = f"/Users/danywaller/Projects/mercury/extreme/"
plot_folder_ts = os.path.join(indir, f"timeseries_beta_{slice_tag}/{base}/")
output_file = os.path.join(indir, f"{base}_timeseries_beta.gif")

duration = 100  # Frame duration (ms)
loop = 1  # Loop count (0 = infinite)

# supported image types
valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# sort for chronological sequence
files = sorted(
    [f for f in os.listdir(plot_folder_ts) if f.lower().endswith(valid_ext)]
)

if not files:
    raise RuntimeError("No images found in directory")

frames = []
for f in files:
    img = Image.open(os.path.join(plot_folder_ts, f))
    frames.append(img.convert("RGB"))  # Ensure consistent encoding

# Save GIF
frames[0].save(
    output_file,
    save_all=True,
    append_images=frames[1:],
    duration=duration,
    loop=loop,
)
print(f"GIF saved as: {output_file}")
