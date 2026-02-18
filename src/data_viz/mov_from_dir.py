#!/usr/bin/env python3
import os
import cv2
from PIL import Image

case = "RPN"

indir = f"/Users/danywaller/Projects/mercury/extreme/density_lonlat/{case}_HNHV_end/1.00-1.05_RM/"
outdir = f"/Users/danywaller/Projects/mercury/extreme/density_lonlat/"
output_file = os.path.join(outdir, f"{case}_HNHV_density_full_timeseries_1.00-1.05_RM.mov")

fps = 5  # Frames per second (adjust for desired playback speed)

# Supported image types
valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Sort for chronological sequence
files = sorted(
    [f for f in os.listdir(indir) if f.lower().endswith(valid_ext)]
)

if not files:
    raise RuntimeError("No images found in directory")

# Get full paths
image_files = [os.path.join(indir, f) for f in files]

print(f"Found {len(image_files)} images")

# Get dimensions from first image
first_img = Image.open(image_files[0])
width, height = first_img.size
print(f"Video dimensions: {width}x{height}")

# Initialize video writer
# For .mov with H.264: use 'avc1' or 'mp4v'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

if not out.isOpened():
    raise RuntimeError("Failed to open video writer")

# Write frames
for i, img_path in enumerate(image_files):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read {img_path}")
        continue
    out.write(img)
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(image_files)} frames")

out.release()
print(f"MOV file saved as: {output_file}")
