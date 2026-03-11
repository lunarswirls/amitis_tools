import os
import re
import numpy as np
import xarray as xr
from collections import defaultdict

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR = "/proj/nobackup/amitis/dany/CPN_BNIV_extendedX/plane"
OUTPUT_DIR = "/proj/nobackup/amitis/dany/CPN_BNIV_extendedX/cube"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Temp memmap directory
MEMMAP_DIR = os.path.join(OUTPUT_DIR, "memmap_tmp")
os.makedirs(MEMMAP_DIR, exist_ok=True)

# Regex to extract timestep and GPU id
FILENAME_RE = re.compile(r".*_(\d+)_G\d+\.npz$")

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def group_files_by_timestep(data_dir):
    files_by_step = defaultdict(list)
    for fname in os.listdir(data_dir):
        if fname.endswith(".npz"):
            match = FILENAME_RE.match(fname)
            if match:
                timestep = int(match.group(1))
                files_by_step[timestep].append(os.path.join(data_dir, fname))
    return files_by_step

def determine_simulation_domain(files_by_step):
    """Scan first timestep to get full simulation domain sizes."""
    first_timestep = sorted(files_by_step.keys())[0]
    sample_files = files_by_step[first_timestep]
    loaded = [dict(np.load(f)) for f in sample_files]

    nx = int(loaded[0]["nx"][0])
    ny = int(loaded[0]["ny"][0])
    nz = int(loaded[0]["nz"][0])

    coordx_max = max(int(d["coordx"][0]) for d in loaded)
    coordy_max = max(int(d["coordy"][0]) for d in loaded)
    coordz_max = max(int(d["coordz"][0]) for d in loaded)

    NX = (coordx_max + 1) * nx
    NY = (coordy_max + 1) * ny
    NZ = (coordz_max + 1) * nz

    return NX, NY, NZ, nx, ny, nz

def build_memmap_cubes(sample, NX, NY, NZ):
    """Create memmap arrays for each field."""
    xz_fields = [k for k in sample.keys() if "_xz_" in k]
    yz_fields = [k for k in sample.keys() if "_yz_" in k]

    all_fields = set([f.split("_xz_")[0] for f in xz_fields] +
                     [f.split("_yz_")[0] for f in yz_fields])

    memmaps = {}
    for field in all_fields:
        dtype = None
        for key in sample.keys():
            if key.startswith(field):
                dtype = sample[key].dtype
                break
        filename = os.path.join(MEMMAP_DIR, f"{field}.dat")
        memmaps[field] = np.memmap(filename, mode="w+", shape=(NX, NY, NZ), dtype=dtype)
        memmaps[field][:] = 0  # initialize to zeros
    return memmaps

def merge_planes_into_memmap(loaded, memmaps, nx, ny, nz):
    """Merge XZ and YZ planes into memory-mapped cube."""
    for d in loaded:
        cx = int(d["coordx"][0])
        cy = int(d["coordy"][0])
        cz = int(d["coordz"][0])

        xs, xe = cx*nx, (cx+1)*nx
        ys, ye = cy*ny, (cy+1)*ny
        zs, ze = cz*nz, (cz+1)*nz

        # XZ planes
        for key in [k for k in d.keys() if "_xz_" in k]:
            field, y_idx = key.split("_xz_")
            y_idx = int(y_idx) - 1 + cy*ny
            memmaps[field][:, y_idx, zs:ze] = d[key]

        # YZ planes
        for key in [k for k in d.keys() if "_yz_" in k]:
            field, x_idx = key.split("_yz_")
            x_idx = int(x_idx) - 1 + cx*nx
            memmaps[field][x_idx, :, zs:ze] = d[key]

def save_memmap_to_netcdf(memmaps, timestep, output_dir):
    """Save memmap cubes to NetCDF via xarray."""
    data_vars = {field: (("X", "Y", "Z"), memmaps[field]) for field in memmaps}
    ds = xr.Dataset(data_vars)
    out_file = os.path.join(output_dir, f"cube_{timestep:06d}.nc")
    ds.to_netcdf(out_file)
    print(f"Saved NetCDF: {out_file}")

# -----------------------------
# MAIN PROCESS
# -----------------------------
def main():
    files_by_step = group_files_by_timestep(DATA_DIR)
    timesteps = sorted(files_by_step.keys())
    print(f"Found timesteps: {timesteps}")

    # Determine domain
    NX, NY, NZ, nx, ny, nz = determine_simulation_domain(files_by_step)
    print(f"Simulation domain: NX={NX}, NY={NY}, NZ={NZ}")

    for t in timesteps:
        print(f"\nProcessing timestep {t}...")
        file_list = files_by_step[t]
        loaded = [dict(np.load(f)) for f in file_list]

        # Create memmap cubes for this timestep
        memmaps = build_memmap_cubes(loaded[0], NX, NY, NZ)

        merge_planes_into_memmap(loaded, memmaps, nx, ny, nz)
        save_memmap_to_netcdf(memmaps, t, OUTPUT_DIR)

        # Clean up memmaps to free disk/memory
        del memmaps

if __name__ == "__main__":
    main()