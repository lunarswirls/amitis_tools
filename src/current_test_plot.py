import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# -----------------------------
# User settings
# -----------------------------
nc_path = "/Users/danywaller/Projects/mercury/test9_2025dec09/Amitis_field_100000.nc"
time_index = 0

threshold_frac = 0.2          # Jmag > threshold_frac * nanmax(Jmag)
subsample = 2                  # subsample grid for speed (2–6 typical)
n_seeds = 500                  # number of streamline seeds to use
max_steps = 1000                # integration steps per streamline
ds_step = None                 # if None, choose automatically from grid spacing
min_speed_frac = 1e-6          # stop if |J| < min_speed_frac * max(|J| in subsampled grid)

rk4_dt_factor = 0.8           # dt = rk4_dt_factor * min(dx,dy,dz)
line_width = 0.7
alpha = 0.9

# Optional: keep streamlines within a bounding box subset
# (useful if you only care about dayside, for example)
use_bbox = False
bbox = dict(
    xmin=-5, xmax=5,
    ymin=-5, ymax=5,
    zmin=-5, zmax=5,
)

# -----------------------------
# Load dataset
# -----------------------------
ds = xr.open_dataset(nc_path)

x = ds["Nx"].values / 2440
y = ds["Ny"].values / 2440
z = ds["Nz"].values / 2440

Jx = ds["Jx"].isel(time=time_index).values
Jy = ds["Jy"].isel(time=time_index).values
Jz = ds["Jz"].isel(time=time_index).values

# Expect (Nz, Ny, Nx) after time slice
dims = ds["Jx"].isel(time=time_index).dims
if dims != ("Nz", "Ny", "Nx"):
    raise ValueError(f"Expected ('Nz','Ny','Nx') after time slice; got {dims}")

# -----------------------------
# Subsample for tractability
# -----------------------------
x_s = x[::subsample]
y_s = y[::subsample]
z_s = z[::subsample]

Jx_s = Jx[::subsample, ::subsample, ::subsample]
Jy_s = Jy[::subsample, ::subsample, ::subsample]
Jz_s = Jz[::subsample, ::subsample, ::subsample]

Jmag_s = np.sqrt(Jx_s**2 + Jy_s**2 + Jz_s**2)
Jmax = np.nanmax(Jmag_s)
if not np.isfinite(Jmax) or Jmax <= 0:
    raise ValueError("Jmag max is non-finite or <= 0. Check data contents.")

thr = threshold_frac * Jmax
mask = np.isfinite(Jmag_s) & (Jmag_s > thr)

if use_bbox:
    Xg, Yg, Zg = np.meshgrid(x_s, y_s, z_s, indexing="xy")  # (Ny, Nx, Nz) order
    # Convert to (Nz,Ny,Nx) order by constructing separately:
    Zg_zyx, Yg_zyx, Xg_zyx = np.meshgrid(z_s, y_s, x_s, indexing="ij")
    if bbox["xmin"] is not None: mask &= (Xg_zyx >= bbox["xmin"])
    if bbox["xmax"] is not None: mask &= (Xg_zyx <= bbox["xmax"])
    if bbox["ymin"] is not None: mask &= (Yg_zyx >= bbox["ymin"])
    if bbox["ymax"] is not None: mask &= (Yg_zyx <= bbox["ymax"])
    if bbox["zmin"] is not None: mask &= (Zg_zyx >= bbox["zmin"])
    if bbox["zmax"] is not None: mask &= (Zg_zyx <= bbox["zmax"])

inds = np.argwhere(mask)
if inds.size == 0:
    raise ValueError(
        "No seed locations satisfy the Jmag threshold after subsampling. "
        "Lower threshold_frac or subsample less."
    )

# -----------------------------
# Build interpolators: Jx(x,y,z), Jy(x,y,z), Jz(x,y,z)
# IMPORTANT: arrays are (Nz,Ny,Nx) so interpolator axes are (z,y,x)
# -----------------------------
interp_Jx = RegularGridInterpolator((z_s, y_s, x_s), Jx_s, bounds_error=False, fill_value=np.nan)
interp_Jy = RegularGridInterpolator((z_s, y_s, x_s), Jy_s, bounds_error=False, fill_value=np.nan)
interp_Jz = RegularGridInterpolator((z_s, y_s, x_s), Jz_s, bounds_error=False, fill_value=np.nan)

# -----------------------------
# Choose seed points from high-J region (spread through the mask)
# -----------------------------
n_avail = len(inds)
pick = np.linspace(0, n_avail - 1, min(n_seeds, n_avail)).astype(int)
sel = inds[pick]

# sel indices are (iz, iy, ix) in subsampled grid
seed_x = x_s[sel[:, 2]]
seed_y = y_s[sel[:, 1]]
seed_z = z_s[sel[:, 0]]
seeds = np.column_stack([seed_x, seed_y, seed_z])

# -----------------------------
# Integration parameters
# -----------------------------
dx = np.nanmedian(np.diff(x_s))
dy = np.nanmedian(np.diff(y_s))
dz = np.nanmedian(np.diff(z_s))
dmin = np.nanmin([abs(dx), abs(dy), abs(dz)])
if not np.isfinite(dmin) or dmin <= 0:
    raise ValueError("Non-finite or non-positive grid spacing after subsampling.")

dt = rk4_dt_factor * dmin if ds_step is None else float(ds_step)
speed_min = min_speed_frac * Jmax

x_min, x_max = np.nanmin(x_s), np.nanmax(x_s)
y_min, y_max = np.nanmin(y_s), np.nanmax(y_s)
z_min, z_max = np.nanmin(z_s), np.nanmax(z_s)

def in_bounds(p):
    return (x_min <= p[0] <= x_max) and (y_min <= p[1] <= y_max) and (z_min <= p[2] <= z_max)

def J_vec(p):
    # p = (x,y,z) but interpolators expect (z,y,x)
    q = (p[2], p[1], p[0])
    jx = interp_Jx(q)
    jy = interp_Jy(q)
    jz = interp_Jz(q)
    if np.any(~np.isfinite([jx, jy, jz])):
        return None
    return np.array([jx, jy, jz], dtype=float)

def rk4_step(p, h):
    v1 = J_vec(p)
    if v1 is None: return None
    k1 = v1

    v2 = J_vec(p + 0.5 * h * k1)
    if v2 is None: return None
    k2 = v2

    v3 = J_vec(p + 0.5 * h * k2)
    if v3 is None: return None
    k3 = v3

    v4 = J_vec(p + h * k3)
    if v4 is None: return None
    k4 = v4

    return p + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# -----------------------------
# Integrate streamlines (forward)
# -----------------------------
lines = []
for s in seeds:
    p = s.astype(float).copy()
    if not in_bounds(p):
        continue

    pts = [p.copy()]
    for _ in range(max_steps):
        v = J_vec(p)
        if v is None:
            break
        spd = np.linalg.norm(v)
        if not np.isfinite(spd) or spd < speed_min:
            break

        p_next = rk4_step(p, dt)
        if p_next is None or not in_bounds(p_next):
            break

        pts.append(p_next.copy())
        p = p_next

    if len(pts) >= 6:
        lines.append(np.vstack(pts))

if len(lines) == 0:
    raise ValueError(
        "No streamlines were produced. Try lowering threshold_frac, "
        "reducing subsample, increasing max_steps, or increasing dt slightly."
    )

# -----------------------------
# Plot with Matplotlib 3D (robust for variable-length streamlines)
# -----------------------------
fig = plt.figure(figsize=(10, 8), dpi=160)
ax = fig.add_subplot(111, projection="3d")

# Set bounds first (prevents autoscale issues)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

# Plot each streamline as a 3D line
for ln in lines:
    ax.plot(ln[:, 0], ln[:, 1], ln[:, 2], linewidth=line_width, alpha=alpha)

ax.set_title(f"3D streamlines of J seeded where |J| > {threshold_frac:.2f}·max(|J|)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

try:
    ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))
except Exception:
    pass

plt.tight_layout()
plt.show()
