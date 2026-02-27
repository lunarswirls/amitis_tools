import numpy as np

# =========================
# from Amitis.inp file
# =========================
B_T = 25e-9                                                    # IMF magnitude [T]
species = np.array(["H+"])
sim_ppc = np.array([9], dtype=float)                 # macroparticles per cell
sim_den = np.array([19.0e6], dtype=float)        # number density [m^-3]
sim_vel = np.array([990.0e3], dtype=float)     # bulk speed magnitude [m/s]
species_mass = np.array([1.0], dtype=float)        # mass in amu
species_charge = np.array([1.0], dtype=float)      # charge state Z (|q| = Z e)
T_sw_K = np.array([1.4e5], dtype=float)          # species temperature [K]

# Alternatively, set T_sw_K to None and use T_sw_eV instead
T_sw_eV = None
# T_sw_eV = np.array([12.1, 48.3], dtype=float)

# How to define "background" density for Va/c_ms
mach_density_mode = "total_mass"   # "total_mass" or "proton_only"

# Dynamic pressure convention (space physics uses rho*v^2; fluid dynamic pressure is 0.5*rho*v^2)
pdyn_convention = "rho_v2"         # "rho_v2" or "half_rho_v2"

# =========================
# Constants
# =========================
gamma = 5.0 / 3.0              # adiabatic index
mu0 = 4e-7 * np.pi
kB = 1.380649e-23
e_charge = 1.602176634e-19
m_p = 1.67262192369e-27
amu = 1.66053906660e-27

# =========================
# Helpers
# =========================
def fmt_sci(x, sig=2):
    """Compact scientific formatting for table cells."""
    if x == 0 or not np.isfinite(x):
        return f"{x}"
    exp = int(np.floor(np.log10(abs(x))))
    if -2 <= exp <= 3:
        return f"{x:.{sig}g}"
    mant = x / (10**exp)
    return f"{mant:.{sig}g}×10$^{exp}$"

def to_ev(T_K):
    return kB * T_K / e_charge

def to_K(T_eV):
    return T_eV * e_charge / kB

def md_table(header, units_row, rows):
    # header: list[str], units_row: list[str], rows: list[list[str]]
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    if units_row is not None:
        lines.append("| " + " | ".join(units_row) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)

# =========================
# Derived quantities
# =========================
n = sim_den.astype(float)                    # [m^-3]
v = sim_vel.astype(float)                    # [m/s]
mi = species_mass * amu                      # [kg]
Zi = species_charge.astype(float)

# Temperature handling: if user provided T_sw_K, compute eV; otherwise convert from eV.
if "T_sw_K" in globals() and T_sw_K is not None:
    T_K = np.array(T_sw_K, dtype=float)
    T_eV = to_ev(T_K)
elif "T_sw_eV" in globals() and T_sw_eV is not None:
    T_eV = np.array(T_sw_eV, dtype=float)
    T_K = to_K(T_eV)
else:
    raise ValueError("Provide either T_sw_K or T_sw_eV.")

# Choose density used for Alfvén speed / magnetosonic speed
rho_i = n * mi                                # species mass density [kg/m^3]
rho_total = np.sum(rho_i)                     # scalar total mass density
rho_proton = rho_i[0] if len(rho_i) > 0 else np.nan

if mach_density_mode == "total_mass":
    rho_for_mach = rho_total
elif mach_density_mode == "proton_only":
    rho_for_mach = rho_proton
else:
    raise ValueError("mach_density_mode must be 'total_mass' or 'proton_only'.")

# Alfvén speed and Alfvén Mach number: vA = B/sqrt(mu0*rho), MA = V/vA
vA = B_T / np.sqrt(mu0 * rho_for_mach)
M_A = v / vA  # per-species (since v can be per-species), but same if all v identical

# Sound speed: cs = sqrt(gamma * kB * T / mi)
cs = np.sqrt(gamma * kB * T_K / mi)

# Magnetosonic speed (fast, perpendicular simplification): c_ms = sqrt(vA^2 + cs^2)
c_ms = np.sqrt(vA**2 + cs**2)
M_s = v / c_ms

# Plasma beta: beta = (n kB T) / (B^2/(2 mu0)) = 2 mu0 n kB T / B^2
beta = 2 * mu0 * (n * kB * T_K) / (B_T**2)

# Dynamic pressure
# If pdyn_convention == "rho_v2": Pdyn = rho_total * V^2 (common in solar-wind contexts)
# If pdyn_convention == "half_rho_v2": q = 0.5 * rho_total * V^2 (fluid-dynamics "dynamic pressure")
V_for_pdyn = np.mean(v)  # one representative speed; change if you want per-species pdyn
if pdyn_convention == "rho_v2":
    Pdyn_Pa = rho_total * V_for_pdyn**2
elif pdyn_convention == "half_rho_v2":
    Pdyn_Pa = 0.5 * rho_total * V_for_pdyn**2
else:
    raise ValueError("pdyn_convention must be 'rho_v2' or 'half_rho_v2'.")
Pdyn_nPa = Pdyn_Pa * 1e9

# =========================
# Build markdown table
# =========================
# Display density in cm^-3 and speed in km/s
n_cm3 = n * 1e-6
v_kms = v * 1e-3

header = [
    "**Species**", "**Density**", r"**$\lVert \mathbf{v}_{sw} \rVert$**", "**T$_{sw}$**", "**T$_{sw}$**",
    "**ppc**", "**M$_{A}$**", "**M$_{s}$**", "**β**", "**P$_{dyn}$**"
]
units = ["", "[cm⁻³]", "[km/s]", "[K]", "[eV]", "", "", "", "", "[nPa]"]

rows = []
for i in range(len(species)):
    rows.append([
        f"{species[i]}",
        f"{n_cm3[i]:.1f}",
        f"{v_kms[i]:.1f}",
        fmt_sci(T_K[i], sig=2),
        f"{T_eV[i]:.3g}",
        f"{sim_ppc[i]:.3g}",
        f"{M_A[i]:.3g}",
        f"{M_s[i]:.3g}",
        f"{beta[i]:.3g}",
        f"{Pdyn_nPa:.3g}",   # same value repeated per row
    ])

print(md_table(header, units, rows))
