import os
import pandas as pd

# --------------------------
# SETTINGS
# --------------------------
cases = ["RPS", "RPN", "CPS", "CPN"]
step = 115000
output_folder = "/Users/danywaller/Projects/mercury/extreme/bfield_topology/"
output_tex_nlat = os.path.join(output_folder, f"ocb_north_latitude_metrics_{step}.tex")
output_tex_slat = os.path.join(output_folder, f"ocb_south_latitude_metrics_{step}.tex")
output_tex_asym = os.path.join(output_folder, f"ocb_asymmetry_metrics_{step}.tex")

# --------------------------
# METRICS
# --------------------------
north_latitude_metrics = [
    "ocb_north_mean_lat", "ocb_north_min_lat", "ocb_north_max_lat", "ocb_north_std_lat",
]
south_latitude_metrics = [
    "ocb_south_mean_lat", "ocb_south_min_lat", "ocb_south_max_lat", "ocb_south_std_lat"
]

asymmetry_metrics = [
    "polar_cap_area_asymmetry", "mean_lat_asymmetry", "connectedness_index"
]

metric_names = {
    "ocb_north_mean_lat": "North mean $\\phi$ [deg]",
    "ocb_north_min_lat": "North min $\\phi$ [deg]",
    "ocb_north_max_lat": "North max $\\phi$ [deg]",
    "ocb_north_std_lat": "North $\\sigma_\\phi$ [deg]",
    "ocb_south_mean_lat": "South mean $\\phi$ [deg]",
    "ocb_south_min_lat": "South min $\\phi$ [deg]",
    "ocb_south_max_lat": "South max $\\phi$ [deg]",
    "ocb_south_std_lat": "South $\\sigma_\\phi$ [deg]",
    "ocb_north_eccentricity": "North Eccentricity",
    "ocb_north_polar_cap_area_km2": "North Polar Cap Area [km$^2$]",
    "ocb_south_eccentricity": "South Eccentricity",
    "ocb_south_polar_cap_area_km2": "South Polar Cap Area [km$^2$]",
    "polar_cap_area_asymmetry": "MCAA",
    "mean_lat_asymmetry": "MLA",
    "connectedness_index": "C",
}

# --------------------------
# READ CSV FILES AND MERGE
# --------------------------
rows = []

for case in cases:
    input_folder = f"/Users/danywaller/Projects/mercury/extreme/bfield_topology/{case}_Base/"
    csv_file = os.path.join(input_folder, f"{case}_{step}_connectedness_metrics.csv")
    if not os.path.exists(csv_file):
        print(f"Warning: CSV not found for {case}: {csv_file}")
        continue

    df = pd.read_csv(csv_file)
    if df.empty:
        print(f"Warning: CSV is empty for {case}")
        continue

    # Take first row (single-timestep CSV)
    row = {"Case": case}
    for col in north_latitude_metrics + south_latitude_metrics + asymmetry_metrics:
        row[col] = df[col].values[0] if col in df.columns else float('nan')
    rows.append(row)

df_all = pd.DataFrame(rows)

# Round numeric columns
for col in north_latitude_metrics + south_latitude_metrics + asymmetry_metrics:
    if col in df_all.columns:
        df_all[col] = df_all[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "--")

# Rename for LaTeX
df_all = df_all.rename(columns=metric_names)

# --------------------------
# TABLE 1: NORTH LATITUDE METRICS
# --------------------------
df_nlat = df_all[["Case"] + [metric_names[m] for m in north_latitude_metrics]]
latex_table_nlat = df_nlat.to_latex(
    index=False,
    caption="Latitude Metrics for North OCB",
    label="tab:ocb_south_latitude",
    escape=False
)

with open(output_tex_nlat, "w") as f:
    f.write(latex_table_nlat)
print(f"Saved LaTeX latitude table to {output_tex_nlat}")

# --------------------------
# TABLE 2: SOUTH LATITUDE METRICS
# --------------------------
df_slat = df_all[["Case"] + [metric_names[m] for m in south_latitude_metrics]]
latex_table_slat = df_slat.to_latex(
    index=False,
    caption="Latitude Metrics for South OCB",
    label="tab:ocb_north_latitude",
    escape=False
)

with open(output_tex_slat, "w") as f:
    f.write(latex_table_slat)
print(f"Saved LaTeX latitude table to {output_tex_slat}")

# --------------------------
# TABLE 2: ECCENTRICITY & ASYMMETRY
# --------------------------
df_asym = df_all[["Case"] + [metric_names[m] for m in asymmetry_metrics]]
latex_table_asym = df_asym.to_latex(
    index=False,
    caption="Eccentricity, Polar Cap Area Asymmetry, and Mean Latitude Asymmetry",
    label="tab:ocb_asymmetry",
    escape=False
)

with open(output_tex_asym, "w") as f:
    f.write(latex_table_asym)
print(f"Saved LaTeX asymmetry table to {output_tex_asym}")
