#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# --- CONFIG ---
TABLE_DIR = Path("/Users/danywaller/Projects/mercury/extreme/surface_precipitation/csv_files/")
output_folder = "/Users/danywaller/Projects/mercury/extreme/surface_precipitation/"
PATTERN = "*.csv"  # Changed to CSV
# ----------------

def phase_from_filename(fname: str) -> str:
    s = fname.lower()
    if "pre" in s and "transient" in s:
        return "Pre-transient"
    if "post" in s and "transient" in s:
        return "Post-transient"
    if "transient" in s and "post" not in s and "pre" not in s:
        return "Transient"
    if "new" in s and "state" in s:
        return "New state"
    return Path(fname).stem

# --- Ingest all CSV tables and combine ---

dfs = []
for path in TABLE_DIR.glob(PATTERN):
    df = pd.read_csv(path)
    phase = phase_from_filename(path.name)
    df["phase"] = phase
    dfs.append(df)

if not dfs:
    raise RuntimeError("No matching .csv tables found.")

all_df = pd.concat(dfs, ignore_index=True)

# Map CSV columns to previous names (based on attached CSV structure)
col_map = {
    "case_name": "Case",
    "total_integrated_flux": "Total Flux",
    "peak_flux_value": "Peak Flux",
    "peak_flux_lat": "Peak Lat",
    "peak_flux_lon": "Peak Lon",
    "spatial_extent_percentage": "Precip. Area",
    "hemispheric_asymmetry_ratio": "N/S",
    "dayside_nightside_ratio": "Day/Night",
}
all_df = all_df.rename(columns=col_map)

# --- Ordered phase axis ---
phase_order = ["Pre-transient", "Transient", "Post-transient", "New state"]
all_df["phase"] = pd.Categorical(all_df["phase"], categories=phase_order, ordered=True)
all_df = all_df.sort_values(["Case", "phase"])

# --- Multi-subplot figure ---

case_order = ["RPN", "CPN", "RPS", "CPS"]
cases = [c for c in case_order if c in all_df["Case"].unique()]

case_styles = {
    "RPN": {"color": "salmon", "marker": "o"},
    "CPN": {"color": "cornflowerblue", "marker": "s"},
    "RPS": {"color": "mediumorchid", "marker": "^"},
    "CPS": {"color": "forestgreen", "marker": "D"},
}

phases_num = range(len(phase_order))

fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True)
axes = axes.ravel()

variables = [
    ("Total Flux", "(a) Total integrated flux", "[particles s$^{-1}$]", True),
    ("Peak Flux", "(b) Peak flux", "[cm$^{-2}$ s$^{-1}$]", True),
    ("Precip. Area", "(c) Precipitating area", "[%]", False),
    ("Day/Night", "(d) Dayside/Nightside ratio", "", True),
    ("Peak Lat", "(e) Peak latitude", "[$^{\\circ}$]", False),
    ("N/S", "(f) North/South ratio", "", False),
]

for ax, (var, title, unit, logy) in zip(axes, variables):  # rename ylabel → title
    for case in cases:
        sub = all_df[all_df["Case"] == case]
        x = [phase_order.index(p) for p in sub["phase"]]
        y = sub[var].values
        style = case_styles.get(case, {"color": "black", "marker": "o"})
        ax.plot(x, y,
                label=case,
                color=style["color"],
                marker=style["marker"])
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(f"{var} {unit}", fontsize=11)  # short version on y-axis
    ax.set_xticks(list(phases_num))
    ax.set_xticklabels(phase_order, rotation=45, ha="right")
    if logy:
        ax.set_yscale("log")

# Shared legend at bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Case",
           loc="lower center", ncol=len(cases), bbox_to_anchor=(0.5, 0.01))

fig.align_ylabels()
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
# plt.show()
outfile = os.path.join(output_folder, f"all_cases_all_species_timeseries_plot.png")
plt.savefig(outfile, dpi=200)
