import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
debug = False
base_dir = "/Users/danywaller/Projects/mercury/extreme/surface_flux_timeseries"

cases = [
    "RPN_HNHV",
    "CPN_HNHV",
    "RPS_HNHV",
    "CPS_HNHV"
]

case_styles = {
    "RPN_HNHV": dict(color="forestgreen", ls="-"),
    "CPN_HNHV": dict(color="cornflowerblue",    ls="-"),
    "RPS_HNHV": dict(color="darkorchid",  ls="-"),
    "CPS_HNHV": dict(color="hotpink",       ls="-"),
}

# ---------------------------------------------------------
# Containers for DataFrames
# ---------------------------------------------------------

total_stats = {}
proton_stats = {}
alpha_stats = {}

# ---------------------------------------------------------
# Load CSV files
# ---------------------------------------------------------

for case in cases:

    case_dir = os.path.join(base_dir, case)

    total_file  = os.path.join(case_dir, f"{case}_stats_total.csv")
    proton_file = os.path.join(case_dir, f"{case}_stats_protons_sum.csv")
    alpha_file  = os.path.join(case_dir, f"{case}_stats_alphas_sum.csv")

    total_stats[case]  = pd.read_csv(total_file)
    proton_stats[case] = pd.read_csv(proton_file)
    alpha_stats[case]  = pd.read_csv(alpha_file)

# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------

print("Loaded files for:", list(total_stats.keys()))

if debug:
    for case in cases:
        print("\nCase:", case)
        print("\tTotal stats:")
        print(total_stats[case].columns)

        print("\tProton stats:")
        print(proton_stats[case].columns)

        print("\tAlpha stats:")
        print(alpha_stats[case].columns)


# -------------------------------------------------
# Parameters
# -------------------------------------------------
variables = [
    "hemispheric_asymmetry_ratio",
    "dayside_nightside_ratio",
    "dawn_dusk_ratio",
    "spatial_extent_percentage",
]

titles = [
    "(A) North/South precipitation ratio",
    "(B) Day/Night precipitation ratio",
    "(C) Dawn/Dusk precipitation ratio",
    "(D) Precipitation spatial extent",
]

ylabels = [
    "N/S ratio",
    "Day/Night ratio",
    "Dawn/Dusk ratio",
    "Surface area (%)",
]

# -------------------------------------------------
# total stats figure
# -------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True)
axes = axes.flatten()

axes = axes.flatten()

lines = {}

for case, df in total_stats.items():

    style = case_styles[case]

    for i, var in enumerate(variables):

        line, = axes[i].plot(
            df["time"],
            df[var],
            color=style["color"],
            linestyle=style["ls"],
            linewidth=1.0,
        )

        if case not in lines:
            lines[case] = line

# Axis formatting
for i, ax in enumerate(axes):

    ax.set_title(titles[i], fontweight="bold")
    ax.set_ylabel(ylabels[i])
    ax.grid(alpha=0.3)

    if variables[i] == "peak_flux_value":
        ax.set_yscale("log")

    if variables[i] == "peak_flux_lat":
        ax.set_ylim([-95, 95])

    if variables[i] == "hemispheric_asymmetry_ratio":
        ax.set_ylim([0.0, 1.6])

    if variables[i] == "dayside_nightside_ratio":
        ax.set_ylim([0.0, 50.0])

    if variables[i] == "dawn_dusk_ratio":
        ax.set_ylim([0.25, 2.25])
        ax.set_xlabel("Time (s)")

    if variables[i] == "spatial_extent_percentage":
        ax.set_ylim([-0.5, 40.0])
        ax.set_xlabel("Time (s)")

    # --- Add black dotted vertical lines ---
    for x in [240, 360]:
        ax.axvline(x, color='k', linestyle='--', linewidth=1.0)
        # Add '*' at the top of the line
        ylim = ax.get_ylim()
        ax.text(
            x+0.015*x, ylim[1]-0.07*ylim[1], '*', color='k', fontsize=14,
            ha='left', va='bottom', fontweight='bold'
        )

# Global legend (bottom)
handles = []
labels_legend = []

for case in case_styles.keys():

    handles.append(lines[case])
    labels_legend.append(case.split("_")[0])

leg = fig.legend(
    handles,
    labels_legend,
    loc="lower center",
    ncol=4,
    fontsize=12,
    bbox_to_anchor=(0.5, 0.01)
)
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
# bulk-set the properties of all lines and texts
plt.setp(leg_lines, linewidth=2.5)

fig.align_ylabels()

# Save / show
plt.suptitle("Total Precipitation Extent Metrics", fontsize=16, fontweight="bold", y=0.99)
plt.tight_layout()

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_png = os.path.join(base_dir, "all_cases_total_precip_stats.png")
plt.savefig(out_png, dpi=300)
plt.show()


# -------------------------------------------------
# proton stats figure
# -------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True)

axes = axes.flatten()

lines = {}

for case, df in proton_stats.items():

    style = case_styles[case]

    for i, var in enumerate(variables):

        line, = axes[i].plot(
            df["time"],
            df[var],
            color=style["color"],
            linestyle=style["ls"],
            linewidth=1.2,
        )

        if case not in lines:
            lines[case] = line

# Axis formatting
for i, ax in enumerate(axes):

    ax.set_title(titles[i], fontweight="bold")
    ax.set_ylabel(ylabels[i])
    ax.grid(alpha=0.3)

    # --- Add black dotted vertical lines ---
    ax.axvline(240, color='k', linestyle='--', linewidth=1.0)
    ax.axvline(360, color='k', linestyle='--', linewidth=1.0)

    if variables[i] == "peak_flux_value":
        ax.set_yscale("log")

    if variables[i] == "peak_flux_lat":
        ax.set_ylim([-95, 95])

    if variables[i] == "hemispheric_asymmetry_ratio":
        ax.set_ylim([0.0, 1.6])

    if variables[i] == "dayside_nightside_ratio":
        ax.set_ylim([0.0, 50.0])

    if variables[i] == "dawn_dusk_ratio":
        ax.set_ylim([0.25, 2.25])
        ax.set_xlabel("Time (s)")

    if variables[i] == "spatial_extent_percentage":
        ax.set_ylim([-0.5, 40.0])
        ax.set_xlabel("Time (s)")

# Global legend (bottom)
handles = []
labels_legend = []

for case in case_styles.keys():

    handles.append(lines[case])
    labels_legend.append(case.split("_")[0])

fig.legend(
    handles,
    labels_legend,
    loc="lower center",
    ncol=4,
    fontsize=12,
    bbox_to_anchor=(0.5, 0.01)
)

fig.align_ylabels()

# Save / show
plt.suptitle("Proton precipitation statistics vs time", fontsize=16, fontweight="bold")

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_png = os.path.join(base_dir, "all_cases_proton_precip_stats.png")
plt.savefig(out_png, dpi=300)
plt.show()


# -------------------------------------------------
# alpha stats figure
# -------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True)

axes = axes.flatten()

lines = {}

for case, df in alpha_stats.items():

    style = case_styles[case]

    for i, var in enumerate(variables):

        line, = axes[i].plot(
            df["time"],
            df[var],
            color=style["color"],
            linestyle=style["ls"],
            linewidth=1.2,
        )

        if case not in lines:
            lines[case] = line

# -------------------------------------------------
# Axis formatting
# -------------------------------------------------

for i, ax in enumerate(axes):

    ax.set_title(titles[i])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabels[i])
    ax.grid(alpha=0.3)

    if variables[i] == "peak_flux_value":
        ax.set_yscale("log")

    if variables[i] == "peak_flux_lat":
        ax.set_ylim([-95, 95])

    if variables[i] == "hemispheric_asymmetry_ratio":
        ax.set_ylim([0.0, 1.6])

    if variables[i] == "dayside_nightside_ratio":
        ax.set_ylim([0.0, 50.0])

    if variables[i] == "dawn_dusk_ratio":
        ax.set_ylim([0.5, 5.0])

    if variables[i] == "spatial_extent_percentage":
        ax.set_ylim([-0.5, 40.0])

# Global legend (bottom)
handles = []
labels_legend = []

for case in case_styles.keys():

    handles.append(lines[case])
    labels_legend.append(case.split("_")[0])

fig.legend(
    handles,
    labels_legend,
    loc="lower center",
    ncol=4,
    fontsize=12,
    bbox_to_anchor=(0.5, 0.01)
)

fig.align_ylabels()

# Save / show
plt.suptitle("Alpha precipitation statistics vs time", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_png = os.path.join(base_dir, "all_cases_alpha_precip_stats.png")
plt.savefig(out_png, dpi=300)
plt.show()

# -------------------------------------------------
# total stats figure - zoom on transient
# -------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True)

axes = axes.flatten()

lines = {}

for case, df in total_stats.items():

    style = case_styles[case]

    for i, var in enumerate(variables):

        line, = axes[i].plot(
            df["time"],
            df[var],
            color=style["color"],
            linestyle=style["ls"],
            linewidth=1.2,
        )

        if case not in lines:
            lines[case] = line

# Axis formatting
for i, ax in enumerate(axes):

    zoom_title = titles[i].replace(")", "*)")
    ax.set_title(zoom_title, fontweight="bold")
    ax.set_ylabel(ylabels[i])
    ax.grid(alpha=0.3)

    ax.set_xlim([240, 360])

    if variables[i] == "peak_flux_value":
        ax.set_yscale("log")

    if variables[i] == "peak_flux_lat":
        ax.set_ylim([-95, 95])

    if variables[i] == "hemispheric_asymmetry_ratio":
        ax.set_ylim([0.0, 1.6])

    if variables[i] == "dayside_nightside_ratio":
        ax.set_ylim([0.0, 50.0])

    if variables[i] == "dawn_dusk_ratio":
        ax.set_ylim([0.25, 2.25])
        ax.set_xlabel("Time (s)")

    if variables[i] == "spatial_extent_percentage":
        ax.set_ylim([-0.5, 40.0])
        ax.set_xlabel("Time (s)")

# global legends
handles = []
labels_legend = []

for case in case_styles.keys():

    handles.append(lines[case])
    labels_legend.append(case.split("_")[0])

leg = fig.legend(
    handles,
    labels_legend,
    loc="lower center",
    ncol=4,
    fontsize=12,
    bbox_to_anchor=(0.5, 0.01)
)
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
# bulk-set the properties of all lines and texts
plt.setp(leg_lines, linewidth=2.5)

fig.align_ylabels()

# Save / show
plt.suptitle("Transient Zoom", fontsize=16, fontweight="bold", y=0.99)

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_png = os.path.join(base_dir, "all_cases_total_precip_stats_transient_zoom.png")
plt.savefig(out_png, dpi=300)
plt.show()