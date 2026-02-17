# %%
import os
from datetime import datetime
import numpy as np
import pandas as pd
import random
import xarray as xr
from src.field_topology.topology_utils import *
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# --------------------------
# SETTINGS
# --------------------------
case = "CPN"
step = 350000  # 115000 for base or 350000 for hnhv

input_folder = f"/Volumes/data_backup/mercury/extreme/High_HNHV/{case}_HNHV/10/out/"
ncfile = os.path.join(input_folder, f"Amitis_{case}_HNHV_{step}.nc")

# input_folder = f"/Volumes/data_backup/mercury/extreme/{case}_Base/05/out/"
# ncfile = os.path.join(input_folder, f"Amitis_{case}_Base_{step}.nc")

output_folder = f"/Users/danywaller/Projects/mercury/extreme/jfield_topology/{case}/"
os.makedirs(output_folder, exist_ok=True)

# Planet parameters
RM = 2440.0          # Mercury radius [km]
RC = 2400.0          # depth within conductive layer [km]

plot_depth = RM

# Seed settings
n_lat = 60
n_lon = n_lat*2
max_steps = 5000
h_step = 50.0
surface_tol = 75.0

max_lines = 250  # downsample trajectory points for plotting

# Clustering settings
clustering_method = "hierarchical"  # "dbscan" or "hierarchical"
auto_select_clusters = True  # automatically find optimal number of clusters
max_clusters_to_test = 10  # maximum number of clusters to test

# %%
# --------------------------
# CREATE SEEDS ON SPHERE
# --------------------------
lats_surface = np.linspace(-90, 90, n_lat)
lons_surface = np.linspace(-180, 180, n_lon)
seeds = []
for lat in lats_surface:
    for lon in lons_surface:
        phi = np.radians(lat)
        theta = np.radians(lon)
        x_s = plot_depth*np.cos(phi)*np.cos(theta)
        y_s = plot_depth*np.cos(phi)*np.sin(theta)
        z_s = plot_depth*np.sin(phi)
        seeds.append(np.array([x_s, y_s, z_s]))
seeds = np.array(seeds)

# --------------------------
# LOAD VECTOR FIELD FROM NETCDF
# --------------------------
def load_field(ncfile):
    ds = xr.open_dataset(ncfile)
    x = ds["Nx"].values
    y = ds["Ny"].values
    z = ds["Nz"].values

    # Extract fields (drop time dimension) and transpose:  Nz, Ny, Nx --> Nx, Ny, Nz
    Jx = np.transpose(ds["Jx"].isel(time=0).values, (2,1,0))
    Jy = np.transpose(ds["Jy"].isel(time=0).values, (2,1,0))
    Jz = np.transpose(ds["Jz"].isel(time=0).values, (2,1,0))
    ds.close()
    return x, y, z, Jx, Jy, Jz

# --------------------------
# FEATURE EXTRACTION
# --------------------------
def extract_trajectory_features(traj_fwd, traj_bwd):
    """
    Extract geometric and topological features from a field line trajectory.
    """
    full_traj = np.vstack([traj_bwd[::-1], traj_fwd])

    features = []

    # Spatial extent features
    features.append(np.min(full_traj[:, 0]))      # min X
    features.append(np.max(full_traj[:, 0]))      # max X
    features.append(np.mean(full_traj[:, 0]))     # mean X

    features.append(np.min(full_traj[:, 1]))      # min Y
    features.append(np.max(full_traj[:, 1]))      # max Y
    features.append(np.mean(full_traj[:, 1]))     # mean Y

    features.append(np.min(full_traj[:, 2]))      # min Z
    features.append(np.max(full_traj[:, 2]))      # max Z
    features.append(np.mean(full_traj[:, 2]))     # mean Z

    # Radial features
    r = np.linalg.norm(full_traj, axis=1)
    features.append(np.min(r))                     # min radial distance
    features.append(np.max(r))                     # max radial distance
    features.append(np.mean(r))                    # mean radial distance

    # Shape features
    features.append(np.std(full_traj[:, 0]))      # X spread
    features.append(np.std(full_traj[:, 1]))      # Y spread
    features.append(np.std(full_traj[:, 2]))      # Z spread

    # Asymmetry features
    features.append(np.abs(np.mean(full_traj[:, 2])))  # Z asymmetry

    # Arc length
    diffs = np.diff(full_traj, axis=0)
    arc_length = np.sum(np.linalg.norm(diffs, axis=1))
    features.append(arc_length)

    # Tail vs dayside preference
    features.append(np.sum(full_traj[:, 0] < 0) / len(full_traj))

    # High latitude measure
    features.append(np.sum(np.abs(full_traj[:, 2]) > RM) / len(full_traj))

    return np.array(features)

# --------------------------
# OPTIMAL CLUSTER SELECTION
# --------------------------
def find_optimal_eps_dbscan(features_scaled, min_samples=5):
    """
    Find optimal epsilon for DBSCAN using k-distance plot (elbow method).
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(features_scaled)
    distances, indices = neighbors_fit.kneighbors(features_scaled)

    # Sort distances
    distances = np.sort(distances[:, -1], axis=0)

    # Find elbow using maximum curvature
    x = np.arange(len(distances))
    y = distances

    # Normalize
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Find point with maximum distance to line connecting first and last points
    # Convert to 3D vectors to avoid NumPy 2.0 deprecation warning
    p1_3d = np.array([x_norm[0], y_norm[0], 0.0])
    p2_3d = np.array([x_norm[-1], y_norm[-1], 0.0])

    max_dist = 0
    elbow_idx = 0
    for i in range(len(x_norm)):
        p_3d = np.array([x_norm[i], y_norm[i], 0.0])
        # Cross product of 3D vectors
        cross = np.cross(p2_3d - p1_3d, p1_3d - p_3d)
        dist = np.linalg.norm(cross) / np.linalg.norm(p2_3d - p1_3d)
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i

    optimal_eps = distances[elbow_idx]

    # Optional: plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.axhline(y=optimal_eps, color='r', linestyle='--', label=f'Optimal eps = {optimal_eps:.3f}')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel(f'{min_samples}-NN Distance')
    plt.title('K-distance Graph for Optimal Epsilon Selection')
    plt.legend()
    plt.grid(True)

    return optimal_eps, plt


def evaluate_clustering(features_scaled, labels):
    """
    Evaluate clustering quality using multiple metrics.
    Returns dict with scores.
    """
    # Filter out noise points (label -1) for DBSCAN
    mask = labels != -1
    if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
        return None

    scores = {}

    # Silhouette Score: [-1, 1], higher is better
    scores['silhouette'] = silhouette_score(features_scaled[mask], labels[mask])

    # Calinski-Harabasz Index (Variance Ratio): higher is better
    scores['calinski_harabasz'] = calinski_harabasz_score(features_scaled[mask], labels[mask])

    # Davies-Bouldin Index: lower is better
    scores['davies_bouldin'] = davies_bouldin_score(features_scaled[mask], labels[mask])

    scores['n_clusters'] = len(np.unique(labels[mask]))
    scores['n_noise'] = np.sum(labels == -1)

    return scores


def find_optimal_n_clusters_hierarchical(features_scaled, max_k=10):
    """
    Find optimal number of clusters for hierarchical clustering.
    Tests multiple metrics across different k values.
    """
    results = []

    for k in range(2, max_k + 1):
        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = clustering.fit_predict(features_scaled)

        scores = evaluate_clustering(features_scaled, labels)
        if scores is not None:
            scores['k'] = k
            results.append(scores)

    # Create results dataframe
    df_results = pd.DataFrame(results)

    # Normalize scores for comparison (0-1 scale)
    df_results['silhouette_norm'] = (df_results['silhouette'] - df_results['silhouette'].min()) / \
                                     (df_results['silhouette'].max() - df_results['silhouette'].min())
    df_results['ch_norm'] = (df_results['calinski_harabasz'] - df_results['calinski_harabasz'].min()) / \
                             (df_results['calinski_harabasz'].max() - df_results['calinski_harabasz'].min())
    # DB index: invert since lower is better
    df_results['db_norm'] = 1 - (df_results['davies_bouldin'] - df_results['davies_bouldin'].min()) / \
                                 (df_results['davies_bouldin'].max() - df_results['davies_bouldin'].min())

    # Combined score (equal weighting)
    df_results['combined_score'] = (df_results['silhouette_norm'] +
                                     df_results['ch_norm'] +
                                     df_results['db_norm']) / 3

    optimal_k = df_results.loc[df_results['combined_score'].idxmax(), 'k']

    # Plot validation metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(df_results['k'], df_results['silhouette'], 'o-')
    axes[0, 0].axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Number of Clusters')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].set_title('Silhouette Score (higher is better)')
    axes[0, 0].grid(True)

    axes[0, 1].plot(df_results['k'], df_results['calinski_harabasz'], 'o-', color='green')
    axes[0, 1].axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('Calinski-Harabasz Index')
    axes[0, 1].set_title('Calinski-Harabasz Index (higher is better)')
    axes[0, 1].grid(True)

    axes[1, 0].plot(df_results['k'], df_results['davies_bouldin'], 'o-', color='orange')
    axes[1, 0].axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Number of Clusters')
    axes[1, 0].set_ylabel('Davies-Bouldin Index')
    axes[1, 0].set_title('Davies-Bouldin Index (lower is better)')
    axes[1, 0].grid(True)

    axes[1, 1].plot(df_results['k'], df_results['combined_score'], 'o-', color='purple')
    axes[1, 1].axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7,
                       label=f'Optimal k = {int(optimal_k)}')
    axes[1, 1].set_xlabel('Number of Clusters')
    axes[1, 1].set_ylabel('Combined Score')
    axes[1, 1].set_title('Combined Validation Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    return int(optimal_k), df_results, plt


def cluster_trajectories_optimal(trajectory_pairs, method="hierarchical",
                                  max_k=10, min_samples=5):
    """
    Cluster trajectories with automatic optimal parameter selection.
    """
    # Extract features
    features = np.array([extract_trajectory_features(fwd, bwd)
                         for fwd, bwd in trajectory_pairs])

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if method == "dbscan":
        # Find optimal epsilon
        optimal_eps, kdist_plot = find_optimal_eps_dbscan(features_scaled, min_samples)

        # Cluster with optimal parameters
        clustering = DBSCAN(eps=optimal_eps, min_samples=min_samples)
        labels = clustering.fit_predict(features_scaled)

        scores = evaluate_clustering(features_scaled, labels)

        return labels, scores, kdist_plot

    elif method == "hierarchical":
        # Find optimal number of clusters
        optimal_k, results_df, validation_plot = find_optimal_n_clusters_hierarchical(
            features_scaled, max_k)

        # Cluster with optimal k
        clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
        labels = clustering.fit_predict(features_scaled)

        scores = evaluate_clustering(features_scaled, labels)

        return labels, scores, validation_plot, results_df

    else:
        raise ValueError(f"Unknown clustering method: {method}")


# function to lightly smooth field lines
def smooth_traj(traj, k=5, order=2):
    if traj.shape[0] < k:
        return traj
    return np.column_stack([
        savgol_filter(traj[:, i], k, order, mode="interp")
        for i in range(3)
    ])

# %%
x, y, z, Jx, Jy, Jz = load_field(ncfile)

start = datetime.now()
print(f"Loaded {ncfile} at {str(start)}")

# %%
# --------------------------
# TRACE FIELD LINES
# --------------------------
lines_by_topo = {"closed": [], "open": []}
trajectory_pairs = {"closed": [], "open": []}

for seed in seeds:
    traj_fwd, exit_fwd_y = trace_field_line_rk(seed, Jx, Jy, Jz, x, y, z, plot_depth,
                                                max_steps=max_steps, h=h_step)
    traj_bwd, exit_bwd_y = trace_field_line_rk(seed, Jx, Jy, Jz, x, y, z, plot_depth,
                                                max_steps=max_steps, h=-h_step)
    topo = classify(traj_fwd, traj_bwd, plot_depth + surface_tol, exit_fwd_y, exit_bwd_y)

    if topo in ["closed", "open"]:
        lines_by_topo[topo].append(traj_fwd)
        lines_by_topo[topo].append(traj_bwd)
        trajectory_pairs[topo].append((traj_fwd, traj_bwd))

classtime = datetime.now()
print(f"Classified all lines at {str(classtime)}")
print(f"Found {len(trajectory_pairs['closed'])} closed field lines")
print(f"Found {len(trajectory_pairs['open'])} open field lines")

# %%
# --------------------------
# CLUSTER OPEN VS CLOSED TRAJECTORIES WITH OPTIMAL PARAMETERS
# --------------------------
cluster_labels = {"closed": None, "open": None}
lines_by_cluster = {}
all_results = {}
all_scores = {}

for topo in ["closed", "open"]:
    if len(trajectory_pairs[topo]) == 0:
        continue

    print(f"\n{'='*60}")
    print(f"Clustering {topo} field lines...")
    print(f"{'='*60}")

    if clustering_method == "hierarchical":
        labels, scores, validation_plot, results_df = cluster_trajectories_optimal(
            trajectory_pairs[topo],
            method=clustering_method,
            max_k=max_clusters_to_test
        )

        # Save validation plots
        validation_plot.savefig(
            os.path.join(output_folder, f"{case}_{step}_{topo}_validation_metrics.png"),
            dpi=150, bbox_inches='tight'
        )
        plt.close()

        # Save results table
        results_df.to_csv(
            os.path.join(output_folder, f"{case}_{step}_{topo}_clustering_scores.csv"),
            index=False
        )

        all_results[topo] = results_df
        all_scores[topo] = scores

    else:  # DBSCAN
        labels, scores, kdist_plot = cluster_trajectories_optimal(
            trajectory_pairs[topo],
            method=clustering_method
        )

        # Save k-distance plot
        kdist_plot.savefig(
            os.path.join(output_folder, f"{case}_{step}_{topo}_kdistance_plot.png"),
            dpi=150, bbox_inches='tight'
        )
        plt.close()

        all_scores[topo] = scores

    # Print results
    print(f"\nOptimal clustering results:")
    print(f"  Number of clusters: {scores['n_clusters']}")
    if 'n_noise' in scores:
        n_noise = scores['n_noise']
        n_total = len(trajectory_pairs[topo])
        pct_noise = 100 * n_noise / n_total
        print(f"  Noise points: {n_noise} ({pct_noise:.1f}%)")
    print(f"  Silhouette Score: {scores['silhouette']:.3f}")
    print(f"  Calinski-Harabasz Index: {scores['calinski_harabasz']:.1f}")
    print(f"  Davies-Bouldin Index: {scores['davies_bouldin']:.3f}")

    cluster_labels[topo] = labels
    unique_labels = np.unique(labels)

    # Calculate cluster sizes
    cluster_sizes = []
    for cluster_id in unique_labels:
        if cluster_id == -1:  # Skip noise for DBSCAN
            continue
        mask = labels == cluster_id
        count = np.sum(mask)
        cluster_sizes.append((cluster_id, count))

    # Sort by size for better readability
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Cluster sizes (sorted by count):")

    # Organize trajectories by cluster
    for cluster_id, count in cluster_sizes:
        cluster_key = f"{topo}_cluster_{cluster_id}"
        lines_by_cluster[cluster_key] = []

        mask = labels == cluster_id
        for idx, (traj_fwd, traj_bwd) in enumerate(trajectory_pairs[topo]):
            if mask[idx]:
                lines_by_cluster[cluster_key].append(traj_fwd)
                lines_by_cluster[cluster_key].append(traj_bwd)

        pct = 100 * count / len(trajectory_pairs[topo])
        print(f"    Cluster {cluster_id}: {count:4d} field lines ({pct:5.1f}%)")

    # Add noise to dictionary if using DBSCAN
    if clustering_method == "dbscan" and -1 in unique_labels:
        noise_key = f"{topo}_noise"
        lines_by_cluster[noise_key] = []

        mask = labels == -1
        for idx, (traj_fwd, traj_bwd) in enumerate(trajectory_pairs[topo]):
            if mask[idx]:
                lines_by_cluster[noise_key].append(traj_fwd)
                lines_by_cluster[noise_key].append(traj_bwd)

clustertime = datetime.now()
print(f"\nCompleted open vs closed clustering at {str(clustertime)}")

# Print summary statistics
print(f"\n{'='*60}")
print(f"CLUSTERING SUMMARY")
print(f"{'='*60}")

for topo in ["closed", "open"]:
    if cluster_labels[topo] is not None:
        scores = all_scores[topo]
        labels = cluster_labels[topo]
        n_clusters = scores['n_clusters']
        n_total = len(trajectory_pairs[topo])
        n_clustered = np.sum(labels != -1)

        print(f"\n{topo.upper()}:")
        print(f"  Total field lines: {n_total}")
        print(f"  Clustered: {n_clustered} ({100*n_clustered/n_total:.1f}%)")
        if 'n_noise' in scores:
            print(f"  Noise: {scores['n_noise']} ({100*scores['n_noise']/n_total:.1f}%)")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Quality metrics:")
        print(f"    Silhouette: {scores['silhouette']:.3f} (range: -1 to 1, higher is better)")
        print(f"    Calinski-Harabasz: {scores['calinski_harabasz']:.1f} (higher is better)")
        print(f"    Davies-Bouldin: {scores['davies_bouldin']:.3f} (lower is better)")

        # Check for highly imbalanced clusters
        cluster_counts = []
        for cluster_id in np.unique(labels):
            if cluster_id != -1:
                cluster_counts.append(np.sum(labels == cluster_id))

        if len(cluster_counts) > 0:
            largest = max(cluster_counts)
            smallest = min(cluster_counts)
            imbalance_ratio = largest / smallest if smallest > 0 else float('inf')

            if imbalance_ratio > 100:
                print(f"  ⚠️  WARNING: Highly imbalanced clusters detected!")
                print(f"      Largest cluster: {largest} | Smallest cluster: {smallest}")
                print(f"      Consider adjusting clustering parameters or trying hierarchical method.")

# %%
# --------------------------
# PLOT 3D FIELD LINES WITH SEPARATE LEGENDS
# --------------------------
def generate_colors(n):
    """Generate visually distinct colors using tab10-style palette"""
    import matplotlib.colors as mcolors

    # Tab10 colors (10 maximally distinct colors)
    tab10_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
    ]

    # If we need more than 10 colors, cycle through with slight variations
    colors = []
    for i in range(n):
        base_color = tab10_colors[i % 10]

        if i < 10:
            colors.append(base_color)
        else:
            # For additional colors, darken or lighten the base colors
            rgb = mcolors.hex2color(base_color)
            factor = 0.7 if (i // 10) % 2 == 0 else 1.3
            rgb_adjusted = tuple(min(1.0, max(0.0, c * factor)) for c in rgb)
            colors.append(f'rgb({int(rgb_adjusted[0]*255)},{int(rgb_adjusted[1]*255)},{int(rgb_adjusted[2]*255)})')

    return colors


# Separate clusters by topology
closed_clusters = sorted([k for k in lines_by_cluster.keys() if k.startswith('closed')],
                        key=lambda x: int(x.split('_')[-1]))
open_clusters = sorted([k for k in lines_by_cluster.keys() if k.startswith('open')],
                      key=lambda x: int(x.split('_')[-1]))

# Generate colors for each topology separately
closed_colors = generate_colors(len(closed_clusters))
open_colors = generate_colors(len(open_clusters))

colors_dict = {}
for key, color in zip(closed_clusters, closed_colors):
    colors_dict[key] = color
for key, color in zip(open_clusters, open_colors):
    colors_dict[key] = color

# Calculate cluster statistics
cluster_stats = {}
for topo in ["closed", "open"]:
    if cluster_labels[topo] is None:
        continue
    labels = cluster_labels[topo]
    unique_labels = np.unique(labels[labels != -1])

    for cluster_id in unique_labels:
        cluster_key = f"{topo}_cluster_{cluster_id}"
        n_lines = np.sum(labels == cluster_id)
        cluster_stats[cluster_key] = n_lines

# Prepare title statistics
title_stats = []
if cluster_labels["closed"] is not None:
    closed_scores = all_results.get("closed")
    if closed_scores is not None:
        closed_sil = closed_scores.loc[closed_scores['combined_score'].idxmax(), 'silhouette']
        closed_ch = closed_scores.loc[closed_scores['combined_score'].idxmax(), 'calinski_harabasz']
    else:
        closed_sil = closed_ch = 0
    title_stats.append(f"Closed: {len(closed_clusters)} clusters (S={closed_sil:.2f}, CH={closed_ch:.0f})")

if cluster_labels["open"] is not None:
    open_scores = all_results.get("open")
    if open_scores is not None:
        open_sil = open_scores.loc[open_scores['combined_score'].idxmax(), 'silhouette']
        open_ch = open_scores.loc[open_scores['combined_score'].idxmax(), 'calinski_harabasz']
    else:
        open_sil = open_ch = 0
    title_stats.append(f"Open: {len(open_clusters)} clusters (S={open_sil:.2f}, CH={open_ch:.0f})")

fig = go.Figure()

# Add planet sphere
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 200)
theta, phi = np.meshgrid(theta, phi)

xs = plot_depth * np.sin(theta) * np.cos(phi)
ys = plot_depth * np.sin(theta) * np.sin(phi)
zs = plot_depth * np.cos(theta)

eps = 0
mask_pos = xs >= -eps
mask_neg = xs <= eps

# Dayside hemisphere
fig.add_trace(go.Surface(
    x=np.where(mask_pos, xs, np.nan),
    y=np.where(mask_pos, ys, np.nan),
    z=np.where(mask_pos, zs, np.nan),
    surfacecolor=np.ones_like(xs),
    colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
    cmin=0, cmax=1,
    showscale=False,
    lighting=dict(ambient=1, diffuse=0, specular=0),
    hoverinfo='skip',
    legendgroup='planet',
    showlegend=False
))

# Nightside hemisphere
fig.add_trace(go.Surface(
    x=np.where(mask_neg, xs, np.nan),
    y=np.where(mask_neg, ys, np.nan),
    z=np.where(mask_neg, zs, np.nan),
    surfacecolor=np.zeros_like(xs),
    colorscale=[[0, 'black'], [1, 'black']],
    cmin=0, cmax=1,
    showscale=False,
    lighting=dict(ambient=1, diffuse=0, specular=0),
    hoverinfo='skip',
    legendgroup='planet',
    showlegend=False
))

# Add closed field lines
print("\nPlotting closed field lines...")
for cluster_key in closed_clusters:
    lines = lines_by_cluster[cluster_key]
    if len(lines) == 0:
        continue

    cluster_id = int(cluster_key.split('_')[-1])
    n_lines = cluster_stats[cluster_key]
    legend_label = f"Closed {cluster_id} (n={n_lines})"

    first = True

    if len(lines) > max_lines:
        lines_to_plot = random.sample(lines, max_lines)
    else:
        lines_to_plot = lines

    for traj in lines_to_plot:
        traj_s = smooth_traj(traj)

        fig.add_trace(go.Scatter3d(
            x=traj_s[:, 0],
            y=traj_s[:, 1],
            z=traj_s[:, 2],
            mode='lines',
            line=dict(color=colors_dict[cluster_key], width=2),
            name=legend_label,
            legendgroup=cluster_key,
            legendgrouptitle_text="Closed" if cluster_key == closed_clusters[0] else None,
            showlegend=first,
            hovertemplate=(
                f'<b>{legend_label}</b><br>' +
                'X: %{x:.0f} km<br>' +
                'Y: %{y:.0f} km<br>' +
                'Z: %{z:.0f} km<br>' +
                '<extra></extra>'
            )
        ))
        first = False

    print(f"  {legend_label}: plotted {len(lines_to_plot)} line segments")

# Add open field lines
print("\nPlotting open field lines...")
for cluster_key in open_clusters:
    lines = lines_by_cluster[cluster_key]
    if len(lines) == 0:
        continue

    cluster_id = int(cluster_key.split('_')[-1])
    n_lines = cluster_stats[cluster_key]
    legend_label = f"Open {cluster_id} (n={n_lines})"

    first = True

    if len(lines) > max_lines:
        lines_to_plot = random.sample(lines, max_lines)
    else:
        lines_to_plot = lines

    for traj in lines_to_plot:
        traj_s = smooth_traj(traj)

        fig.add_trace(go.Scatter3d(
            x=traj_s[:, 0],
            y=traj_s[:, 1],
            z=traj_s[:, 2],
            mode='lines',
            line=dict(color=colors_dict[cluster_key], width=2),
            name=legend_label,
            legendgroup=cluster_key,
            legendgrouptitle_text="Open" if cluster_key == open_clusters[0] else None,
            showlegend=first,
            hovertemplate=(
                f'<b>{legend_label}</b><br>' +
                'X: %{x:.0f} km<br>' +
                'Y: %{y:.0f} km<br>' +
                'Z: %{z:.0f} km<br>' +
                '<extra></extra>'
            )
        ))
        first = False

    print(f"  {legend_label}: plotted {len(lines_to_plot)} line segments")

# Update layout
fig.update_layout(
    template="plotly_white",
    width=1400,
    height=1000,
    scene=dict(
        xaxis=dict(
            title='X [km]',
            range=[-12 * RM, 4.5 * RM],
            backgroundcolor="rgb(230, 230, 230)",
            gridcolor="white",
            showbackground=True
        ),
        yaxis=dict(
            title='Y [km]',
            range=[-4.5 * RM, 4.5 * RM],
            backgroundcolor="rgb(230, 230, 230)",
            gridcolor="white",
            showbackground=True
        ),
        zaxis=dict(
            title='Z [km]',
            range=[-4.5 * RM, 4.5 * RM],
            backgroundcolor="rgb(230, 230, 230)",
            gridcolor="white",
            showbackground=True
        ),
        aspectmode='manual',
        aspectratio=dict(x=2.9, y=1.8, z=1.8),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2),
            center=dict(x=0, y=0, z=0)
        )
    ),
    legend=dict(
        groupclick="togglegroup",
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10),
        itemsizing='constant',
        x=1.02,
        y=0.5,
        xanchor='left',
        yanchor='middle',
        tracegroupgap=10
    ),
    title=dict(
        text=(
            f"<b>{case} Current Field Clustered Topology</b><br>"
            f"<sup>t = {step*0.002} s | {' | '.join(title_stats)}</sup>"
        ),
        x=0.5,
        xanchor='center',
        font=dict(size=16)
    ),
    margin=dict(l=0, r=200, t=100, b=0)
)

# Save outputs
out_html = f"{case}_{step}_J_openvclosed_clustered_topology_optimal.html"
out_png = out_html.replace(".html", ".png")

fig.write_html(os.path.join(output_folder, out_html), include_plotlyjs="cdn")
fig.write_image(os.path.join(output_folder, out_png), scale=2)

plottime = datetime.now()
print(f"\nSaved open vs closed figure at {str(plottime)}")
print(f"  HTML: {out_html}")
print(f"  PNG:  {out_png}")

# Create summary statistics file
summary_stats = []
for topo in ["closed", "open"]:
    if cluster_labels[topo] is None:
        continue

    labels = cluster_labels[topo]
    unique_labels = np.unique(labels[labels != -1])

    for cluster_id in unique_labels:
        cluster_key = f"{topo}_cluster_{cluster_id}"
        n_lines = cluster_stats[cluster_key]
        summary_stats.append({
            'topology': topo,
            'cluster_id': cluster_id,
            'n_field_lines': n_lines
        })

df_summary = pd.DataFrame(summary_stats)
summary_file = f"{case}_{step}_openvclosed_cluster_summary.csv"
df_summary.to_csv(os.path.join(output_folder, summary_file), index=False)
print(f"  Summary: {summary_file}")

# %%
# --------------------------
# CLUSTER ALL TRAJECTORIES TOGETHER WITH OPTIMAL PARAMETERS
# --------------------------
lines_by_cluster_all = {}

# Combine all trajectory pairs regardless of topology
all_trajectory_pairs = []
all_topology_labels = []  # Keep track of original topology for reference

for topo in ["closed", "open"]:
    for traj_pair in trajectory_pairs[topo]:
        all_trajectory_pairs.append(traj_pair)
        all_topology_labels.append(topo)

print(f"\n{'='*60}")
print(f"Clustering all {len(all_trajectory_pairs)} field lines together...")
print(f"{'='*60}")

if clustering_method == "hierarchical":
    all_labels, all_scores, all_validation_plot, all_results_df = cluster_trajectories_optimal(
        all_trajectory_pairs,
        method=clustering_method,
        max_k=max_clusters_to_test
    )

    # Save validation plots
    all_validation_plot.savefig(
        os.path.join(output_folder, f"{case}_{step}_all_validation_metrics.png"),
        dpi=150, bbox_inches='tight'
    )
    plt.close()

    # Save results table
    all_results_df.to_csv(
        os.path.join(output_folder, f"{case}_{step}_all_clustering_scores.csv"),
        index=False
    )

else:  # DBSCAN
    all_labels, all_scores, all_kdist_plot = cluster_trajectories_optimal(
        all_trajectory_pairs,
        method=clustering_method
    )

    # Save k-distance plot
    all_kdist_plot.savefig(
        os.path.join(output_folder, f"{case}_{step}_all_kdistance_plot.png"),
        dpi=150, bbox_inches='tight'
    )
    plt.close()

# Print results
print(f"\nOptimal clustering results:")
print(f"  Number of clusters: {all_scores['n_clusters']}")
if 'n_noise' in all_scores:
    n_noise = all_scores['n_noise']
    n_total = len(all_trajectory_pairs)
    pct_noise = 100 * n_noise / n_total
    print(f"  Noise points: {n_noise} ({pct_noise:.1f}%)")
print(f"  Silhouette Score: {all_scores['silhouette']:.3f}")
print(f"  Calinski-Harabasz Index: {all_scores['calinski_harabasz']:.1f}")
print(f"  Davies-Bouldin Index: {all_scores['davies_bouldin']:.3f}")

all_unique_labels = np.unique(all_labels)

# Analyze topology composition of each cluster
print(f"\nCluster composition:")
cluster_composition = []
for cluster_id in all_unique_labels:
    if cluster_id == -1:  # Skip noise for DBSCAN
        continue

    mask = all_labels == cluster_id
    cluster_topos = [all_topology_labels[i] for i in range(len(all_topology_labels)) if mask[i]]
    n_closed = cluster_topos.count("closed")
    n_open = cluster_topos.count("open")
    total = n_closed + n_open

    cluster_composition.append((cluster_id, total, n_closed, n_open))
    print(f"  Cluster {cluster_id}: {total} field lines "
          f"({n_closed} closed, {n_open} open)")

# Sort by total size for better organization
cluster_composition.sort(key=lambda x: x[1], reverse=True)

# Organize trajectories by cluster
for cluster_id, total, n_closed, n_open in cluster_composition:
    cluster_key = f"all_cluster_{cluster_id}"
    lines_by_cluster_all[cluster_key] = []

    mask = all_labels == cluster_id
    for idx, (traj_fwd, traj_bwd) in enumerate(all_trajectory_pairs):
        if mask[idx]:
            lines_by_cluster_all[cluster_key].append(traj_fwd)
            lines_by_cluster_all[cluster_key].append(traj_bwd)

clustertime = datetime.now()
print(f"\nCompleted all-trajectory clustering at {str(clustertime)}")

# Save cluster assignments with topology information
all_cluster_data = []
for idx, label in enumerate(all_labels):
    all_cluster_data.append({
        'trajectory_idx': idx,
        'cluster_id': label,
        'original_topology': all_topology_labels[idx]
    })

df_all = pd.DataFrame(all_cluster_data)
df_all.to_csv(os.path.join(output_folder, f"{case}_{step}_alltraj_cluster_assignments.csv"), index=False)

# %%
# --------------------------
# PLOT 3D FIELD LINES BY CLUSTER (ALL TRAJECTORIES)
# --------------------------
def generate_colors(n):
    """
    Generate visually distinct colors using tab10-style palette
    """
    import matplotlib.colors as mcolors

    # Tab10 colors (10 maximally distinct colors)
    tab10_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
    ]

    # If we need more than 10 colors, cycle through with slight variations
    colors = []
    for i in range(n):
        base_color = tab10_colors[i % 10]

        if i < 10:
            colors.append(base_color)
        else:
            # For additional colors, darken or lighten the base colors
            rgb = mcolors.hex2color(base_color)
            factor = 0.7 if (i // 10) % 2 == 0 else 1.3
            rgb_adjusted = tuple(min(1.0, max(0.0, c * factor)) for c in rgb)
            colors.append(f'rgb({int(rgb_adjusted[0]*255)},{int(rgb_adjusted[1]*255)},{int(rgb_adjusted[2]*255)})')

    return colors


# Sort clusters by ID for consistent ordering
sorted_all_clusters = sorted(lines_by_cluster_all.keys(),
                             key=lambda x: int(x.split('_')[-1]))

all_cluster_colors = generate_colors(len(sorted_all_clusters))
all_colors_dict = {key: color for key, color in zip(sorted_all_clusters, all_cluster_colors)}

# Calculate cluster statistics for legend labels
all_cluster_stats = {}
for cluster_key in sorted_all_clusters:
    cluster_id = int(cluster_key.split('_')[-1])
    mask = all_labels == cluster_id
    cluster_topos = [all_topology_labels[i] for i in range(len(all_topology_labels)) if mask[i]]
    n_closed = cluster_topos.count("closed")
    n_open = cluster_topos.count("open")
    n_total = n_closed + n_open
    all_cluster_stats[cluster_key] = {
        'total': n_total,
        'closed': n_closed,
        'open': n_open
    }

fig_all = go.Figure()

# Add planet sphere
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 200)
theta, phi = np.meshgrid(theta, phi)

xs = plot_depth * np.sin(theta) * np.cos(phi)
ys = plot_depth * np.sin(theta) * np.sin(phi)
zs = plot_depth * np.cos(theta)

eps = 0
mask_pos = xs >= -eps
mask_neg = xs <= eps

# Dayside hemisphere (light grey)
fig_all.add_trace(go.Surface(
    x=np.where(mask_pos, xs, np.nan),
    y=np.where(mask_pos, ys, np.nan),
    z=np.where(mask_pos, zs, np.nan),
    surfacecolor=np.ones_like(xs),
    colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
    cmin=0, cmax=1,
    showscale=False,
    lighting=dict(ambient=1, diffuse=0, specular=0),
    hoverinfo='skip',
    name='Mercury (dayside)'
))

# Nightside hemisphere (black)
fig_all.add_trace(go.Surface(
    x=np.where(mask_neg, xs, np.nan),
    y=np.where(mask_neg, ys, np.nan),
    z=np.where(mask_neg, zs, np.nan),
    surfacecolor=np.zeros_like(xs),
    colorscale=[[0, 'black'], [1, 'black']],
    cmin=0, cmax=1,
    showscale=False,
    lighting=dict(ambient=1, diffuse=0, specular=0),
    hoverinfo='skip',
    name='Mercury (nightside)'
))

# Add field lines by cluster
print("\nPlotting all-trajectory field lines...")
for cluster_key in sorted_all_clusters:
    lines = lines_by_cluster_all[cluster_key]
    if len(lines) == 0:
        continue

    stats = all_cluster_stats[cluster_key]
    cluster_id = int(cluster_key.split('_')[-1])

    # Create informative legend label
    legend_label = (f"Cluster {cluster_id} "
                   f"(n={stats['total']}: "
                   f"{stats['closed']}C/{stats['open']}O)")

    first = True

    # Downsample lines if necessary
    if len(lines) > max_lines:
        lines_to_plot = random.sample(lines, max_lines)
    else:
        lines_to_plot = lines

    for traj in lines_to_plot:
        traj_s = smooth_traj(traj)

        fig_all.add_trace(go.Scatter3d(
            x=traj_s[:, 0],
            y=traj_s[:, 1],
            z=traj_s[:, 2],
            mode='lines',
            line=dict(color=all_colors_dict[cluster_key], width=2),
            name=legend_label,
            legendgroup=cluster_key,
            showlegend=first,
            hovertemplate=(
                f'<b>{legend_label}</b><br>' +
                'X: %{x:.0f} km<br>' +
                'Y: %{y:.0f} km<br>' +
                'Z: %{z:.0f} km<br>' +
                '<extra></extra>'
            )
        ))
        first = False

    print(f"  {legend_label}: plotted {len(lines_to_plot)} line segments")

# Update layout with better styling
fig_all.update_layout(
    template="plotly_white",
    width=1400,
    height=1000,
    scene=dict(
        xaxis=dict(
            title='X [km]',
            range=[-12 * RM, 4.5 * RM],
            backgroundcolor="rgb(230, 230, 230)",
            gridcolor="white",
            showbackground=True
        ),
        yaxis=dict(
            title='Y [km]',
            range=[-4.5 * RM, 4.5 * RM],
            backgroundcolor="rgb(230, 230, 230)",
            gridcolor="white",
            showbackground=True
        ),
        zaxis=dict(
            title='Z [km]',
            range=[-4.5 * RM, 4.5 * RM],
            backgroundcolor="rgb(230, 230, 230)",
            gridcolor="white",
            showbackground=True
        ),
        aspectmode='manual',
        aspectratio=dict(x=2.9, y=1.8, z=1.8),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2),
            center=dict(x=0, y=0, z=0)
        )
    ),
    legend=dict(
        groupclick="togglegroup",
        title=dict(
            text="<b>Field Line Clusters</b><br>(C=Closed, O=Open)",
            font=dict(size=12)
        ),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10),
        itemsizing='constant',
        x=1.02,
        y=0.5,
        xanchor='left',
        yanchor='middle'
    ),
    title=dict(
        text=f"<b>{case} Current Field Clustered Topology (All Trajectories)</b><br>" +
             f"<sup>t = {step*0.002} s | {len(sorted_all_clusters)} clusters | " +
             f"Silhouette: {all_scores['silhouette']:.3f} | " +
             f"CH Index: {all_scores['calinski_harabasz']:.1f}</sup>",
        x=0.5,
        xanchor='center',
        font=dict(size=16)
    ),
    margin=dict(l=0, r=200, t=80, b=0)
)

# Save outputs
out_html_all = f"{case}_{step}_J_alltraj_clustered_topology_optimal.html"
out_png_all = out_html_all.replace(".html", ".png")

fig_all.write_html(os.path.join(output_folder, out_html_all), include_plotlyjs="cdn")
fig_all.write_image(os.path.join(output_folder, out_png_all), scale=2)

plottime_all = datetime.now()
print(f"\nSaved all trajectory figure at {str(plottime_all)}")
print(f"  HTML: {out_html_all}")
print(f"  PNG:  {out_png_all}")

# Create summary statistics file for all-trajectory clustering
summary_stats_all = []
for cluster_key in sorted_all_clusters:
    cluster_id = int(cluster_key.split('_')[-1])
    stats = all_cluster_stats[cluster_key]
    summary_stats_all.append({
        'cluster_id': cluster_id,
        'n_total': stats['total'],
        'n_closed': stats['closed'],
        'n_open': stats['open'],
        'pct_closed': 100 * stats['closed'] / stats['total'],
        'pct_open': 100 * stats['open'] / stats['total']
    })

df_summary_all = pd.DataFrame(summary_stats_all)
summary_file_all = f"{case}_{step}_alltraj_cluster_summary.csv"
df_summary_all.to_csv(os.path.join(output_folder, summary_file_all), index=False)
print(f"  Summary: {summary_file_all}")
