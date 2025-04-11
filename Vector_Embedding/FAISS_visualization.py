import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Create figure
fig = plt.figure(figsize=(18, 14))
fig.suptitle('FAISS: Inner Workings and Efficiency', fontsize=20, fontweight='bold')

# Define layout
gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1.2])

# 1. Original Vector Space
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.set_title('1. Original High-Dimensional Vectors')

# Generate random vectors
np.random.seed(42)
n_vectors = 200
dim = 3  # For visualization, real FAISS handles much higher dimensions
vectors = np.random.randn(n_vectors, dim)

# Plot vectors
ax1.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], c='blue', alpha=0.6)
ax1.set_xlabel('Dim 1')
ax1.set_ylabel('Dim 2')
ax1.set_zlabel('Dim 3')
ax1.view_init(30, 45)

# 2. Coarse Quantization (IVF)
ax2 = fig.add_subplot(gs[0, 1:])
ax2.set_title('2. Inverted File Index (IVF) Clustering')

# Simulate K-means clustering for IVF
n_clusters = 5
# Pick centroids from the data for simplicity
centroids_idx = np.random.choice(n_vectors, n_clusters, replace=False)
centroids = vectors[centroids_idx, :2]  # Use only 2 dimensions for visualization

# Assign vectors to clusters (simplified)
distances = np.array([[np.linalg.norm(v[:2] - c) for c in centroids] for v in vectors[:, :2]])
cluster_assignments = np.argmin(distances, axis=1)

# Colors for clusters
cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

# Plot clusters and vectors
for i in range(n_clusters):
    cluster_vectors = vectors[cluster_assignments == i, :2]
    ax2.scatter(cluster_vectors[:, 0], cluster_vectors[:, 1],
                color=cluster_colors[i], alpha=0.6, label=f'Cluster {i + 1}')
    ax2.scatter(centroids[i, 0], centroids[i, 1], color=cluster_colors[i],
                s=200, marker='*', edgecolor='black', linewidth=1.5)

    # Draw convex hull around cluster (simplified with a circle)
    circle = plt.Circle((centroids[i, 0], centroids[i, 1]),
                        np.max(np.linalg.norm(cluster_vectors - centroids[i], axis=1)) * 0.9,
                        color=cluster_colors[i], fill=False, linestyle='--', alpha=0.4)
    ax2.add_patch(circle)

ax2.legend(loc='upper right')
ax2.set_xlabel('Dimension 1')
ax2.set_ylabel('Dimension 2')

# 3. Product Quantization (PQ)
ax3 = fig.add_subplot(gs[1, 0:2])
ax3.set_title('3. Product Quantization (PQ)')

# Define parameters for PQ illustration
n_subquantizers = 4  # Number of sub-vectors
n_subcentroids = 8  # Centroids per sub-quantizer
segment_width = 0.8  # Width of each segment
padding = 0.1  # Padding between segments

# Create colormap for codebook values
cmap = LinearSegmentedColormap.from_list('custom_cmap',
                                         ['#E0F7FA', '#006064'],
                                         N=n_subcentroids)

# Draw original vector split into segments
for i in range(8):  # Show 8 example vectors
    y_pos = i * 2

    # Draw original vector label
    ax3.text(-3, y_pos, f"Vector {i + 1}", fontsize=10, ha='center', va='center')

    # Original vector box
    ax3.add_patch(
        patches.Rectangle((-1, y_pos - segment_width / 2),
                          4, segment_width, fill=True,
                          color='#E3F2FD', alpha=0.8,
                          edgecolor='black', linewidth=1)
    )

    # Vector splits into sub-vectors
    for j in range(n_subquantizers):
        x_pos = j
        # Draw sub-vector section
        ax3.add_patch(
            patches.Rectangle((x_pos, y_pos - segment_width / 2),
                              1 - padding, segment_width, fill=True,
                              color='#BBDEFB', edgecolor='black',
                              linewidth=1, alpha=0.8)
        )

    # Draw quantized representation (right side)
    ax3.text(5, y_pos, "→", fontsize=16, ha='center', va='center')

    # For each sub-vector, show its quantized index
    for j in range(n_subquantizers):
        subq_idx = np.random.randint(0, n_subcentroids)
        x_pos = j + 6

        # Draw quantized sub-vector representation
        ax3.add_patch(
            patches.Rectangle((x_pos, y_pos - segment_width / 2),
                              1 - padding, segment_width, fill=True,
                              color=cmap(subq_idx / n_subcentroids),
                              edgecolor='black', linewidth=1)
        )
        ax3.text(x_pos + 0.5 - padding / 2, y_pos, f"{subq_idx}",
                 fontsize=10, ha='center', va='center')

# Add a title for PQ codebook section
ax3.text(10.5, -1, "PQ Codebook", fontsize=12, ha='center', va='center', fontweight='bold')

# Show example of codebook (right side)
for i in range(n_subquantizers):
    for j in range(n_subcentroids):
        x_pos = i + 10
        y_pos = -2 - j * 0.75

        # Codebook entry box
        ax3.add_patch(
            patches.Rectangle((x_pos, y_pos - segment_width / 2),
                              1 - padding, segment_width / 1.5, fill=True,
                              color=cmap(j / n_subcentroids),
                              edgecolor='black', linewidth=1)
        )

        # Codebook index
        if j == 0:
            ax3.text(x_pos + 0.5 - padding / 2, y_pos - 0.7, f"Subspace {i + 1}",
                     fontsize=8, ha='center', va='center', rotation=0)

# Set axis limits and remove ticks
ax3.set_xlim(-4, 14)
ax3.set_ylim(-8, 16)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)

# 4. Approximate Distance Computation
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_title('4. Approximate Distance Calculation')

# Create distance table visual
table_size = 8
table_data = np.random.rand(table_size, table_size) * 4  # Random distances
distances = np.zeros((table_size, table_size))

# Calculate pairwise distances for some vectors (simplified)
for i in range(table_size):
    for j in range(table_size):
        subvec_i = i % 4  # subvector index
        code_i = i // 4  # code index in subvector
        subvec_j = j % 4  # subvector index
        code_j = j // 4  # code index in subvector
        if subvec_i == subvec_j:
            distances[i, j] = table_data[i, j]

# Create the heatmap for the distance tables
sns.heatmap(distances, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=.5, ax=ax4)
ax4.set_xlabel('Subquantizer Centroid Index')
ax4.set_ylabel('Subquantizer Centroid Index')
ax4.set_xticks(np.arange(0, table_size, 2) + 0.5)
ax4.set_xticklabels([f'SQ{i // 2 + 1}C{i % 2}' for i in range(0, table_size, 2)])
ax4.set_yticks(np.arange(0, table_size, 2) + 0.5)
ax4.set_yticklabels([f'SQ{i // 2 + 1}C{i % 2}' for i in range(0, table_size, 2)])

# 5. Bottom Panel - Performance Comparison and Architecture
ax5 = fig.add_subplot(gs[2, :])
ax5.set_title('5. FAISS Architecture & Performance Comparison')

# Define performance data
index_types = ['Flat', 'IVF Flat', 'IVF + PQ', 'HNSW', 'OPQ + IVFADC']
query_times = [100, 45, 8, 2.5, 1.5]  # Normalized query times
memory_usage = [100, 90, 20, 35, 15]  # Normalized memory usage
accuracy = [100, 95, 85, 92, 80]  # Approximate recall@1 percentage

# Create bar chart for comparison
bar_width = 0.25
x = np.arange(len(index_types))

ax5.bar(x - bar_width, query_times, bar_width, label='Query Time', color='#1976D2')
ax5.bar(x, memory_usage, bar_width, label='Memory Usage', color='#FFA000')
ax5.bar(x + bar_width, accuracy, bar_width, label='Accuracy', color='#388E3C')

# Add labels and legend
ax5.set_xticks(x)
ax5.set_xticklabels(index_types)
ax5.set_ylabel('Normalized Values')
ax5.legend(loc='upper right')

# Add architecture visualization on the same axis
# Calculate y-position for architecture diagram
y_arch_start = -40

# Architecture components
components = [
    {"name": "Query Vector", "x": 1, "y": y_arch_start + 40, "width": 2, "height": 10, "color": "#E3F2FD"},
    {"name": "IVF Index", "x": 5, "y": y_arch_start + 20, "width": 2, "height": 10, "color": "#BBDEFB"},
    {"name": "Coarse Quantizer\n(K-means)", "x": 5, "y": y_arch_start + 35, "width": 2, "height": 10,
     "color": "#90CAF9"},
    {"name": "PQ Codebooks", "x": 9, "y": y_arch_start + 20, "width": 2, "height": 10, "color": "#64B5F6"},
    {"name": "Distance Tables", "x": 9, "y": y_arch_start + 35, "width": 2, "height": 10, "color": "#42A5F5"},
    {"name": "Top-K Results", "x": 13, "y": y_arch_start + 40, "width": 2, "height": 10, "color": "#E1F5FE"}
]

# Draw architecture components
for component in components:
    rect = patches.Rectangle((component["x"], component["y"]),
                             component["width"], component["height"],
                             linewidth=1, edgecolor='black', facecolor=component["color"])
    ax5.add_patch(rect)
    ax5.text(component["x"] + component["width"] / 2, component["y"] + component["height"] / 2,
             component["name"], ha='center', va='center', fontsize=9)

# Add arrows between components
arrows = [
    {"start": (3, y_arch_start + 45), "end": (5, y_arch_start + 40)},
    {"start": (3, y_arch_start + 45), "end": (5, y_arch_start + 25)},
    {"start": (7, y_arch_start + 40), "end": (9, y_arch_start + 40)},
    {"start": (7, y_arch_start + 25), "end": (9, y_arch_start + 25)},
    {"start": (11, y_arch_start + 40), "end": (13, y_arch_start + 45)}
]

# Draw arrows
for arrow in arrows:
    ax5.annotate("", xy=arrow["end"], xytext=arrow["start"],
                 arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5))

# Set axis limits for architecture
ax5.set_xlim(0, 16)
ax5.set_ylim(y_arch_start, 105)

# Annotate the diagram
ax5.text(8, y_arch_start + 10, "FAISS Search Pipeline", fontsize=12, ha='center', fontweight='bold')

# Add explanatory text
description = """
The FAISS pipeline efficiently processes similarity searches by:
1. Clustering vectors into coarse partitions (IVF)
2. Quantizing vectors into compact codes (PQ) 
3. Using precomputed distance tables for fast approximate matching
4. Optimizing with SIMD instructions and GPU acceleration
"""
ax5.text(8, y_arch_start + 55, description, fontsize=10, ha='center',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.savefig('faiss_visualization.png', dpi=300, bbox_inches='tight')
plt.show()