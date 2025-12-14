from pathlib import Path
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hdbscan
import time
from sklearn.cluster import DBSCAN
import torch
import open3d as o3d
import random
from umap import UMAP
import plotly.express as px

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

instance_pc_path = Path(
    "/home/se86kimy/Dropbox/05_productive/01_code/15_ContrastiveFruitNeRF/CF-NeRF/debug/outputs/01_apple_tree_1024x1024_#300/cf-nerf/2024-05-31_132409/cf-nerf/instance_feature_vec.ply")

instance_pc = PyntCloud.from_file(instance_pc_path.__str__())
data_points = instance_pc.points

feature_list = data_points.columns.tolist()[3:]
feature_data_vectors_pandas = data_points[feature_list]
feature_data_vectors = np.asarray(feature_data_vectors_pandas)

xyz_list = data_points.columns.tolist()[:3]
xyz_data_vectors_pandas = data_points[xyz_list]
xyz_data_vectors = np.asarray(xyz_data_vectors_pandas)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

num_points = 150000
selected_indices = torch.randperm(feature_data_vectors.shape[0])[:num_points]

cluster_vectors = feature_data_vectors[selected_indices]
xyz_vectors = xyz_data_vectors[selected_indices]

# vector = np.hstack([xyz_vectors, cluster_vectors])

cluster_vectors_normalized = torch.nn.functional.normalize(torch.asarray(cluster_vectors), dim=1)
# cluster_vectors_normalized = xyz_vectors #np.hstack([xyz_vectors, cluster_vectors_normalized])
clustering_type = 'HDBSCAN'

t1 = time.time()
if clustering_type == 'DBSCAN':
    clustering = DBSCAN(eps=0.001,
                        min_samples=500,
                        metric="cosine").fit(cluster_vectors)
elif clustering_type == 'HDBSCAN':
    clustering = hdbscan.HDBSCAN(min_cluster_size=50,
                                 min_samples=5,
                                 prediction_data=False,
                                 allow_single_cluster=False).fit(cluster_vectors)
else:
    raise NotImplementedError

t2 = time.time()
print("Clustering took {} seconds".format(t2 - t1))
points = xyz_data_vectors[selected_indices]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

color = np.zeros_like(points)

# color_list = [[0,0,0], [1,0,0], [1,1,0], [0,0,1]]
num_detected_clusters = np.unique(clustering.labels_)
print(num_detected_clusters)

# Remove class -1. It gets assigned manually with black
color_l = (torch.randint(0, 256, size=(num_detected_clusters.shape[0] - 1, 3)) / 255).tolist()

color_list = [[0, 0, 0]]
color_list.extend(color_l)

for i in range(-1, color_l.__len__()):
    color[np.where(clustering.labels_ == i)] = color_list[i + 1]
    print(i)

pcd.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcd])

# Visualize Clustering
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
tsne_results = tsne.fit_transform(cluster_vectors_normalized)

df_subset = pd.DataFrame()
df_subset['x'] = tsne_results[:, 0]
df_subset['y'] = tsne_results[:, 1]
# df_subset['y'] = color

# proj_vec = torch.randn((cluster_vectors_normalized.shape[1], 3))
# projected = torch.einsum('pd,dc->pc', cluster_vectors_normalized, proj_vec)  # instance-vec @ projec-vec
# projected = projected - projected.min(dim=0, keepdim=True).values
# projected = projected / projected.max(dim=0, keepdim=True).values
# color_random = projected.double().cpu().numpy()


import matplotlib
import tikzplotlib

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#    "pgf.texsystem": "pdflatex",
#    'font.family': 'serif',
#    'text.usetex': True,
#    'pgf.rcfonts': False,
# })

fig = plt.figure(figsize=(16, 16))
sns.scatterplot(
    x="x", y="y",
    color=color,
    palette=sns.color_palette("hls", 25),
    data=df_subset,
    legend="full",
    alpha=0.5,
)
# plt.scatter(
#    x=tsne_results[:, 0][::30], y=tsne_results[:, 1][::30],
#    color=color[::30],
#    alpha=0.5,
# )
plt.axis('off')
ax = fig.gca()
ax.set_xticks(np.arange(-150, 150, 50))
ax.set_yticks(np.arange(-150, -150, 50))
plt.grid()
plt.savefig('t-SNE_apple.pgf')
tikzplotlib.clean_figure()
# tikzplotlib.save("t-SNE_apple.tex")
plt.show()

if False:
    df_subset = pd.DataFrame()
    df_subset['x'] = tsne_results[:, 0][::7]
    df_subset['y'] = tsne_results[:, 1][::7]
    # df_subset['y'] = color
    ax = fig.gca()
    fig = plt.figure(figsize=(12, 12))
    # Customize the plot appearance

    sns.scatterplot(
        x="x", y="y",
        color=color[::7],
        data=df_subset,
        legend="full",
        alpha=0.9,
    )

    ax.set_facecolor('lightsteelblue')  # Set background color to blue
    ax.grid(True, color='white')  # Set grid color to white
    plt.grid(True, color='white')  # Ensure grid color is white
    ax.tick_params(axis='both', which='both', length=0)  # Remove tick marks

    plt.savefig('t-SNE_apple.pdf')
    plt.show()

features = cluster_vectors_normalized

umap_2d = UMAP(n_components=2, init='random', random_state=0)
umap_3d = UMAP(n_components=3, init='random', random_state=0)

proj_2d = umap_2d.fit_transform(features)
proj_3d = umap_3d.fit_transform(features)

df_subset = pd.DataFrame()
df_subset['x'] = proj_2d[:, 0]
df_subset['y'] = proj_2d[:, 1]

plt.figure(figsize=(16, 10))
sns.scatterplot(
    x="x", y="y",
    color=color,
    palette=sns.color_palette("hls", 25),
    data=df_subset,
    legend="full",
    alpha=0.5,
)
plt.show()

fig_3d = px.scatter_3d(
    proj_3d, x=0, y=1, z=2,
    color=color, labels={'color': 'species'}
)
fig_3d.update_traces(marker_size=5)
fig_3d.show()

if False:
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(cluster_vectors_normalized)

    pcd_df_subset = pd.DataFrame()
    pcd_df_subset['pca_x'] = pca_result[:, 0]
    pcd_df_subset['pca_y'] = pca_result[:, 1]
    pcd_df_subset['pca_z'] = pca_result[:, 2]
    # df_subset['y'] = color

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca_x", y="pca_y",
        color=color_random,
        palette=sns.color_palette("hls", 10),
        data=pcd_df_subset,
        legend="full",
        alpha=0.5
    )
    plt.show()

    ax = plt.figure(figsize=(16, 10)).add_subplot(projection='3d')
    ax.scatter(
        xs=pcd_df_subset["pca_x"],
        ys=pcd_df_subset["pca_y"],
        zs=pcd_df_subset["pca_z"],
        c=color_random,
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()
