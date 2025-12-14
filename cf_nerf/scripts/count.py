# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !/usr/bin/env python
"""Processes a video or image sequence to a nerfstudio compatible dataset."""
from docutils.parsers.rst.directives.misc import Unicode
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import warnings
from typing import Union
import copy

import tyro
from typing_extensions import Annotated
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast
import glob
from nerfstudio.utils.rich_utils import CONSOLE
import open3d as o3d
import torch
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import time
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, MeanShift, HDBSCAN, OPTICS
import hdbscan
from hausdorff import hausdorff_distance
import wandb
import json
from sklearn.metrics.pairwise import pairwise_distances
import pickle
from sklearn.cluster import HDBSCAN as HDBSCAN_learn
import random

#torch.seed = 42
#np.random.seed(42)
#random.seed(42)


def pickle_clustering(data_clustering, median_cluster, file_path: Path):
    if data_clustering:
        with open(file_path / 'clustering.pkl', 'wb') as file:
            pickle.dump(data_clustering, file)

    with open(file_path / 'median_cluster.npy', 'wb') as file:
        np.save(file, median_cluster)


def unpickle_clustering(file_path: Path):
    with open(file_path / 'clustering.pkl', 'rb') as file:
        return pickle.load(file)


def visualize_cluster(input_xyz_vector, clustering):
    pcd_instance = o3d.geometry.PointCloud()
    pcd_instance.points = o3d.utility.Vector3dVector(input_xyz_vector)

    color = np.zeros_like(input_xyz_vector)

    # color_list = [[0,0,0], [1,0,0], [1,1,0], [0,0,1]]
    num_detected_clusters = np.unique(clustering.labels_)
    # print(num_detected_clusters)

    # Remove class -1. It gets assigned manually with black
    color_l = (torch.randint(0, 256, size=(num_detected_clusters.shape[0] - 1, 3)) / 255).tolist()

    color_list = [[0, 0, 0]]
    color_list.extend(color_l)

    cluster_list = []
    cluster_center_list = []

    cluster_vector = []

    for i in range(-1, color_l.__len__()):
        color[np.where(clustering.labels_ == i)] = color_list[i + 1]

    pcd_instance.colors = o3d.utility.Vector3dVector(color)

    o3d.visualization.draw_geometries([pcd_instance])

    return pcd_instance


def staged_cluster_point_cloud(input_vector,
                               input_xyz_vector,
                               cluster_type="manual_HDBSCAN",
                               lambda_e=1,
                               lambda_c=1,
                               transformation=None,
                               n_clusters=30,
                               cluster_file_path: Path = './'):
    input_vector_normalized = torch.nn.functional.normalize(torch.asarray(input_vector), dim=1)
    input_vector_normalized = np.asarray(input_vector_normalized)

    # global_clustering = HDBSCAN(min_cluster_size=10_000,
    #                             min_samples=1,
    #                             metric="euclidean",
    #                             allow_single_cluster=False).fit(input_xyz_vector.astype('float64'))

    # global_clustering = DBSCAN(eps=0.001,
    #                    min_samples=20,
    #                    metric="euclidean").fit(input_xyz_vector)
    # ToDo: Try spectral clustering
    global_clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(input_xyz_vector)

    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(input_xyz_vector)
    pcd_tmp = pcd_tmp.translate(transformation["t"])
    pcd_tmp = pcd_tmp.rotate(transformation["R"])

    input_xyz_vector = np.asarray(pcd_tmp.points)

    # global_clustering = DBSCAN(eps=0.1,
    #                           min_samples=200,
    #                           metric="euclidean").fit(input_xyz_vector)

    #visualize_cluster(input_xyz_vector, global_clustering)

    fruit_list = []
    fruit_center_list = []
    fruit_feature_list = []
    fruit_counter = 0

    # Iterate over all 50 clusters and cluster every one again to determine single fruits
    for global_cluster_idx in np.unique(global_clustering.labels_):
        global_selected_points_idx = np.where(global_clustering.labels_ == global_cluster_idx)
        current_feature_vector = input_vector_normalized[global_selected_points_idx]
        current_xyz_vector = input_xyz_vector[global_selected_points_idx]

        # Visualization
        if False:
            fruit_cluster_pcd = o3d.geometry.PointCloud()
            fruit_cluster_pcd.points = o3d.utility.Vector3dVector(current_xyz_vector)
            o3d.visualization.draw_geometries([fruit_cluster_pcd])

        cosine_dist = pairwise_distances(current_feature_vector, metric="cosine", n_jobs=-1)
        eucl_dist = pairwise_distances(current_xyz_vector, metric="euclidean", n_jobs=-1)

        # distance = cosine + 0.5*dist
        distance = lambda_c * cosine_dist + lambda_e * eucl_dist

        local_clustering = hdbscan.HDBSCAN(min_cluster_size=150,  # 150
                                           min_samples=150,
                                           allow_single_cluster=False,
                                           metric="precomputed",
                                           store_centers="both").fit(distance.astype('float64'))

        #visualize_cluster(current_xyz_vector, local_clustering)

        #print("Num fruits in sub cluster:", np.sum(np.unique(local_clustering.labels_) != -1))

        current_fruit_count = np.sum(np.unique(local_clustering.labels_) != -1)
        fruit_counter += current_fruit_count
        # print("Detected {} clusters".format(current_fruit_count))

        for local_cluster_idx in np.unique(local_clustering.labels_):
            # Skip unassigned points (-1)
            if local_cluster_idx == -1:
                continue

            # Select single fruit/cluster
            local_selected_points_idx = np.where(local_clustering.labels_ == local_cluster_idx)
            current_local_feature_vector = current_feature_vector[local_selected_points_idx]
            cluster_feature_median = np.median(current_local_feature_vector, axis=0)
            # cluster_feature_mean = np.mean(current_local_feature_vector, axis=0) # ToDo: whats best?

            current_local_xyz_vector = current_xyz_vector[local_selected_points_idx]
            cluster_xyz_median = np.median(current_local_xyz_vector, axis=0)

            # Visualization
            single_fruit_cluster_pcd = o3d.geometry.PointCloud()
            single_fruit_cluster_pcd.points = o3d.utility.Vector3dVector(current_local_xyz_vector)
            fruit_list.append(single_fruit_cluster_pcd)
            # print("Num points in cluster: ", np.asarray(single_fruit_cluster_pcd.points).shape[0])

            # if fruit_center_list.__len__() != 0:
            #    dist = np.sqrt(np.sum((np.vstack(fruit_center_list) - cluster_xyz_median)**2, axis=1))

            #    if dist[dist < 0.05].shape[0] != 0:
            #        print(dist)

            fruit_center_list.append(cluster_xyz_median)
            fruit_feature_list.append(cluster_feature_median)
            # o3d.visualization.draw_geometries([single_fruit_cluster_pcd])

    pcd_instance = o3d.geometry.PointCloud()

    color_l = (torch.randint(0, 256, size=(len(fruit_list) - 1, 3)) / 255).tolist()
    random.shuffle(color_l)

    cluster_vector = np.hstack([fruit_center_list, fruit_feature_list])

    for pcd, color in zip(fruit_list, color_l):
        pcd.paint_uniform_color(color)
        pcd_instance += pcd

    print("Total Count: ", fruit_counter)

    pickle_clustering(data_clustering=None,
                      median_cluster=np.vstack(cluster_vector),
                      file_path=cluster_file_path)

    return None, pcd_instance, None, fruit_center_list


def cluster_point_cloud(input_vector,
                        input_xyz_vector,
                        cluster_type="manual_HDBSCAN",
                        lambda_eucl_dist=0.2,
                        transformation=None,
                        cluster_file_path: Path = './'):
    input_vector_normalized = torch.nn.functional.normalize(torch.asarray(input_vector), dim=1)
    input_vector_normalized = np.asarray(input_vector_normalized)

    # centmean, centstd = input_vector_normalized.mean(dim=0), input_vector_normalized.std(dim=0)
    # outlier_mask = np.all(np.abs(input_vector_normalized - centmean[None]) < 3 * centstd, axis=1)
    # centers_filtered = features[outlier_mask]

    t1 = time.time()
    if cluster_type == 'DBSCAN':
        clustering = DBSCAN(eps=0.00001,
                            min_samples=50,
                            metric="cosine").fit(input_vector_normalized)
        #visualize_cluster(input_xyz_vector, clustering)

    elif cluster_type == 'HDBSCAN':
        clustering = HDBSCAN(min_cluster_size=50,
                             min_samples=1,
                             metric="euclidean",
                             allow_single_cluster=False).fit(input_vector_normalized.astype('float64'))
    elif cluster_type == 'OPTICS':
        cosine = pairwise_distances(input_vector_normalized, metric="cosine", n_jobs=-1)
        dist = pairwise_distances(input_xyz_vector, metric="euclidean", n_jobs=-1)

        # distance = cosine + 0.5*dist
        distance = cosine + 0.3 * dist

        clustering = OPTICS(min_samples=50, metric='precomputed').fit(distance)

    elif cluster_type == 'manual_HDBSCAN':

        if np.linalg.norm(input_vector_normalized, axis=1).mean() != 1.0:
            warnings.warn("Feature vectors are not normalized!. Mean normalization value: {}".format(
                np.linalg.norm(input_vector_normalized, axis=1).mean()))

        # cosine = 1 - np.dot(input_vector_normalized, input_vector_normalized.T)
        cosine_dist = pairwise_distances(input_vector_normalized, metric="cosine", n_jobs=-1)
        eucl_dist = pairwise_distances(input_xyz_vector, metric="euclidean", n_jobs=-1)

        # distance = cosine + 0.5*dist
        distance = cosine_dist + lambda_eucl_dist * eucl_dist

        clustering = hdbscan.HDBSCAN(min_cluster_size=50,
                                     min_samples=1,
                                     allow_single_cluster=False,
                                     metric="precomputed",
                                     store_centers="both").fit(distance.astype('float64'))
        #visualize_cluster(input_xyz_vector, clustering)

        if False:
            input_vector = np.hstack([input_xyz_vector, input_vector_normalized])

            def own_metric(X, Y, **kwargs):
                x_c = X[:3]
                y_c = Y[:3]
                f_x = X[3:]
                f_y = Y[3:]

                cosine = 1 - np.dot(f_x, f_y)
                dist = np.linalg.norm(x_c - y_c)
                mu = 0.3
                return cosine + mu * dist

            clustering = hdbscan.HDBSCAN(min_cluster_size=50,
                                         min_samples=1,
                                         core_dist_n_jobs=-1,
                                         prediction_data=True,
                                         # allow_single_cluster=False,
                                         metric=own_metric).fit(input_vector.astype('float64'))

    elif cluster_type == 'KMeans':
        clustering = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(input_vector_normalized)
    elif cluster_type == 'MeanShift':
        clustering = MeanShift(cluster_all=False, bin_seeding=True, min_bin_freq=10).fit(input_vector_normalized)
    elif cluster_type == 'SC':
        clustering = SpectralClustering(n_clusters=2,
                                        assign_labels='discretize',
                                        random_state=0).fit(input_vector_normalized)

    else:
        raise NotImplementedError

    t2 = time.time()
    print("Clustering took {} seconds".format(t2 - t1))

    pcd_instance = o3d.geometry.PointCloud()
    pcd_instance.points = o3d.utility.Vector3dVector(input_xyz_vector)

    color = np.zeros_like(input_xyz_vector)

    # color_list = [[0,0,0], [1,0,0], [1,1,0], [0,0,1]]
    num_detected_clusters = np.unique(clustering.labels_)
    print(num_detected_clusters)

    # Remove class -1. It gets assigned manually with black
    color_l = (torch.randint(0, 256, size=(num_detected_clusters.shape[0] - 1, 3)) / 255).tolist()
    random.shuffle(color_l)

    color_list = [[0, 0, 0]]
    color_list.extend(color_l)

    cluster_list = []
    cluster_center_list = []

    cluster_vector = []

    for i in range(-1, color_l.__len__()):
        color[np.where(clustering.labels_ == i)] = color_list[i + 1]

        if i == -1:
            continue
        else:
            cluster_xyz = input_xyz_vector[np.where(clustering.labels_ == i)]
            cluster_xyz_median = np.median(cluster_xyz, axis=0)

            cluster_feature = input_vector_normalized[np.where(clustering.labels_ == i)]
            cluster_feature_median = np.median(cluster_feature, axis=0)

            cluster_list.append(cluster_xyz)
            cluster_center_list.append(cluster_xyz_median)

            cluster_vector.append(np.hstack([cluster_xyz_median, cluster_feature_median]))

    pcd_instance.colors = o3d.utility.Vector3dVector(color)

    # Pickle clustering to visualize in instance viewer
    pickle_clustering(data_clustering=clustering,
                      median_cluster=np.vstack(cluster_vector),
                      file_path=cluster_file_path)

    return clustering, pcd_instance, cluster_list, cluster_center_list


def load_obj_file(filename):
    vertices = []
    faces = []

    clusters = []
    init_flag = True

    with open(filename, 'r') as obj_file:
        for line in obj_file:
            # Split the line into components
            parts = line.strip().split()

            if len(parts) > 0:
                # new object data starts with 'o'
                if parts[0] == 'o':
                    if init_flag:
                        init_flag = False
                    else:
                        clusters.append(vertices_cluster)
                    vertices_cluster = []
                # Vertex data starts with 'v'
                elif parts[0] == 'v':
                    vertex = list(map(float, parts[1:]))
                    vertices.append(vertex)
                    vertices_cluster.append(vertex)
                # Face data starts with 'f'
                elif parts[0] == 'f':
                    face = [int(index.split('/')[0]) - 1 for index in
                            parts[1:]]  # Subtract 1 because OBJ indices start from 1
                    faces.append(face)
        clusters.append(vertices_cluster)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    cluster_center = []
    cluster_pcd = []
    for c in clusters:
        cluster_center.append(np.asarray(c).mean(axis=0))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(c)
        cluster_pcd.append(pcd)

    return mesh, cluster_center, cluster_pcd


@dataclass
class Counter:
    load_pcd: Path
    """Path to the point cloud files."""
    output_dir: Path
    """Path to the output directory."""
    gt_pcd_file: Optional[Union[Path, str]] = None
    """Name of the gt fruit file."""
    lambda_eucl_dist: float = 1.2
    """euclidean term for distance metric."""
    lambda_cosine: float = 0.2
    """cosine term for distance metric."""
    distance_threshold: float = 0.05
    """Distance (non metric) to assign to gt fruit."""
    staged_max_points: int = 600_000
    """Maximum number of points for staged clustering"""
    clustering_variant: str = "staged"
    staged_num_clusters: int =30


@dataclass
class CountInstancePointCloud(Counter):
    """Count instance point cloud."""

    # def __init__(self):
    #    super(Counter).__init__()
    #    self.wandb_logger = wandb.init()

    def load_pcd_from_file(self, path, num_points=-1, denoise_nb_neighbors=20, denoise_std_ratio=1.5):
        instance_pc = PyntCloud.from_file(path.__str__())
        data_points = instance_pc.points

        # Spcial coordinates
        xyz_list = data_points.columns.tolist()[:3]
        xyz_data_vectors_pandas = data_points[xyz_list]
        xyz_data_vectors = np.asarray(xyz_data_vectors_pandas)

        # Feature space
        feature_list = data_points.columns.tolist()[3:]
        feature_data_vectors_pandas = data_points[feature_list]
        feature_data_vectors = np.asarray(feature_data_vectors_pandas)

        if False:
            # Masking
            semantics_cropped_path = Path(path).parent / "semantic_cropped.ply"
            if semantics_cropped_path.exists():
                semantics_cropped = o3d.io.read_point_cloud(semantics_cropped_path.__str__())

                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(xyz_data_vectors)

                dist = pcd1.compute_point_cloud_distance(semantics_cropped)
                dists = np.asarray(dist)
                ind = np.where(dists == 0)[0]

                xyz_data_vectors = xyz_data_vectors[ind]
                feature_data_vectors = feature_data_vectors[ind]

        # ToDo: Why did i hard code the transformation here in the first place?
        if 'tree_02' in path:
            # center: (-0.612588, 1.67182, 0.357728), extent: 3.38911, 2.44513, 2.51941)
            R = np.array([[-0.10712452, -0.99415193, 0.01364871],
                          [0.65568129, -0.06031995, 0.75262444],
                          [-0.74739975, 0.08957373, 0.65830856]])
            obb = o3d.geometry.OrientedBoundingBox(center=(-0.275354, 1.15213, 0.332916),
                                                   extent=(3.10039, 3.3985, 3.93047), R=R)
            valid_indices = obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(xyz_data_vectors))

            xyz_data_vectors = xyz_data_vectors[valid_indices]
            feature_data_vectors = feature_data_vectors[valid_indices]

        elif "fuji" in path.lower():
            line_set = o3d.io.read_line_set(self.gt_pcd_file.__str__())
            obb = line_set.get_oriented_bounding_box()
            obb.color = [0, 0, 1]
            obb.extent = obb.extent + 0.05
            self.distance_threshold = 0.15
            valid_indices = obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(xyz_data_vectors))

            xyz_data_vectors = xyz_data_vectors[valid_indices]
            feature_data_vectors = feature_data_vectors[valid_indices]
            # xyz_data_vectors_shrink = copy.deepcopy(xyz_data_vectors)
            # xyz_data_vectors_shrink_pcd = o3d.geometry.PointCloud()
            # xyz_data_vectors_shrink_pcd.points = o3d.utility.Vector3dVector(xyz_data_vectors_shrink)

            # obb_2 = copy.copy(obb)
            # rotate = obb.R.T
            # obb_2= obb_2.rotate(rotate)
            # line_set= line_set.rotate(rotate)
            # translate = -obb_2.get_center()
            # obb_2.translate(translate)
            # line_set.translate(translate)

            # T = np.asarray([[1, 0, 0],
            #                [0, 1, 0],
            #                [0, 0, 0.9]])
            # obb_2 = obb_2.rotate(T)
            # line_set = line_set.rotate(T)

            # line_set.translate(-translate)
            # line_set= line_set.rotate(rotate.T)
            # o3d.visualization.draw_geometries([line_set, o3d.geometry.TriangleMesh().create_coordinate_frame(), obb, obb_2, xyz_data_vectors_shrink_pcd])
        else:
            obb = o3d.geometry.OrientedBoundingBox()

        def transform_points(points, bounding_box):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            obb_2 = copy.copy(bounding_box)

            # Rotate
            rotate = bounding_box.R.T
            obb_2 = obb_2.rotate(rotate)
            pcd = pcd.rotate(rotate)

            # Translate
            translate = -obb_2.get_center()
            obb_2.translate(translate)
            pcd.translate(translate)

            # o3d.visualization.draw_geometries([o3d.geometry.TriangleMesh().create_coordinate_frame(), bounding_box, obb_2, pcd])

            # pcd= pcd.translate(-translate)
            # pcd= pcd.rotate(bounding_box.R)

            # o3d.visualization.draw_geometries([o3d.geometry.TriangleMesh().create_coordinate_frame(), bounding_box, obb_2, pcd])

            return np.asarray(pcd.points), {"t": -translate, "R": bounding_box.R}

        xyz_data_vectors, transformation = transform_points(points=xyz_data_vectors, bounding_box=obb)

        # Remove outliers
        pcd_noisy = o3d.geometry.PointCloud()
        pcd_noisy.points = o3d.utility.Vector3dVector(xyz_data_vectors)

        # obb.color = [0,0,1]
        # o3d.visualization.draw_geometries([line_set,pcd_noisy, obb])

        # o3d.io.write_point_cloud("./croppy.ply", pcd_noisy)
        # o3d.visualization.draw_geometries([obb, pcd_noisy])
        # o3d.visualization.draw_geometries([pcd_noisy])

        if denoise_nb_neighbors != -1:
            cl, ind = pcd_noisy.remove_statistical_outlier(nb_neighbors=denoise_nb_neighbors,
                                                           std_ratio=denoise_std_ratio)
            xyz_data_vectors = xyz_data_vectors[ind]
            feature_data_vectors = feature_data_vectors[ind]
            o3d.visualization.draw_geometries([cl])

        if num_points != -1:
            selected_indices = torch.randperm(feature_data_vectors.shape[0])[:num_points]

            cluster_vectors = feature_data_vectors[selected_indices]
            xyz_vectors = xyz_data_vectors[selected_indices]
        else:
            cluster_vectors = feature_data_vectors
            xyz_vectors = xyz_data_vectors

        return cluster_vectors, xyz_vectors, transformation

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        available_pcd_files = glob.glob((self.load_pcd / "*.ply").__str__())
        instance_pcd_path = None

        for pcd_path in available_pcd_files:
            if "instance_feature_vec.ply" in pcd_path:
                instance_pcd_path = pcd_path
                break

        variant = self.clustering_variant
        if variant == 'single':
            self.num_points = 42000
        elif variant == 'staged':
            self.num_points = self.staged_max_points  # 500000

        print('$\lambda_e$: {}, $\lambda_c$: {}'.format(self.lambda_eucl_dist, self.lambda_cosine))

        cluster_vectors, xyz_vectors, transformation = self.load_pcd_from_file(path=instance_pcd_path,
                                                                               num_points=self.num_points,
                                                                               denoise_nb_neighbors=50,
                                                                               denoise_std_ratio=0.5)

        if variant == 'single':
            clustering_result, pcd_instance, cluster_list, cluster_center_list = cluster_point_cloud(
                input_vector=cluster_vectors,
                input_xyz_vector=xyz_vectors,
                cluster_file_path=self.load_pcd,
                lambda_eucl_dist=self.lambda_eucl_dist,
                transformation=transformation)
        elif variant == 'staged':
            clustering_result, pcd_instance, cluster_list, cluster_center_list = staged_cluster_point_cloud(
                input_vector=cluster_vectors,
                input_xyz_vector=xyz_vectors,
                cluster_file_path=self.load_pcd,
                lambda_e=self.lambda_eucl_dist,
                lambda_c=self.lambda_cosine,
                n_clusters=self.staged_num_clusters,
                transformation=transformation)

        # o3d.visualization.draw_geometries([pcd_instance])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_vectors)

        # Start evaluation
        if self.gt_pcd_file:
            result_json = self.output_dir.joinpath('count_result.json')

            if "obj" in self.gt_pcd_file.__str__():
                mesh, mesh_cluster_center, cluster_pcd = load_obj_file(self.gt_pcd_file)
            else:
                line_set = o3d.io.read_line_set(self.gt_pcd_file.__str__())
                line_points = np.asarray(line_set.points)
                line_boxes = line_points.reshape((line_points.shape[0] // 8, 8, 3))
                mesh_cluster_center = line_boxes.mean(axis=1)

                cluster_pcd = []
                for idx, box in enumerate(line_boxes):
                    lineset_bb = o3d.geometry.OrientedBoundingBox()
                    lineset_bb = lineset_bb.create_from_points(o3d.utility.Vector3dVector(line_boxes[idx]))
                    lineset_bb.color = (1, 1, 0)
                    cluster_pcd.append(lineset_bb)

            estimated_center = np.vstack(cluster_center_list)
            gt_center = np.vstack(mesh_cluster_center)

            hausdorff_dist_result = hausdorff_distance(XA=gt_center, XB=estimated_center, distance='euclidean')

            print(
                f"Hausdorff distance test: {hausdorff_dist_result}")

            gt_center_copy = gt_center.copy()

            distance_list = []
            number_of_correctly_counted_objects = 0
            false_positive = 0
            for curr_cluster_center in cluster_center_list:

                if gt_center_copy.__len__() == 0:
                    false_positive += 1
                    continue

                distance2centers = np.linalg.norm(gt_center_copy - curr_cluster_center, axis=1)
                nearest_center_id = np.argmin(distance2centers)
                distance2nearest_center = distance2centers[nearest_center_id]

                if distance2nearest_center < self.distance_threshold:  # FUJI: 0.15 Diameter (not metric!) of an apple # 0.05 for synthetic
                    distance_list.append(distance2nearest_center)
                    gt_center_copy = np.delete(gt_center_copy, nearest_center_id, axis=0)
                    cluster_pcd.pop(nearest_center_id)
                    number_of_correctly_counted_objects += 1
                else:
                    print("Distance of false positive: {}".format(distance2nearest_center))
                    false_positive += 1

            false_negative = gt_center_copy.shape[0]  # Remaining unassigned centers
            true_positive = number_of_correctly_counted_objects + 1e-12

            number_of_gt_objects = gt_center.shape[0]
            Precision = true_positive / (true_positive + false_positive)
            Recall = true_positive / (true_positive + false_negative)
            F1 = 2 * Precision * Recall / (Precision + Recall)

            print("Counting result: {}/{}".format(cluster_center_list.__len__(), number_of_gt_objects))
            print("True positive: {}".format(int(true_positive)))
            print("False positive: {}".format(false_positive))
            print("False negative: {}".format(false_negative))
            print("Precision: {}".format(Precision))
            print("Recall: {}".format(Recall))
            print("F1: {}".format(F1))

            results_dict = {
                'gt_apples': number_of_gt_objects,
                'counted_apples': cluster_center_list.__len__(),
                'hausdorff_distance': hausdorff_dist_result,
                'F1': F1,
                'Recall': Recall,
                'Precision': Precision,
                'TP': true_positive,
                'FP': false_positive,
                'FN': false_negative
            }

            with open(result_json, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=4)

            CONSOLE.print(f"[bold green]:white_check_mark:  Saved count results. Path: %s" % result_json)

            all_pcd = []
            all_pcd.extend(cluster_pcd)
            all_pcd.append(pcd_instance)

            # o3d.visualization.draw_geometries(all_pcd)


        else:
            print("Visualizing clustered point cloud result")
            o3d.visualization.draw_geometries([pcd_instance])

        o3d.io.write_point_cloud(self.output_dir.joinpath('instance_clustered.ply').__str__(), pcd_instance)


Commands = Union[
    Annotated[CountInstancePointCloud, tyro.conf.subcommand(name="fruits")],
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # type: ignore
