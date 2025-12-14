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

"""
Export utils such as structs, point cloud generation, and rendering code.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
# import pymeshlab
from pathlib import Path
import torch
from jaxtyping import Float
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from pyntcloud import PyntCloud
import pandas as pd
import time
from sklearn.cluster import DBSCAN
import hdbscan


def sample_volume(
        pipeline: Pipeline,
        num_points: int,
        output_dir: pathlib.Path = None,
        config=None,
        transform_json: dict = None,
        instance_embedding: bool = False,
        instance_feature_dim: int = 12
) -> dict:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points_per_side: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.
        std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.
        output_dir: save pcds to output dir.

    Returns:
        Point cloud.
    """

    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )

    points_sem = []
    points_only_sem = []
    points_den = []
    points_sem_colormap = []
    color_semantics = []
    color_only_semantics = []
    color_semantics_colormap = []
    instance_embedding_vectors_semantic = []
    instance_embedding_vectors = []
    densities = []

    rgb_flag = True
    # sample_points_along_edge = num_points_per_side # num_points_per_side
    with progress as progress_bar:
        task = progress_bar.add_task("Generating Point Cloud", total=num_points)
        while not progress_bar.finished:
            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_sample_volume(0)
                outputs = pipeline.model(ray_bundle)

            # Sampled volume points
            sampled_point_position = outputs['point_location']
            points_3d = sampled_point_position.reshape((-1, 3))

            # Semantic & Density value
            semantic = outputs['semantics'].reshape((-1, 1)).repeat((1, 3))
            semantics_colormap = outputs['semantics_colormap'].reshape((-1, 1)).repeat((1, 3))
            density = outputs['density'].reshape((-1, 1)).repeat((1, 3)) #
            rgb = outputs['rgb'].reshape((-1, 3))

            if instance_embedding:
                instances = outputs['instance'].reshape((-1, instance_feature_dim))

            # Mask irrelevant semantic masks and density values
            mask_sem = semantic >= 3  # 20
            mask_den = density >= 70  # 10
            mask_sem_colormap = semantics_colormap >= 0.999

            mask_only_sem = semantics_colormap >= 0.99  # 9

            # Semantic colormap
            points_3d_semantic_colormap = points_3d[
                mask_sem_colormap.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            if rgb_flag:
                color_semantic_colormap = rgb[mask_sem_colormap.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            else:
                color_semantic_colormap = semantics_colormap[
                    mask_sem_colormap.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]

            color_semantic_colormap = torch.hstack([color_semantic_colormap, torch.sigmoid(
                semantic[mask_sem_colormap.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)][:, 0]).unsqueeze(-1)])
            points_sem_colormap.append(points_3d_semantic_colormap.cpu())
            color_semantics_colormap.append(color_semantic_colormap.cpu())

            # Semantic
            points_3d_semantic = points_3d[mask_sem.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            if rgb_flag:
                color_semantic = rgb[mask_sem.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            else:
                color_semantic = semantic[mask_sem.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
            color_semantic = torch.hstack([color_semantic, torch.sigmoid(
                semantic[mask_sem.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)][:, 0]).unsqueeze(-1)])
            points_sem.append(points_3d_semantic.cpu())  # & mask_den.sum(dim=1).to(bool)
            color_semantics.append(color_semantic.cpu())  # & mask_den.sum(dim=1).to(bool)

            # RGB
            points_3d_density = points_3d[mask_den.sum(dim=1).to(bool)]
            if rgb_flag:
                density_color = rgb[mask_den.sum(dim=1).to(bool)]
            else:
                density_color = density[mask_den.sum(dim=1).to(bool)]
            density_color = torch.hstack(
                [density_color, torch.sigmoid(density[mask_den.sum(dim=1).to(bool)][:, 0]).unsqueeze(-1)])

            # rgb_color = rgb[mask_den.sum(dim=1).to(bool)]
            points_den.append(points_3d_density.cpu())
            # densities.append(rgb_color.cpu())
            densities.append(density_color.cpu())

            # Instance vectors

            if instance_embedding:
                vector_instance = instances[mask_sem_colormap.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)]
                instance_embedding_vectors_semantic.append(vector_instance.cpu())

                vector_instance = instances[mask_den.sum(dim=1).to(bool)]
                instance_embedding_vectors.append(vector_instance.cpu())

            if False:
                # Semantic only
                points_3d_only_semantic_colormap = points_3d[mask_only_sem.sum(dim=1).to(bool)]

                if rgb_flag:
                    sem_color_only = rgb[mask_only_sem.sum(dim=1).to(bool)]
                else:
                    sem_color_only = semantic[mask_only_sem.sum(dim=1).to(bool)]

                # sem_color_only = torch.hstack(
                #    [sem_color_only, torch.sigmoid(
                #    semantic[mask_only_sem.sum(dim=1).to(bool)][:, 0]).unsqueeze(-1)])

                points_only_sem.append(points_3d_only_semantic_colormap.cpu())
                color_only_semantics.append(sem_color_only.cpu())

            torch.cuda.empty_cache()
            progress.advance(task, sampled_point_position.shape[0])

    pcd_list = {}

    def convert_to_pcd(points, color, normalize_color: bool = False):
        points_np = points.detach().double().cpu().numpy()
        color_np = color.detach().double().cpu().numpy()[:, :3]

        if color_np.shape[0] != 0 and normalize_color:
            color_np /= color_np.max()

        poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points_np.shape[0], axis=0)[:, :3, :]
        poses[:, :3, 3] = points_np

        poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
            torch.from_numpy(poses)
        )
        pcd_points_orig_space = poses[:, :3, 3].numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points_orig_space)
        pcd.colors = o3d.utility.Vector3dVector(color_np)

        return pcd

    def prepare_instance_point_cloud(instance_embedding_vec, points):
        # 3D points
        poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
        poses[:, :3, 3] = points

        poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
            torch.from_numpy(poses)
        )
        pcd_points_orig_space = poses[:, :3, 3].numpy()

        # Feature vec
        instance_vec = torch.cat(instance_embedding_vec, dim=0)
        instance_vec_normalized = torch.nn.functional.normalize(instance_vec, dim=1)

        # Random projection for debugging
        proj_vec = torch.randn((instance_vec_normalized.shape[1], 3))
        projected = torch.einsum('pd,dc->pc', instance_vec_normalized, proj_vec)  # instance-vec @ projec-vec
        projected = projected - projected.min(dim=0, keepdim=True).values
        projected = projected / projected.max(dim=0, keepdim=True).values
        vec = projected.double().cpu().numpy()

        pcd_instance_rand = o3d.geometry.PointCloud()
        pcd_instance_rand.points = o3d.utility.Vector3dVector(pcd_points_orig_space)
        pcd_instance_rand.colors = o3d.utility.Vector3dVector(vec)

        # Raw point cloud
        columns = ["x", "y", "z"]

        for i in range(instance_feature_dim):
            columns.append("f_{}".format(i))

        pcd_instance_raw = PyntCloud(
            pd.DataFrame(
                data=np.asarray(torch.hstack((torch.asarray(pcd_points_orig_space), instance_vec))),
                columns=columns))

        return pcd_instance_rand, pcd_instance_raw, proj_vec

    # Semantic colormap
    points_sem_colormap = torch.cat(points_sem_colormap, dim=0)
    color_sem_colormap = torch.cat(color_semantics_colormap, dim=0)

    pcd_sem_col = convert_to_pcd(points=points_sem_colormap, color=color_sem_colormap)
    pcd_list.update(
        {'semantic_colormap': {
            'pcd': pcd_sem_col,
            'path': str(output_dir / config.load_dir.parts[-3] / 'semantic_colormap.ply')
        }})

    # Semantic
    points_sem = torch.cat(points_sem, dim=0)
    color_semantic = torch.cat(color_semantics, dim=0)

    pcd_sem = convert_to_pcd(points_sem, color_semantic, normalize_color=True)
    pcd_list.update({'semantic': {
        'pcd': pcd_sem,
        'path': str(output_dir / config.load_dir.parts[-3] / 'semantic.ply')
    }})

    # Density
    points_den = torch.cat(points_den, dim=0)
    color_density = torch.cat(densities, dim=0)

    pcd_density = convert_to_pcd(points_den, color_density, normalize_color=True)
    pcd_list.update({'density': {
        'pcd': pcd_density,
        'path': str(output_dir / config.load_dir.parts[-3] / 'density.ply')
    }})

    if instance_embedding:

        # Instance point cloud with semantic points
        pcd_instance_rand, pcd_instance_raw, proj_vec = prepare_instance_point_cloud(
            instance_embedding_vec=instance_embedding_vectors_semantic, points=points_sem_colormap)

        pcd_list.update({'instance_feature_vec': {
            'pcd': pcd_instance_raw,
            'path': str(output_dir / config.load_dir.parts[-3] / 'instance_feature_vec.ply')
        }})

        pcd_list.update({'instance_random': {
            'pcd': pcd_instance_rand,
            'path': str(output_dir / config.load_dir.parts[-3] / 'instance_random.ply')
        }})

        if False:
            points_np = points_sem_colormap.detach().double().cpu().numpy()
            color_np = color_semantic.detach().double().cpu().numpy()[:, :3]

            if color_np.shape[0] != 0:
                color_np /= color_np.max()

            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points_np.shape[0], axis=0)[:, :3, :]
            poses[:, :3, 3] = points_np

            poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
                torch.from_numpy(poses)
            )
            pcd_points_orig_space = poses[:, :3, 3].numpy()

            pcd_sem_col_ = o3d.geometry.PointCloud()
            pcd_sem_col_.points = o3d.utility.Vector3dVector(pcd_points_orig_space)
            pcd_sem_col_.colors = o3d.utility.Vector3dVector(color_np)

            cl, ind = pcd_sem_col_.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)

            # Feature vec
            instance_vec = torch.cat(instance_embedding_vectors_semantic, dim=0)
            instance_vec_normalized = torch.nn.functional.normalize(instance_vec, dim=1)
            instance_vec_normalized_noise_removed = instance_vec_normalized[ind]
            points_instance = np.asarray(pcd_sem_col_.points)[ind]

            num_points = 150000  # 250000
            selected_indices = torch.randperm(instance_vec_normalized_noise_removed.shape[0])[:num_points]

            cluster_vectors = instance_vec_normalized_noise_removed[selected_indices]
            clustering_type = 'HDBSCAN'

            t1 = time.time()
            if clustering_type == 'DBSCAN':
                clustering = DBSCAN(eps=0.001,
                                    min_samples=500,
                                    metric="cosine").fit(cluster_vectors)
            elif clustering_type == 'HDBSCAN':
                clustering = hdbscan.HDBSCAN(min_cluster_size=50,
                                             min_samples=1,
                                             prediction_data=False,
                                             allow_single_cluster=False).fit(cluster_vectors)
            else:
                raise NotImplementedError

            t2 = time.time()
            print("Clustering took {} seconds".format(t2 - t1))
            points = points_instance[selected_indices]
            pcd_instance_clustered = o3d.geometry.PointCloud()
            pcd_instance_clustered.points = o3d.utility.Vector3dVector(points)

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
            pcd_instance_clustered.colors = o3d.utility.Vector3dVector(color)

            o3d.visualization.draw_geometries([pcd_instance_clustered])

            pcd_list.update({'instance_clustered': {
                'pcd': pcd_instance_clustered,
                'path': str(output_dir / config.load_dir.parts[-3] / 'instance_clustered.ply')
            }})

        # Instance point cloud with all density points
        instance_vec = torch.cat(instance_embedding_vectors, dim=0)
        instance_vec = torch.nn.functional.normalize(instance_vec, dim=1)

        # Just projection
        # proj_vec = torch.randn((instance_vec.shape[1], 3))
        projected = torch.einsum('pd,dc->pc', instance_vec, proj_vec)  # instance-vec @ projec-vec
        projected = projected - projected.min(dim=0, keepdim=True).values
        projected = projected / projected.max(dim=0, keepdim=True).values
        vec = projected.double().cpu().numpy()

        points_np = points_den.detach().double().cpu().numpy()

        poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points_np.shape[0], axis=0)[:, :3, :]
        poses[:, :3, 3] = points_np

        poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
            torch.from_numpy(poses)
        )
        pcd_points_orig_space = poses[:, :3, 3].numpy()

        pcd_instance_all = o3d.geometry.PointCloud()
        pcd_instance_all.points = o3d.utility.Vector3dVector(pcd_points_orig_space)
        pcd_instance_all.colors = o3d.utility.Vector3dVector(vec)

        pcd_list.update({'instance_all': {
            'pcd': pcd_instance_all,
            'path': str(output_dir / config.load_dir.parts[-3] / 'instance_all.ply')
        }})

    return pcd_list
