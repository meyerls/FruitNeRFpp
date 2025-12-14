"""
Semantic dataset.
"""
import copy
from typing import Dict
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import pairwise_distances

import torch
from pathlib import Path
import numpy as np
from PIL import Image
from typing import List, Tuple, Union
# import torchpairwise
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset


class CFNeRFDataset(InputDataset):
    """Dataset that returns images and semantics and masks.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["mask", "semantics"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

        assert "semantics" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["semantics"],
                                                                                Semantics), "No semantic istance could be found! Is a semantic folder included in the input folder and transform.json file?"
        self.semantics = self.metadata["semantics"]

        if "semantics_gt" in self.metadata.keys():
            self.semantics_gt = self.metadata["semantics_gt"]
            self.semantics_gt_available = True
        else:
            self.semantics_gt_available = False

    def get_metadata(self, data: Dict) -> Dict:
        # handle mask
        filepath = self.semantics.filenames[data["image_idx"]]

        # Load image
        pil_image = Image.open(filepath)

        # Convert to torch tensor
        instance_segmentation_image = torch.from_numpy(np.array(pil_image, dtype="int16"))

        if self.semantics_gt_available:
            filepath_gt = self.semantics_gt.filenames[data["image_idx"]]
            # Load image
            pil_image_gt = Image.open(filepath_gt)
            # Convert to torch tensor
            instance_segmentation_image_gt = torch.from_numpy(np.array(pil_image_gt, dtype="int16"))

        # Convert instance segmentation to semantic image
        segmentation_image = copy.deepcopy(instance_segmentation_image)
        segmentation_image[instance_segmentation_image > 0] = 1
        #segmentation_image = segmentation_image.to(torch.uint8)

        # Get unique ids and remove id 0 as it represents the background
        unique_instances = torch.unique(instance_segmentation_image)[1:]
        number_of_instances_available = unique_instances.shape[0]  # Get total number of unique ids
        instances_available = unique_instances.view(1, 1, number_of_instances_available)

        # Create a mask of dim Width x Height x Num_Instances. Every channel holds information where a certain instance
        # id is present.
        instances_available_mask = instance_segmentation_image.view(instance_segmentation_image.shape[0],
                                                                    instance_segmentation_image.shape[1],
                                                                    1) == instances_available

        # Max number of ID's visible in a image. ToDo: Increase number.
        max_number_of_instances_per_image = 1200
        max_number_of_neighbouring_instances = 50

        if unique_instances.shape[0] == 0:

            # nearest_neighbor_matrix set all to -1
            nearest_neighbor_matrix = torch.ones(
                (max_number_of_instances_per_image, max_number_of_neighbouring_instances), dtype=torch.long, device="cpu") * -1
        else:

            # Here we have to tackle the problem that some instance images have missing ids and thus the number of instances
            # has problems with indexing
            available_number_of_instances_in_image = unique_instances.shape[0]

            if unique_instances.max() > max_number_of_instances_per_image:
                raise ValueError("Too many instances per image. Please increase manually")

            center_position_fruits = torch.empty((max_number_of_instances_per_image, 2)).to(torch.int16)
            center_position_fruits[:, :] = -30000

            # nearest_neighbor_matrix has a larger size as collate functions needs fixed matrix size
            nearest_neighbor_matrix = torch.ones(
                (max_number_of_instances_per_image, max_number_of_neighbouring_instances), dtype=torch.long)

            # Iterate over all instances
            for idx, cluster_center_id in enumerate(unique_instances):
                # Select all pixels for one instance id and compute median location
                cluster_center_pixels = torch.where(instances_available_mask[..., idx])
                cluster_center_pixels_pos = [int(cluster_center_pixels[0].median()),
                                             int(cluster_center_pixels[1].median())]
                center_position_fruits[cluster_center_id, 0] = cluster_center_pixels_pos[0]
                center_position_fruits[cluster_center_id, 1] = cluster_center_pixels_pos[1]

            # Generate distance matrix
            # center_points = torch.asarray(center_position_fruits, device="cuda")
            # eucl_pdist = torch.nn.PairwiseDistance(p=2.0, eps=1e-06)
            # dist_matrix = eucl_pdist(center_points, center_points)

            eucl_dist = pairwise_distances(center_position_fruits, metric="euclidean", n_jobs=-1)
            eucl_dist = torch.asarray(eucl_dist, device="cpu")

            # Fill with inf on diaginal as self distance is 0
            eucl_dist.fill_diagonal_(torch.inf)

            # Sort
            sorted_indices = torch.argsort(eucl_dist, dim=1)[:, :max_number_of_neighbouring_instances]
            nearest_neighbor_matrix = sorted_indices

            #del(eucl_dist)

        dataset_params = {"semantic": segmentation_image,
                          "instances": instance_segmentation_image,
                          "nearest_neighbour_matrix": nearest_neighbor_matrix,
                          }

        if self.semantics_gt_available:
            dataset_params.update({"semantics_gt": instance_segmentation_image_gt})

        return dataset_params
