import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Type, Union

import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor

from nerfstudio.configs.base_config import InstantiateConfig
# from nerfstudio.data.pixel_samplers import PixelSamplerConfig
from nerfstudio.data.pixel_samplers import PixelSampler as PixelSamplerReference
from typing import Literal


@dataclass
class PixelSamplerConfig(InstantiateConfig):
    """Configuration for pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: PixelSampler)
    """Target class to instantiate."""
    num_rays_per_batch: int = 4096
    """Number of rays to sample per batch."""
    keep_full_image: bool = False
    """Whether or not to include a reference to the full image in returned batch."""
    is_equirectangular: bool = False
    """List of whether or not camera i is equirectangular."""
    ignore_mask: bool = False
    """Whether to ignore the masks when sampling."""
    fisheye_crop_radius: Optional[float] = None
    """Set to the radius (in pixels) for fisheye cameras."""
    rejection_sample_mask: bool = True
    """Whether or not to use rejection sampling when sampling images with masks"""
    max_num_iterations: int = 100
    """If rejection sampling masks, the maximum number of times to sample"""


class PixelSampler(PixelSamplerReference):
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: PixelSamplerConfig

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if
            key != "image_idx" and value is not None and "neighbour" not in key
        }
        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices
        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch


@dataclass
class SemanticPixelSamplerConfig(PixelSamplerConfig):
    """Configuration for pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: SemanticPixelSampler)
    """Target class to instantiate."""
    bg_to_instance_ratio: float = 0.85
    """Number of background pixels in comparison to instance pixels"""


#### Base Class
@dataclass
class ContrastivePixelSamplerConfig(PixelSamplerConfig):
    """Configuration for pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: ImageContrastivePixelSampler)
    """Target class to instantiate."""
    minimum_number_of_selected_indices: int = 256
    """Minimum number of selected instances to sample from. Must be a power of 2"""


@dataclass
class ImageContrastivePixelSamplerConfig(ContrastivePixelSamplerConfig):
    """Configuration for pixel sampler instantiation."""

    """Target class to instantiate."""
    _target: Type = field(default_factory=lambda: ImageContrastivePixelSampler)


@dataclass
class ClusterContrastivePixelSamplerConfig(ContrastivePixelSamplerConfig):
    """Configuration for pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: ClusterContrastivePixelSampler)
    """Target class to instantiate."""
    num_clusters: int = 8
    """ Number of clusters"""
    num_fruits_per_cluster: int = 4
    """Number of fruits per cluster"""

class SemanticPixelSampler:
    """
    Contrastive pixel sampler class for the entire image
    """

    config: SemanticPixelSamplerConfig

    def __init__(self, config: PixelSamplerConfig, **kwargs) -> None:
        self.kwargs = kwargs
        self.config = config
        self.config.num_rays_per_batch = self.kwargs.get("num_rays_per_batch", self.config.num_rays_per_batch)
        self.config.keep_full_image = self.kwargs.get("keep_full_image", self.config.keep_full_image)
        self.set_num_rays_per_batch(self.config.num_rays_per_batch)
        self.bg_to_instance_ratio = self.config.bg_to_instance_ratio

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        indices = self.sample_method_semantic(num_rays_per_batch,
                                              num_images,
                                              image_height,
                                              image_width,
                                              # rgb_images=batch["image"],
                                              semantic_images=batch['semantic'],
                                              # instance_images=batch['instances'],
                                              device=device)

        # Convert to image id, x and y position in image
        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()

        # Create batch
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if
            key != "image_idx" and value is not None and "neighbour" not in key
        }
        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices
        # if keep_full_image:
        if True:
            collated_batch["full_image"] = batch["image"]
            collated_batch["full_image_sem"] = batch["semantic"]

        return collated_batch

    def sample_method_semantic(
            self,
            batch_size: int,
            num_images: int,
            image_height: int,
            image_width: int,
            semantic_images: Optional[Tensor],
            device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        """
        Pixel sampler, samples across for image one instances and chooses positive (same instance) and negative
        (every other instance) pixel pairs

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
        """

        # Select random image indices
        num_selected_images = 16
        selected_image_index = torch.randint(0, num_images, size=(num_selected_images,))
        selected_images = semantic_images[selected_image_index][..., 0]

        # Select for every image the maximum value
        max_over_image_dim = selected_images.max(dim=1).values.max(dim=1).values

        bg_to_instance_ratio = self.bg_to_instance_ratio
        # If no semantic pixel is present in an image
        if int(max_over_image_dim.min()) == 0:
            # If one semantic img with no masks is present we choose to pick only background pixels for the entire batch
            bg_to_instance_ratio = 1

        pixels_per_bg_class = int((batch_size // num_selected_images) * bg_to_instance_ratio)
        pixels_per_fruit_class = (batch_size // num_selected_images) - pixels_per_bg_class

        # Create a boolean mask for all instances. True indicates the presence of the element
        bg_mask = selected_images == 0
        fruit_mask = selected_images == 1

        indices = torch.zeros(
            size=(num_selected_images, pixels_per_bg_class + pixels_per_fruit_class, 3), dtype=torch.long, device=device
        )

        # Iterate over alle images and select  background ind fruit pixel locations
        for idx, selected_image_idx in enumerate(selected_image_index):
            bg_pixel_indices = torch.nonzero(bg_mask[idx], as_tuple=False)
            fruit_pixel_indices = torch.nonzero(fruit_mask[idx], as_tuple=False)

            bg_ids = torch.randperm(bg_pixel_indices.shape[0])[:pixels_per_bg_class]
            fruit_ids = torch.randperm(fruit_pixel_indices.shape[0])[:pixels_per_fruit_class]

            indices[idx, :, 0] = selected_image_idx
            indices[idx, :pixels_per_bg_class, 1:] = bg_pixel_indices[bg_ids]
            indices[idx:, pixels_per_bg_class:, 1:] = fruit_pixel_indices[fruit_ids]

        return indices.view(batch_size, 3)

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a torch.Tensor")
        return pixel_batch


class ImageContrastivePixelSampler:
    """
    Contrastive pixel sampler class for the entire image
    """

    config: ImageContrastivePixelSamplerConfig

    def __init__(self, config: ContrastivePixelSamplerConfig, **kwargs) -> None:
        self.kwargs = kwargs
        self.config = config
        self.config.num_rays_per_batch = self.kwargs.get("num_rays_per_batch", self.config.num_rays_per_batch)
        self.config.keep_full_image = self.kwargs.get("keep_full_image", self.config.keep_full_image)
        self.set_num_rays_per_batch(self.config.num_rays_per_batch)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        indices = self.sample_method_stratified(num_rays_per_batch,
                                                num_images,
                                                image_height,
                                                image_width,
                                                instance_images=batch['instances'],
                                                device=device)
        indices_split = indices
        indices = indices.view(-1, 3)

        # Convert to image id, x and y position in image
        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()

        # Create batch
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if
            key != "image_idx" and value is not None and "neighbour" not in key
        }
        assert collated_batch["image"].shape[0] == num_rays_per_batch

        collated_batch["pixel_indices_split"] = indices_split

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices
        # if keep_full_image:
        if True:
            collated_batch["full_image"] = batch["image"]
            collated_batch["full_image_sem"] = batch["semantic"]
            collated_batch["full_image_instance"] = batch["instances"]

        return collated_batch

    def sample_method_stratified(
            self,
            batch_size: int,
            num_images: int,
            image_height: int,
            image_width: int,
            instance_images: Optional[Tensor],
            device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:

        selected_image_index = torch.randint(0, num_images, size=(1,))
        selected_images = instance_images[selected_image_index][0]

        # How many instances are present in the image
        unique_instances = torch.unique(selected_images)
        number_of_instances_available = unique_instances.shape[0]
        # Round to the next possible [2, 4, 8, 16, 32, 64]. If less than recommendation are present select nearest. Batch size must be 2 to the power of x
        num_instances = torch.randint(
            min(number_of_instances_available, self.config.minimum_number_of_selected_indices),
            number_of_instances_available + 1, size=(1,))
        num_instances = int(2 ** torch.floor(torch.log2(num_instances)))
        # Select random indices
        mask_ids = torch.randperm(number_of_instances_available)[:num_instances].cuda()
        if int((mask_ids == 0).sum()) == 0:
            mask_ids[0] = 0

        instances_available = unique_instances.view(1, 1, number_of_instances_available)

        number_of_pixels_per_instance = batch_size // num_instances
        # Init indices. Write same image id in tensor
        indices = torch.zeros(size=(num_instances, number_of_pixels_per_instance, 3), dtype=torch.long, device=device)
        indices[:, :, 0] = selected_image_index

        # Create a boolean mask for all instances. True indicates the presence of the element
        instance_mask = selected_images.view(selected_images.shape[0], selected_images.shape[1],
                                             1) == instances_available

        # Number of pixel for every instance
        num_pixel_per_instance = instance_mask.sum(dim=0).sum(dim=0)

        for idx, selected_instance in enumerate(mask_ids):
            pixel_indices = torch.nonzero(instance_mask[..., selected_instance], as_tuple=False)
            number_of_pixels_in_instance = pixel_indices.shape[0]
            selected_indecies = torch.randperm(number_of_pixels_in_instance)[:number_of_pixels_per_instance]

            if selected_indecies.shape[0] < indices.shape[1]:
                selected_indecies = selected_indecies.repeat(int(indices.shape[1] / selected_indecies.shape[0]) + 1)[
                                    :indices.shape[1]]

            indices[idx, :, 1:] = pixel_indices[selected_indecies]

        return indices

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a torch.Tensor")
        return pixel_batch


class ClusterContrastivePixelSampler:
    """
    Contrastive pixel sampler class for a patch.
    """

    config: ClusterContrastivePixelSamplerConfig

    def __init__(self, config: ContrastivePixelSamplerConfig, **kwargs) -> None:
        self.kwargs = kwargs
        self.config = config
        self.num_clusters = self.config.num_clusters
        self.num_fruits_per_cluster  = self.config.num_fruits_per_cluster
        self.config.num_rays_per_batch = self.kwargs.get("num_rays_per_batch", self.config.num_rays_per_batch)
        self.config.keep_full_image = self.kwargs.get("keep_full_image", self.config.keep_full_image)
        self.set_num_rays_per_batch(self.config.num_rays_per_batch)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        indices = self.sample_method_clustered_stratified(num_rays_per_batch,
                                                          num_images,
                                                          image_height,
                                                          image_width,
                                                          neighbour_matrix=batch["nearest_neighbour_matrix"],
                                                          instance_images=batch['instances'],
                                                          device=device)

        if isinstance(indices, type(None)):
            indices = self.sample_method_stratified(num_rays_per_batch,
                                                    num_images,
                                                    image_height,
                                                    image_width,
                                                    instance_images=batch['instances'],
                                                    device=device)

        indices_split = indices
        indices = indices.view(-1, 3)

        # Convert to image id, x and y position in image
        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()

        # Create batch
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if
            key != "image_idx" and value is not None and "neighbour" not in key
        }
        assert collated_batch["image"].shape[0] == num_rays_per_batch

        collated_batch["pixel_indices_split"] = indices_split

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices
        # if keep_full_image:
        if keep_full_image:
            collated_batch["full_image"] = batch["image"]
            collated_batch["full_image_sem"] = batch["semantic"]
            collated_batch["full_image_instance"] = batch["instances"]

        return collated_batch

    def sample_method_clustered_stratified(
            self,
            batch_size: int,
            num_images: int,
            image_height: int,
            image_width: int,
            neighbour_matrix: torch.Tensor,
            instance_images: Optional[Tensor],
            device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:

        debug = False

        sub_cluster_size = batch_size // self.num_clusters  # Number per Pixels per cluster
        sub_cluster_neighbours = self.num_fruits_per_cluster  # Number of neighbours (fruit) per selected instance (center fruit)
        sub_cluster_num_pixel_per_instance = sub_cluster_size // sub_cluster_neighbours  # Number of pixels per fruit

        selected_image_index = torch.randint(0, num_images, size=(1,))  # Select random image index
        selected_images = instance_images[selected_image_index][0]  # Select random image
        selected_neighbour_matrix = neighbour_matrix[
            int(selected_image_index)]  # Select corresponding kd_tree for image

        unique_instances = torch.unique(selected_images)  # Get all available instances in current image
        number_of_instances_available = unique_instances.shape[0]  # Get number of instance ids in image
        instances_available = unique_instances.view(1, 1, number_of_instances_available)

        # If to less pixels are available the function exists and a different pixel sampler is applied
        if unique_instances.shape[0] <= (sub_cluster_neighbours * self.num_clusters):
            # import matplotlib.pyplot as plt
            # plt.imshow(selected_images.cpu())
            # plt.show()
            return None

        # Get random permutation of instances present in image
        select_random_cluster_center_ids_all = torch.randperm(unique_instances.shape[0])
        select_random_cluster_center_ids_all = select_random_cluster_center_ids_all[
            select_random_cluster_center_ids_all.nonzero()].flatten()  # Remove 0 and flatten tensor

        # Select first 'self.num_clusters' of random selected cluster ids
        idx_select_random_cluster_center_ids_backup = select_random_cluster_center_ids_all[self.num_clusters:].to(
            selected_images.device)
        idx_select_random_cluster_center_ids = select_random_cluster_center_ids_all[:self.num_clusters].to(
            selected_images.device)

        # this tensor is needed as not all ids are set and might lead to an error
        instance_ids_arranged = torch.arange(0, instances_available.max() + 1, 1)
        # Create a mask of dim Width x Height x Num_Instances. Every channel holds information where a certain instance
        # id is present in current image
        instances_available_mask = selected_images.view(selected_images.shape[0],
                                                        selected_images.shape[1],
                                                        1) == instance_ids_arranged.to(instance_images.device)

        # Init selected indices for pixel sampler of shape (Num_Clusters, Num_Cluster_Neighbours, Pixel_per_fruit, (image_index, pixel_pos_x, pixel_pos_y)))
        indices = torch.zeros(size=(self.num_clusters, sub_cluster_neighbours, sub_cluster_num_pixel_per_instance, 3),
                              dtype=torch.long, device=device)
        indices[..., 0] = selected_image_index

        # Selected instance ids
        select_random_cluster_center_ids = instances_available[0, 0][idx_select_random_cluster_center_ids]
        select_random_cluster_center_ids_backup = instances_available[0, 0][idx_select_random_cluster_center_ids_backup]

        if debug:
            vis_mask = torch.zeros((selected_images.shape[0], selected_images.shape[1], 3))
            vis_mask[torch.nonzero(selected_images).cpu()[:, 0], torch.nonzero(selected_images).cpu()[:, 1], :] = 1

            viridis_color = [[0.98, 0.9, 0.14],
                             [0.63, 0.85, 0.22],
                             [0.2891, 0.7539, 0.4258],
                             [0.1211, 0.6289, 0.5273],
                             [0.1523, 0.4961, 0.5547],
                             [0.2109, 0.3594, 0.5508],
                             [0.2734, 0.1953, 0.4922],
                             [0.2656, 0.0039, 0.3281]]

        selected_instance_ids = []
        backup_counter = 0  # In the case an id has already been selected

        # Iterate over all selected instance ids
        for cluster_idx, cluster_center_id in enumerate(select_random_cluster_center_ids):
            # In the case the id has already been selected in previous rounds
            if cluster_center_id in selected_instance_ids:
                # If and only if the number of available indicis is too low another pixel sampler is selected.
                if select_random_cluster_center_ids_backup.shape[0] == 0 or backup_counter > \
                        select_random_cluster_center_ids_backup.shape[0]:
                    return None
                else:
                    cluster_center_id = int(select_random_cluster_center_ids_backup[backup_counter])
                    backup_counter += 1

            # Query kd tree for nearest (distance) sub_cluster_neighbours*8 (why this number?) fruits
            ind = [int(cluster_center_id)]
            ind.extend(selected_neighbour_matrix[cluster_center_id].cpu().long().numpy().tolist())

            # import matplotlib.pyplot as plt
            # plt.imshow(instances_available_mask[..., ind[0,:sub_cluster_neighbours]].sum(dim=-1).cpu().numpy())
            # plt.show()

            if False:
                import matplotlib.pyplot as plt
                selected_x[selected_x < 0] = 0
                plt.scatter(selected_x[:, 1].cpu(), -selected_x[:, 0].cpu())
                plt.scatter(center_position[1].cpu(), -center_position[0].cpu(), marker="x")
                plt.scatter(selected_x[ind, 1].cpu(), -selected_x[ind, 0].cpu(), marker="v")
                plt.show()

                import matplotlib.pyplot as plt
                plt.imshow(
                    instances_available_mask[..., selected_mapping[ind[0]].to(int)].sum(dim=-1).sum(
                        dim=-1).cpu().numpy())
                plt.show()

            # Select only selected fruit ids
            current_selected_ids = ind[:sub_cluster_neighbours]
            if selected_instance_ids.__len__() == 0:
                selected_instance_ids.extend(current_selected_ids)  # If no instances are assigned save in list
            else:
                current_selected_ids_set = set(ind[:sub_cluster_neighbours])
                already_selected_ids_set = set(selected_instance_ids)
                intersection = current_selected_ids_set.intersection(already_selected_ids_set)
                if intersection.__len__() != 0:
                    # Mask out already chosen fruit id
                    mask = ~torch.isin(torch.asarray(ind).T, torch.asarray(list(intersection)))
                    ind = torch.asarray(ind)[mask].tolist()  # current_selected_ids

                    # Set new selected fruit ids
                    current_selected_ids = ind[:sub_cluster_neighbours]

                    if len(current_selected_ids) < sub_cluster_neighbours:
                        raise ValueError("To less neighbors found!")
                        # return None should return None and go to backup strategy?

                selected_instance_ids.extend(current_selected_ids)

            # import matplotlib.pyplot as plt
            # plt.imshow(instances_available_mask[..., ind[0]].sum(dim=-1).cpu().numpy())
            # plt.show()

            if debug:
                mm = instances_available_mask[..., current_selected_ids].cpu().sum(dim=-1)

                if debug:
                    import matplotlib.pyplot as plt

                    plt.imshow(mm.numpy())
                    plt.show()

                c = viridis_color[cluster_idx]

                vis_mask[torch.nonzero(mm)[:, 0], torch.nonzero(mm)[:, 1], 0] = c[0]
                vis_mask[torch.nonzero(mm)[:, 0], torch.nonzero(mm)[:, 1], 1] = c[1]
                vis_mask[torch.nonzero(mm)[:, 0], torch.nonzero(mm)[:, 1], 2] = c[2]

            # Iterate over selected center and neighbouring clusters
            for neighbour_idx in range(sub_cluster_neighbours):
                # Select all pixels of fruit
                pixel_indices = torch.nonzero(instances_available_mask[..., int(ind[neighbour_idx])], as_tuple=False)
                selected_indecies = torch.randperm(pixel_indices.shape[0])[
                                    :sub_cluster_num_pixel_per_instance]  # Draw number of pixels

                # If number of pixels is lower than required -> repeat pixels
                if selected_indecies.shape[0] < sub_cluster_num_pixel_per_instance:
                    selected_indecies = selected_indecies.repeat(
                        int(sub_cluster_num_pixel_per_instance / selected_indecies.shape[0]) + 1)[
                                        :sub_cluster_num_pixel_per_instance]

                # Save pixel indices to tensor
                indices[cluster_idx, neighbour_idx, :, 1:] = pixel_indices[selected_indecies]

        if debug:
            import matplotlib.pyplot as plt
            img = plt.imshow(vis_mask, interpolation='nearest')
            img.set_cmap('hot')
            plt.axis('off')
            plt.show()
            # plt.savefig("./vis_mask_cluster.png", bbox_inches='tight')

        return indices.view(-1, sub_cluster_num_pixel_per_instance, 3)

    def sample_method_stratified(
            self,
            batch_size: int,
            num_images: int,
            image_height: int,
            image_width: int,
            instance_images: Optional[Tensor],
            device: Union[torch.device, str] = "cuda",
    ) -> Int[Tensor, "batch_size 3"]:

        selected_image_index = torch.randint(0, num_images, size=(1,))
        selected_images = instance_images[selected_image_index][0]

        # Choose new instance images until no invalid image is chosen (invalid is an image with zero instance pixels)
        while int(selected_images.max()) == 0:
            selected_image_index = torch.randint(0, num_images, size=(1,))
            selected_images = instance_images[selected_image_index][0]

        # How many instances are present in the image
        unique_instances = torch.unique(selected_images)
        number_of_instances_available = unique_instances.shape[0]
        # Round to the next possible [2, 4, 8, 16, 32, 64]. If less than recommendation are present select nearest. Batch size must be 2 to the power of x

        # number of instances in this image
        num_instances = torch.randint(
            min(number_of_instances_available, self.config.minimum_number_of_selected_indices),
            number_of_instances_available + 1, size=(1,))
        # round to the next lowes 2**n
        num_instances = int(2 ** torch.floor(torch.log2(num_instances)))
        # Select random indices
        mask_ids = torch.randperm(number_of_instances_available)[:num_instances].cuda()
        if int((mask_ids == 0).sum()) == 0:
            mask_ids[0] = 0

        instances_available = unique_instances.view(1, 1, number_of_instances_available)

        number_of_pixels_per_instance = batch_size // num_instances
        # Init indices. Write same image id in tensor
        indices = torch.zeros(size=(num_instances, number_of_pixels_per_instance, 3), dtype=torch.long, device=device)
        indices[:, :, 0] = selected_image_index

        # Create a boolean mask for all instances. True indicates the presence of the element
        instance_mask = selected_images.view(selected_images.shape[0], selected_images.shape[1],
                                             1) == instances_available

        # Number of pixel for every instance
        num_pixel_per_instance = instance_mask.sum(dim=0).sum(dim=0)

        for idx, selected_instance in enumerate(mask_ids):
            pixel_indices = torch.nonzero(instance_mask[..., selected_instance], as_tuple=False)
            number_of_pixels_in_instance = pixel_indices.shape[0]
            selected_indecies = torch.randperm(number_of_pixels_in_instance)[:number_of_pixels_per_instance]

            if selected_indecies.shape[0] < indices.shape[1]:
                selected_indecies = selected_indecies.repeat(int(indices.shape[1] / selected_indecies.shape[0]) + 1)[
                                    :indices.shape[1]]

            indices[idx, :, 1:] = pixel_indices[selected_indecies]

        # Check if correct clusters are selected: selected_images[indices[0].view(-1, 3)[:, 1:][:, 0], indices[0].view(-1, 3)[:, 1:][:, 1]]

        return indices

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a torch.Tensor")
        return pixel_batch
