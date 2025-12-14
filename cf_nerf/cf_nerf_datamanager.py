"""
Template DataManager
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
)

import torch
from typing_extensions import TypeVar

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion

from cf_nerf.components.cf_nerf_dataparser import CFNeRFDataParserConfig
from cf_nerf.components.cf_nerf_dataset import CFNeRFDataset
from cf_nerf.components.pixel_samplers import ImageContrastivePixelSampler, ContrastivePixelSamplerConfig, \
    ImageContrastivePixelSamplerConfig, SemanticPixelSamplerConfig, SemanticPixelSampler, PixelSampler
# from nerfstudio.data.pixel_samplers import PixelSampler
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.utils.dataloaders import CacheDataloader, FixedIndicesEvalDataloader, RandIndicesEvalDataloader
from nerfstudio.utils.rich_utils import CONSOLE
from cf_nerf.components.pixel_samplers import PixelSamplerConfig
from cf_nerf.components.ray_generators import OrthographicRayGenerator
# from nerfstudio.data.pixel_samplers import PatchPixelSamplerConfig, PixelSamplerConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from cf_nerf.components.cf_nerf_collate import cf_nerf_collate

TDataset = TypeVar("TDataset", bound=CFNeRFDataset, default=CFNeRFDataset)


def get_corners_of_aabb(aabb, device):
    """
    Get the 3D locations of the corners of the Axis-Aligned Bounding Box (AABB).

    Parameters:
        aabb (numpy array): AABB representing the bounding box with shape (2, 3),
                            where the first row contains the minimum coordinates (x, y, z) and
                            the second row contains the maximum coordinates (x, y, z).

    Returns:
        numpy array: Array of shape (8, 3) containing the 3D coordinates of the corners.
    """
    min_coords = aabb[0]  # Minimum coordinates (x, y, z)
    max_coords = aabb[1]  # Maximum coordinates (x, y, z)

    corners = torch.asarray([
        [min_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], min_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]],
        [max_coords[0], max_coords[1], max_coords[2]]
    ], device=device)

    return corners


def sample_surface_points(aabb, n, device, noise=False):
    """
    Sample points on a single surface of the Axis-Aligned Bounding Box (AABB).

    Parameters:
        aabb (numpy array): AABB representing the bounding box with shape (2, 3).
        n (int): Number of points to sample along each axis (total points = n*n).
        device: Device Type.

    Returns:
        torch tensor: Tensor of shape (num_points, 3) containing the sampled 3D coordinates.
    """
    # select three corners (must be adjacent!)
    corner_1 = aabb[0]  # x
    corner_2 = aabb[1]  # y
    corner_3 = aabb[2]  # z

    # Check if elements are to far away (check if adjacent)
    # assert torch.abs(torch.sum(corner_1 - corner_2)) == 2.0
    # assert torch.abs(torch.sum(corner_1 - corner_3)) == 2.0

    dx_y_z = torch.abs(torch.max(aabb, axis=0).values - torch.min(aabb, axis=0).values)

    # Part where the coordinate does not change
    constant_axis_part_pos = int(
        torch.argmax(torch.logical_and((corner_1 == corner_2), (corner_2 == corner_3)).to(int)))

    # Generate meshgrid along XY plane
    start_x_pos = torch.argmax(torch.abs(corner_1 - corner_2))
    x = torch.linspace(corner_1[start_x_pos], corner_2[start_x_pos],
                       int(dx_y_z[0] / dx_y_z[constant_axis_part_pos] * n), dtype=torch.float32, device=device)
    start_y_pos = torch.argmax(torch.abs(corner_1 - corner_3))
    y = torch.linspace(corner_1[start_y_pos], corner_3[start_y_pos],
                       int(dx_y_z[1] / dx_y_z[constant_axis_part_pos] * n), dtype=torch.float32, device=device)

    xx, yy = torch.meshgrid(x, y)

    # Flatten the meshgrid and set Z coordinate to the minimum Z value of the AABB
    surface_points = torch.column_stack(
        (xx.flatten(),
         yy.flatten(),
         torch.full_like(xx.flatten(), corner_3[constant_axis_part_pos])))

    # Convert to torch tensor
    surface_points_tensor = surface_points.clone()

    corner_4 = aabb[-1]
    plane_vector = torch.asarray([[0, 0, torch.sign(corner_4[constant_axis_part_pos]) * torch.abs(
        corner_1[constant_axis_part_pos]) + torch.abs(corner_4[constant_axis_part_pos])]], dtype=torch.float32,
                                 device=device)

    return surface_points_tensor, plane_vector


@dataclass
class CFNeRFDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: CFNeRFDataManager)
    dataparser = CFNeRFDataParserConfig()
    eval_pixel_sampler: PixelSamplerConfig = field(default_factory=PixelSamplerConfig)
    rgb_pixel_sampler: PixelSamplerConfig = field(default_factory=PixelSamplerConfig)
    # semantic_pixel_sampler: SemanticPixelSamplerConfig = field(
    #    default_factory=SemanticPixelSamplerConfig)
    semantic_pixel_sampler: PixelSamplerConfig = field(
        default_factory=PixelSamplerConfig)
    instance_pixel_sampler: ContrastivePixelSamplerConfig = field(
        default_factory=ImageContrastivePixelSamplerConfig)
    collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(cf_nerf_collate))
    """Specifies the collate function to use for the train and eval dataloaders."""


class CFNeRFDataManager(VanillaDataManager):
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config = CFNeRFDataManagerConfig# AnnotatedDataParserUnion = field(default_factory=CFNeRFDataManagerConfig)
    train_dataset: TDataset
    eval_dataset: TDataset

    train_instance_pixel_sampler: Optional[SemanticPixelSampler] = None
    train_semantic_pixel_sampler: Optional[ImageContrastivePixelSampler] = None
    train_rgb_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
            self,
            config: CFNeRFDataManagerConfig,
            device: Union[torch.device, str] = "cuda",
            test_mode: Literal["test", "val", "inference"] = "val",
            world_size: int = 1,
            local_rank: int = 0,
            **kwargs,  # pylint: disable=unused-argument
    ):

        config.dataparser = CFNeRFDataParserConfig()
        config.dataparser.linear_probe = kwargs['linear_probe']
        config.dataparser.linear_probe_gt_path = kwargs['linear_probe_gt_path']
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_rgb_pixel_sampler = self.config.rgb_pixel_sampler.setup(
            num_rays_per_batch=self.config.train_num_rays_per_batch)
        self.train_semantic_pixel_sampler = self.config.semantic_pixel_sampler.setup(
            num_rays_per_batch=self.config.train_num_rays_per_batch)
        self.train_instance_pixel_sampler = self.config.instance_pixel_sampler.setup(
            num_rays_per_batch=self.config.train_num_rays_per_batch)
        self.eval_pixel_sampler = self.config.eval_pixel_sampler.setup(
            num_rays_per_batch=self.config.train_num_rays_per_batch)

        # Maybe modify ray generator
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

    def setup_inference(self, aabb, num_points):
        num_points_per_edge = num_points
        corners = get_corners_of_aabb(aabb=aabb, device=self.device)
        surface_points, plane_vector = sample_surface_points(corners,
                                                             n=num_points_per_edge,
                                                             device=self.device,
                                                             noise=False)
        self.orthographic_ray_generator = OrthographicRayGenerator(surface_points=surface_points,
                                                                   plane_normal=plane_vector,
                                                                   ray_batch_size=self.config.eval_num_rays_per_batch,
                                                                   device=self.device,
                                                                   aabb=aabb)

        num_rays = surface_points.shape[0]

        return num_rays

    def next_sample_volume(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""

        self.train_count += 1
        ray_bundle = self.orthographic_ray_generator(count=self.train_count)
        return ray_bundle, None

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return CFNeRFDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return CFNeRFDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def next_train(self, step: int, training_steps) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)

        if image_batch['image'].shape[-1] == 4:
            image_batch['image'] = image_batch['image'][..., :3]

        if image_batch['semantic'].dim() == 3:
            image_batch['semantic'] = image_batch['semantic'].unsqueeze(3).float()

        assert self.train_rgb_pixel_sampler is not None
        assert self.train_semantic_pixel_sampler is not None
        assert self.train_instance_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        if step < training_steps['semantic']:
            batch = self.train_rgb_pixel_sampler.sample(image_batch)
        elif step < training_steps['instance']:
            batch = self.train_semantic_pixel_sampler.sample(image_batch)
        else:
            batch = self.train_instance_pixel_sampler.sample(image_batch)

        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)

        assert self.train_rgb_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_rgb_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch
