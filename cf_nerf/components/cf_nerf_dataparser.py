from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Type, Union
import json
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
import imageio
from PIL import Image
from rich.prompt import Confirm
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs, Semantics
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.utils.io import load_from_json

MAX_AUTO_RESOLUTION = 1200


@dataclass
class CFNeRFDataParserConfig(DataParserConfig):
    """CF-NeRF dataset config"""
    _target: Type = field(default_factory=lambda: CFNeRFDataParser)
    """Directory specifying location of data."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    image_folder: str = "images"
    """Image folder name relative to `data` directory."""
    semantic_folder: Path = "semantics"
    """Semantic folder name relative to `data` directory."""
    semantic_folder_gt: str = ''
    """GT Semantic folder name relative to `data` directory (only for synthetic data)."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    train_split_fraction: float = 0.99
    """The fraction of images to use for training. The remaining images are for eval."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""



class CFNeRFDataParser(DataParser):
    """CF-NeRF DatasetParser."""

    config: CFNeRFDataParserConfig
    downscale_factor: Optional[int] = None

    def __init__(self, config: CFNeRFDataParserConfig):
        super().__init__(config)
        self.data: Path = config.data
        self.image_path: Path = config.data / config.image_folder
        self.semantic_path: Path = config.data / config.semantic_folder
        # self.semantic_gt_path: Path = config.data / config.semantic_folder_gt
        self.config = config
        self.scale_factor: float = config.scale_factor
        self.linear_probe = self.config.linear_probe
        self.linear_probe_gt_path = Path(self.config.linear_probe_gt_path) if config.linear_probe else None

    @abstractmethod
    def _generate_dataparser_outputs(self, split: str = "train", **kwargs) -> DataparserOutputs:
        """Abstract method that returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break

        image_filenames = []
        semantic_filenames = []
        semantic_gt_filenames = []
        poses = []
        distort = []

        for frame in meta["frames"]:
            # Image base name
            file_name = frame["file_path"].split("/")[-1]
            semantic_file_name = frame["semantic_path"].split("/")[-1]

            self.semantic_path = self.semantic_path.parent / frame["semantic_path"].split('/')[0]

            # Path to rgb and semantic image
            rgb_fname = self.image_path / file_name
            semantic_fname = self.semantic_path / semantic_file_name
            if self.linear_probe:
                gt_semantic_fname = self.linear_probe_gt_path / file_name

            semantic_prefix = self.semantic_path.stem

            rgb_fname = self._get_fname(rgb_fname.as_posix().replace('\\', '/'), data_dir)
            semantic_fname = self._get_fname(semantic_fname.as_posix().replace('\\', '/'),
                                             data_dir,
                                             downsample_folder_prefix="{}_".format(semantic_prefix)
                                             )

            image_filenames.append(rgb_fname)
            semantic_filenames.append(semantic_fname)
            if self.linear_probe:
                semantic_gt_filenames.append(gt_semantic_fname)

            #if self.config.semantic_folder_gt:
            #    semantic_gt_fname = self.config.semantic_folder_gt / file_name
            #    semantic_gt_filenames.append(semantic_gt_fname)

            poses.append(np.array(frame["transform_matrix"]))

            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

        assert len(semantic_filenames) == 0 or (
                len(semantic_filenames) == len(image_filenames)
        ), """
        Different number of image and semantic filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        num_train_images = math.ceil(num_images * self.config.train_split_fraction)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.asarray(poses).to(torch.float32)
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=self.config.center_method,
        )
        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor



        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        cx = float(meta["cx"])
        cy = float(meta["cy"])
        fl_x = float(meta["fl_x"])
        fl_y = float(meta["fl_y"])
        image_height= int(meta["h"])
        image_width = int(meta["w"])
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        image_filenames = [image_filenames[i] for i in indices]
        semantic_filenames = [semantic_filenames[i] for i in indices] if len(semantic_filenames) > 0 else []

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        classes = ['fruit', 'stuff']
        semantics = Semantics(filenames=semantic_filenames,
                              classes=classes,
                              colors=torch.asarray([0, 1]),
                              mask_classes=[])

        metadata = {'semantics': semantics}

        if self.linear_probe:
            semantics_gt = Semantics(filenames=semantic_gt_filenames,
                                     classes=classes,
                                     colors=torch.asarray([0, 1]),
                                     mask_classes=[])

            metadata.update({'semantics_gt': semantics_gt})

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=fl_x,
            fy=fl_y,
            cx=cx,
            cy=cy,
            height=image_height,
            width=image_width,
            distortion_params=distortion_params,
            camera_type=CameraType.PERSPECTIVE,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata=metadata,
        )

        return dataparser_outputs

    def _get_fname(self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if isinstance(filepath, str):
            filepath = Path(filepath)

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"{downsample_folder_prefix}{2 ** (df + 1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2 ** df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return data_dir / filepath
