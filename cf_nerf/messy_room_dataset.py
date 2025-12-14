from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
import json
import numpy as np
import pyquaternion as pyquat
from PIL import Image
import glob

from nerfstudio.process_data import colmap_utils, hloc_utils, process_data_utils
from nerfstudio.process_data.base_converter_to_nerfstudio_dataset import BaseConverterToNerfstudioDataset
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils import install_checks
from nerfstudio.utils.rich_utils import CONSOLE






import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()





def read_cameras(meta, H, W):
    '''
    Code from https://github.com/yashbhalgat/Contrastive-Lift/blob/main/dataset/many_object_scenes.py
    :param meta:
    :param H:
    :param W:
    :return:
    '''
    K = np.array(meta["camera"]["K"])  # 3x3
    K[0] *= W  # multiplying first row by W
    K[1] *= H  # multiplying second row by H
    K = np.abs(K)  # a bit hacky... :/

    poses = []
    for i in range(len(meta["camera"]["positions"])):
        pose = np.eye(4)
        t = np.array(meta["camera"]["positions"][i])
        q = np.array(meta["camera"]["quaternions"][i])
        rot = pyquat.Quaternion(*q).rotation_matrix
        pose[:3, :3] = rot
        pose[:3, 3] = t
        # # we may need to convert blender convention to opencv convention
        #blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #pose = pose @ blender2opencv
        poses.append(pose)
    return K, poses


def convert_image_to_int16_array(image: Image.Image) -> np.ndarray:
    # Convert image to numpy array
    image_array = np.array(image)

    # Reshape image array to 2D array of RGB colors
    reshaped_array = image_array.reshape(-1, 3)

    # Get unique colors and map them to unique integer IDs starting from 0
    unique_colors, ids = np.unique(reshaped_array, axis=0, return_inverse=True)

    # Convert the image array to the corresponding ID array
    int16_array = ids.reshape(image_array.shape[:2]).astype(np.int16)

    return int16_array


@dataclass
class MessyRoomDatasetToNerfstudioDataset(BaseConverterToNerfstudioDataset):
    num_downscales: int = 0
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3 will downscale the
       images by 2x, 4x, and 8x."""
    gpu: bool = True
    """If True, use GPU."""
    metadata: str = 'metadata.json'
    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Portion of the image to crop. All values should be in [0,1]. (top, bottom, left, right)"""

    same_dimensions: bool = True
    """Whether to assume all images are same dimensions and so to use fast downscaling with no autorotation."""

    @property
    def rgb_image_path(self) -> Path:
        return self.data / 'color'

    @property
    def visualized_semantic_path(self) -> Path:
        return self.data / 'visualized_semantic'

    @property
    def visualized_instance_path(self) -> Path:
        return self.data / 'visualized_instance_new'

    @property
    def metadata_path(self) -> Path:
        return self.data / self.metadata

    @property
    def output_path(self) -> Path:
        return self.output_dir / 'messy_room'

    @property
    def output_image_path(self) -> Path:
        return self.output_path / 'images'

    @property
    def output_instance_path(self) -> Path:
        return self.output_path / 'semantics'

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""
        summary_log = []
        image_rename_map: Optional[dict[str, str]] = None

        image_rename_map_paths = process_data_utils.copy_images(
            self.rgb_image_path,
            image_dir=self.output_image_path,
            crop_factor=self.crop_factor,
            image_prefix="frame_train_" if self.eval_data is not None else "frame_",
            verbose=self.verbose,
            num_downscales=self.num_downscales,
            same_dimensions=self.same_dimensions,
            keep_image_dir=False,
        )

        image_rename_map = dict(
            (a.relative_to(self.data).as_posix(), b.name) for a, b in image_rename_map_paths.items()
        )
        num_frames_rgb = len(image_rename_map)
        summary_log.append(f"Starting with {num_frames_rgb} images")

        self.output_instance_path.mkdir(exist_ok=True)

        for instance_path in sorted(glob.glob((self.visualized_instance_path / '*').__str__())):
            image = Image.open(instance_path)
            np_image = convert_image_to_int16_array(image)


            image_id = int(Path(instance_path).name.split('.')[0]) + 1
            image_suffix = Path(instance_path).name.split('.')[-1]
            output_instance_path = self.output_instance_path / ('frame_{:05d}.{}'.format(image_id, image_suffix))
            gray_image = Image.fromarray(np_image)
            gray_image.save(output_instance_path)

        # image_rename_map_paths_instance = process_data_utils.copy_images(
        #    self.visualized_instance_path,
        #    image_dir=self.output_instance_path,
        #    crop_factor=self.crop_factor,
        #    image_prefix="frame_train_" if self.eval_data is not None else "frame_",
        #    verbose=self.verbose,
        #    num_downscales=self.num_downscales,
        #    same_dimensions=self.same_dimensions,
        #    keep_image_dir=False,
        # )

        # image_rename_map_instance = dict(
        #    (a.relative_to(self.data).as_posix(), b.name) for a, b in image_rename_map_paths_instance.items()
        # )
        # num_frames_instance = len(image_rename_map_instance)
        # summary_log.append(f"Starting with {num_frames_instance} semantic images")

        with open(self.metadata_path) as f:
            self.metadata_file = json.load(f)

        img_h, img_w = np.array(Image.open(next(iter(image_rename_map_paths)))).shape[:2]

        K, P = read_cameras(meta=self.metadata_file, H=img_h, W=img_w)

        if True:
            visualizer = CameraPoseVisualizer([-10, 10], [-10, 10], [0, 10])

            for p in P:
                visualizer.extrinsic2pyramid(p, 'c', 1)
            visualizer.show()

        self.transform = {
            "camera_angle_x": 2 * np.arctan(img_w / K[0, 0]),
            "camera_angle_y": 2 * np.arctan(img_h / K[1, 1]),
            "fl_x": K[0, 0],
            "fl_y": K[1, 1],
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "cx": K[0, 2],
            "cy": K[1, 2],
            "w": img_w,
            "h": img_h,
            "aabb_scale": 16,
            'frames': [],
            "semantics": [
                "stuff",
                "test"
            ]
        }
        for idx, image_path in enumerate(image_rename_map.keys()):
            element = {
                "file_path": "images/{}".format(image_rename_map[image_path]),
                "transform_matrix": [
                    P[idx][0].tolist(), P[idx][1].tolist(), P[idx][2].tolist(), P[idx][3].tolist()
                ],
                "semantic_path": "semantics/{}".format(image_rename_map[image_path])
            }

            self.transform['frames'].append(element)

        json.dump(self.transform, open(self.output_path / 'transforms.json', 'w+'), indent=4)

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.log(summary)
